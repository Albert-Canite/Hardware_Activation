import argparse
import copy
import csv
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.ao.quantization import QConfig, convert, disable_fake_quant, enable_fake_quant, prepare_qat
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.observer import MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from train_vgg11_mnist_qat import QuantizableVGG11MNIST


def scale_to_signed_unit(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0 - 1.0


class UniformQuantizerSTE(nn.Module):
    def __init__(self, qmin: float, qmax: float, levels: int):
        super().__init__()
        if levels < 2:
            raise ValueError("levels must be >= 2")
        self.qmin = qmin
        self.qmax = qmax
        self.levels = levels
        self.step = (qmax - qmin) / (levels - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_clamped = torch.clamp(x, self.qmin, self.qmax)
        x = x + (x_clamped - x).detach()
        q = (x - self.qmin) / self.step
        q_rounded = torch.round(q)
        q = q + (q_rounded - q).detach()
        return q * self.step + self.qmin


def load_lut_points_from_csv(path: str):
    xs, ys = [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"LUT csv has no header: {path}")
        lowered = {name.strip().lower(): name for name in reader.fieldnames if name is not None}
        x_col = lowered.get("input")
        y_col = lowered.get("output")
        if x_col is None or y_col is None:
            cols = [c for c in reader.fieldnames if c and c.strip()]
            if len(cols) < 2:
                raise RuntimeError(f"LUT csv must contain at least two columns: {path}")
            x_col, y_col = cols[0], cols[1]
        for row in reader:
            try:
                xs.append(float(row[x_col]))
                ys.append(float(row[y_col]))
            except (ValueError, TypeError, KeyError):
                continue
    if len(xs) < 2:
        raise RuntimeError("LUT csv needs at least 2 numeric rows.")
    x = torch.tensor(xs, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.float32)
    order = torch.argsort(x)
    return x[order], y[order]


def interpolate_lut_levels(lut_x: torch.Tensor, lut_y: torch.Tensor, levels: int):
    x_target = torch.linspace(-1.0, 1.0, steps=levels)
    idx = torch.searchsorted(lut_x, x_target, right=False)
    idx = torch.clamp(idx, 1, lut_x.numel() - 1)
    x0 = lut_x[idx - 1]
    x1 = lut_x[idx]
    y0 = lut_y[idx - 1]
    y1 = lut_y[idx]
    t = (x_target - x0) / (x1 - x0 + 1e-12)
    return y0 + t * (y1 - y0)


class HardwareMeasuredLUTActivation(nn.Module):
    def __init__(self, lut_levels: torch.Tensor, ema_momentum: float = 0.95, adaptive_scale: bool = True):
        super().__init__()
        self.levels = int(lut_levels.numel())
        self.input_quant = UniformQuantizerSTE(-1.0, 1.0, self.levels)
        self.adaptive_scale = adaptive_scale
        self.ema_momentum = ema_momentum
        self.register_buffer("lut_levels", lut_levels.float().contiguous())
        self.register_buffer("running_input_absmax", torch.tensor(1.0))
        self.register_buffer("running_output_absmax", torch.tensor(1.0))

    def _update(self, x: torch.Tensor, name: str):
        with torch.no_grad():
            v = x.detach().abs().amax().clamp(min=1e-3)
            buf = getattr(self, name)
            buf.mul_(self.ema_momentum).add_(v * (1.0 - self.ema_momentum))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        was_quantized = x.is_quantized
        q_dtype, q_scale, q_zero = None, None, None
        if was_quantized:
            qscheme = x.qscheme()
            if qscheme in (torch.per_tensor_affine, torch.per_tensor_symmetric):
                q_dtype = x.dtype
                q_scale = float(x.q_scale())
                q_zero = int(x.q_zero_point())
            x = x.dequantize()

        if self.adaptive_scale and self.training:
            self._update(x, "running_input_absmax")
        in_scale = self.running_input_absmax.clamp(min=1e-3) if self.adaptive_scale else x.new_tensor(1.0)
        x_norm = x / in_scale
        xq = self.input_quant(x_norm)

        idx = torch.round((torch.clamp(xq, -1.0, 1.0) + 1.0) * 0.5 * (self.levels - 1)).to(torch.long)
        idx = torch.clamp(idx, 0, self.levels - 1)
        y_lut = self.lut_levels[idx]
        # STE: forward uses LUT values, backward approximates identity around normalized input.
        y_norm = x_norm + (y_lut - x_norm).detach()

        if self.adaptive_scale and self.training:
            self._update(y_norm, "running_output_absmax")
        out_scale = self.running_output_absmax.clamp(min=1e-3) if self.adaptive_scale else x.new_tensor(1.0)
        out = y_norm * out_scale

        if was_quantized and q_dtype is not None:
            out = torch.quantize_per_tensor(out, scale=q_scale, zero_point=q_zero, dtype=q_dtype)
        return out


class QuantizableVGG11MNISTHwLUT(nn.Module):
    def __init__(self, lut_levels: torch.Tensor, adaptive_activation_scale: bool = True):
        super().__init__()
        base = models.vgg11_bn(weights=None)
        base.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        base.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        base.classifier[6] = nn.Linear(4096, 10)
        self.lut_levels = lut_levels
        self.adaptive_activation_scale = adaptive_activation_scale
        base.features = self._replace_relu(base.features)
        base.classifier = self._replace_relu(base.classifier)
        self.quant = torch.ao.quantization.QuantStub()
        self.vgg = base
        self.dequant = torch.ao.quantization.DeQuantStub()

    def _replace_relu(self, module: nn.Module):
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU):
                setattr(module, name, HardwareMeasuredLUTActivation(self.lut_levels, adaptive_scale=self.adaptive_activation_scale))
            else:
                self._replace_relu(child)
        return module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.vgg(x)
        x = self.dequant(x)
        return x


@dataclass
class EpochStats:
    loss: float
    accuracy: float


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, epochs, log_interval=50, val_loader=None, val_probe_batches=0):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    step_logs = []
    for step, (inputs, targets) in enumerate(loader, start=1):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * targets.size(0)
        correct += outputs.argmax(dim=1).eq(targets).sum().item()
        total += targets.size(0)
        if log_interval > 0 and (step % log_interval == 0 or step == len(loader)):
            train_loss = running_loss / total
            train_acc = 100.0 * correct / total
            row = {
                "epoch": epoch,
                "step": step,
                "total_steps": len(loader),
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_probe_loss": "",
                "val_probe_acc": "",
                "val_probe_batches": val_probe_batches if val_probe_batches > 0 else "",
            }
            msg = (
                f"Epoch {epoch}/{epochs} Step {step}/{len(loader)} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%"
            )
            if val_loader is not None and val_probe_batches > 0:
                probe = evaluate(model, val_loader, criterion, device, max_batches=val_probe_batches)
                msg += f" | Val Probe Loss: {probe.loss:.4f}, Val Probe Acc: {probe.accuracy:.2f}% ({val_probe_batches} batches)"
                row["val_probe_loss"] = probe.loss
                row["val_probe_acc"] = probe.accuracy
            step_logs.append(row)
            print(msg)
    return EpochStats(running_loss / total, 100.0 * correct / total), step_logs


@torch.no_grad()
def evaluate(model, loader, criterion, device, max_batches=None):
    was_training = model.training
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for idx, (inputs, targets) in enumerate(loader):
        if max_batches is not None and idx >= max_batches:
            break
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        running_loss += loss.item() * targets.size(0)
        correct += outputs.argmax(dim=1).eq(targets).sum().item()
        total += targets.size(0)
    if was_training:
        model.train()
    return EpochStats(running_loss / total, 100.0 * correct / total)


def set_fake_quant_enabled(model: nn.Module, enabled: bool):
    if enabled:
        model.apply(enable_fake_quant)
    else:
        model.apply(disable_fake_quant)


def main():
    parser = argparse.ArgumentParser(description="Train VGG11 in two modes: ideal quantized-ReLU LUT or measured hardware LUT.")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./artifacts/hw_lut_4bit")
    parser.add_argument("--lut-csv", type=str, default="./LUT_ReLU.csv")
    parser.add_argument("--mode", type=str, choices=["ideal", "hardware"], default="hardware")
    parser.add_argument("--activation-bits", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--qat-start-epoch", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=50, help="Print training stats every N steps.")
    parser.add_argument("--val-probe-batches", type=int, default=0, help="If >0, run quick val probe every log interval.")
    parser.add_argument("--no-adaptive-activation-scale", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    levels = 2 ** args.activation_bits

    lut_x, lut_y = load_lut_points_from_csv(args.lut_csv)
    lut_levels = interpolate_lut_levels(lut_x, lut_y, levels=levels)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(scale_to_signed_unit),
    ])
    train_set = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=args.data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    if args.mode == "hardware":
        model = QuantizableVGG11MNISTHwLUT(
            lut_levels=lut_levels,
            adaptive_activation_scale=(not args.no_adaptive_activation_scale),
        ).to(device)
    else:
        model = QuantizableVGG11MNIST(
            activation_bits=args.activation_bits,
            adaptive_activation_scale=(not args.no_adaptive_activation_scale),
        ).to(device)

    act_qmin, act_qmax = 0, (2 ** args.activation_bits) - 1
    wt_qmin, wt_qmax = -(2 ** (args.activation_bits - 1)), (2 ** (args.activation_bits - 1)) - 1
    model.qconfig = QConfig(
        activation=FakeQuantize.with_args(
            observer=MovingAverageMinMaxObserver,
            quant_min=act_qmin,
            quant_max=act_qmax,
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine,
        ),
        weight=FakeQuantize.with_args(
            observer=MovingAveragePerChannelMinMaxObserver,
            quant_min=wt_qmin,
            quant_max=wt_qmax,
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
            ch_axis=0,
        ),
    )
    prepare_qat(model, inplace=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    os.makedirs(args.output_dir, exist_ok=True)
    step_metrics_csv = os.path.join(
        args.output_dir,
        f"train_step_metrics_{args.mode}_lut_{args.activation_bits}bit.csv",
    )
    epoch_metrics_csv = os.path.join(
        args.output_dir,
        f"train_epoch_metrics_{args.mode}_lut_{args.activation_bits}bit.csv",
    )
    with open(step_metrics_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "epoch",
            "step",
            "total_steps",
            "train_loss",
            "train_acc",
            "val_probe_loss",
            "val_probe_acc",
            "val_probe_batches",
        ])
    with open(epoch_metrics_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "epoch",
            "fake_quant_enabled",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "best_val_acc_so_far",
            "lr",
        ])

    if args.mode == "hardware":
        with open(os.path.join(args.output_dir, f"hardware_lut_interpolated_for_training_{args.activation_bits}bit.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["index", "input", "output"])
            x_levels = torch.linspace(-1.0, 1.0, steps=levels)
            for i, (xv, yv) in enumerate(zip(x_levels.tolist(), lut_levels.tolist())):
                w.writerow([i, f"{xv:.8f}", f"{yv:.8f}"])

    best_acc = 0.0
    print(f"Using device: {device}")
    print(f"Training mode: {args.mode}")
    print(f"Activation bits: {args.activation_bits} (levels={levels})")
    if args.mode == "hardware":
        print(f"LUT source: {args.lut_csv}")
    print(f"QAT fake quant starts at epoch {args.qat_start_epoch}")

    for epoch in range(1, args.epochs + 1):
        fake_quant_enabled = (epoch >= args.qat_start_epoch)
        set_fake_quant_enabled(model, enabled=fake_quant_enabled)
        train_stats, step_logs = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch=epoch,
            epochs=args.epochs,
            log_interval=args.log_interval,
            val_loader=test_loader,
            val_probe_batches=args.val_probe_batches,
        )
        with open(step_metrics_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for r in step_logs:
                w.writerow([
                    r["epoch"],
                    r["step"],
                    r["total_steps"],
                    f"{r['train_loss']:.6f}",
                    f"{r['train_acc']:.4f}",
                    "" if r["val_probe_loss"] == "" else f"{r['val_probe_loss']:.6f}",
                    "" if r["val_probe_acc"] == "" else f"{r['val_probe_acc']:.4f}",
                    r["val_probe_batches"],
                ])
        val_stats = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_stats.loss:.4f}, Train Acc: {train_stats.accuracy:.2f}% | "
            f"Val Loss: {val_stats.loss:.4f}, Val Acc: {val_stats.accuracy:.2f}%"
        )
        if epoch >= args.qat_start_epoch and val_stats.accuracy > best_acc:
            best_acc = val_stats.accuracy
            torch.save(
                model.state_dict(),
                os.path.join(args.output_dir, f"vgg11_mnist_qat_best_{args.mode}_lut_{args.activation_bits}bit.pth"),
            )
        with open(epoch_metrics_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                epoch,
                int(fake_quant_enabled),
                f"{train_stats.loss:.6f}",
                f"{train_stats.accuracy:.4f}",
                f"{val_stats.loss:.6f}",
                f"{val_stats.accuracy:.4f}",
                f"{best_acc:.4f}",
                f"{optimizer.param_groups[0]['lr']:.10f}",
            ])

    final_path = os.path.join(args.output_dir, f"vgg11_mnist_qat_final_{args.mode}_lut_{args.activation_bits}bit.pth")
    torch.save(model.state_dict(), final_path)

    model_cpu = copy.deepcopy(model).to("cpu").eval()
    quantized_model = convert(model_cpu, inplace=False)
    traced = torch.jit.trace(quantized_model, torch.randn(1, 1, 32, 32))
    traced_path = os.path.join(args.output_dir, f"vgg11_mnist_int8_traced_{args.mode}_lut_{args.activation_bits}bit.pt")
    traced.save(traced_path)

    print("Training complete.")
    print(f"Best Val Acc: {best_acc:.2f}%")
    print(f"Saved model: {final_path}")
    print(f"Saved traced model: {traced_path}")
    print(f"Saved step metrics: {step_metrics_csv}")
    print(f"Saved epoch metrics: {epoch_metrics_csv}")


if __name__ == "__main__":
    main()
