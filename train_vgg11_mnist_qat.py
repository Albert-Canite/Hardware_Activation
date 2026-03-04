import argparse
import copy
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


class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


class ClampSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, qmin: float, qmax: float) -> torch.Tensor:
        return torch.clamp(x, qmin, qmax)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None, None


def scale_to_signed_unit(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0 - 1.0


class Uniform8BitQuantizer(nn.Module):
    def __init__(self, qmin: float, qmax: float, levels: int = 256):
        super().__init__()
        self.qmin = qmin
        self.qmax = qmax
        self.step = (qmax - qmin) / (levels - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # STE clamp/round implemented with detach() so traced model can be exported.
        x_clamped = torch.clamp(x, self.qmin, self.qmax)
        x = x + (x_clamped - x).detach()
        x = (x - self.qmin) / self.step
        x_rounded = torch.round(x)
        x = x + (x_rounded - x).detach()
        return x * self.step + self.qmin


class HardwareLUTReLUSim(nn.Module):
    """
    Hardware activation simulation with adaptive normalization.

    Input is adaptively normalized to avoid saturation, then quantized to [-1,1] 8-bit,
    passed through ReLU, quantized to [0,1] 8-bit, and rescaled back.
    """

    def __init__(self, ema_momentum: float = 0.95, activation_bits: int = 8, adaptive_scale: bool = True):
        super().__init__()
        levels = 2 ** activation_bits
        self.input_quant = Uniform8BitQuantizer(-1.0, 1.0, levels)
        self.relu = nn.ReLU(inplace=False)
        self.output_quant = Uniform8BitQuantizer(0.0, 1.0, levels)
        self.ema_momentum = ema_momentum
        self.activation_bits = activation_bits
        self.adaptive_scale = adaptive_scale
        self.register_buffer("running_input_absmax", torch.tensor(1.0))
        self.register_buffer("running_output_absmax", torch.tensor(1.0))
        # Optional external LUT override used for hardware-path inference without replacing module objects.
        self.register_buffer("external_lut_levels", torch.empty(0), persistent=False)

    def _update_running_absmax(self, x: torch.Tensor, buffer_name: str):
        with torch.no_grad():
            current = x.detach().abs().amax().clamp(min=1e-3)
            buf = getattr(self, buffer_name)
            buf.mul_(self.ema_momentum).add_(current * (1.0 - self.ema_momentum))

    def set_external_lut_levels(self, lut_levels: torch.Tensor):
        if lut_levels.numel() < 2:
            raise ValueError("external LUT levels must contain at least two points")
        self.external_lut_levels = lut_levels.detach().float().to(self.external_lut_levels.device)

    def clear_external_lut_levels(self):
        self.external_lut_levels = torch.empty(0, device=self.external_lut_levels.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        was_quantized = x.is_quantized
        q_dtype = None
        q_scale = None
        q_zero_point = None
        if was_quantized:
            qscheme = x.qscheme()
            if qscheme in (torch.per_tensor_affine, torch.per_tensor_symmetric):
                q_dtype = x.dtype
                q_scale = float(x.q_scale())
                q_zero_point = int(x.q_zero_point())
            x = x.dequantize()

        if self.adaptive_scale and self.training:
            self._update_running_absmax(x, "running_input_absmax")
        input_scale = self.running_input_absmax.clamp(min=1e-3) if self.adaptive_scale else x.new_tensor(1.0)

        x_norm = x / input_scale
        xq = self.input_quant(x_norm)
        if self.external_lut_levels.numel() >= 2:
            levels = int(self.external_lut_levels.numel())
            x_clamped = torch.clamp(xq, -1.0, 1.0)
            idx = torch.round((x_clamped + 1.0) * 0.5 * (levels - 1)).to(torch.long)
            idx = torch.clamp(idx, 0, levels - 1)
            yq = self.external_lut_levels[idx]
        else:
            y = self.relu(xq)
            yq = self.output_quant(y)

        if self.adaptive_scale and self.training:
            self._update_running_absmax(yq, "running_output_absmax")
        output_scale = self.running_output_absmax.clamp(min=1e-3) if self.adaptive_scale else x.new_tensor(1.0)
        out = yq * output_scale
        if was_quantized and q_dtype is not None and q_scale is not None and q_zero_point is not None:
            out = torch.quantize_per_tensor(out, scale=q_scale, zero_point=q_zero_point, dtype=q_dtype)
        return out


class QuantizableVGG11MNIST(nn.Module):
    def __init__(self, activation_bits: int = 8, adaptive_activation_scale: bool = True):
        super().__init__()
        self.activation_bits = activation_bits
        self.adaptive_activation_scale = adaptive_activation_scale
        base_model = models.vgg11_bn(weights=None)

        base_model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        base_model.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        base_model.classifier[6] = nn.Linear(4096, 10)

        base_model.features = self._replace_relu(base_model.features)
        base_model.classifier = self._replace_relu(base_model.classifier)

        self.quant = torch.ao.quantization.QuantStub()
        self.vgg = base_model
        self.dequant = torch.ao.quantization.DeQuantStub()

    def _replace_relu(self, module: nn.Module) -> nn.Module:
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU):
                setattr(
                    module,
                    name,
                    HardwareLUTReLUSim(
                        activation_bits=self.activation_bits,
                        adaptive_scale=self.adaptive_activation_scale,
                    ),
                )
            else:
                self._replace_relu(child)
        return module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.vgg(x)
        x = self.dequant(x)
        return x


def export_hardware_relu_lut(output_path: str, activation_bits: int = 8):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    levels = 2 ** activation_bits
    quant_in = Uniform8BitQuantizer(-1.0, 1.0, levels)
    quant_out = Uniform8BitQuantizer(0.0, 1.0, levels)
    xs = torch.linspace(-1.0, 1.0, steps=levels)
    with torch.no_grad():
        xq = quant_in(xs)
        yq = quant_out(torch.relu(xq))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("index,input,output\n")
        for idx, (xv, yv) in enumerate(zip(xq.tolist(), yq.tolist())):
            f.write(f"{idx},{xv:.8f},{yv:.8f}\n")


@dataclass
class EpochStats:
    loss: float
    accuracy: float


def train_one_epoch(model, loader, optimizer, criterion, device, max_batches=None):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * targets.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return EpochStats(loss=running_loss / total, accuracy=100.0 * correct / total)


def evaluate(model, loader, criterion, device, max_batches=None):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return EpochStats(loss=running_loss / total, accuracy=100.0 * correct / total)


def set_fake_quant_enabled(model: nn.Module, enabled: bool):
    if enabled:
        model.apply(enable_fake_quant)
    else:
        model.apply(disable_fake_quant)


def main():
    parser = argparse.ArgumentParser(description="Train VGG-11 on MNIST with hardware-like LUT ReLU and bit-aligned QAT")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./artifacts")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--qat-start-epoch", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument("--activation-bits", type=int, default=8, help="Activation quantization bit-width for hardware LUT simulation")
    parser.add_argument(
        "--adaptive-activation-scale",
        dest="adaptive_activation_scale",
        action="store_true",
        help="Enable adaptive per-layer input/output scale estimation for LUT-domain normalization (default: enabled).",
    )
    parser.add_argument(
        "--no-adaptive-activation-scale",
        dest="adaptive_activation_scale",
        action="store_false",
        help="Disable adaptive scaling and force fixed LUT-domain scale=1.0.",
    )
    parser.set_defaults(adaptive_activation_scale=True)
    args = parser.parse_args()

    if args.activation_bits < 2:
        raise ValueError("--activation-bits must be >= 2")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(scale_to_signed_unit),
    ])
    train_set = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=args.data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    model = QuantizableVGG11MNIST(
        activation_bits=args.activation_bits,
        adaptive_activation_scale=args.adaptive_activation_scale,
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
    best_acc = 0.0

    print(f"Using device: {device}")
    print(
        f"Hardware ReLU simulation: input [-1,1] {args.activation_bits}-bit -> ReLU -> "
        f"output [0,1] {args.activation_bits}-bit "
        f"({'adaptive input/output scales' if args.adaptive_activation_scale else 'fixed scale=1'})"
    )
    print(f"QAT fake quant starts at epoch {args.qat_start_epoch} (after first {args.qat_start_epoch - 1} epochs)")

    for epoch in range(1, args.epochs + 1):
        set_fake_quant_enabled(model, enabled=(epoch >= args.qat_start_epoch))

        train_stats = train_one_epoch(model, train_loader, optimizer, criterion, device, args.limit_train_batches)
        val_stats = evaluate(model, test_loader, criterion, device, args.limit_val_batches)
        scheduler.step()

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_stats.loss:.4f}, Train Acc: {train_stats.accuracy:.2f}% | "
            f"Val Loss: {val_stats.loss:.4f}, Val Acc: {val_stats.accuracy:.2f}%"
        )

        if epoch >= args.qat_start_epoch and val_stats.accuracy > best_acc:
            best_acc = val_stats.accuracy
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"vgg11_mnist_qat_best_{args.activation_bits}bit.pth"))

    qat_path = os.path.join(args.output_dir, f"vgg11_mnist_qat_final_{args.activation_bits}bit.pth")
    torch.save(model.state_dict(), qat_path)

    model_cpu = copy.deepcopy(model).to("cpu").eval()
    quantized_model = convert(model_cpu, inplace=False)
    # Use trace instead of script because custom autograd STE ops are not script-exportable.
    example_input = torch.randn(1, 1, 32, 32)
    traced = torch.jit.trace(quantized_model, example_input)
    int8_path = os.path.join(args.output_dir, f"vgg11_mnist_int8_traced_{args.activation_bits}bit.pt")
    traced.save(int8_path)

    lut_path = os.path.join(args.output_dir, f"hardware_relu_lut_{args.activation_bits}bit.csv")
    export_hardware_relu_lut(lut_path, activation_bits=args.activation_bits)

    print("Training complete.")
    print(f"Best Val Acc: {best_acc:.2f}%")
    print(f"Saved QAT model: {qat_path}")
    print(f"Saved INT8 model: {int8_path}")
    print(f"Saved hardware LUT: {lut_path}")


if __name__ == "__main__":
    main()
