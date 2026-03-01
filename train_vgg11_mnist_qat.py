import argparse
import copy
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.ao.quantization import QConfig, convert, prepare_qat
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.observer import (
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
)
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
        x = ClampSTE.apply(x, self.qmin, self.qmax)
        x = (x - self.qmin) / self.step
        x = RoundSTE.apply(x)
        return x * self.step + self.qmin


class HardwareLUTReLUSim(nn.Module):
    """
    Hardware activation simulation:
    - input quantized to [-1, 1] with 8-bit (256 points)
    - ReLU
    - output quantized to [0, 1] with 8-bit (256 points)
    """

    def __init__(self):
        super().__init__()
        self.input_quant = Uniform8BitQuantizer(-1.0, 1.0, 256)
        self.relu = nn.ReLU(inplace=False)
        self.output_quant = Uniform8BitQuantizer(0.0, 1.0, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_quant(self.relu(self.input_quant(x)))


class QuantizableVGG11MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = models.vgg11(weights=None)

        # standard vgg11 adapted for MNIST
        base_model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        base_model.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        base_model.classifier[6] = nn.Linear(4096, 10)

        # replace every ReLU with hardware-like simulated activation
        base_model.features = self._replace_relu(base_model.features)
        base_model.classifier = self._replace_relu(base_model.classifier)

        self.quant = torch.ao.quantization.QuantStub()
        self.vgg = base_model
        self.dequant = torch.ao.quantization.DeQuantStub()

    @staticmethod
    def _replace_relu(module: nn.Module) -> nn.Module:
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU):
                setattr(module, name, HardwareLUTReLUSim())
            else:
                QuantizableVGG11MNIST._replace_relu(child)
        return module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.vgg(x)
        x = self.dequant(x)
        return x


def export_hardware_relu_lut(output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    quant_in = Uniform8BitQuantizer(-1.0, 1.0, 256)
    quant_out = Uniform8BitQuantizer(0.0, 1.0, 256)

    xs = torch.linspace(-1.0, 1.0, steps=256)
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
    running_loss = 0.0
    correct = 0
    total = 0

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
    running_loss = 0.0
    correct = 0
    total = 0

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


def main():
    parser = argparse.ArgumentParser(description="Train VGG-11 on MNIST with hardware-like 8-bit ReLU and QAT")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./artifacts")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    args = parser.parse_args()

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

    model = QuantizableVGG11MNIST().to(device)
    model.qconfig = QConfig(
        activation=FakeQuantize.with_args(
            observer=MovingAverageMinMaxObserver,
            quant_min=0,
            quant_max=255,
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine,
        ),
        weight=FakeQuantize.with_args(
            observer=MovingAveragePerChannelMinMaxObserver,
            quant_min=-128,
            quant_max=127,
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
    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, criterion, device, args.limit_train_batches)
        val_stats = evaluate(model, test_loader, criterion, device, args.limit_val_batches)
        scheduler.step()

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_stats.loss:.4f}, Train Acc: {train_stats.accuracy:.2f}% | "
            f"Val Loss: {val_stats.loss:.4f}, Val Acc: {val_stats.accuracy:.2f}%"
        )

        if val_stats.accuracy > best_acc:
            best_acc = val_stats.accuracy
            torch.save(model.state_dict(), os.path.join(args.output_dir, "vgg11_mnist_qat_best.pth"))

    qat_path = os.path.join(args.output_dir, "vgg11_mnist_qat_final.pth")
    torch.save(model.state_dict(), qat_path)

    model_cpu = copy.deepcopy(model).to("cpu").eval()
    quantized_model = convert(model_cpu, inplace=False)
    scripted = torch.jit.script(quantized_model)
    int8_path = os.path.join(args.output_dir, "vgg11_mnist_int8_scripted.pt")
    scripted.save(int8_path)

    lut_path = os.path.join(args.output_dir, "hardware_relu_lut.csv")
    export_hardware_relu_lut(lut_path)

    print("Training complete.")
    print(f"Best Val Acc: {best_acc:.2f}%")
    print(f"Saved QAT model: {qat_path}")
    print(f"Saved INT8 model: {int8_path}")
    print(f"Saved hardware LUT: {lut_path}")


if __name__ == "__main__":
    main()
