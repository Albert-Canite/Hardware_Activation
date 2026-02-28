import argparse
import copy
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


class Quantizer8Bit:
    """Symmetric 8-bit quantizer for values in [-1, 1]."""

    def __init__(self, min_val: float = -1.0, max_val: float = 1.0, levels: int = 256):
        self.min_val = min_val
        self.max_val = max_val
        self.levels = levels

    def quantize_to_index(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, self.min_val, self.max_val)
        scale = (self.levels - 1) / (self.max_val - self.min_val)
        idx = torch.round((x - self.min_val) * scale)
        return idx.long()

    def dequantize_from_index(self, idx: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        scale = (self.max_val - self.min_val) / (self.levels - 1)
        return idx.to(dtype=dtype, device=device) * scale + self.min_val


class QuantizedReLU(nn.Module):
    """ReLU with 8-bit input/output quantization over [-1, 1]."""

    def __init__(self, quantizer: Quantizer8Bit):
        super().__init__()
        self.quantizer = quantizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx_in = self.quantizer.quantize_to_index(x)
        x_qin = self.quantizer.dequantize_from_index(idx_in, x.dtype, x.device)
        y = torch.relu(x_qin)
        idx_out = self.quantizer.quantize_to_index(y)
        y_qout = self.quantizer.dequantize_from_index(idx_out, x.dtype, x.device)
        return y_qout


class LUTActivation(nn.Module):
    """8-bit LUT activation over [-1, 1]. Input/output both quantized to 8-bit."""

    def __init__(self, quantizer: Quantizer8Bit, lut_values_8bit: np.ndarray):
        super().__init__()
        self.quantizer = quantizer
        if lut_values_8bit.shape[0] != quantizer.levels:
            raise ValueError("LUT length must equal quantization levels.")
        self.register_buffer("lut", torch.tensor(lut_values_8bit, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx_in = self.quantizer.quantize_to_index(x)
        y = self.lut[idx_in]
        idx_out = self.quantizer.quantize_to_index(y)
        y_qout = self.quantizer.dequantize_from_index(idx_out, x.dtype, x.device)
        return y_qout


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_and_interpolate_lut(lut_path: Path, levels: int = 256) -> Tuple[np.ndarray, pd.DataFrame]:
    df = pd.read_excel(lut_path, header=None)
    df = df.dropna(how="any")
    if df.shape[1] < 2:
        raise ValueError("LUT_ReLU.xlsx must have at least 2 columns: input, output.")

    x = df.iloc[:, 0].astype(float).to_numpy()
    y = df.iloc[:, 1].astype(float).to_numpy()

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    x_dense = np.linspace(-1.0, 1.0, levels)
    y_dense = np.interp(x_dense, x, y)
    y_dense = np.clip(y_dense, -1.0, 1.0)

    lut_df = pd.DataFrame({"input": x_dense, "output": y_dense})
    return y_dense.astype(np.float32), lut_df


def replace_relu_with_quantized(module: nn.Module, quantizer: Quantizer8Bit) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, QuantizedReLU(quantizer))
        else:
            replace_relu_with_quantized(child, quantizer)


def replace_quantized_with_lut(module: nn.Module, quantizer: Quantizer8Bit, lut_values_8bit: np.ndarray) -> None:
    for name, child in module.named_children():
        if isinstance(child, QuantizedReLU):
            setattr(module, name, LUTActivation(quantizer, lut_values_8bit))
        else:
            replace_quantized_with_lut(child, quantizer, lut_values_8bit)


def build_vgg11(num_classes: int, in_channels: int, quantizer: Quantizer8Bit) -> nn.Module:
    model = models.vgg11_bn(weights=None)
    if in_channels != 3:
        old_conv = model.features[0]
        model.features[0] = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    replace_relu_with_quantized(model, quantizer)
    return model


def get_dataloaders(dataset_name: str, batch_size: int, data_dir: Path) -> Tuple[DataLoader, DataLoader, int, int]:
    if dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_set = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
        return (
            DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
            DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True),
            10,
            1,
        )

    if dataset_name == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
        test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
        return (
            DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
            DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True),
            10,
            3,
        )

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


def train_dataset(
    dataset_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    data_dir: Path,
    output_dir: Path,
    quantizer: Quantizer8Bit,
    lut_values_8bit: np.ndarray,
    device: torch.device,
) -> Dict[str, float]:
    train_loader, test_loader, num_classes, in_channels = get_dataloaders(dataset_name, batch_size, data_dir)

    model = build_vgg11(num_classes=num_classes, in_channels=in_channels, quantizer=quantizer).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_acc = evaluate(model, test_loader, device)
        print(f"[{dataset_name.upper()}] Epoch {epoch}/{epochs} | loss={train_loss:.4f} | test_acc={test_acc:.2f}%")

    model_path = output_dir / f"vgg11_{dataset_name}_quant_relu.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved trained model: {model_path}")

    acc_standard = evaluate(model, test_loader, device)

    lut_model = copy.deepcopy(model)
    replace_quantized_with_lut(lut_model, quantizer, lut_values_8bit)
    acc_lut = evaluate(lut_model, test_loader, device)

    return {
        "standard_relu_accuracy": acc_standard,
        "lut_relu_accuracy": acc_lut,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VGG-11 on MNIST/CIFAR10 with 8-bit quantized ReLU and compare LUT ReLU inference.")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs per dataset.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs"))
    parser.add_argument("--lut-path", type=Path, default=Path("./LUT_ReLU.xlsx"))
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    lut_values_8bit, lut_df = load_and_interpolate_lut(args.lut_path, levels=256)
    lut_8bit_path = args.output_dir / "LUT_ReLU_8bit_interpolated.csv"
    lut_df.to_csv(lut_8bit_path, index=False)
    print(f"Saved interpolated 8-bit LUT: {lut_8bit_path}")

    quantizer = Quantizer8Bit(min_val=-1.0, max_val=1.0, levels=256)

    results = {}
    for dataset_name in ["mnist", "cifar10"]:
        results[dataset_name] = train_dataset(
            dataset_name=dataset_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            quantizer=quantizer,
            lut_values_8bit=lut_values_8bit,
            device=device,
        )

    print("\n===== Accuracy Comparison =====")
    for dataset_name in ["mnist", "cifar10"]:
        acc_standard = results[dataset_name]["standard_relu_accuracy"]
        acc_lut = results[dataset_name]["lut_relu_accuracy"]
        print(
            f"{dataset_name.upper()} | Standard ReLU Acc: {acc_standard:.2f}% | LUT ReLU Acc: {acc_lut:.2f}%"
        )


if __name__ == "__main__":
    main()
