import argparse
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class FakeQuantizer(nn.Module):
    def __init__(self, num_bits: int = 8, min_val: float = -1.0, max_val: float = 1.0):
        super().__init__()
        self.num_bits = num_bits
        self.min_val = min_val
        self.max_val = max_val
        self.levels = 2**num_bits - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_clamped = torch.clamp(x, self.min_val, self.max_val)
        scale = (self.max_val - self.min_val) / self.levels
        q = torch.round((x_clamped - self.min_val) / scale)
        x_q = q * scale + self.min_val
        # Straight-through estimator
        return x + (x_q - x).detach()


class QuantizedReLU(nn.Module):
    """
    每次激活都做两次 8-bit 映射：
    1) ReLU 输入先映射到 [-1, 1] 的 8-bit 离散值
    2) ReLU 输出再映射到 [-1, 1] 的 8-bit 离散值
    """

    def __init__(self):
        super().__init__()
        self.pre_quant = FakeQuantizer(num_bits=8, min_val=-1.0, max_val=1.0)
        self.relu = nn.ReLU(inplace=False)
        self.post_quant = FakeQuantizer(num_bits=8, min_val=-1.0, max_val=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_quant(x)
        x = self.relu(x)
        x = self.post_quant(x)
        return x


class QuantConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_quant = FakeQuantizer(num_bits=8, min_val=-1.0, max_val=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_weight = self.weight_quant(self.weight)
        return self._conv_forward(x, q_weight, self.bias)


class QuantLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_quant = FakeQuantizer(num_bits=8, min_val=-1.0, max_val=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_weight = self.weight_quant(self.weight)
        return nn.functional.linear(x, q_weight, self.bias)


class QuantizedVGG11(nn.Module):
    """VGG-11 结构（A 配置），用于 MNIST (1x32x32) + 8-bit fake quant 训练。"""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.input_quant = FakeQuantizer(num_bits=8, min_val=-1.0, max_val=1.0)

        cfg = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]
        layers = []
        in_channels = 1
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(QuantConv2d(in_channels, v, kernel_size=3, padding=1))
                layers.append(QuantizedReLU())
                in_channels = v

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            QuantLinear(512 * 1 * 1, 4096),
            QuantizedReLU(),
            nn.Dropout(p=0.5),
            QuantLinear(4096, 4096),
            QuantizedReLU(),
            nn.Dropout(p=0.5),
            QuantLinear(4096, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_quant(x)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


@dataclass
class EpochResult:
    loss: float
    acc: float


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    return EpochResult(loss=total_loss / total_samples, acc=total_correct / total_samples)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    return EpochResult(loss=total_loss / total_samples, acc=total_correct / total_samples)


def build_dataloaders(data_dir: str, batch_size: int, num_workers: int):
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    train_set = datasets.MNIST(root=data_dir, train=True, transform=transform, download=True)
    val_set = datasets.MNIST(root=data_dir, train=False, transform=transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="Train VGG11 on MNIST with 8-bit fake quantization")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--save-dir", type=str, default="./artifacts")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.save_dir, exist_ok=True)

    train_loader, val_loader = build_dataloaders(args.data_dir, args.batch_size, args.num_workers)

    model = QuantizedVGG11(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_result = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_result = validate(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"Train Loss: {train_result.loss:.4f} Train Acc: {train_result.acc:.4f} | "
            f"Val Loss: {val_result.loss:.4f} Val Acc: {val_result.acc:.4f}"
        )

        if val_result.acc > best_val_acc:
            best_val_acc = val_result.acc
            best_path = os.path.join(args.save_dir, "vgg11_mnist_quant8_best.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": best_val_acc,
                    "config": vars(args),
                },
                best_path,
            )

    final_path = os.path.join(args.save_dir, "vgg11_mnist_quant8_last.pth")
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": best_val_acc,
            "config": vars(args),
        },
        final_path,
    )

    print(f"Training done. Best model saved to: {os.path.join(args.save_dir, 'vgg11_mnist_quant8_best.pth')}")
    print(f"Last model saved to: {final_path}")


if __name__ == "__main__":
    main()
