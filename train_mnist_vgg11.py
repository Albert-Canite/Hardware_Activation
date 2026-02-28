import argparse
import os
from pathlib import Path

from env_check import ensure_runtime_compatibility

PROJECT_DIR = Path(__file__).resolve().parent


def train(args):
    ensure_runtime_compatibility()

    import torch
    import torch.nn as nn
    import torch.optim as optim

    from vgg11_data_utils import build_vgg11_quantrelu, evaluate, get_mnist_loaders, set_seed

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_mnist_loaders(batch_size=args.batch_size, data_root=args.data_root)
    model = build_vgg11_quantrelu(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)
        val_acc = evaluate(model, test_loader, device)
        print(f"[MNIST] Epoch {epoch}/{args.epochs} | loss={train_loss:.4f} | val_acc={val_acc * 100:.2f}%")

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    import torch as _torch

    _torch.save({"state_dict": model.state_dict()}, args.save_path)
    print(f"Saved MNIST VGG11 quantized-ReLU model to: {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VGG11 on MNIST with 8-bit QuantReLU")
    parser.add_argument("--data-root", type=str, default=str(PROJECT_DIR / "data"))
    parser.add_argument("--save-path", type=str, default=str(PROJECT_DIR / "checkpoints" / "mnist_vgg11_quantrelu8.pth"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)
