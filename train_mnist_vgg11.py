import argparse
import os
import subprocess
import sys
from pathlib import Path

from env_check import ensure_runtime_compatibility

PROJECT_DIR = Path(__file__).resolve().parent
WINDOWS_ACCESS_VIOLATION_CODES = {-1073741819, 3221225477, -1}


def _train_worker(args):
    ensure_runtime_compatibility()

    if args.device == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

    import torch
    import torch.nn as nn
    import torch.optim as optim

    from vgg11_data_utils import build_vgg11_quantrelu, evaluate, get_mnist_loaders, set_seed

    use_cuda = args.device == "cuda"
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available or CUDA runtime is broken.")
    device = torch.device("cuda" if use_cuda else "cpu")
    set_seed(args.seed, use_cuda=use_cuda)

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
    torch.save({"state_dict": model.state_dict()}, args.save_path)
    print(f"Saved MNIST VGG11 quantized-ReLU model to: {args.save_path}")


def _run_with_crash_guard():
    cmd = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--_worker"]
    ret = subprocess.run(cmd).returncode
    if ret in WINDOWS_ACCESS_VIOLATION_CODES:
        print("\n[CRASH GUARD] Worker exited with code indicating Windows access violation (0xC0000005).")
        print(f"[CRASH GUARD] Raw worker return code: {ret}")
        print("[CRASH GUARD] This usually means native runtime conflict (PyTorch/CUDA/MKL DLL), not Python logic.")
        print("[CRASH GUARD] Please try in your PyONN env:")
        print("  1) conda install -y cpuonly pytorch torchvision pytorch-cuda=none -c pytorch")
        print("  2) or reinstall a matching CUDA stack for torch 2.2.2 + cu121")
        print("  3) set IDE Run configuration env: CUDA_VISIBLE_DEVICES=-1")
        return 1
    if ret != 0:
        print(f"[CRASH GUARD] Worker failed with non-zero exit code: {ret}")
        print("[CRASH GUARD] If this is on Windows IDE, check interpreter path and try running from terminal once.")
    return ret


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train VGG11 on MNIST with 8-bit QuantReLU")
    parser.add_argument("--data-root", type=str, default=str(PROJECT_DIR / "data"))
    parser.add_argument("--save-path", type=str, default=str(PROJECT_DIR / "checkpoints" / "mnist_vgg11_quantrelu8.pth"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()
    if args._worker:
        _train_worker(args)
    else:
        raise SystemExit(_run_with_crash_guard())
