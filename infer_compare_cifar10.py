import argparse
import os
import subprocess
import sys
from pathlib import Path

from env_check import ensure_runtime_compatibility

PROJECT_DIR = Path(__file__).resolve().parent
WINDOWS_ACCESS_VIOLATION_CODES = {-1073741819, 3221225477, -1}


def _run_worker(args):
    ensure_runtime_compatibility()

    if args.device == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

    import torch

    from relu_quant_lut import (
        interpolate_lut_to_8bit,
        load_lut_from_excel,
        replace_quantrelu_with_lutrelu,
        save_lut_8bit_csv,
    )
    from vgg11_data_utils import build_vgg11_quantrelu, evaluate, get_cifar10_loaders

    use_cuda = args.device == "cuda"
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available or CUDA runtime is broken.")
    device = torch.device("cuda" if use_cuda else "cpu")

    _, test_loader = get_cifar10_loaders(batch_size=args.batch_size, data_root=args.data_root)
    ckpt = torch.load(args.model_path, map_location="cpu")

    model_standard = build_vgg11_quantrelu(num_classes=10)
    model_standard.load_state_dict(ckpt["state_dict"])
    model_standard = model_standard.to(device)
    acc_standard = evaluate(model_standard, test_loader, device)

    xs, ys = load_lut_from_excel(args.lut_xlsx)
    x_256, y_256 = interpolate_lut_to_8bit(xs, ys)
    os.makedirs(os.path.dirname(args.interp_lut_csv), exist_ok=True)
    save_lut_8bit_csv(x_256, y_256, args.interp_lut_csv)

    model_lut = build_vgg11_quantrelu(num_classes=10)
    model_lut.load_state_dict(ckpt["state_dict"])
    replace_quantrelu_with_lutrelu(model_lut, y_256)
    model_lut = model_lut.to(device)
    acc_lut = evaluate(model_lut, test_loader, device)

    print("[CIFAR10] Accuracy comparison")
    print(f"  Standard QuantReLU inference acc : {acc_standard * 100:.2f}%")
    print(f"  LUT-ReLU inference acc          : {acc_lut * 100:.2f}%")
    print(f"  Interpolated 8-bit LUT saved to : {args.interp_lut_csv}")


def _run_with_crash_guard():
    cmd = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--_worker"]
    ret = subprocess.run(cmd).returncode
    if ret in WINDOWS_ACCESS_VIOLATION_CODES:
        print("\n[CRASH GUARD] Worker exited with code indicating Windows access violation (0xC0000005).")
        print(f"[CRASH GUARD] Raw worker return code: {ret}")
        print("[CRASH GUARD] This usually means native runtime conflict (PyTorch/CUDA/MKL DLL), not Python logic.")
        return 1
    if ret != 0:
        print(f"[CRASH GUARD] Worker failed with non-zero exit code: {ret}")
        print("[CRASH GUARD] If this is on Windows IDE, check interpreter path and try running from terminal once.")
    return ret


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CIFAR-10 inference: standard vs LUT-ReLU")
    parser.add_argument("--data-root", type=str, default=str(PROJECT_DIR / "data"))
    parser.add_argument("--model-path", type=str, default=str(PROJECT_DIR / "checkpoints" / "cifar10_vgg11_quantrelu8.pth"))
    parser.add_argument("--lut-xlsx", type=str, default=str(PROJECT_DIR / "LUT_ReLU.xlsx"))
    parser.add_argument("--interp-lut-csv", type=str, default=str(PROJECT_DIR / "checkpoints" / "lut_relu_8bit_cifar10.csv"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()
    if args._worker:
        _run_worker(args)
    else:
        raise SystemExit(_run_with_crash_guard())
