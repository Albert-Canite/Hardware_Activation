import argparse
import os
from pathlib import Path

import torch

from relu_quant_lut import (
    interpolate_lut_to_8bit,
    load_lut_from_excel,
    replace_quantrelu_with_lutrelu,
    save_lut_8bit_csv,
)
from vgg11_data_utils import build_vgg11_quantrelu, evaluate, get_cifar10_loaders


PROJECT_DIR = Path(__file__).resolve().parent


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 inference: standard vs LUT-ReLU")
    parser.add_argument("--data-root", type=str, default=str(PROJECT_DIR / "data"))
    parser.add_argument("--model-path", type=str, default=str(PROJECT_DIR / "checkpoints" / "cifar10_vgg11_quantrelu8.pth"))
    parser.add_argument("--lut-xlsx", type=str, default=str(PROJECT_DIR / "LUT_ReLU.xlsx"))
    parser.add_argument("--interp-lut-csv", type=str, default=str(PROJECT_DIR / "checkpoints" / "lut_relu_8bit_cifar10.csv"))
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    run(args)
