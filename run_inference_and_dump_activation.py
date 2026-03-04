import argparse
import copy
import csv
import os

import torch
import torch.nn as nn
from torch.ao.quantization import QConfig, prepare_qat
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.observer import MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from train_vgg11_mnist_qat import HardwareLUTReLUSim, QuantizableVGG11MNIST, scale_to_signed_unit


def load_lut_from_csv(path: str):
    xs, ys = [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"LUT csv has no header: {path}")

        lowered = {name.strip().lower(): name for name in reader.fieldnames if name is not None}
        x_col = lowered.get("input")
        y_col = lowered.get("output")

        # Fallback: if header is non-standard, use first two columns.
        if x_col is None or y_col is None:
            valid_cols = [c for c in reader.fieldnames if c is not None and c.strip() != ""]
            if len(valid_cols) < 2:
                raise RuntimeError(f"LUT csv must contain at least two columns: {path}")
            x_col, y_col = valid_cols[0], valid_cols[1]

        for row in reader:
            try:
                xs.append(float(row[x_col]))
                ys.append(float(row[y_col]))
            except (TypeError, ValueError, KeyError):
                continue

    if len(xs) < 2:
        raise RuntimeError(f"LUT csv must contain at least two numeric rows: {path}")

    x = torch.tensor(xs, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.float32)
    order = torch.argsort(x)
    return x[order], y[order]


def resample_lut(lut_x: torch.Tensor, lut_y: torch.Tensor, target_points: int):
    if target_points < 2:
        raise ValueError("target_points must be >= 2")
    x_target = torch.linspace(float(lut_x[0]), float(lut_x[-1]), steps=target_points)
    idx = torch.searchsorted(lut_x, x_target, right=False)
    idx = torch.clamp(idx, 1, lut_x.numel() - 1)

    x0 = lut_x[idx - 1]
    x1 = lut_x[idx]
    y0 = lut_y[idx - 1]
    y1 = lut_y[idx]
    t = (x_target - x0) / (x1 - x0 + 1e-12)
    y_target = y0 + t * (y1 - y0)
    return x_target, y_target


def build_lut_levels(
    lut_x: torch.Tensor,
    lut_y: torch.Tensor,
    target_levels: int,
):
    if target_levels < 2:
        raise ValueError("target_levels must be >= 2")

    _, out = resample_lut(lut_x, lut_y, target_levels)
    return out


def save_lut_levels_csv(path: str, lut_levels: torch.Tensor):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    x_levels = torch.linspace(-1.0, 1.0, steps=int(lut_levels.numel()))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "input", "output"])
        for i, (xv, yv) in enumerate(zip(x_levels.tolist(), lut_levels.tolist())):
            writer.writerow([i, f"{xv:.8f}", f"{yv:.8f}"])


def apply_external_hardware_lut(model, lut_y_levels: torch.Tensor):
    for child in model.modules():
        if isinstance(child, HardwareLUTReLUSim):
            child.set_external_lut_levels(lut_y_levels.to(next(child.parameters(), child.running_input_absmax).device))


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int = None):
    model.eval()
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        pred = logits.argmax(dim=1)
        correct += pred.eq(targets).sum().item()
        total += targets.size(0)
    return 100.0 * correct / max(total, 1)


def build_trained_qat_model(model_path: str, device: torch.device, activation_bits: int):
    model = QuantizableVGG11MNIST(activation_bits=activation_bits, adaptive_activation_scale=True).to(device)
    act_qmin, act_qmax = 0, (2 ** activation_bits) - 1
    wt_qmin, wt_qmax = -(2 ** (activation_bits - 1)), (2 ** (activation_bits - 1)) - 1
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
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description="Run two LUT-based inference evaluations and save ACC comparison CSV.")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--activation-bits", type=int, default=7, help="Bit-width for selecting model/LUT and ideal LUT interpolation points")
    parser.add_argument("--artifacts-root", type=str, default="./artifacts")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--hardware-lut-csv", type=str, default="./LUT_ReLU.csv")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--result-csv", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-test-batches", type=int, default=None)
    args = parser.parse_args()

    if args.activation_bits < 2 or args.activation_bits > 8:
        raise ValueError("--activation-bits must be in [2, 8]")

    bit = args.activation_bits
    default_model = os.path.join(args.artifacts_root, f"{bit}bit", f"vgg11_mnist_qat_best_{bit}bit.pth")
    output_dir = args.output_dir or os.path.join(args.artifacts_root, f"{bit}bit")
    result_csv_name = args.result_csv or f"inference_acc_comparison_{bit}bit.csv"
    model_path = args.model_path or default_model
    hardware_lut_csv = args.hardware_lut_csv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(scale_to_signed_unit),
    ])
    test_set = datasets.MNIST(root=args.data_dir, train=False, download=True, transform=transform)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    levels = 2 ** bit
    hw_x_raw, hw_y_raw = load_lut_from_csv(hardware_lut_csv)

    # Hardware branch: measured LUT_ReLU.csv (no manual level merge).
    hw_levels = build_lut_levels(
        hw_x_raw,
        hw_y_raw,
        target_levels=levels,
    )
    hw_interp_lut_path = os.path.join(output_dir, f"hardware_lut_interpolated_for_inference_{bit}bit.csv")
    save_lut_levels_csv(hw_interp_lut_path, hw_levels)

    # Ideal branch: exactly the same path as training/validation checkpoint model.
    model_ideal = build_trained_qat_model(model_path, device, activation_bits=bit)
    acc_ideal = evaluate_accuracy(model_ideal, test_loader, device, args.limit_test_batches)
    # Hardware branch: only replace activation LUT on top of the same loaded model.
    model_hw = copy.deepcopy(model_ideal)
    apply_external_hardware_lut(model_hw, hw_levels)
    acc_hw = evaluate_accuracy(model_hw, test_loader, device, args.limit_test_batches)

    os.makedirs(output_dir, exist_ok=True)
    result_csv_path = os.path.join(output_dir, result_csv_name)
    with open(result_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["bit", "method", "model_path", "lut_source", "lut_points", "test_acc_percent"])
        writer.writerow([bit, "hardware_lut_relu_csv", model_path, hardware_lut_csv, int(hw_levels.numel()), f"{acc_hw:.4f}"])
        writer.writerow([bit, "ideal_trained_qat_model", model_path, "checkpoint_internal_activation", int(levels), f"{acc_ideal:.4f}"])

    print(f"Using device: {device}")
    print(f"Bit: {bit}")
    print(f"Model: {model_path}")
    print(f"[1/2] ACC with ideal trained model (no activation replacement): {acc_ideal:.4f}%")
    print(f"[2/2] ACC with hardware LUT (LUT_ReLU.csv): {acc_hw:.4f}%")
    print(f"Saved comparison CSV: {result_csv_path}")
    print(f"Saved interpolated hardware LUT used for replacement: {hw_interp_lut_path}")


if __name__ == "__main__":
    main()
