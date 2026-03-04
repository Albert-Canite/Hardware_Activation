import argparse
import csv
import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from run_inference_and_dump_activation import (
    apply_external_hardware_lut,
    build_lut_levels,
    build_trained_qat_model,
    load_lut_from_csv,
)
from train_vgg11_mnist_qat import HardwareLUTReLUSim, scale_to_signed_unit


def build_mnist_test_set(data_dir: str):
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(scale_to_signed_unit),
        ]
    )
    return datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)


def pick_one_sample_of_digit(dataset, digit: int) -> Tuple[torch.Tensor, int]:
    for idx in range(len(dataset)):
        x, y = dataset[idx]
        if int(y) == digit:
            return x, idx
    raise RuntimeError(f"No sample with label={digit} found in MNIST test set.")


def collect_activation_modules(model: torch.nn.Module, num_layers: int) -> List[Tuple[str, HardwareLUTReLUSim]]:
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, HardwareLUTReLUSim):
            layers.append((name, module))
    if len(layers) < num_layers:
        raise RuntimeError(f"Requested {num_layers} activation layers, but model has only {len(layers)}.")
    return layers[:num_layers]


def run_and_capture(
    model: torch.nn.Module,
    x: torch.Tensor,
    target_layers: List[Tuple[str, HardwareLUTReLUSim]],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    captured: Dict[str, torch.Tensor] = {}
    hooks = []

    def make_hook(layer_name: str):
        def hook_fn(_module, _inputs, output):
            out = output.dequantize() if getattr(output, "is_quantized", False) else output
            captured[layer_name] = out.detach().cpu()

        return hook_fn

    for layer_name, layer_mod in target_layers:
        hooks.append(layer_mod.register_forward_hook(make_hook(layer_name)))

    with torch.no_grad():
        logits = model(x)

    for h in hooks:
        h.remove()

    return logits.detach().cpu(), captured


def sanitize_name(name: str) -> str:
    return name.replace(".", "_").replace("/", "_")


def save_feature_map(path: str, fmap_2d: torch.Tensor, vmin: float, vmax: float):
    plt.figure(figsize=(3.2, 3.2))
    plt.imshow(fmap_2d.numpy(), cmap="RdBu", vmin=vmin, vmax=vmax)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize first 8 activation layers (first 8 channels each) for one MNIST digit-8 sample: ideal vs hardware LUT."
    )
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--model-path", type=str, default="./artifacts/4bit/vgg11_mnist_qat_best_4bit.pth")
    parser.add_argument("--hardware-lut-csv", type=str, default="./LUT_ReLU.csv")
    parser.add_argument("--output-dir", type=str, default="./artifacts/4bit_featuremap_vis")
    parser.add_argument("--activation-bits", type=int, default=4)
    parser.add_argument("--digit", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--channels-per-layer", type=int, default=8)
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu; default auto")
    args = parser.parse_args()

    if args.activation_bits != 4:
        raise ValueError("This script is intended for 4-bit checkpoints; please pass --activation-bits 4.")

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    test_set = build_mnist_test_set(args.data_dir)
    x_single, sample_idx = pick_one_sample_of_digit(test_set, args.digit)
    y_true = args.digit
    x_batch = x_single.unsqueeze(0).to(device)

    # Build ideal branch exactly as inference script does.
    model_ideal = build_trained_qat_model(args.model_path, device=device, activation_bits=args.activation_bits)
    model_ideal.eval()

    # Build hardware branch from identical checkpoint, only replacing activation LUT path.
    model_hw = build_trained_qat_model(args.model_path, device=device, activation_bits=args.activation_bits)
    model_hw.eval()
    levels = 2 ** args.activation_bits
    lut_x, lut_y = load_lut_from_csv(args.hardware_lut_csv)
    lut_levels = build_lut_levels(lut_x, lut_y, target_levels=levels)
    apply_external_hardware_lut(model_hw, lut_levels)

    target_layers_ideal = collect_activation_modules(model_ideal, args.num_layers)
    target_layers_hw = collect_activation_modules(model_hw, args.num_layers)
    layer_names_ideal = [n for n, _ in target_layers_ideal]
    layer_names_hw = [n for n, _ in target_layers_hw]
    if layer_names_ideal != layer_names_hw:
        raise RuntimeError("Layer order mismatch between ideal and hardware models.")

    logits_ideal, acts_ideal = run_and_capture(model_ideal, x_batch, target_layers_ideal)
    logits_hw, acts_hw = run_and_capture(model_hw, x_batch, target_layers_hw)

    pred_ideal = int(torch.argmax(logits_ideal, dim=1).item())
    pred_hw = int(torch.argmax(logits_hw, dim=1).item())
    probs_ideal = torch.softmax(logits_ideal, dim=1).squeeze(0)
    probs_hw = torch.softmax(logits_hw, dim=1).squeeze(0)

    # Save original input with RdBu colormap.
    original_dir = os.path.join(args.output_dir, "original")
    os.makedirs(original_dir, exist_ok=True)
    save_feature_map(
        os.path.join(original_dir, f"sample_idx_{sample_idx}_label_{y_true}_input_rdbu.png"),
        x_single[0].cpu(),
        vmin=-1.0,
        vmax=1.0,
    )

    # Save per-layer maps: fixed first N channels, identical order for ideal/hardware.
    for layer_idx, layer_name in enumerate(layer_names_ideal, start=1):
        layer_dir = os.path.join(args.output_dir, f"layer_{layer_idx:02d}_{sanitize_name(layer_name)}")
        os.makedirs(layer_dir, exist_ok=True)

        feat_i = acts_ideal[layer_name].squeeze(0)
        feat_h = acts_hw[layer_name].squeeze(0)
        if feat_i.ndim != 3 or feat_h.ndim != 3:
            # This script focuses on first 8 feature-map layers (C,H,W); skip non-image activations.
            continue

        c = min(args.channels_per_layer, feat_i.shape[0], feat_h.shape[0])
        for ch in range(c):
            fmap_i = feat_i[ch]
            fmap_h = feat_h[ch]
            vmin = float(torch.min(torch.stack([fmap_i.min(), fmap_h.min()])))
            vmax = float(torch.max(torch.stack([fmap_i.max(), fmap_h.max()])))
            if abs(vmax - vmin) < 1e-12:
                vmax = vmin + 1e-6

            save_feature_map(
                os.path.join(layer_dir, f"ch_{ch:02d}_ideal.png"),
                fmap_i,
                vmin=vmin,
                vmax=vmax,
            )
            save_feature_map(
                os.path.join(layer_dir, f"ch_{ch:02d}_hardware.png"),
                fmap_h,
                vmin=vmin,
                vmax=vmax,
            )

    summary_csv = os.path.join(args.output_dir, "summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model_path", "hardware_lut_csv", "activation_bits", "sample_idx", "true_label", "pred_ideal", "pred_hardware"])
        w.writerow([args.model_path, args.hardware_lut_csv, args.activation_bits, sample_idx, y_true, pred_ideal, pred_hw])

    class_scores_csv = os.path.join(args.output_dir, "class_scores_logits_and_probs.csv")
    with open(class_scores_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "ideal_logit", "ideal_prob", "hardware_logit", "hardware_prob"])
        logits_ideal_row = logits_ideal.squeeze(0).tolist()
        logits_hw_row = logits_hw.squeeze(0).tolist()
        probs_ideal_row = probs_ideal.tolist()
        probs_hw_row = probs_hw.tolist()
        for label in range(10):
            w.writerow(
                [
                    label,
                    f"{logits_ideal_row[label]:.8f}",
                    f"{probs_ideal_row[label]:.8f}",
                    f"{logits_hw_row[label]:.8f}",
                    f"{probs_hw_row[label]:.8f}",
                ]
            )

    print(f"Using device: {device}")
    print(f"Model: {args.model_path}")
    print(f"Hardware LUT: {args.hardware_lut_csv}")
    print(f"Sample chosen: idx={sample_idx}, label={y_true}")
    print(f"Pred ideal={pred_ideal}, pred hardware={pred_hw}")
    print(f"Saved visualization root: {args.output_dir}")
    print(f"Saved summary: {summary_csv}")
    print(f"Saved class scores: {class_scores_csv}")


if __name__ == "__main__":
    main()
