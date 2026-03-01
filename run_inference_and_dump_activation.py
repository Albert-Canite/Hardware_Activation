import argparse
import csv
import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from train_vgg11_mnist_qat import HardwareLUTReLUSim, QuantizableVGG11MNIST, scale_to_signed_unit


def main():
    parser = argparse.ArgumentParser(description="Run one-sample inference and dump activation pre/post LUT values")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--model-path", type=str, default="./artifacts/vgg11_mnist_qat_best.pth")
    parser.add_argument("--output-dir", type=str, default="./artifacts")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-points", type=int, default=4096)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(scale_to_signed_unit),
    ])
    test_set = datasets.MNIST(root=args.data_dir, train=False, download=True, transform=transform)

    model = QuantizableVGG11MNIST().to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # pick sample
    image, label = test_set[args.sample_index]
    image = image.unsqueeze(0).to(device)

    # hook first hardware activation
    capture = {}

    def hook_fn(module, inputs, output):
        capture["pre"] = inputs[0].detach().cpu().flatten()
        capture["post"] = output.detach().cpu().flatten()

    first_act = None
    for m in model.modules():
        if isinstance(m, HardwareLUTReLUSim):
            first_act = m
            break

    if first_act is None:
        raise RuntimeError("No HardwareLUTReLUSim layer found.")

    handle = first_act.register_forward_hook(hook_fn)

    with torch.no_grad():
        logits = model(image)
        pred = int(torch.argmax(logits, dim=1).item())

    handle.remove()

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "inference_activation_pre_post.csv")

    pre = capture["pre"]
    post = capture["post"]
    n = min(args.max_points, pre.numel(), post.numel())

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pre_activation", "post_activation"])
        for i in range(n):
            writer.writerow([f"{pre[i].item():.8f}", f"{post[i].item():.8f}"])

    print(f"Using device: {device}")
    print(f"Model: {args.model_path}")
    print(f"Sample index: {args.sample_index}, label: {label}, prediction: {pred}")
    print(f"Saved activation csv: {csv_path}")
    print(f"pre range: [{pre.min().item():.6f}, {pre.max().item():.6f}]")
    print(f"post range: [{post.min().item():.6f}, {post.max().item():.6f}]")
    print(f"post unique count: {torch.unique(post).numel()} (should be <= 256)")


if __name__ == "__main__":
    main()
