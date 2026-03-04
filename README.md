# Hardware Activation Validation Pipeline (KAUST IPL Related)

This repository is related to KAUST IPL's paper:
**"Transfer-free Van der Waals Integration for Reconfigurable Optical Neural Networks"**.

The goal is to evaluate whether a measured hardware LUT activation can replace the ideal quantized ReLU path in a QAT-trained VGG11-MNIST model, and to quantify the accuracy/feature-map differences across bit-widths.

## Purpose

This project implements an end-to-end workflow for:

1. Training QAT VGG11 models with bit-aligned activation simulation.
2. Comparing inference accuracy between:
   - `ideal` branch: checkpoint internal activation path (same as training).
   - `hardware` branch: same checkpoint, but activation LUT replaced by measured `LUT_ReLU.csv`.
3. Exporting interpolated LUT tables used during inference/training.
4. Visualizing layer-wise activation feature maps for one fixed input sample (digit `8`) under both branches.
5. Exporting per-class logits/probabilities (`10` classes) for direct branch comparison.

## Repository Scripts (Current)

- `train_vgg11_mnist_qat.py`
  - Baseline QAT training (single path with internal quantized ReLU/LUT simulation).
  - Supports configurable `--activation-bits`.

- `train_vgg11_mnist_qat_hw_lut4bit.py`
  - Dual-mode 4-bit training script:
    - `--mode ideal` (internal quantized ReLU/LUT)
    - `--mode hardware` (measured LUT from `LUT_ReLU.csv`)
  - Exports step/epoch CSV metrics.
  - Note: eval/train mode switching bug for probe validation has been fixed in the current version.

- `run_inference_and_dump_activation.py`
  - Loads trained checkpoint at selected bit-width.
  - Computes accuracy for both branches on the same test loader:
    - ideal = unchanged trained QAT model
    - hardware = same model + external LUT replacement
  - Exports per-bit comparison CSV and LUT used for hardware branch.

- `visualize_4bit_featuremaps_ideal_vs_hardware.py`
  - Loads a trained 4-bit checkpoint.
  - Uses one MNIST test sample of digit `8`.
  - Saves first `8` activation layers and first `8` channels per layer for both branches with `RdBu`.
  - Exports:
    - summary metadata CSV
    - class logits and softmax probabilities CSV (both branches, same sample).

## Environment

Install dependencies:

```bash
pip install -r requirements.txt
```

Main dependency set used by scripts includes:
- `torch`, `torchvision`
- `matplotlib`

## Data Convention

- Dataset: MNIST.
- Input transform: resized to `32x32`, then scaled from `[0,1]` to `[-1,1]` using:
  - `x -> 2*x - 1`

## Workflow A: Baseline QAT Training (Single Script)

Run:

```bash
python train_vgg11_mnist_qat.py \
  --activation-bits 4 \
  --epochs 20 \
  --qat-start-epoch 4 \
  --output-dir artifacts/4bit
```

Typical outputs in `artifacts/4bit/`:
- `vgg11_mnist_qat_best_4bit.pth`
- `vgg11_mnist_qat_final_4bit.pth`
- `vgg11_mnist_int8_traced_4bit.pt`
- `hardware_relu_lut_4bit.csv`

## Workflow B: Dual-Mode 4-bit Training (Ideal vs Hardware LUT)

Run one mode manually:

```bash
python train_vgg11_mnist_qat_hw_lut4bit.py \
  --mode ideal \
  --activation-bits 4 \
  --epochs 10 \
  --qat-start-epoch 4 \
  --output-dir artifacts/4bit_ideal_train
```

```bash
python train_vgg11_mnist_qat_hw_lut4bit.py \
  --mode hardware \
  --activation-bits 4 \
  --lut-csv ./LUT_ReLU.csv \
  --epochs 10 \
  --qat-start-epoch 4 \
  --output-dir artifacts/4bit_hardware_train
```

Outputs include:
- `vgg11_mnist_qat_best_{mode}_lut_4bit.pth`
- `vgg11_mnist_qat_final_{mode}_lut_4bit.pth`
- `train_step_metrics_{mode}_lut_4bit.csv`
- `train_epoch_metrics_{mode}_lut_4bit.csv`
- `hardware_lut_interpolated_for_training_4bit.csv` (hardware mode)

## Workflow C: Inference Accuracy Comparison (Ideal vs Hardware)

Single bit example:

```bash
python run_inference_and_dump_activation.py \
  --activation-bits 4 \
  --artifacts-root ./artifacts \
  --hardware-lut-csv ./LUT_ReLU.csv \
  --output-dir artifacts/4bit \
  --result-csv inference_acc_comparison_4bit.csv
```

Key outputs:
- `artifacts/4bit/inference_acc_comparison_4bit.csv`
- `artifacts/4bit/hardware_lut_interpolated_for_inference_4bit.csv`

The CSV has two rows for the same model checkpoint:
- `ideal_trained_qat_model`
- `hardware_lut_relu_csv`

## Workflow D: 4-bit Feature-Map Visualization (Ideal vs Hardware)

Run:

```bash
python visualize_4bit_featuremaps_ideal_vs_hardware.py \
  --model-path ./artifacts/4bit/vgg11_mnist_qat_best_4bit.pth \
  --hardware-lut-csv ./LUT_ReLU.csv \
  --output-dir ./artifacts/4bit_featuremap_vis \
  --activation-bits 4 \
  --digit 8 \
  --num-layers 8 \
  --channels-per-layer 8
```

Outputs:
- `artifacts/4bit_featuremap_vis/original/*.png`
- `artifacts/4bit_featuremap_vis/layer_01_.../ch_00_ideal.png`, `ch_00_hardware.png`, ...
- `artifacts/4bit_featuremap_vis/summary.csv`
- `artifacts/4bit_featuremap_vis/class_scores_logits_and_probs.csv`

The class-score CSV exports both branches in one file:
- `ideal_logit`, `ideal_prob`
- `hardware_logit`, `hardware_prob`

## SLURM Submission Scripts

- `jobscript_train_hw_lut4bit.slurm`
  - Array job over two modes (`ideal`, `hardware`) for 4-bit dual-mode training.

- `jobscript_infer.slurm`
  - Array job over `8 -> 2` bits for inference comparison.
  - Aggregates per-bit CSV rows into `artifacts/inference_acc_comparison_all_bits.csv`.

- `jobscript_visualize_4bit_featuremaps.slurm`
  - Runs the 4-bit feature-map visualization workflow.

## Notes on Current Version

1. Ideal vs hardware inference comparison is implemented by replacing only the activation LUT while keeping the same checkpoint weights for fair comparison.
2. `LUT_ReLU.csv` is treated as measured hardware data; higher/lower level tables are generated by interpolation inside scripts.
3. For training logs with probe validation in `train_vgg11_mnist_qat_hw_lut4bit.py`, model mode restoration after probe evaluation is fixed.
