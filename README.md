# VGG-11 + MNIST + 8-bit QAT

该目录包含一个可直接在 IDE 里运行的脚本：

- `train_vgg11_mnist_qat.py`

## 功能说明

- 使用标准 VGG-11 结构（基于 `torchvision.models.vgg11`）并适配 MNIST（1 通道、10 类），输入 resize 到 32×32 以匹配 VGG-11 的 5 次池化。
- 所有激活函数均为 ReLU（通过 `QuantizedReLU` 封装）。
- 每次进入激活函数前后都进行 `[-1, 1]` 范围的 8-bit（256级）离散映射，保证激活输入/输出不是连续值。
- 激活离散化使用 STE（Straight-Through Estimator）：前向保持离散值，反向允许梯度传播（修复“准确率异常低”的关键问题）。
- 同时启用 PyTorch QAT（Quantization Aware Training）进行 8-bit 量化训练（权重/激活 fake quant）。
- 训练 + 验证流程一体化。
- 最终保存两个模型：
  - `artifacts/vgg11_mnist_qat_final.pth`（QAT 训练后的模型权重）
  - `artifacts/vgg11_mnist_int8_scripted.pt`（转换后的 INT8 TorchScript 模型）

## 为什么之前准确率低

主要原因有两点：

1. 原脚本默认只有 3 个 epoch，VGG11 + QAT + 激活离散化的训练强度明显不够，严重欠拟合。
2. 原激活离散化直接用 `torch.round`，其梯度几乎处处为 0，导致梯度难以穿过激活量化层，学习能力大幅下降。

当前版本修复为：

- 默认 `--epochs 20`
- 离散化采用 STE
- 增加 `CosineAnnealingLR`

## 运行方式

```bash
python train_vgg11_mnist_qat.py --num-workers 0
```

> 说明：默认 `--num-workers 0` 以避免 Windows 下 DataLoader 多进程 pickling 问题。

可选快速验证（少量 batch）：

```bash
python train_vgg11_mnist_qat.py --epochs 1 --limit-train-batches 5 --limit-val-batches 2
```
