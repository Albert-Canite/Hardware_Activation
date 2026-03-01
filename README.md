# VGG-11 + MNIST + 8-bit Hardware ReLU Simulation

一个脚本，IDE 里直接 Run 即可完成：训练 + 验证 + 导出模型 + 导出硬件 LUT。

- `train_vgg11_mnist_qat.py`

## 为什么之前会一直 10% 左右

之前实现中两件事叠加导致训练塌缩：

1. 每层激活都强约束到硬件域，如果输入分布不匹配，很多层会饱和，梯度信息很差。
2. 从第 1 个 epoch 就开启全量 QAT fake-quant，优化会更难。

本版修复：
- 激活改为**自适应归一化 + 硬件量化**：先把每层输入按运行统计缩放到接近 `[-1,1]`，再做 8-bit LUT 约束，最后再缩放回网络尺度。
- 增加 `--qat-start-epoch`（默认 6）：前几轮先学到可用特征，再进入 QAT。
- 使用 `vgg11_bn`（仍是 VGG11 主干）提升稳定性。

## 满足的核心需求

- 每一层激活都模拟硬件 ReLU：输入 `[-1,1]` 8-bit -> ReLU -> 输出 `[0,1]` 8-bit。
- 训练流程仍是一键运行，默认参数可直接跑。
- 训练完成导出：
  - `artifacts/vgg11_mnist_qat_best.pth`
  - `artifacts/vgg11_mnist_qat_final.pth`
  - `artifacts/vgg11_mnist_int8_scripted.pt`
  - `artifacts/hardware_relu_lut.csv`

## 直接运行

```bash
python train_vgg11_mnist_qat.py
```

默认值：
- `epochs=25`
- `lr=3e-4`
- `qat-start-epoch=6`
- `num-workers=0`
