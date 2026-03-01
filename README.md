# VGG-11 + MNIST + 8-bit Hardware ReLU Simulation

一个脚本，IDE 里直接 Run 即可完成：训练 + 验证 + 导出模型 + 导出硬件 LUT。

- `train_vgg11_mnist_qat.py`

## 为什么之前训练会卡在 10% 左右

核心原因不是“硬件激活无法实现”，而是之前实现把每层激活都**硬压缩到 [0,1] 且不做尺度恢复**，导致深层网络信号幅度塌缩、梯度和特征表达能力严重受限。

本版本修复：
- 仍严格执行硬件激活约束：输入 `[-1,1]` 8-bit -> ReLU -> 输出 `[0,1]` 8-bit。
- 增加每层激活后的**可学习尺度恢复**（trainable scale recovery），保持硬件 LUT 约束同时让网络可训练。

## 满足的核心需求

- 标准 VGG-11（`torchvision.models.vgg11`）训练 MNIST（输入 resize 到 32×32）。
- 每一层激活都模拟硬件 ReLU：
  - 输入量化：`[-1,1]`，8-bit，256 点
  - 输出量化：`[0,1]`，8-bit，256 点
- 启用 QAT（8-bit fake quant），训练后导出 INT8 TorchScript。
- 导出硬件 ReLU LUT（256 行）：`artifacts/hardware_relu_lut.csv`

## 直接运行

```bash
python train_vgg11_mnist_qat.py
```

> 默认值已设置为可直接运行：`epochs=25`、`lr=3e-4`、`num_workers=0`（兼容 Windows IDE 运行）。

## 产物

- `artifacts/vgg11_mnist_qat_best.pth`
- `artifacts/vgg11_mnist_qat_final.pth`
- `artifacts/vgg11_mnist_int8_scripted.pt`
- `artifacts/hardware_relu_lut.csv`
