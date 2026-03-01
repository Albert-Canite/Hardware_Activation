# VGG-11 + MNIST + 8-bit Hardware ReLU Simulation

一个脚本，IDE 里直接 Run 即可完成：训练 + 验证 + 导出模型 + 导出硬件 LUT。

- `train_vgg11_mnist_qat.py`

## 满足的核心需求

- 标准 VGG-11（`torchvision.models.vgg11`）训练 MNIST（输入 resize 到 32×32）。
- 每一层激活函数都是 ReLU，但在 ReLU 前后都模拟硬件约束：
  - 输入量化：`[-1,1]`，8-bit，256 点
  - 输出量化：`[0,1]`，8-bit，256 点
- 训练同时启用 QAT（8-bit fake quant），训练后导出 INT8 TorchScript。
- 导出硬件 ReLU LUT（256 行）：`artifacts/hardware_relu_lut.csv`

## 直接运行

```bash
python train_vgg11_mnist_qat.py
```

> 默认值已设置为可直接运行：`epochs=20`、`num_workers=0`（兼容 Windows IDE 运行）。

## 产物

- `artifacts/vgg11_mnist_qat_best.pth`
- `artifacts/vgg11_mnist_qat_final.pth`
- `artifacts/vgg11_mnist_int8_scripted.pt`
- `artifacts/hardware_relu_lut.csv`
