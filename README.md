# VGG-11 + MNIST + 8-bit QAT

该目录包含一个可直接在 IDE 里运行的脚本：

- `train_vgg11_mnist_qat.py`

## 功能说明

- 使用标准 VGG-11 结构（基于 `torchvision.models.vgg11`）并适配 MNIST（1 通道、10 类）。
- 所有激活函数均为 ReLU（通过 `QuantizedReLU` 封装）。
- 每次进入激活函数前后都进行 `[-1, 1]` 范围的 8-bit（256级）离散映射，保证激活输入/输出不是连续值。
- 同时启用 PyTorch QAT（Quantization Aware Training）进行 8-bit 量化训练（权重/激活 fake quant）。
- 训练 + 验证流程一体化。
- 最终保存两个模型：
  - `artifacts/vgg11_mnist_qat_final.pth`（QAT 训练后的模型权重）
  - `artifacts/vgg11_mnist_int8_scripted.pt`（转换后的 INT8 TorchScript 模型）

## 运行方式

```bash
python train_vgg11_mnist_qat.py
```

可选快速验证（少量 batch）：

```bash
python train_vgg11_mnist_qat.py --epochs 1 --limit-train-batches 5 --limit-val-batches 2
```
