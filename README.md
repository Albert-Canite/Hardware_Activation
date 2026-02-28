# VGG-11 MNIST 8-bit Quantized Training

该目录提供一个可直接运行的脚本：`train_vgg11_mnist_qat.py`。

## 功能
- 使用 VGG-11（标准 A 配置卷积层堆叠）在 MNIST 上进行训练与验证。
- 所有激活函数均为 ReLU。
- 每次进入 ReLU 前、以及 ReLU 输出后，都做一次 **[-1, 1] 区间的 8-bit 离散映射**（fake quantization）。
- 卷积层和全连接层权重也在前向中执行 8-bit fake quantization。
- 自动保存两个模型：
  - `vgg11_mnist_quant8_best.pth`（验证集最佳）
  - `vgg11_mnist_quant8_last.pth`（最后一个 epoch）

## 运行
```bash
pip install -r requirements.txt
python train_vgg11_mnist_qat.py --epochs 5 --batch-size 128
```

如果在 IDE 中运行，直接运行 `train_vgg11_mnist_qat.py` 即可（默认参数会自动下载 MNIST 并训练验证）。
