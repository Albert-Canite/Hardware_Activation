# Hardware ReLU LUT 验证流程（VGG-11 + MNIST/CIFAR-10）

本目录包含 4 个主脚本，满足你要求的“训练+验证+保存”以及“标准激活 vs LUT 激活推理对比”流程：

1. `train_mnist_vgg11.py`：训练/验证/保存 MNIST 模型
2. `infer_compare_mnist.py`：MNIST 标准激活与 LUT 激活推理对比
3. `train_cifar10_vgg11.py`：训练/验证/保存 CIFAR-10 模型
4. `infer_compare_cifar10.py`：CIFAR-10 标准激活与 LUT 激活推理对比

## 核心实现说明
- 网络：标准 `torchvision.models.vgg11(weights=None)`，类别数 10。
- 激活：训练和标准推理阶段，所有 `ReLU` 均替换为 `QuantReLU8bit`。
- `QuantReLU8bit` 约束：
  - 激活输入先量化到 `[-1, 1]` 的 256 个离散值。
  - 经过 ReLU 后再量化到同样 256 个离散值。
- LUT 推理：
  - 读取 `LUT_ReLU.xlsx` 的前两列（输入、输出）。
  - 对原始 16 点（或任意点数）在 `[-1,1]` 线性插值到 256 点。
  - 将模型内 `QuantReLU8bit` 替换为 `LUTReLU8bit` 完成查表激活推理。

## 避免 0xC0000005 的设计
- 所有 `DataLoader` 固定 `num_workers=0`。
- 主入口均采用 `if __name__ == "__main__":`。
- 不使用多进程数据加载，减少 Windows 环境访问冲突风险。

## 运行方式（直接复制）


> 现在 4 个脚本都已经改成“**默认使用脚本所在目录**”作为路径基准。
> 所以在 IDE 中直接点 Run（不加任何参数）即可开始对应流程，不依赖你当前工作目录。

### 1) MNIST 训练并保存
```bash
python train_mnist_vgg11.py
```

### 2) MNIST 推理对比（得到 2 个准确度）
```bash
python infer_compare_mnist.py
```

### 3) CIFAR-10 训练并保存
```bash
python train_cifar10_vgg11.py
```

### 4) CIFAR-10 推理对比（得到 2 个准确度）
```bash
python infer_compare_cifar10.py
```

最终你会得到 4 个准确度值：
- MNIST 标准激活推理准确度
- MNIST LUT 激活推理准确度
- CIFAR-10 标准激活推理准确度
- CIFAR-10 LUT 激活推理准确度

## 输出文件
- 模型权重：
  - `checkpoints/mnist_vgg11_quantrelu8.pth`
  - `checkpoints/cifar10_vgg11_quantrelu8.pth`
- 插值后 8bit LUT：
  - `checkpoints/lut_relu_8bit_mnist.csv`
  - `checkpoints/lut_relu_8bit_cifar10.csv`


## 若出现 `0xC0000005`（Windows）

这个错误通常不是你代码逻辑问题，而是 **PyTorch 2.2.x 与 NumPy 2.x ABI 不兼容** 导致的原生库崩溃。

建议在你的环境中执行：

```bash
pip uninstall -y numpy
pip install "numpy<2"
```

然后重新运行脚本。

本项目脚本现在会在最开始做环境检查：如果检测到 `numpy>=2.0` 会直接给出明确提示并退出，避免无提示崩溃。

