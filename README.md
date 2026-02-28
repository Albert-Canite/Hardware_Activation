# Hardware Activation Validation (VGG-11 + MNIST/CIFAR-10)

这个目录提供一个**一键可运行**脚本：

- 训练 `VGG-11`（轻量化版本，适配 32x32 输入）
- 数据集：`MNIST` 与 `CIFAR-10`
- 训练时每层激活都使用 **8-bit 量化 ReLU**（输入/输出范围固定在 `[-1, 1]`，共 256 个离散映射）
- 读取 `LUT_ReLU.xlsx`（16 点）并插值得到 8-bit LUT（256 点）
- 将模型中的标准 ReLU 激活替换为 LUT 激活进行推理
- 打印 4 个精度值：
  - MNIST 标准 ReLU 推理精度
  - MNIST LUT ReLU 推理精度
  - CIFAR-10 标准 ReLU 推理精度
  - CIFAR-10 LUT ReLU 推理精度

## 环境

已按你提供的版本约束设计：

- Python 3.8.18
- PyTorch 2.2.2
- CUDA 12.1

额外依赖：`torchvision`, `pandas`, `openpyxl`, `numpy`

## 运行方式（直接一条命令）

```bash
python train_and_validate.py --epochs 10 --batch-size 32
```

> 说明：直接在 IDE 里点 run 也可以，不需要命令行参数。脚本默认 `num_workers=0`，是为了避免 Windows + CUDA 下某些环境出现 `0xC0000005` 原生崩溃。

脚本新增了 **CUDA 子进程烟雾测试**：

- 如果检测到 CUDA 可能不稳定（包括你遇到的 `0xC0000005` 类型崩溃风险），会自动回退到 CPU，避免直接闪退。
- 你仍可手工指定：
  - `--device auto`（默认，推荐）
  - `--device cuda`
  - `--device cpu`

如果你**必须使用 GPU**，请使用：

```bash
python train_and_validate.py --device cuda --require-gpu
```

并保持默认安全模式（`--gpu-safe-mode`，默认开启），该模式会禁用 `cudnn` 高风险路径但仍然使用 CUDA 计算，专门用于规避部分 Windows 机器上的 `0xC0000005`。

> 如果你想先快速验证流程是否通畅，可用：

```bash
python train_and_validate.py --epochs 1 --max-train-batches 10 --max-test-batches 10
```

## 输出

运行后会生成：

- `outputs/relu_lut_8bit.csv`：由 `LUT_ReLU.xlsx` 插值得到的 8-bit LUT
- `outputs/vgg11_mnist_quant_relu.pth`：MNIST 模型权重
- `outputs/vgg11_cifar10_quant_relu.pth`：CIFAR-10 模型权重

并在终端中打印两个数据集在标准 ReLU 与 LUT ReLU 下的精度对比。
