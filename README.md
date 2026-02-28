# Hardware LUT ReLU Verification (VGG-11 + MNIST/CIFAR-10)

该目录提供一个可直接运行的脚本，完成以下完整流程：

1. 读取 `LUT_ReLU.xlsx`（两列：输入、输出，原始 16 个点）。
2. 将 LUT 在 `[-1, 1]` 上插值为 8-bit 的 256 个映射点，并保存到 `outputs/LUT_ReLU_8bit_interpolated.csv`。
3. 使用 **VGG-11** 分别在 **MNIST** 和 **CIFAR-10** 上训练（每层激活为 ReLU，且激活输入/输出都做 8-bit 量化到 `[-1,1]`）。
4. 保存两个模型权重：
   - `outputs/vgg11_mnist_quant_relu.pth`
   - `outputs/vgg11_cifar10_quant_relu.pth`
5. 分别对两个模型做推理对比：
   - 标准 ReLU（量化输入/输出）
   - LUT ReLU（使用插值后的 8-bit 查找表）
6. 打印总计 4 个准确率结果（MNIST 2 个 + CIFAR-10 2 个）。

## 一键运行

```bash
python train_vgg11_quant_lut.py --epochs 1 --batch-size 128 --lr 1e-3
```

> 说明：默认 `--epochs 1` 主要用于快速验证流程。若要更高精度，请增大 epoch（例如 20+）。

## 主要参数

- `--epochs`: 每个数据集训练轮数。
- `--batch-size`: 批大小。
- `--lr`: 学习率。
- `--lut-path`: LUT 文件路径（默认 `./LUT_ReLU.xlsx`）。
- `--data-dir`: 数据集下载目录（默认 `./data`）。
- `--output-dir`: 输出目录（默认 `./outputs`）。
- `--num-workers`: DataLoader 进程数（默认 `0`，Windows 更稳定）。
- `--device`: `auto/cpu/cuda`。

## Windows / CUDA 崩溃排查（如出现 0xC0000005）

如果出现你反馈的类似：

```text
Using device: cuda
Process finished with exit code -1073741819 (0xC0000005)
```

建议优先使用下面命令验证流程：

```bash
python train_vgg11_quant_lut.py --device cpu --num-workers 0 --epochs 1
```

如要继续用 GPU，再尝试：

```bash
python train_vgg11_quant_lut.py --device cuda --num-workers 0 --batch-size 64
```

## 输出示例

```text
MNIST | Standard ReLU Acc: xx.xx% | LUT ReLU Acc: xx.xx%
CIFAR10 | Standard ReLU Acc: xx.xx% | LUT ReLU Acc: xx.xx%
```

