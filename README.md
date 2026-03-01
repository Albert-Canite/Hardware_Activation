# VGG-11 + MNIST + 8-bit Hardware ReLU Simulation

一个脚本，IDE 里直接 Run 即可完成：训练 + 验证 + 导出模型 + 导出硬件 LUT。

- 训练脚本：`train_vgg11_mnist_qat.py`
- 推理与激活导出脚本：`run_inference_and_dump_activation.py`

## 默认训练配置（按你要求）

- 总 epoch：`10`
- QAT fake quant 开始：`第 4 个 epoch`（`--qat-start-epoch 4`）
- 其余默认：`lr=3e-4`、`num-workers=0`

## 核心约束

- 每一层激活都模拟硬件 ReLU：
  - 输入量化：`[-1,1]`，8-bit（256点）
  - 通过 ReLU
  - 输出量化：`[0,1]`，8-bit（256点）

## 直接训练

```bash
python train_vgg11_mnist_qat.py
```

训练后产物：
- `artifacts/vgg11_mnist_qat_best.pth`
- `artifacts/vgg11_mnist_qat_final.pth`
- `artifacts/vgg11_mnist_int8_scripted.pt`
- `artifacts/hardware_relu_lut.csv`

## 典型样本推理 + 激活前后 CSV 导出

```bash
python run_inference_and_dump_activation.py
```

默认会：
- 加载 `artifacts/vgg11_mnist_qat_best.pth`
- 对测试集第 0 个样本做推理并打印预测结果
- 导出 `artifacts/inference_activation_pre_post.csv`

CSV 两列：
1. `pre_activation`：进入激活函数前的数据
2. `post_activation`：通过硬件激活仿真后的输出

你可以直接检查：
- `post_activation` 值域是否在 `[0,1]`
- `post_activation` 的离散点数是否不超过 256（8bit）
