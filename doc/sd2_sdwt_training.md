# 从原版 SD2 直接训练：SDWT 二阶段方案

本方案从原始 Stable Diffusion 2.1 权重直接开始一次连续训练，不依赖一阶段
checkpoint。前 8,000 步使用 latent MSE、pixel L1 和梯度损失建立基本深度映射；
8,000 至 12,000 步逐渐加入本文的 **Spatially Debiased Window Transport
(SDWT) Loss** 并解冻 VAE decoder。

SDWT 受到 Marigold V2 局部最优传输思路的启发，但不是 SinkLoss 的改名版本。
它针对原方法的四个限制重新定义了代价、目标函数、窗口划分和聚合方式。

## 原局部匹配的限制

1. **忽略空间位置。** 仅使用深度差构造代价时，同一窗口内任意像素排列都可能
   获得近似相同的损失。它能容忍边缘标注错位，也可能接受错误的局部几何。
2. **熵偏差。** 带熵正则的交叉传输成本在预测等于真值时通常仍不为零，损失的
   数值和梯度会受温度影响。
3. **固定窗口接缝。** 非重叠窗口无法匹配落在相邻窗口两侧的同一条边缘。
4. **稀疏窗口权重过大。** 按非空窗口平均时，只有一个有效像素的窗口与完整窗口
   权重相同。

## SDWT

对窗口中的预测深度 `p_i`、目标深度 `g_j` 和归一化坐标 `x_i`、`x_j`，
SDWT 使用联合代价

```text
C_ij = rho(p_i - g_j) + lambda_s * ||x_i - x_j||_1
rho(e) = sqrt(e^2 + epsilon^2) - epsilon
```

其中空间项限制远距离交换，Charbonnier 项在零点平滑，同时保留对异常深度误差
的鲁棒性。无效像素在构造传输矩阵之前被精确移除，不再用极大哨兵值参与
Sinkhorn 归一化。

令带熵正则的局部传输目标为 `OT_tau(A, B)`，最终目标使用去偏形式

```text
SDWT(A, B) = OT_tau(A, B)
             - 0.5 * OT_tau(A, A)
             - 0.5 * OT_tau(B, B)
```

因此相同预测和目标的损失为零。每张图同时计算原始窗口网格和偏移半个窗口的
网格，以覆盖原网格边界；最后按每个窗口的有效像素数聚合。默认参数为 5×5
窗口、10 次传输归一化、温度 0.1、空间权重 0.15。

去偏项和双网格需要额外的传输计算，因此 SDWT 会比单次局部匹配更慢。实现按
有效像素数分组、分块计算并使用激活重算，目标自传输项不建立梯度图，以限制显存
增长。论文实验应同时报告训练吞吐与峰值显存。

## 总训练目标

令已完成更新数为 `s`：

```text
r = clip((s - 8000) / 4000, 0, 1)

L = (1 - 0.5r) L_latent_MSE
  + (1 - 0.75r) L_pixel_L1
  + (0.5 - 0.4r) L_gradient
  + r L_SDWT
```

- `s < 8000`：训练 U-Net，VAE 保持冻结。
- `8000 <= s < 12000`：逐渐加入 SDWT，并开始微调 decoder 和
  `post_quant_conv`。
- `s >= 12000`：保留全部基础重建项，SDWT 权重保持 1.0。

推荐配置额外将 latent FFT 从 0 渐增到 0.2。FFT 约束全局频谱，SDWT 处理带
有限位移容忍度的局部几何，两者作用不同。上述超参数是 SD2 的实验起点，需要
通过验证集和消融实验确定最终论文设置。

## 训练

推荐的 SDWT + FFT：

```bash
python train.py --config config/train_sd2_sdwt_fft.yaml --no_wandb
```

不使用 FFT 的对照：

```bash
python train.py --config config/train_sd2_sdwt.yaml --no_wandb
```

这两个配置都不能传入 `--init_checkpoint`。恢复训练：

```bash
python train.py \
  --resume_run output/train_sd2_sdwt_fft/checkpoint/latest --no_wandb
```

推理时组合原始 SD2.1、训练后的 U-Net 和 VAE：

```bash
python run.py --checkpoint /path/to/stable-diffusion-2-1 \
  --training_checkpoint output/train_sd2_sdwt_fft/checkpoint/iter_021000 \
  --input_rgb_dir input/example-1 --output_dir output/sd2_sdwt_fft \
  --ensemble_size 1 --processing_res 0
```

## 论文消融

所有实验应保持初始化、数据、随机种子、训练步数与评估协议一致。

| 配置 | 验证内容 |
| --- | --- |
| `config/train_sd2_sdwt_fft.yaml` | 完整 SDWT + FFT |
| `config/train_sd2_sdwt.yaml` | 去除 FFT |
| `config/ablation/sd2_sdwt_no_transport.yaml` | 去除整个 SDWT 项 |
| `config/ablation/sd2_sdwt_no_spatial.yaml` | 去除空间位移代价 |
| `config/ablation/sd2_sdwt_no_debias.yaml` | 去除自成本校正 |
| `config/ablation/sd2_sdwt_single_grid.yaml` | 去除半窗口偏移网格 |
| `config/ablation/sd2_sdwt_equal_windows.yaml` | 改回非空窗口等权 |
| `config/ablation/sd2_sdwt_no_decoder.yaml` | 保持 VAE decoder 冻结 |

除常规 AbsRel 和 delta1 外，应报告边缘指标、细结构区域指标、不同有效像素率区间
的误差，以及训练显存和耗时。只有在完整方法相对各项消融取得可重复改进后，才能
把这些设计写成实证贡献。

TensorBoard 记录 `train/sdwt_loss`、`train/sdwt_weight`、
`train/gradient_loss`、`train/latent_mse`、`train/pixel_l1` 和
`train/freq_loss_weight`。

CPU 验证：

```bash
python -m pytest tests/test_sdwt_training.py -q
```

## 方法来源

局部最优传输的研究方向来自：

> Pavlovic et al., “Marigold V2: Revisiting Diffusion Transformers for
> Monocular Depth Estimation,” 2026, arXiv:2609.08084.

论文写作中应明确引用该工作，并将 SDWT 表述为空间约束、去偏、双网格和
有效支持聚合方面的扩展。
