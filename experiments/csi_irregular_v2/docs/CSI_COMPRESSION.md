# CSI 线性子空间与文件格式

这里的压缩只降低预测模型输入/输出维度，原始 `H.npy` 完整保留。它不沿时间平滑、不修改传播相位、不截断信道增益，也不根据预测难度删除轨迹。

## 拟合与秩选择

运行：

```bash
cd /root/autodl-tmp/csi-irregular-benchmark
/root/miniconda3/envs/SimART/bin/python -m benchmark.compression fit
/root/miniconda3/envs/SimART/bin/python -m benchmark.compression project
```

`fit` 等待所有训练/验证轨迹的 `complete.json` 存在才允许最终拟合。每条训练轨迹按时间等宽分层随机抽样默认 128 帧，每帧除以自身 Frobenius 范数，仅用于均衡拟合权重，避免大功率路线支配基底。基底只拟合训练 CSI，不减均值，使用复数随机低秩 SVD。投影实际数据时使用原始 CSI 幅度，而非这些单位能量拟合样本。

候选复数秩默认 32、64、128、256。基底拟合结束后，独立分层抽取训练/验证帧，计算逐帧相对重建平方误差。平均指标为先对线性域逐帧误差求算术平均、再转 dB；95% 分位也先在线性域计算。默认选择训练、验证的平均误差均不高于 −25 dB、95% 分位均不高于 −15 dB 的最小候选秩。测试数据不参与基底拟合或秩选择。

如果没有候选达到门限，保存最大候选秩，但明确记录 `thresholds_met=false`，默认禁止继续投影；应检查报告并增大候选秩，例如 `--ranks 64 128 256 512`。如果研究者明确接受该误差下限，也可以显式 `project --accept-rank-floor`，但必须报告该限制，不能隐藏它。

`data/compression/metadata.json` 保存抽样轨迹/帧编号、随机种子、SVD 参数、训练/验证的各候选秩误差、正交性误差以及基底 SHA256。重复运行不会静默覆盖既有基底，需要显式 `fit --force` 或另选输出目录。

生成期间可运行试探性拟合，但必须使用独立目录：

```bash
python -m benchmark.compression fit --allow-partial --output data/compression_pilot
```

这种结果标记为 `partial_pilot=true`，不能代替全部训练轨迹上的最终拟合。默认投影拒绝使用该类基底。

## 复数约定

`basis.npy` 是复数矩阵 `[8192,K]`。每帧 `[64,128]` CSI 按 C-order 展平。正向投影为 `z = H_flat @ basis.conj()`，重建为 `H_hat_flat = z @ basis.T`。这是有意的共轭约定；不能把其中一个转置任意替换为共轭转置。模型实数特征为 `z.real, z.imag` 逐系数交错排列，维度 `F=2K`，可由连续 complex64 数组的 float32 view 实现。

因为不减均值，零 CSI 仍然投影为零，改变整体复幅度也严格对应系数的相同改变。模型窗口可以按其已观测历史的平均完整 CSI 能量统一缩放，不能用目标帧能量缩放输入。

## 每条轨迹的输出

全部写在既有 `data/routes/<traj_id>/` 内：

| 文件 | 形状与含义 |
|---|---|
| `coefficients.npy` | complex64 `[T,K]`，原始幅度系数 |
| `energy.npy` | float64 `[T]`，每帧全部 64×128 元素的平方幅度之和，不是均值 |
| `residual_energy.npy` | float64 `[T]`，完整 CSI 减去投影重建后的残差能量 |
| `compression.json` | 基底哈希、来源时间、秩、完整轨迹重建误差摘要 |
| `offgrid_coefficients.npz` | 保留真实非栅格查询的 `cutoff_s[C]`、`query_s[C,Q]`、`coefficients[C,Q,K]`、`energy[C,Q]`、`residual_energy[C,Q]`、`los[C,Q]` |

非栅格数据直接读取 `offgrid_queries.npz` 中独立射线追踪生成的原始 H，不由相邻规则帧插值。

训练/评价时，目标系数误差应加上目标的不可表示残差能量，再除以完整目标 CSI 能量，才能得到完整矩阵的预测误差。只报告系数空间误差会遗漏压缩下限。数值上基底在 complex64 精度下近似正交；脚本同时报告实际正交性误差。

`project` 支持安全续跑：仅当当前基底哈希、原始 H 文件修改时间与必需输出一致时跳过已完成路线。最后在基底目录写 `projection_status.json`。未生成完的路线会列出，之后可重跑补全；不会读取仍在写入而没有完成标记的 H 文件。
