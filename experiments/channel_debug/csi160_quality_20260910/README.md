# 160 条轨迹的 CSI 质量检查

数据来源：`csi-multimodal-160-20260910`，每条 5 秒、500 Hz、64×16 复数 CSI。本目录检查原始数据，未清洗、插值修补或平滑任何 CSI。

## 六类完整增益图

增益定义为 `10 log10(mean(|H|²))`，对 64 根天线和 16 个频点平均，不是发射功率或接收功率 dBm。每个小图对应一条完整轨迹，六张图使用相同纵轴范围。蓝色/橙色/黑色曲线分别表示训练/验证/测试轨迹；淡橙底色表示射线结果为 NLoS；底部紫色叉号表示零 CSI，其真实 dB 值为负无穷，不能把叉号所在高度当作功率。

| 类别 | 全部路径 |
|---|---|
| 近距离正常，20 条 | [完整增益图](figures/01_near_normal_all_routes.png) |
| 近距离高机动，20 条 | [完整增益图](figures/02_near_maneuver_all_routes.png) |
| 近距离遮挡，40 条 | [完整增益图](figures/03_near_occlusion_all_routes.png) |
| 远距离正常，20 条 | [完整增益图](figures/04_far_normal_all_routes.png) |
| 远距离高机动，20 条 | [完整增益图](figures/05_far_maneuver_all_routes.png) |
| 远距离遮挡，40 条 | [完整增益图](figures/06_far_occlusion_all_routes.png) |

下载原图可放大查看各轨迹编号和细节。

## 异常线索

- [全部路径增益与相邻跳变热图](figures/07_gain_and_jump_overview.png)：保留全部 160 条路径，跳变为相邻 2 ms 的增益差。
- [相邻 CSI 变化分布](figures/08_adjacent_change_distributions.png)：检查增益、按两帧总能量归一化的复数变化，以及基于归一化复相关度的形状变化。后者忽略整体复数倍率，只检查空间频率形状，不用于替换预测标签。
- [测试误差集中程度](figures/09_test_error_concentration.png)：使用上一轮四模型保存的测试结果，不重新选择 checkpoint。

400,160 帧中有 843 帧零 CSI，分布在以下 5 段；没有发现非有限 CSI 数值。持续时间按零帧数乘 2 ms 统计。

| 路径 | 划分 | 首末零帧时刻/s | 零帧数 | 采样持续时间/ms |
|---|---|---|---:|---:|
| far_occlusion_000 | 训练 | 0–0.248 | 125 | 250 |
| far_occlusion_020 | 训练 | 0–0.528 | 265 | 530 |
| far_occlusion_029 | 测试 | 4.714–5.000 | 144 | 288 |
| far_occlusion_036 | 训练 | 0–0.300 | 151 | 302 |
| far_occlusion_036 | 训练 | 0.444–0.758 | 158 | 316 |

在前后均非零的相邻帧中，804 次增益变化达到 6 dB，659 次达到 10 dB，441 次达到 20 dB。这些只是检查标记，不等于全部都是仿真错误。6 dB 事件中 62 次恰逢 LoS 切换，36 次位于原来每 32 帧的分批边界；因此不能把所有跳变归咎于分批边界。

纯 CSI 测试的 `far_occlusion_001` 单条轨迹贡献约 85.86% 的逐目标 NMSE 总和。这里的贡献是先逐目标除以其真实能量再求和，不是原始平方误差总和，也不是异常样本比例。

## 重点路径的逐时刻检查

以下图同时展示增益、路径数量、LoS、相邻变化。紫色背景是零信道区间；淡灰竖线是原始 32 帧计算批次边界。

- [far_occlusion_001：主导测试误差的轨迹](figures/10_detail_far_occlusion_001.png)
- [far_occlusion_000](figures/10_detail_far_occlusion_000.png)
- [far_occlusion_020](figures/10_detail_far_occlusion_020.png)
- [far_occlusion_029](figures/10_detail_far_occlusion_029.png)
- [far_occlusion_036](figures/10_detail_far_occlusion_036.png)
- [near_occlusion_034：最大单步增益变化](figures/10_detail_near_occlusion_034.png)

## 复算结果

已完成少量片段的同参数复现、种子变化、射线数和路径上限变化、批次上下文变化、边缘绕射及最大深度对照，保持合成阵列，不进行逐天线计算。

- [训练/验证/测试损失集中图](figures/11_train_val_test_concentration.png)
- [九个代表片段的设置敏感性](figures/12_solver_sensitivity.png)
- [高损失轨迹的重点复算](figures/13_high_loss_rechecks.png)

**高影响片段定位。** 用已有最佳模型在固定窗口做事后诊断，纯 CSI 训练划分中 far_occlusion_014、far_occlusion_008、near_occlusion_035 分别贡献 26.52%、19.19%、15.93% 的诊断损失，前三条合计约 61.64%。CSI+运动模型中 near_occlusion_035 占 44.75%。验证集最突出的 far_occlusion_003 占 15.34%，相对分散。这不是对训练过程中历次随机梯度的回溯。非零目标用 NMSE，零目标沿用历史尺度平方误差。

**强路径漏检。** near_occlusion_034 在 1.482 s 原始增益为 −123.35 dB；换种子或 4 倍射线后约 −77.30 dB，找到了原先遗漏的强单次绕射路径。几何 LoS 切换仍在，但相邻 52 dB 跳变缩小至约 0.29 dB。near_occlusion_035 在 1.060 s 从约 −120.09 dB 变到 −78.34 dB。far_occlusion_014 的 1.150 s、far_occlusion_008 的 2.400 s 同样高度敏感，应优先复核候选发现与收敛性。

**原批量结果与同参数复算不一致。** far_occlusion_006 开头的 2 ms 尖峰在当前复算中消失；far_occlusion_031 在 3.200 s 从原始约 −126.99 dB 变为约 −96.82 dB。当前连续重复三次基本一致。差异远大于浮点末位误差，涉及原批量运行的求解状态、候选发现或验证；确切代码触发点尚未锁定，不能称为已修复。

**零信道与传播配置/搜索覆盖有关。** 抽查 far_occlusion_020/029/036 的三个零片段，开启边缘绕射后所有抽查点均出现路径；4 倍射线也恢复了其中 029/036 的抽查点。单独增加深度或路径上限没有恢复。这里只证明配置敏感，不保证开启后全部数据自动可信。

**不能把全部强衰落当坏点。** far_occlusion_001 在 2.646 s 的直达线确实被遮挡，独立线段求交确认了切换。原始相邻增益约 −82.65 → −119.04 dB；4 倍射线后约 −82.64 → −118.24 dB，强衰落仍存在。有限设置尚不能证明完全收敛，但没有依据直接删除。far_occlusion_016 的验证集 LoS 恢复也仍有较大变化。

按 50 ms 滑动、前后各 500 ms 窗口检查，训练/验证/测试分别有 32/0/6 个窗口接触零 CSI，有 1383/182/235 个窗口接触至少一次 10 dB 相邻跳变。接触不等于实际随机抽样选中，也不是删除建议。

下一步应先修复/复算高影响的候选不稳定与漏检片段，再制定清洗规则。增加射线不是处处单调改善；不能统一平滑、删除深衰落，或取每次试验中增益最大的结果拼成新 CSI。原始数据、划分和模型均未修改。

完整统计、脚本和复算数据保留在服务器 `/root/autodl-tmp/csi160-quality-20260910/`。此目录只上传图片与简短说明，原始数据及现有模型没有修改。
