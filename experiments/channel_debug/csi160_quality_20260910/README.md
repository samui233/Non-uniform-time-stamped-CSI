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
- [相邻 CSI 变化分布](figures/08_adjacent_change_distributions.png)：同时检查增益、完整复数变化及忽略一个公共相位后的空间频率形状变化。曲线为分位数概览；形状指标是诊断，不用于替换预测标签。
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

## 判断原则与复算

已经分开检查三类情况：连续零信道、LoS 切换时的强衰落、LoS 不变时的短尖峰。正在对少量片段进行同参数复现、随机种子变化、射线数和路径上限变化、计算批次上下文变化及传播选项对照。仅凭曲线突变不能判定异常；同一位置的结果若对数值求解设置高度敏感，则应先解决仿真收敛问题。

完整统计、脚本和复算数据保留在服务器 `/root/autodl-tmp/csi160-quality-20260910/`。此目录只上传图片与简短说明，原始数据及现有模型没有修改。
