# 六类统一评估接口

入口：`evaluate_family.py --family cruise`，默认执行全部实验、六种学习方法三种子及四种经典方法。GPU默认batch16；若PCA有512实特征，自动降至batch4，维持t-PatchGNN约3GiB的模型峰值预算。单评估进程，CPU线程1。不得在模型尚未训练完成时绕过common.load_model的完成状态/来源检查。

## 实验及目录

`runs/six_category_v1/results/<experiment>/<family>/<label>/` 保存 `per_target.csv`、NPZ、`summary.json`、来源guard `complete.json`、资源统计。学习方法标签为 `<model>_seed17` 等；保持/线性/卡尔曼/谐波为 hold/linear/kalman/harmonic。

- main：TEST 600个固定窗口，seed1917，混合4–32个历史，查询10/20/50/100/200/500ms；所有种子。offgrid与stress同样使用TEST seed1917。
- offgrid：所有真实射线追踪的非网格查询，不把插值当目标；所有种子。
- stress：每项TEST 200个固定窗口，学习方法只用seed17，所有经典方法均计算。
  - `pattern_uniform_n16`、`pattern_random_n16`、`pattern_bursty_n16`、`pattern_dropout_n16`。
  - `count_random_n4`、`count_random_n8`、`count_random_n32`；16点参考为pattern_random_n16。
  - `timestamps_none`、`timestamps_uniform_times`、`timestamps_jitter_20`：同一random16历史/截止/目标；均匀化或20ms扰动仅作用于内部时间，真实首尾锚点固定。噪声截到历史范围后稳定排序，压缩CSI与完整CSI同时重排，保证各方法收到相同的数值—时间配对，保持法不变。这不是全传感器时钟偏差实验。
  - `stale_50`、`stale_100`：从真实[-0.4,-gap]历史区间重新抽16点，目标截止和未来查询不变；根据实际可用陈旧历史重新计算归一化、所有压缩/完整历史及目标尺度，不把当前CSI输入模型。

卡尔曼过程噪声比每类别独立使用VAL600、seed917调参；不使用TEST选择参数。缓存包含数据、代码与调参配置哈希。

精度：CRU使用已验证的完整FP32计算，关闭autocast及TF32；其余学习模型使用原bfloat16 AMP。经典方法接收完整复数CSI，谐波法内部使用complex128。

## 指标与分组

所有预测重建到完整复数CSI后计算逐目标NMSE；先线性平均再转dB。同时保存物理误差/目标能量、幅度误差、形状相关、增益误差，无额外相位对齐。行包含真实channel_group、最后已观察CSI物理能量last_observed_energy、目标与该能量比target_over_last_db、实际历史首尾时间和陈旧间隔。

event_category依据真实传播标签分组：history_and_future_switch、history_switch_only、future_switch_only、stable_los、stable_nlos。历史切换只统计实际观测窗口首尾之间；未来切换统计截止点到实际查询时间。非网格查询使用此前dense LoS加该查询实际射线LoS；不把查询之后的dense状态用于判断。stale未观察的间隙单独由staleness_s描述，不冒充已观察历史切换。

遮挡类bootstrap按channel_group采样，保持同一空间通道所有变体为一个簇；其余类别按独立轨迹。原始轨迹和通道标识均保存。每类测试独立通道数量少，置信区间仅是本仿真数据内部的有限样本不确定性。

## 恢复与完整性

复用旧结果前必须匹配checkpoint内容SHA、数据来源、评估代码和指标代码SHA，并核对CSV/NPZ/summary内容SHA。来源改变的已完成结果移入带旧签名的`.stale_...`目录后重新生成。大型原始CSI数组用大小/mtime及已哈希的生成、投影metadata守卫，不声称已逐字节哈希全部大数组。全部默认实验、六模型、三种子完成后才写`results/<family>_complete.json`。

CPU测试 `test_evaluation.py`：**13 passed in 3.89s**。包括真实VAL六模型接口小批、物理目标与stale尺度一致、时戳扰动保持不变及压缩/完整配对一致、事件与通道聚类、完整矩阵保持误差、输出/恢复guard及scheduler marker守卫。三个现成neural_flow/observed_state_flow/local_dynamics checkpoint另外经common完整来源检查，在真实VAL两样本上CPU推理通过。没有启动正式评估或读取TEST用于开发选择。
