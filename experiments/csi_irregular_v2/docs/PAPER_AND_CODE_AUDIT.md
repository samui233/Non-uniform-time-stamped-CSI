# Time-IMM 单模态方法与 CSI 适配审计

日期：2026-09-05。本文记录已阅读的原论文、固定源码版本、可保留的机制与必要适配。它是实现约束，不等于已经完成训练或复现了论文原数据集结果。

## 1. 以哪个版本为准

主依据是 [Time-IMM 的 NeurIPS 2025 正式论文](https://papers.neurips.cc/paper_files/paper/2025/file/4199594d3c15736df2bf5274fa3155f4-Paper-Datasets_and_Benchmarks_Track.pdf)，尤其第 3.1、3.2、4.1 节及附录 H、K。正式 PDF 已下载到 `third_party/papers/Time_IMM_NeurIPS2025.pdf`，同目录有可检索的文本。

单模态预测模块接收历史数值、真实观测时间及未来查询时间，输出查询时刻的数值预测。论文比较的原生非规则方法为 **CRU、Latent-ODE、Neural Flow、t-PatchGNN**。此外包含经过预对齐的 DLinear、Informer、PatchTST、TimesNet、TimeMixer 等规则序列模型。本轮重点实现前四类连续时间/非规则方法，不把旧实验中的其他网络说成该论文的一阶段方法。

“Stage 1”在本项目指去掉文本编码和融合后的数值预测模块；不是另外发明的 Time-IMM 预训练算法。只做 CSI 时不需要下载大语言模型或文本数据集。

正式版本描述的预对齐保留值、mask 和时间，并把查询作为未观测位置加入；不应把旧 arXiv 版本一句“interpolates”理解成必须对未来 CSI 做数值插值。标签只能来自真实仿真查询，不使用未来真值填充历史。

## 2. 固定源码

所有仓库位于本项目 `third_party/`，获取后未直接修改原文件。

| 项目 | 上游与固定 commit | 本地许可检查 |
|---|---|---|
| IMM-TSF | [官方基准库](https://github.com/blacksnail789521/IMM-TSF/tree/b9e245f2238a9c35d241dbc5bd943c28193a5b90)，`b9e245f2238a9c35d241dbc5bd943c28193a5b90` | 顶层 MIT；子文件原版权仍需保留 |
| CRU | [原作者代码](https://github.com/boschresearch/Continuous-Recurrent-Units/tree/79723e7772102cae8a78109e18d81beff9c7cac2)，`79723e7772102cae8a78109e18d81beff9c7cac2` | AGPL-3.0，另有第三方许可文件 |
| Latent ODE | [原作者代码](https://github.com/YuliaRubanova/latent_ode/tree/c0682d4f52b806fb88d965755892eadd9783f936)，`c0682d4f52b806fb88d965755892eadd9783f936` | MIT |
| Neural Flows | [原作者代码](https://github.com/mbilos/neural-flows-experiments/tree/bd19f7c92461e83521e268c1a235ef845a3dd963)，`bd19f7c92461e83521e268c1a235ef845a3dd963` | 该 commit 未找到独立 LICENSE 文件；不得假定 MIT |
| t-PatchGNN | [原作者代码](https://github.com/usail-hkust/t-PatchGNN/tree/00c94e7bbaf21c71b03ed84ff690ae59e37129e5)，`00c94e7bbaf21c71b03ed84ff690ae59e37129e5` | 该 commit 未找到独立 LICENSE 文件；不得假定 MIT |

上游允许查看不自动意味着任意重新许可。训练所需源码保留在独立第三方目录；若以后发布代码，需要保留各自声明，并单独确认未明确许可的部分。不把整个第三方树当成本项目原创代码上传。

## 3. 四类方法必须保留什么

### CRU

[Modeling Irregular Time Series with Continuous Recurrent Units，ICML 2022](https://proceedings.mlr.press/v162/schirmer22a.html)。原代码：`Continuous-Recurrent-Units/lib/{CRU,CRULayer,CRUCell,encoder,decoder,models}.py`。

核心是观测编码器产生潜在均值和方差，连续时间状态转移在观测之间推进均值/协方差，观测到达时执行 Kalman 更新，再由解码器输出预测分布。实际时间间隔参与矩阵指数，不能以 GRU 拼接时间间隔代替后仍命名 CRU。未来查询应设为未观测，不执行伪观测更新。

原模型有一般 CRU 与快速 f-CRU 变体；采用哪一种需在最终模型配置中记录。以均方误差训练均值属于点预测适配；若声称复现概率版本，应保留其方差及相应似然训练。时间单位可统一缩放，但不能每个样本按自身长短任意缩放而丢失真实跨度。

### Latent ODE

[Latent ODEs for Irregularly-Sampled Time Series，NeurIPS 2019](https://arxiv.org/abs/1907.03907)。原代码：`latent_ode/lib/{encoder_decoder,latent_ode,diffeq_solver,ode_func,base_models,create_latent_ode_model}.py`。

完整模型用反向 ODE-RNN 从历史估计初始隐状态分布，采样初态后沿生成 ODE 推进，再逐查询解码。保留编码器的连续演化、观测更新、初态分布及独立生成动力学。普通 GRU 编码器加 MLP 查询头不等价。原论文还区分 ODE-RNN 与 Latent ODE；二者不能混名。

原方法是变分训练。若采用 Time-IMM 基准的统一 MSE 训练方式，必须明确写“基准式 CSI 点预测适配”，不能称完全复现原 ELBO。推理可用固定种子的多样本均值，或标明使用初态后验均值的确定性近似。

### Neural Flow

[Neural Flows: Efficient Alternative to Neural ODEs，NeurIPS 2021](https://arxiv.org/abs/2110.13040)。原代码：`neural-flows-experiments/nfe/models/flow.py`、`nfe/experiments/latent_ode/lib/`；上游依赖 `stribor==0.1.0`。

Time-IMM 配置对应潜在变量预测框架中的 coupling flow：用显式、时间条件化的可逆变换替代数值 ODE 求解。必须保留交替耦合掩码、尺度/平移结构及零时间恒等映射。普通无约束 MLP 接收“状态+时间”不等价于 Neural Flow。观察间的编码更新与未来查询都应使用一致的时间原点。

该方法通常比逐步 ODE 解算便于批量连续查询，但实际速度以本机测量为准。由于原模型也估计随机初态，单次采样不等于确定性预测。

### t-PatchGNN

[Irregular Multivariate Time Series Forecasting: A Transformable Patching Graph Neural Networks Approach，ICML 2024](https://proceedings.mlr.press/v235/zhang24bw.html)。原代码：`t-PatchGNN/tPatchGNN/model/tPatchGNN.py`，以及 `lib/parse_datasets.py` 的窗口整理。

在固定时间跨度的 patch 内，允许观测数量变化；用依赖数值和实际时间的动态滤波汇聚局部观测，再用 patch Transformer 和随时间变化的变量图传播建模。查询时间编码与历史特征一起进入解码器。需保留 TTCN、patch 时序建模、动态图学习及查询头；仅做均值池化加 Transformer 不是原模型。

动态图含变量维度的平方开销。直接把 64×128 个复数展开成 16384 个实变量，单个邻接矩阵就有约 2.68 亿项，不适合本机批量训练。应在共同 CSI 空间表示上使用真实 t-PatchGNN 机制，并明确节点是压缩系数而非物理天线。

## 4. 代码审计发现的陷阱

以下来自逐文件检查，不依据旧实验猜测。

1. **IMM-TSF 不等于所有原方法的原始训练器。** `IMM-TSF/lib/evaluation.py::compute_all_losses` 调用 `forecasting` 后计算 MSE；不会自动加入 LatentODE/NeuralFlow 的 KL，也不使用 CRU 输出方差。因此“来自官方库”不能代替训练目标的核对。
2. **一次采样不等于确定性。** IMM 的 `models/NeuralFlow.py` 注释称 `n_traj_samples=1` 为 deterministic，但其 core 的 `get_reconstruction` 明确调用高斯采样。必须固定评估策略，避免每次重画图结果变化。
3. **ODE 的初始时间。** 原 `latent_ode/lib/diffeq_solver.py` 将初态交给查询向量的第一个时间点。如果预测调用只提供未来查询而没有包含初態参考时间，会把初态错误地搬到未来。应从历史初始参考时间积分，然后取未来结果；单查询也必须走完整跨度。
4. **不等长 batch 不能擅自共享第一行时间。** 原 Latent ODE 常使用 batch 合并后的全局时间向量及 mask。CSI 新接口可按序列积分或用批量独立间隔，但不能只用 batch 中第一个样本的时间供所有样本使用。
5. **查询集合不应改变单点预测。** 一个时刻独立查询和放入多查询集合时应相符。ODE 要有同一初态原点；CRU 在无观测未来点间重新条件化转移系数可能引入查询划分依赖，需测试并从同一个最后历史后验独立预测。
6. **mask 与填充值。** padding 不能成为伪 CSI 观测；未来值不得进入编码器。共享空间投影后的完整 CSI 系数具有共同时间戳，本轮不声称解决 CSI 元素各自异步的额外任务。
7. **源码导入与设备。** t-PatchGNN 原代码有直接 `.cuda()`；多个仓库使用泛名 `lib`，可能导入冲突。适配器应隔离命名空间并显式传设备，不能靠改变 `sys.path` 顺序碰运气。

## 5. 推荐的矩阵适配与边界

建议四类方法共享相同的、只用训练轨迹建立的空间表示。先考察复数 PCA/SVD 或固定频域基的低秩表示，再根据验证集重建误差决定维度；优先选择可审计的线性投影，不急于引入额外自编码器。

- 原始目标始终保存完整 `complex64 [64,128]`。实部、虚部都参与损失和评估，不只预测幅值或波束。
- 可在训练时用一段历史确定的正比例尺度归一化整窗；这个同一尺度同时作用于全部历史与未来，并可逆还原。不能用未来能量归一化输入，也不能逐帧单位化之后仍声称预测了衰落。
- 空间基只在训练轨迹拟合；阶数选择使用验证集，不看测试预测排名。预测系数解码回完整矩阵后报告 NMSE，同时报告该基在真值上的重建误差下界。
- 若学习空间瓶颈的误差已经接近预测误差，说明实验受表示瓶颈限制，应增加阶数或换表示，而不是解释为模型不会处理非规则时间。
- 坐标、LoS 标签、速度、遮挡事件可用于分组分析，**不输入当前 CSI 单模态模型**。
- 数据集 500 Hz 是 2 ms 真值网格。模型接口可以接收连续浮点查询；若测试只读取已有网格，结论仅覆盖这些离散真值。检验真正 off-grid 泛化需要额外按非网格时刻执行 RT，不能用 CSI 插值当作真实标签。

## 6. 对非均匀时间实验的建议

所有模型使用相同训练/验证/测试轨迹划分，所有对比共享同一组截止时间与查询，避免把轨迹差异误认为时间采样差异。

先固定历史时间跨度和观测数量，比较规则、随机稀疏、抖动、成簇以及末端观测空缺。规则与随机对比应尽量固定最后一次观测时间；否则观测新鲜度与间隔不规则程度混杂。随后单独研究历史数量和最后观测延迟。

另做同一模型的真实时间输入与等间隔伪时间消融，查询保持真实，检验是否真正利用时间。对“任意数量”测试训练区间内多种数量，也标明少于/多于训练范围的外推。模型支持连续查询不意味着无限预测时域；应明确训练与测试的未来时间上限。

简单参照建议为保持、使用真实时间拟合的短窗线性趋势、连续时间常速度 Kalman。Kalman 的状态包含系数值和变化率，转移与过程噪声必须依赖实际间隔，参数仅在验证集选取。保持基线可以直接作用于完整 H；若其他基线在压缩系数空间运行，应同时报告表示差异。

不只给总均值：至少报告不同查询时域、观测数量、采样模式及 LoS/NLoS/过渡状态结果。低增益时逐样本 NMSE 和全局能量加权 NMSE会明显不同，均需定义清楚；无需剔除测试中难预测的真实衰落来美化结果。

## 7. 完成状态

本审计已完成原论文/源码选择与风险核查。实际适配、损失、空间阶数、训练配置、运行结果应由最终实现和运行生成的说明补全；本文件不预先声称某模型优于其他模型，也不把任何尚未训练的模型标为完成。
