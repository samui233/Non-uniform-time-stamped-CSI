# 本地射线几何修正：1 mm共面镜像容差

## 使用范围

正式数据版本使用 `rt_runtime.apply_rt_fix(sim)`，在创建 SimART 的 `OfflineSionnaSimulator` 后调用。它只替换该实例 `sim.solver._image_method`，不修改系统安装的Sionna、其他正在运行的进程、地图、发射机、接收机或传播开关。

```python
from rt_runtime import apply_rt_fix
sim = OfflineSionnaSimulator(SimulationConfig(**frozen_config))
fix_provenance = apply_rt_fix(sim, tolerance_m=0.001)
```

`rt_runtime/provenance.json` 保存原始文件与修改后文件的SHA-256、唯一改动表达式及许可。helper核对本地修正版哈希，发生变化会报错；返回的来源信息也附于 `sim._csi_geometry_fix`。实际配置应保存 `image_coplanarity_tolerance_m: 0.001`，而不只在文字中宣称修复。

## 唯一修改

原 `image_method.py` 的镜面路径回溯有两个接受分支：命中原候选三角面，或者虽命中另一三角面、但镜像位置足够接近。第一分支完全保留；只将第二分支的平方距离阈值从 `1e-4` 改为 `1e-6`，相当于实际距离从1厘米收紧到1毫米。

```python
dr.squared_norm(exp_img_img - new_img_img) < 1e-6
```

同一文件的其他代码逐字保留，NVIDIA原版权与Apache-2.0标记保留。并未改变PlaneHasher量化、候选数量、射线方向、反射/绕射场系数、时延归一化、相位或CSI幅度计算。

## 为什么需要修改

训练轨迹中发现LoS保持时出现约10–11 dB的2 ms增益跳变。逐路径检查显示，多个不同候选通过较宽松的共面接受条件收敛到同一地面反射附近，仍被作为多条近重复路径相干求和。增加四倍搜索不能解决；关闭绕射也不能解决；关闭镜面反射后跳变消失。全部mesh改用面法线也无改善，因此问题不是视觉平滑法线开关。

原候选平面哈希的距离尺度为1毫米，而后续镜像回溯允许1厘米差异，两阶段对“同一平面”的容忍尺度不一致。收紧后处理几何接受条件，避免把近似但不一致的候选重复当成有效镜面路径。这是路径几何验证修正，**不是对生成曲线平滑或限幅**。

完整证据见 `DISCONTINUITY_DIAGNOSIS.md` 与 `results/discontinuity_audit/`。两个原异常点修复后2 ms变化约0.04–0.15 dB；独立遮挡入影/出影检查仍保留LoS→NLoS→LoS和渐变衰落。0.1毫米阈值更容易受当前浮点精度影响，因此本轮统一采用1毫米。

## 可复现性与边界

原文件SHA-256：`f1983dc1b5a6038f95788f331855f798552e216caa04c6361443ec0cc5015d2a`。

修正版SHA-256：`ffecd1fa450ac74497a4b861489a3b1b1d0f122f0c1c1cba0546a0952af2c106`。

这是针对当前Sionna-RT 2.0.1、BigCity简化mesh及实测异常的项目级修正，并非声称原库在所有场景错误，也不保证所有数值重复、多次反射、弱绕射都已完全解决。短窗两预算检查只能支持局部稳定性；正式数据仍应完成全轨迹质量检查。地图本身含高频细节、有限射线候选和路径可见性切换，仍允许真实信道快速起伏。

带近重复候选的原始数据须单独归档。修正版要沿**同一120条轨迹重新追踪**，不混用两版本H，不根据模型性能选择某个版本的样本，也不因为某帧难预测就删除它。

采样分辨率审计须使用与主数据相同的helper。500 Hz真值及额外2000 Hz验证窗口都应采用1毫米容差，否则混入物理版本差异，会把它错误地解释成时间采样误差。
