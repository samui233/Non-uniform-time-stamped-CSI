# 四条测试轨迹的完整四视角 RGB

保留全部原始 400×400 PNG，不裁剪、不放大、不筛除无人机不可见的帧。采样频率为 20 Hz。

|轨迹|类别|空间组|时刻数|全部图像数|时间范围|
|---|---|---|---:|---:|---|
|cruise_010|巡航|—|161|644|0–8 s|
|cruise_018|巡航|—|161|644|0–8 s|
|occlusion_v4_000|遮挡|spatial_19|51|204|0–2.5 s|
|occlusion_v4_001|遮挡|spatial_10|51|204|0–2.5 s|

均来自测试集。巡航选择两条不同轨迹，遮挡选择两个不同空间组；不按预测效果或可见帧比例筛选。它们是最终模型实际使用的 RGB 数据源，模型读取其冻结 CNN 特征。

每条轨迹按 north（北）、east（东）、south（南）、west（西）分为四个独立 ZIP，以满足 GitHub 单文件大小限制。下载四个 ZIP 后解压到同一个目录，即得到该轨迹完整的四视角图片。各方向同名图片对应同一个采样时刻；时间对应关系见 frame_times.csv。

## 下载

### cruise_010

- [north ZIP](https://github.com/samui233/Non-uniform-time-stamped-CSI/raw/refs/heads/main/experiments/rgb_four_test_routes_20260909/cruise_010/cruise_010_north.zip)
- [east ZIP](https://github.com/samui233/Non-uniform-time-stamped-CSI/raw/refs/heads/main/experiments/rgb_four_test_routes_20260909/cruise_010/cruise_010_east.zip)
- [south ZIP](https://github.com/samui233/Non-uniform-time-stamped-CSI/raw/refs/heads/main/experiments/rgb_four_test_routes_20260909/cruise_010/cruise_010_south.zip)
- [west ZIP](https://github.com/samui233/Non-uniform-time-stamped-CSI/raw/refs/heads/main/experiments/rgb_four_test_routes_20260909/cruise_010/cruise_010_west.zip)

### cruise_018

- [north ZIP](https://github.com/samui233/Non-uniform-time-stamped-CSI/raw/refs/heads/main/experiments/rgb_four_test_routes_20260909/cruise_018/cruise_018_north.zip)
- [east ZIP](https://github.com/samui233/Non-uniform-time-stamped-CSI/raw/refs/heads/main/experiments/rgb_four_test_routes_20260909/cruise_018/cruise_018_east.zip)
- [south ZIP](https://github.com/samui233/Non-uniform-time-stamped-CSI/raw/refs/heads/main/experiments/rgb_four_test_routes_20260909/cruise_018/cruise_018_south.zip)
- [west ZIP](https://github.com/samui233/Non-uniform-time-stamped-CSI/raw/refs/heads/main/experiments/rgb_four_test_routes_20260909/cruise_018/cruise_018_west.zip)

### occlusion_v4_000

- [north ZIP](https://github.com/samui233/Non-uniform-time-stamped-CSI/raw/refs/heads/main/experiments/rgb_four_test_routes_20260909/occlusion_v4_000/occlusion_v4_000_north.zip)
- [east ZIP](https://github.com/samui233/Non-uniform-time-stamped-CSI/raw/refs/heads/main/experiments/rgb_four_test_routes_20260909/occlusion_v4_000/occlusion_v4_000_east.zip)
- [south ZIP](https://github.com/samui233/Non-uniform-time-stamped-CSI/raw/refs/heads/main/experiments/rgb_four_test_routes_20260909/occlusion_v4_000/occlusion_v4_000_south.zip)
- [west ZIP](https://github.com/samui233/Non-uniform-time-stamped-CSI/raw/refs/heads/main/experiments/rgb_four_test_routes_20260909/occlusion_v4_000/occlusion_v4_000_west.zip)

### occlusion_v4_001

- [north ZIP](https://github.com/samui233/Non-uniform-time-stamped-CSI/raw/refs/heads/main/experiments/rgb_four_test_routes_20260909/occlusion_v4_001/occlusion_v4_001_north.zip)
- [east ZIP](https://github.com/samui233/Non-uniform-time-stamped-CSI/raw/refs/heads/main/experiments/rgb_four_test_routes_20260909/occlusion_v4_001/occlusion_v4_001_east.zip)
- [south ZIP](https://github.com/samui233/Non-uniform-time-stamped-CSI/raw/refs/heads/main/experiments/rgb_four_test_routes_20260909/occlusion_v4_001/occlusion_v4_001_south.zip)
- [west ZIP](https://github.com/samui233/Non-uniform-time-stamped-CSI/raw/refs/heads/main/experiments/rgb_four_test_routes_20260909/occlusion_v4_001/occlusion_v4_001_west.zip)

