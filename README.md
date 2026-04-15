# 导盲犬机器人数据采集工具

用于模仿学习的人-机器人路径数据采集工具。

## 安装

```bash
conda env create -f environment.yml
conda activate data-env
```

## 使用

```bash
python collect.py --backend 2d
```

## 控制方式

- `↑` 前进
- `↓` 后退  
- `←` 左转
- `→` 右转
- `SPACE` 开始/暂停记录
- `S` 保存当前轨迹
- `R` 重置
- `N` 生成新路径
- `ESC` 退出

## 数据格式

- **Zarr**: 存储轨迹数据 (robot_path, human_path, timestamps)
- **JSON**: 存储元数据 (起点、终点、路径长度等)

## Mid-360 3D 仿真

`feat/mid360` 分支现在额外支持基于 `Mid360_simulation_plugin` 的 Gazebo 3D 采集后端：

```bash
python collect.py --backend mid360 \
  --mid360-plugin-dir /path/to/Mid360_simulation_plugin
```

默认挂载外参已经写入生成的机器人 SDF：

```text
T_base_mid360 =
[
  [-0.707,  0.000,  0.707,  0.170],
  [ 0.000,  1.000,  0.000,  0.000],
  [-0.707,  0.000, -0.707,  0.090],
  [ 0.000,  0.000,  0.000,  1.000]
]
```

运行 `mid360` 模式前需要：

1. 安装并 `source` ROS Noetic + Gazebo11 环境。
2. 编译 `Mid360_simulation_plugin`，保证 `liblivox_laser_simulation.so` 可被发现。
3. 如果库不在默认 catkin 位置，显式传入：

```bash
python collect.py --backend mid360 \
  --mid360-plugin-dir /path/to/Mid360_simulation_plugin \
  --mid360-plugin-lib /path/to/liblivox_laser_simulation.so
```

`mid360` 模式保存的 `trajectory.zarr` 额外包含：

- `robot_base_pose`
- `human_base_pose`
- `mid360_pose`
- `point_cloud_values`
- `point_cloud_offsets`
- `point_cloud_sizes`
- `point_cloud_timestamps`

其中点云使用拼接数组 + offsets 的方式保存每帧 ragged point cloud。

## 开发文档

- [最近一次安全过滤提交说明](docs/commit_82ed297_qp_safety_filter_zh.md)
