# 导盲犬机器人数据采集工具

用于模仿学习的人-机器人路径数据采集工具。

## 安装

```bash
conda env create -f environment.yml
conda activate data-env
```

## 使用

```bash
python collect.py
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

