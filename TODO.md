# 导盲犬机器人数据采集工具 - 开发清单

## 1. 环境配置 ✅
- [x] conda环境创建成功
- [x] 所有依赖可正常导入 (numpy, scipy, pygame, zarr)

## 2. 路径生成器 ✅
- [x] 能生成~50m随机路径 (实测: 50.00m)
- [x] 路径平滑（样条插值, 500点）

## 3. 物理引擎 ✅
- [x] 键盘控制机器人移动
- [x] 绳子牵引人类跟随 (约束1.5m)

## 4. 可视化模块 ✅
- [x] pygame窗口正常显示
- [x] 显示机器人、人类、起点、终点、参考路径

## 5. 数据存储 ✅
- [x] Zarr保存轨迹数据 (robot_path, human_path, timestamps)
- [x] JSON保存元数据 (episode_name, positions, lengths)

## 6. 集成测试 ✅
- [x] 完整流程可运行

---

## 运行命令
```bash
conda activate data-env
python collect.py
```
