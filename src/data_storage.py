"""
数据存储模块 - Zarr + JSON元数据
"""
import os
import json
import numpy as np
import zarr
from datetime import datetime
from typing import Optional, List


class DataStorage:
    """数据存储管理器"""
    
    def __init__(self, base_dir: str = "data"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        
        # 当前会话数据
        self.current_session = None
        self.robot_trajectory = []
        self.human_trajectory = []
        self.timestamps = []
        self.start_time = None
    
    def start_recording(self):
        """开始录制"""
        self.robot_trajectory = []
        self.human_trajectory = []
        self.timestamps = []
        self.start_time = datetime.now()
    
    def record_frame(
        self,
        robot_pos: np.ndarray,
        human_pos: np.ndarray,
        timestamp: Optional[float] = None
    ):
        """记录一帧数据"""
        if timestamp is None:
            timestamp = (datetime.now() - self.start_time).total_seconds()
        
        self.robot_trajectory.append(robot_pos.copy())
        self.human_trajectory.append(human_pos.copy())
        self.timestamps.append(timestamp)
    
    def save_episode(
        self,
        reference_path: np.ndarray,
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        obstacles: Optional[np.ndarray] = None,
        segment_obstacles: Optional[np.ndarray] = None,
        episode_name: Optional[str] = None,
        extra_metadata: Optional[dict] = None
    ) -> str:
        """
        保存一个episode的数据
        
        Returns:
            str: 保存的目录路径
        """
        if len(self.robot_trajectory) == 0:
            raise ValueError("没有录制数据可保存")
        
        # 生成episode名称
        if episode_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            episode_name = f"episode_{timestamp}"
        
        episode_dir = os.path.join(self.base_dir, episode_name)
        os.makedirs(episode_dir, exist_ok=True)
        
        # 保存Zarr数据
        zarr_path = os.path.join(episode_dir, "trajectory.zarr")
        self._save_zarr(zarr_path)
        
        # 保存JSON元数据
        meta_path = os.path.join(episode_dir, "metadata.json")
        self._save_metadata(
            meta_path,
            reference_path=reference_path,
            start_pos=start_pos,
            end_pos=end_pos,
            obstacles=obstacles,
            segment_obstacles=segment_obstacles,
            episode_name=episode_name,
            extra_metadata=extra_metadata
        )
        
        print(f"Episode saved to: {episode_dir}")
        return episode_dir
    
    def _save_zarr(self, path: str):
        """保存轨迹数据到Zarr"""
        # 转换为numpy数组
        robot_arr = np.array(self.robot_trajectory)
        human_arr = np.array(self.human_trajectory)
        time_arr = np.array(self.timestamps)
        
        # 创建Zarr存储
        store = zarr.DirectoryStore(path)
        root = zarr.group(store=store, overwrite=True)
        
        # 保存数据集
        root.create_dataset(
            'robot_path',
            data=robot_arr,
            chunks=(1000, 2),
            dtype='float64'
        )
        root.create_dataset(
            'human_path',
            data=human_arr,
            chunks=(1000, 2),
            dtype='float64'
        )
        root.create_dataset(
            'timestamps',
            data=time_arr,
            chunks=(1000,),
            dtype='float64'
        )
        
        # 添加属性
        root.attrs['num_frames'] = len(self.timestamps)
        root.attrs['duration'] = self.timestamps[-1] if self.timestamps else 0
        root.attrs['created_at'] = datetime.now().isoformat()
    
    def _save_metadata(
        self,
        path: str,
        reference_path: np.ndarray,
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        obstacles: Optional[np.ndarray],
        segment_obstacles: Optional[np.ndarray],
        episode_name: str,
        extra_metadata: Optional[dict] = None
    ):
        """保存元数据到JSON"""
        # 计算轨迹长度
        robot_arr = np.array(self.robot_trajectory)
        robot_length = self._compute_path_length(robot_arr)
        
        human_arr = np.array(self.human_trajectory)
        human_length = self._compute_path_length(human_arr)
        
        ref_length = self._compute_path_length(reference_path)
        
        metadata = {
            'episode_name': episode_name,
            'created_at': datetime.now().isoformat(),
            'num_frames': len(self.timestamps),
            'duration_seconds': self.timestamps[-1] if self.timestamps else 0,
            'start_position': start_pos.tolist(),
            'end_position': end_pos.tolist(),
            'reference_path_length': ref_length,
            'robot_path_length': robot_length,
            'human_path_length': human_length,
            'reference_path': reference_path.tolist(),
        }

        if obstacles is not None and len(obstacles) > 0:
            metadata['obstacles'] = np.asarray(obstacles).tolist()
        if segment_obstacles is not None and len(segment_obstacles) > 0:
            metadata['segment_obstacles'] = np.asarray(segment_obstacles).tolist()
        
        # Add extra metadata (like scores)
        if extra_metadata:
            metadata.update(extra_metadata)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def _compute_path_length(self, path: np.ndarray) -> float:
        """计算路径长度"""
        if len(path) < 2:
            return 0.0
        diffs = np.diff(path, axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))
    
    def clear(self):
        """清除当前录制数据"""
        self.robot_trajectory = []
        self.human_trajectory = []
        self.timestamps = []
        self.start_time = None
    
    def get_num_points(self) -> int:
        """获取当前录制的点数"""
        return len(self.robot_trajectory)
    
    @staticmethod
    def load_episode(episode_dir: str) -> dict:
        """加载一个episode的数据"""
        # 加载Zarr数据
        zarr_path = os.path.join(episode_dir, "trajectory.zarr")
        store = zarr.DirectoryStore(zarr_path)
        root = zarr.open_group(store=store, mode='r')
        
        data = {
            'robot_path': root['robot_path'][:],
            'human_path': root['human_path'][:],
            'timestamps': root['timestamps'][:]
        }
        
        # 加载元数据
        meta_path = os.path.join(episode_dir, "metadata.json")
        with open(meta_path, 'r', encoding='utf-8') as f:
            data['metadata'] = json.load(f)
        
        return data
    
    @staticmethod
    def list_episodes(base_dir: str = "data") -> List[str]:
        """列出所有episode"""
        if not os.path.exists(base_dir):
            return []
        
        episodes = []
        for name in os.listdir(base_dir):
            episode_dir = os.path.join(base_dir, name)
            if os.path.isdir(episode_dir):
                zarr_path = os.path.join(episode_dir, "trajectory.zarr")
                meta_path = os.path.join(episode_dir, "metadata.json")
                if os.path.exists(zarr_path) and os.path.exists(meta_path):
                    episodes.append(name)
        
        return sorted(episodes)


if __name__ == "__main__":
    # 测试数据存储
    storage = DataStorage()
    storage.start_recording()
    
    # 模拟录制数据
    for i in range(100):
        robot_pos = np.array([i * 0.1, np.sin(i * 0.1)])
        human_pos = np.array([i * 0.1 - 1, np.sin(i * 0.1)])
        storage.record_frame(robot_pos, human_pos, i * 0.02)
    
    # 保存
    ref_path = np.array([[0, 0], [5, 0], [10, 0]])
    storage.save_episode(
        reference_path=ref_path,
        start_pos=np.array([0, 0]),
        end_pos=np.array([10, 0])
    )
    
    # 列出所有episode
    print("Episodes:", DataStorage.list_episodes())
