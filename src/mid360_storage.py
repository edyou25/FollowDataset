"""
Mid-360 data storage helpers.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

import numpy as np
import zarr

from .data_storage import DataStorage


class Mid360DataStorage(DataStorage):
    """Store trajectories together with simulated Mid-360 point clouds."""

    def __init__(self, base_dir: str = "data"):
        super().__init__(base_dir=base_dir)
        self.robot_base_poses = []
        self.human_base_poses = []
        self.mid360_poses = []
        self.point_cloud_frames = []
        self.point_cloud_timestamps = []
        self.point_cloud_fields: list[str] | None = None

    def start_recording(self):
        super().start_recording()
        self.robot_base_poses = []
        self.human_base_poses = []
        self.mid360_poses = []
        self.point_cloud_frames = []
        self.point_cloud_timestamps = []
        self.point_cloud_fields = None

    def record_frame(
        self,
        robot_pos: np.ndarray,
        human_pos: np.ndarray,
        timestamp: Optional[float] = None,
        *,
        robot_base_pose: Optional[np.ndarray] = None,
        human_base_pose: Optional[np.ndarray] = None,
        mid360_pose: Optional[np.ndarray] = None,
        point_cloud: Optional[np.ndarray] = None,
        point_cloud_timestamp: Optional[float] = None,
        point_cloud_fields: Optional[list[str]] = None,
    ):
        super().record_frame(robot_pos=robot_pos, human_pos=human_pos, timestamp=timestamp)

        self.robot_base_poses.append(
            np.asarray(robot_base_pose, dtype=np.float64).copy()
            if robot_base_pose is not None
            else np.zeros((7,), dtype=np.float64)
        )
        self.human_base_poses.append(
            np.asarray(human_base_pose, dtype=np.float64).copy()
            if human_base_pose is not None
            else np.zeros((7,), dtype=np.float64)
        )
        self.mid360_poses.append(
            np.asarray(mid360_pose, dtype=np.float64).copy()
            if mid360_pose is not None
            else np.zeros((7,), dtype=np.float64)
        )

        if point_cloud_fields is not None:
            if self.point_cloud_fields is None:
                self.point_cloud_fields = list(point_cloud_fields)
            elif list(point_cloud_fields) != self.point_cloud_fields:
                raise ValueError(
                    f"Inconsistent point-cloud fields: {point_cloud_fields} vs {self.point_cloud_fields}"
                )

        n_fields = len(self.point_cloud_fields or [])
        if point_cloud is None:
            frame = np.zeros((0, n_fields), dtype=np.float32)
        else:
            frame = np.asarray(point_cloud, dtype=np.float32)
            if frame.ndim != 2:
                raise ValueError(f"Expected point cloud with shape (N, F), got {frame.shape}")
            if n_fields == 0:
                n_fields = frame.shape[1]
                self.point_cloud_fields = [f"field_{i}" for i in range(n_fields)]
            if frame.shape[1] != n_fields:
                raise ValueError(
                    f"Point cloud field count mismatch: got {frame.shape[1]}, expected {n_fields}"
                )

        self.point_cloud_frames.append(frame.copy())
        self.point_cloud_timestamps.append(
            float(point_cloud_timestamp) if point_cloud_timestamp is not None else float(self.timestamps[-1])
        )

    def _save_zarr(self, path: str):
        robot_arr = np.asarray(self.robot_trajectory, dtype=np.float64)
        human_arr = np.asarray(self.human_trajectory, dtype=np.float64)
        time_arr = np.asarray(self.timestamps, dtype=np.float64)

        store = zarr.DirectoryStore(path)
        root = zarr.group(store=store, overwrite=True)

        root.create_dataset(
            "robot_path",
            data=robot_arr,
            chunks=(min(max(len(robot_arr), 1), 1000), 2),
            dtype="float64",
        )
        root.create_dataset(
            "human_path",
            data=human_arr,
            chunks=(min(max(len(human_arr), 1), 1000), 2),
            dtype="float64",
        )
        root.create_dataset(
            "timestamps",
            data=time_arr,
            chunks=(min(max(len(time_arr), 1), 1000),),
            dtype="float64",
        )

        if self.robot_base_poses:
            robot_pose_arr = np.asarray(self.robot_base_poses, dtype=np.float64)
            human_pose_arr = np.asarray(self.human_base_poses, dtype=np.float64)
            mid360_pose_arr = np.asarray(self.mid360_poses, dtype=np.float64)
            root.create_dataset(
                "robot_base_pose",
                data=robot_pose_arr,
                chunks=(min(max(len(robot_pose_arr), 1), 1000), 7),
                dtype="float64",
            )
            root.create_dataset(
                "human_base_pose",
                data=human_pose_arr,
                chunks=(min(max(len(human_pose_arr), 1), 1000), 7),
                dtype="float64",
            )
            root.create_dataset(
                "mid360_pose",
                data=mid360_pose_arr,
                chunks=(min(max(len(mid360_pose_arr), 1), 1000), 7),
                dtype="float64",
            )

        n_fields = len(self.point_cloud_fields or [])
        cloud_offsets = []
        cloud_sizes = []
        total = 0
        for frame in self.point_cloud_frames:
            total += int(frame.shape[0])
            cloud_offsets.append(total)
            cloud_sizes.append(int(frame.shape[0]))

        if self.point_cloud_frames:
            if total > 0:
                cloud_values = np.concatenate(self.point_cloud_frames, axis=0).astype(np.float32, copy=False)
            else:
                cloud_values = np.zeros((0, n_fields), dtype=np.float32)
            root.create_dataset(
                "point_cloud_values",
                data=cloud_values,
                chunks=(min(max(len(cloud_values), 1), 50000), max(n_fields, 1)),
                dtype="float32",
            )
            root.create_dataset(
                "point_cloud_offsets",
                data=np.asarray(cloud_offsets, dtype=np.int64),
                chunks=(min(max(len(cloud_offsets), 1), 1000),),
                dtype="int64",
            )
            root.create_dataset(
                "point_cloud_sizes",
                data=np.asarray(cloud_sizes, dtype=np.int64),
                chunks=(min(max(len(cloud_sizes), 1), 1000),),
                dtype="int64",
            )
            root.create_dataset(
                "point_cloud_timestamps",
                data=np.asarray(self.point_cloud_timestamps, dtype=np.float64),
                chunks=(min(max(len(self.point_cloud_timestamps), 1), 1000),),
                dtype="float64",
            )

        root.attrs["num_frames"] = len(self.timestamps)
        root.attrs["duration"] = self.timestamps[-1] if self.timestamps else 0.0
        root.attrs["created_at"] = datetime.now().isoformat()
        root.attrs["point_cloud_fields"] = list(self.point_cloud_fields or [])

    def _save_metadata(
        self,
        path: str,
        reference_path: np.ndarray,
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        obstacles: Optional[np.ndarray],
        segment_obstacles: Optional[np.ndarray],
        episode_name: str,
        extra_metadata: Optional[dict] = None,
    ):
        robot_arr = np.asarray(self.robot_trajectory, dtype=np.float64)
        human_arr = np.asarray(self.human_trajectory, dtype=np.float64)
        ref_path = np.asarray(reference_path, dtype=np.float64)

        metadata = {
            "episode_name": episode_name,
            "created_at": datetime.now().isoformat(),
            "num_frames": len(self.timestamps),
            "duration_seconds": self.timestamps[-1] if self.timestamps else 0.0,
            "start_position": np.asarray(start_pos, dtype=np.float64).tolist(),
            "end_position": np.asarray(end_pos, dtype=np.float64).tolist(),
            "reference_path_length": self._compute_path_length(ref_path),
            "robot_path_length": self._compute_path_length(robot_arr),
            "human_path_length": self._compute_path_length(human_arr),
            "reference_path": ref_path.tolist(),
            "point_cloud_fields": list(self.point_cloud_fields or []),
            "point_cloud_frames": len(self.point_cloud_frames),
        }

        if obstacles is not None and len(obstacles) > 0:
            metadata["obstacles"] = np.asarray(obstacles).tolist()
        if segment_obstacles is not None and len(segment_obstacles) > 0:
            metadata["segment_obstacles"] = np.asarray(segment_obstacles).tolist()
        if extra_metadata:
            metadata.update(extra_metadata)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def clear(self):
        super().clear()
        self.robot_base_poses = []
        self.human_base_poses = []
        self.mid360_poses = []
        self.point_cloud_frames = []
        self.point_cloud_timestamps = []
        self.point_cloud_fields = None
