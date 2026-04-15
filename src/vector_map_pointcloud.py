from __future__ import annotations

"""
Offline point-cloud generation from 2D vector maps.
"""

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import zarr


@dataclass
class VectorMapPointCloudConfig:
    num_rays: int = 360
    min_angle: float = -np.pi
    max_angle: float = np.pi
    min_range: float = 0.05
    max_range: float = 20.0
    z_height: float = 0.0


def compute_path_headings(path: np.ndarray) -> np.ndarray:
    path = np.asarray(path, dtype=np.float32)
    n = len(path)
    headings = np.zeros((n,), dtype=np.float32)
    if n <= 1:
        return headings

    deltas = np.zeros_like(path, dtype=np.float32)
    deltas[0] = path[1] - path[0]
    deltas[-1] = path[-1] - path[-2]
    if n > 2:
        deltas[1:-1] = path[2:] - path[:-2]

    norms = np.linalg.norm(deltas, axis=1)
    valid = norms > 1e-6
    headings[valid] = np.arctan2(deltas[valid, 1], deltas[valid, 0]).astype(np.float32)

    last_heading = 0.0
    for idx in range(n):
        if valid[idx]:
            last_heading = float(headings[idx])
        else:
            headings[idx] = last_heading
    return headings


class VectorMapPointCloudSimulator:
    """Ray-cast point clouds against circle and segment vector-map primitives."""

    def __init__(self, config: Optional[VectorMapPointCloudConfig] = None):
        self.config = config or VectorMapPointCloudConfig()
        self.local_angles = np.linspace(
            self.config.min_angle,
            self.config.max_angle,
            int(self.config.num_rays),
            endpoint=False,
            dtype=np.float32,
        )
        self.local_dirs = np.stack(
            [np.cos(self.local_angles), np.sin(self.local_angles)],
            axis=-1,
        ).astype(np.float32)

    @property
    def fields(self) -> list[str]:
        return ["x", "y", "z", "range", "azimuth"]

    def simulate_frame(
        self,
        robot_pos: np.ndarray,
        heading: float,
        obstacles: Optional[np.ndarray],
        segment_obstacles: Optional[np.ndarray],
    ) -> np.ndarray:
        robot_pos = np.asarray(robot_pos, dtype=np.float32)
        heading = float(heading)

        cos_h = float(np.cos(heading))
        sin_h = float(np.sin(heading))
        rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]], dtype=np.float32)
        world_dirs = self.local_dirs @ rot.T

        ranges = np.full((len(self.local_dirs),), np.inf, dtype=np.float32)

        circle_ranges = self._intersect_circles(robot_pos, world_dirs, obstacles)
        if circle_ranges is not None:
            ranges = np.minimum(ranges, circle_ranges)

        segment_ranges = self._intersect_segments(robot_pos, world_dirs, segment_obstacles)
        if segment_ranges is not None:
            ranges = np.minimum(ranges, segment_ranges)

        valid = np.isfinite(ranges)
        valid &= ranges >= float(self.config.min_range)
        valid &= ranges <= float(self.config.max_range)
        if not np.any(valid):
            return np.zeros((0, len(self.fields)), dtype=np.float32)

        hit_ranges = ranges[valid]
        hit_dirs = self.local_dirs[valid]
        hit_angles = self.local_angles[valid]
        xy = hit_ranges[:, None] * hit_dirs
        z = np.full((len(hit_ranges), 1), float(self.config.z_height), dtype=np.float32)
        return np.concatenate(
            [
                xy.astype(np.float32),
                z,
                hit_ranges[:, None].astype(np.float32),
                hit_angles[:, None].astype(np.float32),
            ],
            axis=1,
        )

    def _intersect_circles(
        self,
        origin: np.ndarray,
        world_dirs: np.ndarray,
        obstacles: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        if obstacles is None or len(obstacles) == 0:
            return None
        circles = np.asarray(obstacles, dtype=np.float32)
        if circles.ndim != 2 or circles.shape[1] < 3:
            return None

        centers = circles[:, :2]
        radii = circles[:, 2]
        oc = origin[None, :] - centers
        b = world_dirs @ oc.T
        c = np.sum(oc * oc, axis=1)[None, :] - radii[None, :] * radii[None, :]
        disc = b * b - c
        valid = disc >= 0.0
        if not np.any(valid):
            return None

        sqrt_disc = np.sqrt(np.clip(disc, 0.0, None)).astype(np.float32, copy=False)
        t1 = -b - sqrt_disc
        t2 = -b + sqrt_disc
        min_range = float(self.config.min_range)
        circle_t = np.where(t1 >= min_range, t1, np.where(t2 >= min_range, t2, np.inf))
        circle_t = np.where(valid, circle_t, np.inf)
        return np.min(circle_t, axis=1).astype(np.float32)

    def _intersect_segments(
        self,
        origin: np.ndarray,
        world_dirs: np.ndarray,
        segment_obstacles: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        if segment_obstacles is None or len(segment_obstacles) == 0:
            return None
        segments = np.asarray(segment_obstacles, dtype=np.float32)
        if segments.ndim != 2 or segments.shape[1] < 4:
            return None

        p1 = segments[:, :2]
        p2 = segments[:, 2:4]
        v = p2 - p1
        w = p1 - origin[None, :]

        denom = self._cross(world_dirs[:, None, :], v[None, :, :])
        with np.errstate(divide="ignore", invalid="ignore"):
            t = self._cross(w[None, :, :], v[None, :, :]) / denom
            u = self._cross(w[None, :, :], world_dirs[:, None, :]) / denom

        valid = np.abs(denom) > 1e-8
        valid &= t >= float(self.config.min_range)
        valid &= u >= 0.0
        valid &= u <= 1.0
        seg_t = np.where(valid, t, np.inf)
        return np.min(seg_t, axis=1).astype(np.float32)

    @staticmethod
    def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def write_pointclouds_to_episode(
    episode_dir: str,
    point_cloud_frames: list[np.ndarray],
    timestamps: np.ndarray,
    point_cloud_fields: list[str],
    config: VectorMapPointCloudConfig,
    *,
    overwrite: bool = True,
) -> None:
    zarr_path = os.path.join(episode_dir, "trajectory.zarr")
    meta_path = os.path.join(episode_dir, "metadata.json")

    store = zarr.DirectoryStore(zarr_path)
    root = zarr.open_group(store=store, mode="a")

    keys = [
        "point_cloud_values",
        "point_cloud_offsets",
        "point_cloud_sizes",
        "point_cloud_timestamps",
    ]
    for key in keys:
        if key in root:
            if not overwrite:
                raise ValueError(f"Episode already contains point cloud data: {episode_dir}")
            del root[key]

    n_fields = len(point_cloud_fields)
    offsets = []
    sizes = []
    total = 0
    for frame in point_cloud_frames:
        total += int(frame.shape[0])
        offsets.append(total)
        sizes.append(int(frame.shape[0]))

    if total > 0:
        values = np.concatenate(point_cloud_frames, axis=0).astype(np.float32, copy=False)
    else:
        values = np.zeros((0, n_fields), dtype=np.float32)

    root.create_dataset(
        "point_cloud_values",
        data=values,
        chunks=(min(max(len(values), 1), 50000), max(n_fields, 1)),
        dtype="float32",
    )
    root.create_dataset(
        "point_cloud_offsets",
        data=np.asarray(offsets, dtype=np.int64),
        chunks=(min(max(len(offsets), 1), 1000),),
        dtype="int64",
    )
    root.create_dataset(
        "point_cloud_sizes",
        data=np.asarray(sizes, dtype=np.int64),
        chunks=(min(max(len(sizes), 1), 1000),),
        dtype="int64",
    )
    root.create_dataset(
        "point_cloud_timestamps",
        data=np.asarray(timestamps, dtype=np.float64),
        chunks=(min(max(len(timestamps), 1), 1000),),
        dtype="float64",
    )
    root.attrs["point_cloud_fields"] = list(point_cloud_fields)
    root.attrs["point_cloud_source"] = "vector_map_raycast"
    root.attrs["point_cloud_generated_at"] = datetime.now().isoformat()

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    metadata["point_cloud_fields"] = list(point_cloud_fields)
    metadata["point_cloud_frames"] = len(point_cloud_frames)
    metadata["point_cloud_source"] = "vector_map_raycast"
    metadata["point_cloud_generated_at"] = datetime.now().isoformat()
    metadata["point_cloud_config"] = {
        key: (float(value) if isinstance(value, np.floating) else int(value) if isinstance(value, np.integer) else value)
        for key, value in asdict(config).items()
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
