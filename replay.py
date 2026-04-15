#!/usr/bin/env python3
from __future__ import annotations

"""
Dataset Replay - Playback and manage recorded trajectories.

Controls:
    UP/DOWN    Select episode
    ENTER      Load selected episode
    BACKSPACE  Delete selected episode
    C          Collect point cloud for selected/current episode
    A          Collect point cloud for all episodes (selection mode)
    SPACE      Play/Pause
    LEFT/RIGHT Step backward/forward (when paused)
    R          Restart from beginning
    +/-        Speed up/down
    WASD       Pan camera
    Scroll     Zoom in/out
    ESC        Exit / Back
"""

import argparse
import json
import os
import shutil
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

try:
    import pygame
except ModuleNotFoundError:
    pygame = None

from src.data_storage import DataStorage
from src.mid360_gazebo import (
    Mid360GazeboConfig,
    Mid360GazeboSession,
    resolve_mid360_plugin_dir,
    resolve_mid360_plugin_library,
)
from src.vector_map_pointcloud import compute_path_headings

try:
    from src.visualizer import Visualizer
except ModuleNotFoundError:
    Visualizer = None


def collect_pointcloud_for_episode(
    episode_dir: str,
    config: Mid360GazeboConfig,
    *,
    overwrite: bool = True,
    verbose: bool = True,
) -> dict:
    data = DataStorage.load_episode(episode_dir)
    meta = data["metadata"]

    robot_path = np.asarray(data["robot_path"], dtype=np.float64)
    human_path = np.asarray(data["human_path"], dtype=np.float64)
    timestamps = np.asarray(data["timestamps"], dtype=np.float64)
    headings = compute_path_headings(robot_path)

    path_data = {
        "path": np.asarray(meta.get("reference_path", robot_path), dtype=np.float64),
        "waypoints": np.asarray(meta.get("reference_path", robot_path), dtype=np.float64),
        "start": np.asarray(meta.get("start_position", robot_path[0]), dtype=np.float64),
        "end": np.asarray(meta.get("end_position", robot_path[-1]), dtype=np.float64),
        "obstacles": np.asarray(meta.get("obstacles", []), dtype=np.float64),
        "segment_obstacles": np.asarray(meta.get("segment_obstacles", []), dtype=np.float64),
    }

    if len(timestamps) >= 2:
        dt = np.diff(timestamps)
        valid_dt = dt[dt > 1e-6]
        estimated_rate = float(1.0 / np.median(valid_dt)) if len(valid_dt) > 0 else float(config.update_rate)
    else:
        estimated_rate = float(config.update_rate)
    replay_config = replace(config, update_rate=max(float(config.update_rate), estimated_rate))

    session = Mid360GazeboSession(path_data=path_data, config=replay_config)
    frames: list[np.ndarray] = []
    frame_timestamps: list[float] = []
    robot_base_poses: list[np.ndarray] = []
    human_base_poses: list[np.ndarray] = []
    mid360_poses: list[np.ndarray] = []
    point_cloud_fields: list[str] = []
    total_points = 0

    try:
        session.start()
        last_seq = None
        for robot_pos, human_pos, ts, heading in zip(robot_path, human_path, timestamps, headings):
            robot_state = SimpleNamespace(position=np.asarray(robot_pos, dtype=np.float64), heading=float(heading))
            human_state = SimpleNamespace(position=np.asarray(human_pos, dtype=np.float64))
            session.update_entities(robot_state, human_state)
            cloud = session.get_pointcloud(
                after_seq=last_seq,
                wait_timeout=max(0.5, 3.0 / float(replay_config.update_rate)),
            )
            if cloud is None:
                cloud = session.get_pointcloud(wait_timeout=max(0.5, 3.0 / float(replay_config.update_rate)))
            if cloud is None:
                raise RuntimeError(f"No Mid360 point cloud generated for frame at t={float(ts):.3f}s")
            last_seq = int(cloud["seq"])
            frame = np.asarray(cloud.get("points", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32)
            frames.append(frame)
            frame_timestamps.append(float(cloud.get("stamp", float(ts))))
            total_points += int(len(frame))
            point_cloud_fields = list(cloud.get("fields", point_cloud_fields))
            robot_base_poses.append(session.get_robot_base_pose(robot_state))
            human_base_poses.append(session.get_human_base_pose(human_state))
            mid360_poses.append(session.get_mid360_pose(robot_state))
    finally:
        session.close()

    _write_mid360_pointclouds_to_episode(
        episode_dir=episode_dir,
        point_cloud_frames=frames,
        point_cloud_timestamps=np.asarray(frame_timestamps, dtype=np.float64),
        point_cloud_fields=point_cloud_fields,
        robot_base_poses=np.asarray(robot_base_poses, dtype=np.float64),
        human_base_poses=np.asarray(human_base_poses, dtype=np.float64),
        mid360_poses=np.asarray(mid360_poses, dtype=np.float64),
        config=config,
        overwrite=overwrite,
    )

    result = {
        "episode_dir": episode_dir,
        "episode_name": meta.get("episode_name", os.path.basename(episode_dir)),
        "num_frames": int(len(robot_path)),
        "total_points": int(total_points),
        "avg_points_per_frame": float(total_points / max(len(robot_path), 1)),
    }
    if verbose:
        print(
            f"[pointcloud] {result['episode_name']}: "
            f"{result['num_frames']} frames, "
            f"{result['total_points']} points, "
            f"{result['avg_points_per_frame']:.1f} pts/frame"
        )
    return result


def collect_pointcloud_for_dataset(
    data_dir: str,
    config: Mid360GazeboConfig,
    *,
    overwrite: bool = True,
    episode_names: list[str] | None = None,
) -> list[dict]:
    episodes = episode_names if episode_names is not None else DataStorage.list_episodes(data_dir)
    results = []
    if not episodes:
        print(f"No episodes found in {data_dir}")
        return results

    total = len(episodes)
    for idx, episode_name in enumerate(episodes, start=1):
        print(f"[{idx}/{total}] collecting point cloud for {episode_name}")
        episode_dir = os.path.join(data_dir, episode_name)
        results.append(
            collect_pointcloud_for_episode(
                episode_dir=episode_dir,
                config=config,
                overwrite=overwrite,
                verbose=True,
            )
        )
    return results


def _write_mid360_pointclouds_to_episode(
    episode_dir: str,
    *,
    point_cloud_frames: list[np.ndarray],
    point_cloud_timestamps: np.ndarray,
    point_cloud_fields: list[str],
    robot_base_poses: np.ndarray,
    human_base_poses: np.ndarray,
    mid360_poses: np.ndarray,
    config: Mid360GazeboConfig,
    overwrite: bool = True,
):
    import zarr

    zarr_path = os.path.join(episode_dir, "trajectory.zarr")
    meta_path = os.path.join(episode_dir, "metadata.json")

    store = zarr.DirectoryStore(zarr_path)
    root = zarr.open_group(store=store, mode="a")

    keys = [
        "robot_base_pose",
        "human_base_pose",
        "mid360_pose",
        "point_cloud_values",
        "point_cloud_offsets",
        "point_cloud_sizes",
        "point_cloud_timestamps",
    ]
    for key in keys:
        if key in root:
            if not overwrite:
                raise ValueError(f"Episode already contains Mid360 data: {episode_dir}")
            del root[key]

    n_fields = len(point_cloud_fields)
    offsets = []
    sizes = []
    total = 0
    for frame in point_cloud_frames:
        total += int(frame.shape[0])
        offsets.append(total)
        sizes.append(int(frame.shape[0]))

    values = (
        np.concatenate(point_cloud_frames, axis=0).astype(np.float32, copy=False)
        if total > 0
        else np.zeros((0, n_fields), dtype=np.float32)
    )

    root.create_dataset("robot_base_pose", data=robot_base_poses, chunks=(min(max(len(robot_base_poses), 1), 1000), 7), dtype="float64")
    root.create_dataset("human_base_pose", data=human_base_poses, chunks=(min(max(len(human_base_poses), 1), 1000), 7), dtype="float64")
    root.create_dataset("mid360_pose", data=mid360_poses, chunks=(min(max(len(mid360_poses), 1), 1000), 7), dtype="float64")
    root.create_dataset("point_cloud_values", data=values, chunks=(min(max(len(values), 1), 50000), max(n_fields, 1)), dtype="float32")
    root.create_dataset("point_cloud_offsets", data=np.asarray(offsets, dtype=np.int64), chunks=(min(max(len(offsets), 1), 1000),), dtype="int64")
    root.create_dataset("point_cloud_sizes", data=np.asarray(sizes, dtype=np.int64), chunks=(min(max(len(sizes), 1), 1000),), dtype="int64")
    root.create_dataset("point_cloud_timestamps", data=np.asarray(point_cloud_timestamps, dtype=np.float64), chunks=(min(max(len(point_cloud_timestamps), 1), 1000),), dtype="float64")
    root.attrs["point_cloud_fields"] = list(point_cloud_fields)
    root.attrs["point_cloud_source"] = "mid360_gazebo_replay"
    root.attrs["point_cloud_generated_at"] = __import__("datetime").datetime.now().isoformat()

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    metadata["point_cloud_fields"] = list(point_cloud_fields)
    metadata["point_cloud_frames"] = len(point_cloud_frames)
    metadata["point_cloud_source"] = "mid360_gazebo_replay"
    metadata["point_cloud_generated_at"] = __import__("datetime").datetime.now().isoformat()
    metadata["mid360_topic"] = config.ros_topic
    metadata["mid360_frame"] = config.frame_name
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


class DatasetReplay:
    """Replay and manage recorded trajectories."""

    def __init__(
        self,
        data_dir: str = "data",
        pointcloud_config: Mid360GazeboConfig | None = None,
    ):
        if pygame is None or Visualizer is None:
            raise RuntimeError(
                "Interactive replay requires pygame. "
                "For batch point-cloud collection, run replay.py with --collect-pointcloud."
            )
        self.data_dir = data_dir
        self.pointcloud_config = pointcloud_config

        self.visualizer = Visualizer()
        self.visualizer.camera_follow = False

        self.episodes = DataStorage.list_episodes(data_dir)
        self.selected_idx = 0

        self.current_data = None
        self.current_meta = None
        self.current_episode = None

        self.playing = False
        self.frame_idx = 0
        self.playback_speed = 1.0
        self.accumulated_time = 0.0

        self.running = True
        self.in_selection = True
        self.confirm_delete = False
        self.delete_target = None

    def _get_frame_point_cloud_world_xy(self) -> np.ndarray | None:
        if self.current_data is None:
            return None
        values = self.current_data.get("point_cloud_values")
        offsets = self.current_data.get("point_cloud_offsets")
        sizes = self.current_data.get("point_cloud_sizes")
        if values is None or offsets is None or sizes is None:
            return None
        if self.frame_idx >= len(offsets) or self.frame_idx >= len(sizes):
            return None

        count = int(sizes[self.frame_idx])
        if count <= 0:
            return None
        end = int(offsets[self.frame_idx])
        start = end - count
        frame = np.asarray(values[start:end], dtype=np.float32)
        if frame.ndim != 2 or frame.shape[0] == 0:
            return None

        field_names = list(
            self.current_data.get(
                "point_cloud_fields",
                self.current_meta.get("point_cloud_fields", []) if self.current_meta else [],
            )
        )
        if field_names:
            try:
                ix = field_names.index("x")
                iy = field_names.index("y")
            except ValueError:
                ix, iy = 0, 1
            iz = field_names.index("z") if "z" in field_names else None
        else:
            ix, iy = 0, 1
            iz = 2 if frame.shape[1] > 2 else None

        local_xy = frame[:, [ix, iy]].astype(np.float32, copy=False)
        if "mid360_pose" in self.current_data:
            pose = np.asarray(self.current_data["mid360_pose"][self.frame_idx], dtype=np.float64)
            if pose.shape[0] >= 7:
                if iz is None or iz >= frame.shape[1]:
                    local_xyz = np.concatenate(
                        [local_xy, np.zeros((len(local_xy), 1), dtype=np.float32)],
                        axis=1,
                    )
                else:
                    local_xyz = frame[:, [ix, iy, iz]].astype(np.float32, copy=False)
                rot = Rotation.from_quat(pose[3:7]).as_matrix().astype(np.float32)
                world_xyz = local_xyz @ rot.T + pose[:3].astype(np.float32)
                return world_xyz[:, :2]

        robot_path = np.asarray(self.current_data["robot_path"], dtype=np.float32)
        robot_pos = robot_path[self.frame_idx]
        headings = compute_path_headings(robot_path)
        heading = float(headings[self.frame_idx]) if len(headings) > 0 else 0.0
        cos_h = float(np.cos(heading))
        sin_h = float(np.sin(heading))
        rot_2d = np.array([[cos_h, -sin_h], [sin_h, cos_h]], dtype=np.float32)
        return local_xy @ rot_2d.T + robot_pos

    def _refresh_episodes(self):
        self.episodes = DataStorage.list_episodes(self.data_dir)
        if self.selected_idx >= len(self.episodes):
            self.selected_idx = max(0, len(self.episodes) - 1)

    def _delete_episode(self, episode_name: str):
        episode_dir = os.path.join(self.data_dir, episode_name)
        try:
            shutil.rmtree(episode_dir)
            print(f"Deleted: {episode_name}")
            self._refresh_episodes()
            return True
        except Exception as exc:
            print(f"Delete failed: {exc}")
            return False

    def _load_episode(self, episode_name: str):
        episode_dir = os.path.join(self.data_dir, episode_name)
        self.current_data = DataStorage.load_episode(episode_dir)
        self.current_meta = self.current_data["metadata"]
        self.current_episode = episode_name
        self.frame_idx = 0
        self.playing = False
        self.accumulated_time = 0.0
        self.in_selection = False

        ref_path = np.array(self.current_meta["reference_path"], dtype=np.float32)
        center = np.mean(ref_path, axis=0)
        self.visualizer.camera_offset = center

        print(f"Loaded: {episode_name}")
        print(
            f"  Frames: {self.current_meta['num_frames']}, "
            f"Duration: {self.current_meta['duration_seconds']:.1f}s"
        )
        scores = self.current_meta.get("scores", {})
        if scores:
            print(f"  Score: {scores.get('total', 'N/A')}/100 (Grade: {scores.get('grade', 'N/A')})")
        if self.current_meta.get("point_cloud_frames"):
            print(
                f"  Point cloud: {self.current_meta.get('point_cloud_frames')} frames, "
                f"fields={self.current_meta.get('point_cloud_fields', [])}"
            )

    def _collect_selected_episode_pointcloud(self):
        if not self.episodes:
            return
        episode_name = self.episodes[self.selected_idx]
        self._collect_episode_pointcloud(episode_name)

    def _collect_current_episode_pointcloud(self):
        if not self.current_episode:
            return
        self._collect_episode_pointcloud(self.current_episode)

    def _collect_episode_pointcloud(self, episode_name: str):
        episode_dir = os.path.join(self.data_dir, episode_name)
        collect_pointcloud_for_episode(
            episode_dir=episode_dir,
            config=self.pointcloud_config,
            overwrite=True,
            verbose=True,
        )
        self._refresh_episodes()
        if self.current_episode == episode_name:
            self._load_episode(episode_name)

    def _collect_all_pointclouds(self):
        collect_pointcloud_for_dataset(
            data_dir=self.data_dir,
            config=self.pointcloud_config,
            overwrite=True,
        )
        self._refresh_episodes()
        if self.current_episode is not None and self.current_episode in self.episodes:
            self._load_episode(self.current_episode)

    def _handle_input(self):
        for event in self.visualizer.get_events():
            if event.type == pygame.QUIT:
                self.running = False

            self.visualizer.handle_event(event)

            if event.type != pygame.KEYDOWN:
                continue

            if self.confirm_delete:
                if event.key == pygame.K_y:
                    self._delete_episode(self.delete_target)
                    self.confirm_delete = False
                    self.delete_target = None
                elif event.key in (pygame.K_n, pygame.K_ESCAPE):
                    self.confirm_delete = False
                    self.delete_target = None
                    print("Delete cancelled")
                continue

            if event.key == pygame.K_ESCAPE:
                if self.in_selection:
                    self.running = False
                else:
                    self.in_selection = True
                    self.current_data = None
                    self.playing = False
                continue

            if self.in_selection:
                if event.key == pygame.K_UP:
                    self.selected_idx = max(0, self.selected_idx - 1)
                elif event.key == pygame.K_DOWN:
                    self.selected_idx = min(len(self.episodes) - 1, self.selected_idx + 1)
                elif event.key == pygame.K_RETURN and self.episodes:
                    self._load_episode(self.episodes[self.selected_idx])
                elif event.key in (pygame.K_DELETE, pygame.K_BACKSPACE) and self.episodes:
                    self.confirm_delete = True
                    self.delete_target = self.episodes[self.selected_idx]
                    print(f"Delete '{self.delete_target}'? Press Y to confirm, N to cancel")
                elif event.key == pygame.K_c and self.episodes:
                    self._collect_selected_episode_pointcloud()
                elif event.key == pygame.K_a and self.episodes:
                    self._collect_all_pointclouds()
            else:
                if event.key == pygame.K_SPACE:
                    self.playing = not self.playing
                elif event.key == pygame.K_r:
                    self.frame_idx = 0
                    self.accumulated_time = 0.0
                elif event.key == pygame.K_LEFT and not self.playing:
                    self.frame_idx = max(0, self.frame_idx - 1)
                elif event.key == pygame.K_RIGHT and not self.playing:
                    max_frame = len(self.current_data["robot_path"]) - 1
                    self.frame_idx = min(max_frame, self.frame_idx + 1)
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                    self.playback_speed = min(4.0, self.playback_speed * 1.5)
                    print(f"Speed: {self.playback_speed:.1f}x")
                elif event.key == pygame.K_MINUS:
                    self.playback_speed = max(0.25, self.playback_speed / 1.5)
                    print(f"Speed: {self.playback_speed:.1f}x")
                elif event.key in (pygame.K_DELETE, pygame.K_BACKSPACE) and self.current_episode:
                    self.confirm_delete = True
                    self.delete_target = self.current_episode
                    print(f"Delete '{self.delete_target}'? Press Y to confirm, N to cancel")
                elif event.key == pygame.K_c and self.current_episode:
                    self._collect_current_episode_pointcloud()

        if not self.in_selection and not self.confirm_delete:
            keys = pygame.key.get_pressed()
            pan_speed = 0.5
            if keys[pygame.K_w]:
                self.visualizer.camera_offset[1] += pan_speed
            if keys[pygame.K_s]:
                self.visualizer.camera_offset[1] -= pan_speed
            if keys[pygame.K_a]:
                self.visualizer.camera_offset[0] -= pan_speed
            if keys[pygame.K_d]:
                self.visualizer.camera_offset[0] += pan_speed

    def _update(self, dt: float):
        if not self.playing or self.current_data is None:
            return

        self.accumulated_time += dt * self.playback_speed
        timestamps = self.current_data["timestamps"]
        while self.frame_idx < len(timestamps) - 1:
            if timestamps[self.frame_idx + 1] <= self.accumulated_time:
                self.frame_idx += 1
            else:
                break

        if self.frame_idx >= len(timestamps) - 1:
            self.playing = False

    def _render_selection(self):
        self.visualizer.screen.fill((25, 25, 35))

        title = self.visualizer.font_large.render("Select Episode", True, (220, 220, 220))
        self.visualizer.screen.blit(title, (self.visualizer.width // 2 - 80, 30))

        count_text = self.visualizer.font.render(
            f"Total: {len(self.episodes)} episodes",
            True,
            (150, 150, 150),
        )
        self.visualizer.screen.blit(count_text, (self.visualizer.width - 180, 35))

        if not self.episodes:
            no_data = self.visualizer.font.render("No episodes found in data/", True, (150, 150, 150))
            self.visualizer.screen.blit(no_data, (self.visualizer.width // 2 - 100, 100))
        else:
            y = 80
            for i, episode_name in enumerate(self.episodes):
                try:
                    meta_path = os.path.join(self.data_dir, episode_name, "metadata.json")
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)

                    scores = meta.get("scores", {})
                    grade = scores.get("grade", "?")
                    total = scores.get("total", 0)
                    frames = meta.get("num_frames", 0)
                    duration = meta.get("duration_seconds", 0.0)
                    pc_frames = meta.get("point_cloud_frames", 0)
                    display_text = (
                        f"{episode_name}  |  {frames} frames  |  {duration:.1f}s  |  "
                        f"Score: {total:.0f} ({grade})  |  PC: {pc_frames}"
                    )
                except Exception:
                    display_text = episode_name

                if i == self.selected_idx:
                    color = (100, 200, 255)
                    pygame.draw.rect(
                        self.visualizer.screen,
                        (40, 60, 80),
                        (40, y - 5, self.visualizer.width - 80, 30),
                        border_radius=5,
                    )
                    indicator = self.visualizer.font.render(">", True, color)
                    self.visualizer.screen.blit(indicator, (50, y))
                else:
                    color = (180, 180, 180)

                text = self.visualizer.font.render(display_text, True, color)
                self.visualizer.screen.blit(text, (70, y))
                y += 35

        if self.confirm_delete:
            overlay = pygame.Surface((self.visualizer.width, self.visualizer.height))
            overlay.fill((0, 0, 0))
            overlay.set_alpha(180)
            self.visualizer.screen.blit(overlay, (0, 0))

            dialog_w, dialog_h = 500, 150
            dialog_x = (self.visualizer.width - dialog_w) // 2
            dialog_y = (self.visualizer.height - dialog_h) // 2

            pygame.draw.rect(
                self.visualizer.screen,
                (50, 50, 60),
                (dialog_x, dialog_y, dialog_w, dialog_h),
                border_radius=10,
            )
            pygame.draw.rect(
                self.visualizer.screen,
                (100, 100, 120),
                (dialog_x, dialog_y, dialog_w, dialog_h),
                2,
                border_radius=10,
            )

            warn_text = self.visualizer.font_large.render("Delete Episode?", True, (255, 100, 100))
            self.visualizer.screen.blit(warn_text, (dialog_x + dialog_w // 2 - 80, dialog_y + 20))

            ep_text = self.visualizer.font.render(self.delete_target, True, (200, 200, 200))
            self.visualizer.screen.blit(ep_text, (dialog_x + dialog_w // 2 - 100, dialog_y + 60))

            hint_text = self.visualizer.font.render(
                "Press Y to confirm, N to cancel",
                True,
                (150, 150, 150),
            )
            self.visualizer.screen.blit(hint_text, (dialog_x + dialog_w // 2 - 130, dialog_y + 100))

        hints = (
            "UP/DOWN: Select  |  ENTER: Load  |  C: Collect PC  |  "
            "A: Collect All PC  |  BACKSPACE: Delete  |  ESC: Exit"
        )
        hint_surface = self.visualizer.font.render(hints, True, (100, 100, 100))
        self.visualizer.screen.blit(
            hint_surface,
            (self.visualizer.width // 2 - 360, self.visualizer.height - 40),
        )

        pygame.display.flip()

    def _render_playback(self):
        if self.current_data is None:
            return

        robot_path = self.current_data["robot_path"]
        human_path = self.current_data["human_path"]
        ref_path = np.array(self.current_meta["reference_path"], dtype=np.float32)
        start_pos = np.array(self.current_meta["start_position"], dtype=np.float32)
        end_pos = np.array(self.current_meta["end_position"], dtype=np.float32)
        obstacles = np.array(self.current_meta.get("obstacles", []), dtype=np.float32)
        segment_obstacles = np.array(self.current_meta.get("segment_obstacles", []), dtype=np.float32)

        robot_pos = robot_path[self.frame_idx]
        human_pos = human_path[self.frame_idx]

        headings = compute_path_headings(robot_path)
        heading = float(headings[self.frame_idx]) if len(headings) > 0 else 0.0

        leash_dist = np.linalg.norm(robot_pos - human_pos)
        leash_tension = min(1.0, leash_dist / 1.5)

        robot_trajectory = robot_path[: self.frame_idx + 1]
        human_trajectory = human_path[: self.frame_idx + 1]
        scores = self.current_meta.get("scores", {})
        point_cloud_xy = self._get_frame_point_cloud_world_xy()

        info = {
            "fps": self.visualizer.clock.get_fps(),
            "path_length": self.current_meta.get("reference_path_length", 0),
            "robot_x": robot_pos[0],
            "robot_y": robot_pos[1],
            "num_points": len(robot_path),
            "recording": False,
            "scores": scores,
        }

        self.visualizer.render(
            robot_pos=robot_pos,
            robot_heading=heading,
            human_pos=human_pos,
            reference_path=ref_path,
            robot_trajectory=robot_trajectory,
            human_trajectory=human_trajectory,
            point_cloud=point_cloud_xy,
            obstacles=obstacles if len(obstacles) > 0 else None,
            segment_obstacles=segment_obstacles if len(segment_obstacles) > 0 else None,
            obstacle_inflation=None,
            start_pos=start_pos,
            end_pos=end_pos,
            leash_tension=leash_tension,
            info=info,
            flip=False,
        )

        self._draw_playback_overlay()
        if self.confirm_delete:
            self._draw_delete_dialog()

        pygame.display.flip()

    def _draw_delete_dialog(self):
        overlay = pygame.Surface((self.visualizer.width, self.visualizer.height))
        overlay.fill((0, 0, 0))
        overlay.set_alpha(180)
        self.visualizer.screen.blit(overlay, (0, 0))

        dialog_w, dialog_h = 500, 150
        dialog_x = (self.visualizer.width - dialog_w) // 2
        dialog_y = (self.visualizer.height - dialog_h) // 2

        pygame.draw.rect(
            self.visualizer.screen,
            (50, 50, 60),
            (dialog_x, dialog_y, dialog_w, dialog_h),
            border_radius=10,
        )
        pygame.draw.rect(
            self.visualizer.screen,
            (100, 100, 120),
            (dialog_x, dialog_y, dialog_w, dialog_h),
            2,
            border_radius=10,
        )

        warn_text = self.visualizer.font_large.render("Delete Episode?", True, (255, 100, 100))
        self.visualizer.screen.blit(warn_text, (dialog_x + dialog_w // 2 - 80, dialog_y + 20))

        ep_text = self.visualizer.font.render(self.delete_target, True, (200, 200, 200))
        self.visualizer.screen.blit(ep_text, (dialog_x + dialog_w // 2 - 100, dialog_y + 60))

        hint_text = self.visualizer.font.render("Press Y to confirm, N to cancel", True, (150, 150, 150))
        self.visualizer.screen.blit(hint_text, (dialog_x + dialog_w // 2 - 130, dialog_y + 100))

    def _draw_playback_overlay(self):
        bar_width = 400
        bar_height = 8
        bar_x = (self.visualizer.width - bar_width) // 2
        bar_y = self.visualizer.height - 60

        pygame.draw.rect(
            self.visualizer.screen,
            (60, 60, 70),
            (bar_x, bar_y, bar_width, bar_height),
            border_radius=4,
        )

        if self.current_data:
            progress = self.frame_idx / max(1, len(self.current_data["robot_path"]) - 1)
            pygame.draw.rect(
                self.visualizer.screen,
                (100, 180, 255),
                (bar_x, bar_y, int(bar_width * progress), bar_height),
                border_radius=4,
            )

        if self.current_data:
            current_time = self.current_data["timestamps"][self.frame_idx]
            total_time = self.current_meta["duration_seconds"]
            time_text = f"{current_time:.1f}s / {total_time:.1f}s"
        else:
            time_text = "0.0s / 0.0s"

        time_surface = self.visualizer.font.render(time_text, True, (200, 200, 200))
        self.visualizer.screen.blit(time_surface, (bar_x + bar_width + 15, bar_y - 5))

        status = "Playing" if self.playing else "Paused"
        status_color = (100, 255, 100) if self.playing else (255, 200, 100)
        status_surface = self.visualizer.font.render(status, True, status_color)
        self.visualizer.screen.blit(status_surface, (bar_x - 80, bar_y - 5))

        speed_text = f"{self.playback_speed:.1f}x"
        speed_surface = self.visualizer.font.render(speed_text, True, (150, 150, 150))
        self.visualizer.screen.blit(speed_surface, (bar_x + bar_width // 2 - 20, bar_y + 15))

        if self.current_data:
            frame_text = f"Frame: {self.frame_idx + 1}/{len(self.current_data['robot_path'])}"
            frame_surface = self.visualizer.font.render(frame_text, True, (150, 150, 150))
            self.visualizer.screen.blit(frame_surface, (bar_x, bar_y + 15))

        if self.current_episode:
            ep_surface = self.visualizer.font.render(self.current_episode, True, (180, 180, 180))
            self.visualizer.screen.blit(ep_surface, (self.visualizer.width // 2 - 80, 15))

        if self.current_meta and self.current_meta.get("point_cloud_frames", 0) > 0:
            pc_sizes = self.current_data.get("point_cloud_sizes")
            current_pc = int(pc_sizes[self.frame_idx]) if pc_sizes is not None else 0
            pc_text = f"PointCloud: {current_pc} pts"
            pc_surface = self.visualizer.font.render(pc_text, True, (150, 220, 180))
            self.visualizer.screen.blit(pc_surface, (bar_x + 180, bar_y + 15))

        hint = (
            "SPACE: Play/Pause | LEFT/RIGHT: Step | R: Restart | +/-: Speed | "
            "C: Collect PC | BACKSPACE: Delete | ESC: Back"
        )
        hint_surface = self.visualizer.font.render(hint, True, (100, 100, 100))
        self.visualizer.screen.blit(hint_surface, (self.visualizer.width // 2 - 360, self.visualizer.height - 25))

    def run(self):
        print("=" * 50)
        print("Dataset Replay")
        print("=" * 50)

        clock = pygame.time.Clock()
        fps = 60

        while self.running:
            dt = clock.tick(fps) / 1000.0
            self._handle_input()

            if self.in_selection:
                self._render_selection()
            else:
                self._update(dt)
                self._render_playback()

        self.visualizer.quit()
        print("Replay exit")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay episodes and regenerate Mid360 point clouds with Gazebo.")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(script_dir, "data")
    parser.add_argument("--data-dir", type=str, default=default_data_dir, help="Episode dataset directory.")
    parser.add_argument("--collect-pointcloud", action="store_true", help="Batch-generate point clouds and exit.")
    parser.add_argument("--all", action="store_true", help="Process all episodes when used with --collect-pointcloud.")
    parser.add_argument("--episode", type=str, default=None, help="Single episode name to process in batch mode.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing point-cloud datasets.")
    parser.add_argument("--mid360-plugin-dir", type=str, default=None, help="Path to the Mid360_simulation_plugin repository.")
    parser.add_argument("--mid360-plugin-lib", type=str, default=None, help="Path to liblivox_laser_simulation.so if outside default locations.")
    parser.add_argument("--mid360-downsample", type=int, default=1, help="Mid-360 plugin downsample factor.")
    parser.add_argument("--mid360-update-rate", type=float, default=10.0, help="Mid-360 update rate for replay generation.")
    parser.add_argument("--mid360-samples", type=int, default=20000, help="Mid-360 samples per frame for replay generation.")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    plugin_dir = resolve_mid360_plugin_dir(args.mid360_plugin_dir)
    plugin_library = resolve_mid360_plugin_library(plugin_dir, args.mid360_plugin_lib)
    pointcloud_config = Mid360GazeboConfig(
        plugin_dir=plugin_dir,
        plugin_library_path=plugin_library,
        downsample=max(1, int(args.mid360_downsample)),
        update_rate=float(args.mid360_update_rate),
        samples=max(1, int(args.mid360_samples)),
        gui=False,
        visualize_laser=False,
    )

    if args.collect_pointcloud:
        if args.episode:
            episode_names = [args.episode]
        elif args.all or not args.episode:
            episode_names = None
        else:
            episode_names = None

        results = collect_pointcloud_for_dataset(
            data_dir=args.data_dir,
            config=pointcloud_config,
            overwrite=bool(args.overwrite),
            episode_names=episode_names,
        )
        print(f"Processed {len(results)} episode(s).")
        return

    replay = DatasetReplay(data_dir=args.data_dir, pointcloud_config=pointcloud_config)
    replay.run()


if __name__ == "__main__":
    main()
