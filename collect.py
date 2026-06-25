#!/usr/bin/env python3
"""
Guide Dog Robot Data Collection Tool.

Controls:
    ↑/↓  Forward/Backward
    ←/→  Turn Left/Right
    SPACE Start/Stop Recording
    S     Save trajectory
    R     Reset position
    N     Generate new path
    ESC   Exit
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pygame

from src.data_storage import DataStorage
from src.mid360_gazebo import (
    Mid360GazeboConfig,
    Mid360GazeboSession,
    resolve_mid360_plugin_dir,
    resolve_mid360_plugin_library,
)
from src.mid360_storage import Mid360DataStorage
from src.path_generator import PathGenerator
from src.physics import PhysicsEngine
from src.scoring import TrajectoryScorer
from src.visualizer import Visualizer


class DataCollector:
    """2D data collection with optional hooks for derived collectors."""

    mode_name = "2d"

    def __init__(
        self,
        path_length: float = 50.0,
        leash_length: float = 1.5,
        robot_speed: float = 2.0,
        fps: int = 60,
    ):
        self.fps = fps
        self.dt = 1.0 / fps
        self.leash_length = leash_length

        self.path_generator = PathGenerator(target_length=path_length)
        self.physics = PhysicsEngine(
            leash_length=leash_length,
            robot_speed=robot_speed,
            dt=self.dt,
        )
        self.visualizer = Visualizer()

        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "data")
        self.storage = self._create_storage(data_dir)

        self.scorer = None
        self.running = True
        self.recording = False
        self.bre = False
        self.current_path_data = None
        self.robot_trajectory = []
        self.human_trajectory = []

        self._generate_new_path()

    def _create_storage(self, data_dir: str):
        return DataStorage(base_dir=data_dir)

    def _get_save_extra_metadata(self, scores: dict) -> dict:
        return {"scores": scores}

    def _get_controls_text(self) -> list[str]:
        return [
            "Arrows: Move",
            "SPACE: Record",
            "S: Save",
            "R: Reset",
            "N: New Path",
            "Scroll: Zoom",
            "ESC: Exit",
        ]

    def _on_path_generated(self):
        """Hook for derived collectors."""

    def _sync_external_sim(self):
        """Hook for derived collectors."""

    def _on_record_frame(self, robot_state, human_state):
        self.storage.record_frame(robot_state.position, human_state.position)

    def _cleanup(self):
        """Hook for derived collectors."""

    def _generate_new_path(self):
        self.current_path_data = self.path_generator.generate()
        self._reset_position()
        self._on_path_generated()
        print(f"New path generated: length={self.current_path_data['length']:.1f}m")

    def _reset_position(self):
        if self.current_path_data is not None:
            start = self.current_path_data["start"]
            self.scorer = TrajectoryScorer(self.current_path_data["path"], self.leash_length)
        else:
            start = np.array([0.0, 0.0], dtype=np.float64)

        self.physics.reset(start)
        self.robot_trajectory = []
        self.human_trajectory = []

        if self.recording:
            self._stop_recording()

        self._sync_external_sim()

    def _start_recording(self):
        self.recording = True
        self.storage.start_recording()
        self.robot_trajectory = []
        self.human_trajectory = []
        if self.scorer:
            self.scorer.reset()
        print("Recording started...")

    def _stop_recording(self):
        self.recording = False
        if self.scorer:
            scores = self.scorer.get_scores()
            print(f"Recording stopped. Points: {self.storage.get_num_points()}")
            print(f"  Score: {scores['total']:.0f}/100 (Grade: {scores['grade']})")

    def _save_episode(self):
        if self.storage.get_num_points() == 0:
            print("No data to save!")
            return

        scores = self.scorer.get_scores() if self.scorer else {}
        if scores.get("total", 0) < 50:
            print(
                f"⚠ Low quality score: {scores.get('total', 0):.0f}/100 "
                f"(Grade: {scores.get('grade', 'F')})"
            )
            print("  Consider discarding this trajectory (press S again to force save)")

        try:
            episode_dir = self.storage.save_episode(
                reference_path=self.current_path_data["path"],
                start_pos=self.current_path_data["start"],
                end_pos=self.current_path_data["end"],
                obstacles=self.current_path_data.get("obstacles"),
                segment_obstacles=self.current_path_data.get("segment_obstacles"),
                extra_metadata=self._get_save_extra_metadata(scores),
            )
            print(
                f"✓ Saved! Score: {scores.get('total', 0):.0f}/100 "
                f"({scores.get('grade', 'N/A')}) -> {episode_dir}"
            )

            self.recording = False
            self.storage.clear()
            self.robot_trajectory = []
            self.human_trajectory = []
            if self.scorer:
                self.scorer.reset()
        except Exception as exc:
            print(f"Save failed: {exc}")

    def _handle_input(self):
        for event in self.visualizer.get_events():
            if event.type == pygame.QUIT:
                self.running = False

            self.visualizer.handle_event(event)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    if self.recording:
                        self._stop_recording()
                    else:
                        self._start_recording()
                elif event.key == pygame.K_s:
                    self._save_episode()
                elif event.key == pygame.K_b:
                    self.bre = not self.bre
                elif event.key == pygame.K_r:
                    self._reset_position()
                    print("Position reset")
                elif event.key == pygame.K_n:
                    self._generate_new_path()

        keys = pygame.key.get_pressed()
        forward = 0.0
        turn = 0.0

        if keys[pygame.K_UP]:
            forward = 1.0
        elif keys[pygame.K_DOWN]:
            forward = -1.0

        if keys[pygame.K_LEFT]:
            turn = 1.0
        elif keys[pygame.K_RIGHT]:
            turn = -1.0

        self.physics.set_control(forward, turn, self.bre)

    def _update(self):
        robot_state, human_state = self.physics.step()

        if self._check_collision():
            return self.physics.robot.copy(), self.physics.human.copy()

        self.robot_trajectory.append(robot_state.position.copy())
        self.human_trajectory.append(human_state.position.copy())

        max_trail = 5000
        if len(self.robot_trajectory) > max_trail:
            self.robot_trajectory = self.robot_trajectory[-max_trail:]
            self.human_trajectory = self.human_trajectory[-max_trail:]

        self._sync_external_sim()

        if self.recording:
            self._on_record_frame(robot_state, human_state)
            if self.scorer:
                self.scorer.update(robot_state.position, human_state.position)

        return robot_state, human_state

    def _check_collision(self) -> bool:
        obstacles = self.current_path_data.get("obstacles") if self.current_path_data else None
        segments = self.current_path_data.get("segment_obstacles") if self.current_path_data else None
        collided, info = self.physics.check_collision(obstacles, segment_obstacles=segments)
        if collided:
            who = info.get("who", "agent")
            idx = info.get("idx")
            obs_type = info.get("type", "obstacle")
            print(f"Collision detected ({who}, {obs_type} {idx}), resetting.")
            self._reset_position()
            return True
        return False

    def _render(self, robot_state, human_state, actual_fps: float):
        scores = self.scorer.get_scores() if self.scorer else {}
        info = {
            "fps": actual_fps,
            "path_length": self.current_path_data["length"] if self.current_path_data else 0.0,
            "robot_x": robot_state.position[0],
            "robot_y": robot_state.position[1],
            "num_points": self.storage.get_num_points(),
            "recording": self.recording,
            "scores": scores,
            "robot_radius": self.physics.robot_radius,
            "human_radius": self.physics.human_radius,
            "mode": self.mode_name,
            "controls": self._get_controls_text(),
        }

        self.visualizer.render(
            robot_pos=robot_state.position,
            robot_heading=robot_state.heading,
            human_pos=human_state.position,
            reference_path=self.current_path_data["path"] if self.current_path_data else None,
            robot_trajectory=self.robot_trajectory,
            human_trajectory=self.human_trajectory,
            obstacles=self.current_path_data.get("obstacles") if self.current_path_data else None,
            segment_obstacles=self.current_path_data.get("segment_obstacles") if self.current_path_data else None,
            obstacle_inflation=(self.physics.robot_radius, self.physics.human_radius),
            robot_radius=self.physics.robot_radius,
            human_radius=self.physics.human_radius,
            start_pos=self.current_path_data["start"] if self.current_path_data else None,
            end_pos=self.current_path_data["end"] if self.current_path_data else None,
            leash_tension=self.physics.get_leash_tension(),
            info=info,
        )

    def run(self):
        print("=" * 50)
        print("Guide Dog Robot Data Collection Tool")
        print("=" * 50)
        print("Controls: Arrows=Move | SPACE=Record | S=Save | R=Reset | N=NewPath | ESC=Exit")
        print(f"Backend: {self.mode_name}")
        print("=" * 50)

        try:
            while self.running:
                self._handle_input()
                robot_state, human_state = self._update()
                actual_fps = self.visualizer.tick(self.fps)
                self._render(robot_state, human_state, actual_fps)
        finally:
            self._cleanup()
            self.visualizer.quit()
            print("Program exit")


class Mid360DataCollector(DataCollector):
    """2.5D path collection with a 3D Gazebo Mid-360 sensor simulation backend."""

    mode_name = "mid360-3d"

    def __init__(
        self,
        *,
        mid360_config: Mid360GazeboConfig,
        path_length: float = 50.0,
        leash_length: float = 1.5,
        robot_speed: float = 1.5,
        fps: int = 10,
    ):
        self.mid360_config = mid360_config
        self.mid360_session: Mid360GazeboSession | None = None
        self._last_recorded_cloud_seq: int | None = None
        super().__init__(
            path_length=path_length,
            leash_length=leash_length,
            robot_speed=robot_speed,
            fps=fps,
        )

    def _create_storage(self, data_dir: str):
        return Mid360DataStorage(base_dir=data_dir)

    def _get_controls_text(self) -> list[str]:
        return [
            "Arrows: Move robot",
            "Gazebo: 3D Mid-360 scene",
            "SPACE: Record traj + cloud",
            "S: Save episode",
            "R: Reset pose",
            "N: Regenerate 3D world",
            "ESC: Exit",
        ]

    def _on_path_generated(self):
        if self.mid360_session is not None:
            self.mid360_session.close()
        self.mid360_session = Mid360GazeboSession(self.current_path_data, self.mid360_config)
        self.mid360_session.start()
        self._sync_external_sim()
        print(f"Mid360 Gazebo runtime: {self.mid360_session.runtime_dir}")

    def _sync_external_sim(self):
        if self.mid360_session is None:
            return
        self.mid360_session.update_entities(self.physics.robot, self.physics.human)

    def _start_recording(self):
        super()._start_recording()
        self._last_recorded_cloud_seq = None

    def _on_record_frame(self, robot_state, human_state):
        if self.mid360_session is None:
            raise RuntimeError("Mid360 Gazebo session is not running.")

        cloud = self.mid360_session.get_pointcloud(
            after_seq=self._last_recorded_cloud_seq,
            wait_timeout=max(self.dt * 1.2, 0.15),
        )
        if cloud is None:
            cloud = {
                "seq": self._last_recorded_cloud_seq if self._last_recorded_cloud_seq is not None else 0,
                "stamp": 0.0,
                "fields": [],
                "points": np.zeros((0, 0), dtype=np.float32),
            }

        self._last_recorded_cloud_seq = int(cloud.get("seq", 0))
        self.storage.record_frame(
            robot_state.position,
            human_state.position,
            robot_base_pose=self.mid360_session.get_robot_base_pose(robot_state),
            human_base_pose=self.mid360_session.get_human_base_pose(human_state),
            mid360_pose=self.mid360_session.get_mid360_pose(robot_state),
            point_cloud=np.asarray(cloud.get("points", np.zeros((0, 0), dtype=np.float32))),
            point_cloud_timestamp=float(cloud.get("stamp", 0.0)),
            point_cloud_fields=list(cloud.get("fields", [])),
        )

    def _get_save_extra_metadata(self, scores: dict) -> dict:
        metadata = super()._get_save_extra_metadata(scores)
        if self.mid360_session is not None:
            metadata.update(self.mid360_session.metadata())
        return metadata

    def _cleanup(self):
        if self.mid360_session is not None:
            self.mid360_session.close()
            self.mid360_session = None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Guide-dog trajectory and Mid-360 simulation collector")
    parser.add_argument(
        "--backend",
        choices=("2d", "mid360"),
        default="mid360",
        help="Collection backend. `mid360` uses Gazebo + Mid-360 simulation; `2d` keeps the legacy pygame-only collector.",
    )
    parser.add_argument("--path-length", type=float, default=50.0, help="Target path length in meters.")
    parser.add_argument("--leash-length", type=float, default=1.5, help="Leash length in meters.")
    parser.add_argument("--robot-speed", type=float, default=1.5, help="Robot forward speed in m/s.")
    parser.add_argument("--fps", type=int, default=None, help="Collector loop frequency.")
    parser.add_argument(
        "--mid360-plugin-dir",
        type=str,
        default=None,
        help="Path to the Mid360_simulation_plugin repository.",
    )
    parser.add_argument(
        "--mid360-plugin-lib",
        type=str,
        default=None,
        help="Path to liblivox_laser_simulation.so if it is outside the default catkin build locations.",
    )
    parser.add_argument(
        "--mid360-downsample",
        type=int,
        default=1,
        help="Mid-360 plugin downsample factor. Larger means fewer points.",
    )
    parser.add_argument(
        "--gazebo-gui",
        action="store_true",
        help="Launch Gazebo with GUI in mid360 mode. Default is headless.",
    )
    parser.set_defaults(mid360_visualize=False)
    parser.add_argument(
        "--mid360-visualize",
        dest="mid360_visualize",
        action="store_true",
        help="Enable laser ray visualization in Gazebo.",
    )
    parser.add_argument(
        "--no-mid360-visualize",
        dest="mid360_visualize",
        action="store_false",
        help="Disable laser ray visualization in Gazebo.",
    )
    return parser


def create_collector_from_args(args: argparse.Namespace):
    if args.backend == "2d":
        fps = args.fps if args.fps is not None else 20
        return DataCollector(
            path_length=args.path_length,
            leash_length=args.leash_length,
            robot_speed=args.robot_speed,
            fps=fps,
        )

    plugin_dir = resolve_mid360_plugin_dir(args.mid360_plugin_dir)
    plugin_library = resolve_mid360_plugin_library(plugin_dir, args.mid360_plugin_lib)
    config = Mid360GazeboConfig(
        plugin_dir=plugin_dir,
        plugin_library_path=plugin_library,
        downsample=max(1, int(args.mid360_downsample)),
        gui=bool(args.gazebo_gui),
        visualize_laser=bool(args.mid360_visualize),
    )

    fps = args.fps if args.fps is not None else int(round(config.update_rate))
    return Mid360DataCollector(
        mid360_config=config,
        path_length=args.path_length,
        leash_length=args.leash_length,
        robot_speed=args.robot_speed,
        fps=max(fps, 1),
    )


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    collector = create_collector_from_args(args)
    collector.run()


if __name__ == "__main__":
    main()
