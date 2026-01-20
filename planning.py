#!/usr/bin/env python3
"""
Guide Dog Robot Planning Tool (model-based simulation)

Controls:
    P     Toggle policy/manual control
    SPACE Pause/Resume
    R     Reset position
    N     Generate new path
    ESC   Exit
    Arrows Manual control (when policy disabled)
"""
import argparse
import copy
import json
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pygame
import torch

# Allow importing project modules when running from the FollowDataset directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Add diffusion_policy submodule to Python path
DIFFUSION_POLICY_DIR = PROJECT_ROOT / "diffusion_policy"
if str(DIFFUSION_POLICY_DIR) not in sys.path:
    sys.path.insert(0, str(DIFFUSION_POLICY_DIR))

from src.path_generator import PathGenerator
from src.physics import PhysicsEngine
from src.visualizer import Visualizer
from src.scoring import TrajectoryScorer

from diffusion_policy.workspace.train_diffusion_unet_lowdim_workspace import (
    TrainDiffusionUnetLowdimWorkspace,
)


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class ModelPlanner:
    """Run simulation and control robot with a trained policy."""

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        device: str = "auto",
        use_ema: bool = True,
        action_mode: Optional[str] = None,
        k_lookahead: Optional[int] = None,
        frame_stride: Optional[int] = None,
        path_length: float = 50.0,
        leash_length: float = 1.5,
        robot_speed: float = 1.5,
        fps: int = 20,
        inference_steps: int = 8,
        turn_gain: float = 1.2,
        curvature_slowdown: bool = True,
        curvature_scale: float = 0.7,
        min_speed_scale: float = 0.25,
        log_path: Optional[Path] = None,
        log_interval: int = 1,
    ):
        self.fps = fps
        self.sim_dt = 1.0 / fps
        self.leash_length = leash_length

        # Initialize modules
        self.path_generator = PathGenerator(target_length=path_length)
        self.visualizer = Visualizer()

        # Load policy if checkpoint is provided
        self.device = resolve_device(device)
        self.workspace = None
        self.policy = None
        if checkpoint_path is not None and checkpoint_path.exists():
            self.workspace = TrainDiffusionUnetLowdimWorkspace.create_from_checkpoint(
                str(checkpoint_path)
            )
            if use_ema and getattr(self.workspace, "ema_model", None) is not None:
                self.policy = self.workspace.ema_model
            else:
                self.policy = self.workspace.model
            self.policy.to(self.device)
            self.policy.eval()
        else:
            # No checkpoint - will use manual control only
            print("Warning: No checkpoint provided. Running in manual control mode only.")
            print("Press 'P' key is disabled. Use arrow keys for manual control.")

        # Load configuration from checkpoint if available, otherwise use defaults
        if self.workspace is not None:
            cfg_action_mode = None
            try:
                cfg_action_mode = self.workspace.cfg.task.dataset.get("action_mode")
            except Exception:
                cfg_action_mode = None
            self.action_mode = action_mode or cfg_action_mode or "forward_heading"

            cfg_robot_frame = None
            try:
                cfg_robot_frame = self.workspace.cfg.task.dataset.get("robot_frame")
            except Exception:
                cfg_robot_frame = None
            self.robot_frame = bool(cfg_robot_frame) if cfg_robot_frame is not None else False
            if self.action_mode == "forward_heading" and not self.robot_frame:
                raise ValueError("action_mode='forward_heading' requires robot_frame=True")

            cfg_robot_state = None
            try:
                cfg_robot_state = self.workspace.cfg.task.dataset.get("robot_state")
            except Exception:
                cfg_robot_state = None
            self.robot_state = str(cfg_robot_state or "zero").lower()

            cfg_k_lookahead = None
            try:
                cfg_k_lookahead = self.workspace.cfg.task.get("k_lookahead")
            except Exception:
                cfg_k_lookahead = None
            if cfg_k_lookahead is None:
                try:
                    cfg_k_lookahead = self.workspace.cfg.task.dataset.get("k_lookahead")
                except Exception:
                    cfg_k_lookahead = None
            if cfg_k_lookahead is None:
                try:
                    cfg_k_lookahead = self.workspace.cfg.task.get("lookahead_stride")
                except Exception:
                    cfg_k_lookahead = None
            if cfg_k_lookahead is None:
                try:
                    cfg_k_lookahead = self.workspace.cfg.task.dataset.get("lookahead_stride")
                except Exception:
                    cfg_k_lookahead = None

            cfg_n_lookahead = None
            try:
                cfg_n_lookahead = self.workspace.cfg.task.get("n_lookahead")
            except Exception:
                cfg_n_lookahead = None
            if cfg_n_lookahead is None:
                try:
                    cfg_n_lookahead = self.workspace.cfg.task.dataset.get("n_lookahead")
                except Exception:
                    cfg_n_lookahead = None

            cfg_n_obstacle_circles = None
            try:
                cfg_n_obstacle_circles = self.workspace.cfg.task.get("n_obstacle_circles")
            except Exception:
                cfg_n_obstacle_circles = None
            if cfg_n_obstacle_circles is None:
                try:
                    cfg_n_obstacle_circles = self.workspace.cfg.task.dataset.get("n_obstacle_circles")
                except Exception:
                    cfg_n_obstacle_circles = None

            cfg_n_obstacle_segments = None
            try:
                cfg_n_obstacle_segments = self.workspace.cfg.task.get("n_obstacle_segments")
            except Exception:
                cfg_n_obstacle_segments = None
            if cfg_n_obstacle_segments is None:
                try:
                    cfg_n_obstacle_segments = self.workspace.cfg.task.dataset.get("n_obstacle_segments")
                except Exception:
                    cfg_n_obstacle_segments = None

            cfg_obstacle_include_radius = None
            try:
                cfg_obstacle_include_radius = self.workspace.cfg.task.dataset.get("obstacle_include_radius")
            except Exception:
                cfg_obstacle_include_radius = None

            cfg_segment_repr = None
            try:
                cfg_segment_repr = self.workspace.cfg.task.dataset.get("segment_repr")
            except Exception:
                cfg_segment_repr = None

            cfg_frame_stride = None
            try:
                cfg_frame_stride = self.workspace.cfg.task.get("frame_stride")
            except Exception:
                cfg_frame_stride = None
            if cfg_frame_stride is None:
                try:
                    cfg_frame_stride = self.workspace.cfg.task.dataset.get("frame_stride")
                except Exception:
                    cfg_frame_stride = None

            self.obs_dim = int(self.policy.obs_dim)
            self.action_dim = int(self.policy.action_dim)
            self.n_obs_steps = int(self.policy.n_obs_steps)
            self.n_action_steps = int(self.policy.n_action_steps)
            self.n_obstacle_circles = max(0, int(cfg_n_obstacle_circles or 0))
            self.n_obstacle_segments = max(0, int(cfg_n_obstacle_segments or 0))
            self.obstacle_include_radius = (
                True if cfg_obstacle_include_radius is None else bool(cfg_obstacle_include_radius)
            )
            self.segment_repr = str(cfg_segment_repr or "endpoints").lower()
            if self.segment_repr not in ("endpoints", "closest_dir"):
                raise ValueError(f"Unsupported segment_repr: {self.segment_repr}")
            circle_dim = 3 if self.obstacle_include_radius else 2
            self.obstacle_obs_dim = (
                self.n_obstacle_circles * circle_dim + self.n_obstacle_segments * 4
            )
            if cfg_n_lookahead is None:
                extra_obs = self.obs_dim - 4 - self.obstacle_obs_dim
                if extra_obs < 0 or extra_obs % 2 != 0:
                    raise ValueError(
                        f"Unsupported obs_dim={self.obs_dim}, expected 4 + 2*N + {self.obstacle_obs_dim}"
                    )
                self.n_lookahead = extra_obs // 2
            else:
                self.n_lookahead = int(cfg_n_lookahead)
                expected_obs_dim = 4 + 2 * self.n_lookahead + self.obstacle_obs_dim
                if self.obs_dim != expected_obs_dim:
                    extra_obs = self.obs_dim - 4 - self.obstacle_obs_dim
                    if extra_obs >= 0 and extra_obs % 2 == 0:
                        self.n_lookahead = extra_obs // 2
                        print(
                            f"[warn] obs_dim mismatch (expected {expected_obs_dim}, got {self.obs_dim}); "
                            f"using derived n_lookahead={self.n_lookahead}"
                        )
                    else:
                        raise ValueError(
                            f"Unsupported obs_dim={self.obs_dim}, expected {expected_obs_dim}"
                        )
            stride = k_lookahead if k_lookahead is not None else cfg_k_lookahead
            if stride is None:
                stride = 5
            self.lookahead_stride = max(1, int(stride))

            stride = frame_stride if frame_stride is not None else cfg_frame_stride
            if stride is None:
                stride = 1
            self.frame_stride = max(1, int(stride))
        else:
            # Default values when no checkpoint is provided
            self.action_mode = action_mode or "forward_heading"
            self.robot_frame = True
            self.robot_state = "vel"
            self.obs_dim = 44  # Default: 4 (robot+human) + 2*20 (lookahead) + 0 (no obstacles)
            self.action_dim = 2
            self.n_obs_steps = 1
            self.n_action_steps = 24
            self.n_obstacle_circles = 0
            self.n_obstacle_segments = 0
            self.obstacle_include_radius = False
            self.segment_repr = "endpoints"
            self.obstacle_obs_dim = 0
            self.n_lookahead = 20
            self.lookahead_stride = k_lookahead if k_lookahead is not None else 5
            self.frame_stride = frame_stride if frame_stride is not None else 1
        self.data_dt = self.sim_dt * self.frame_stride
        self.physics = PhysicsEngine(
            leash_length=leash_length,
            robot_speed=robot_speed,
            dt=self.sim_dt,
        )

        self.scorer = None
        self.current_path_data = None
        self.running = True
        self.paused = False
        self.use_policy = False

        self.robot_trajectory = []
        self.human_trajectory = []
        self.planned_path = None
        self.lookahead_world = None
        self.frame_count = 0
        self.prev_robot_pos = None
        self.data_step_idx = 0
        self.log_fp = None
        self.log_interval = max(1, int(log_interval))

        self.obs_history = deque(maxlen=self.n_obs_steps)
        
        # Performance optimization: cache actions and reduce inference frequency
        self.cached_action_seq = None
        self.cached_action_idx = 0
        self.inference_interval = max(1, self.n_action_steps // 2)  # Infer every N frames
        self.frames_since_inference = 0
        self.cached_control = (0.0, 0.0)
        self.current_action = None
        self.current_delta = None
        self.turn_gain = float(turn_gain)
        self.curvature_slowdown = bool(curvature_slowdown)
        self.curvature_scale = float(curvature_scale)
        self.min_speed_scale = float(min_speed_scale)
        self.current_speed_scale = 1.0
        
        # Reduce inference steps for faster performance
        if self.policy is not None and hasattr(self.policy, 'num_inference_steps'):
            original_steps = self.policy.num_inference_steps
            self.policy.num_inference_steps = inference_steps
            if original_steps != inference_steps:
                print(f"Set inference steps to {inference_steps} (original: {original_steps}) for faster performance")

        if log_path is not None:
            log_path = Path(log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_fp = log_path.open("w", encoding="utf-8")
            self._log_event("init", {"log_path": str(log_path)})
            self._log_event(
                "config",
                {
                    "fps": float(self.fps),
                    "sim_dt": float(self.sim_dt),
                    "data_dt": float(self.data_dt),
                    "frame_stride": int(self.frame_stride),
                    "sim_time_scale": float(self.sim_dt * self.fps),
                    "data_time_scale": float(self.data_dt * self.fps),
                    "robot_speed": float(self.physics.robot_speed),
                    "turn_speed": float(self.physics.turn_speed),
                    "leash_length": float(self.leash_length),
                    "action_mode": self.action_mode,
                    "robot_frame": bool(self.robot_frame),
                    "robot_state": self.robot_state,
                    "n_lookahead": int(self.n_lookahead),
                    "lookahead_stride": int(self.lookahead_stride),
                    "n_obstacle_circles": int(self.n_obstacle_circles),
                    "n_obstacle_segments": int(self.n_obstacle_segments),
                    "obstacle_obs_dim": int(self.obstacle_obs_dim),
                    "obstacle_include_radius": bool(self.obstacle_include_radius),
                    "segment_repr": self.segment_repr,
                    "inference_interval": int(self.inference_interval),
                    "turn_gain": float(self.turn_gain),
                    "curvature_slowdown": bool(self.curvature_slowdown),
                    "curvature_scale": float(self.curvature_scale),
                    "min_speed_scale": float(self.min_speed_scale),
                },
            )

        self._generate_new_path()

    def _generate_new_path(self):
        """Generate new reference path."""
        self.current_path_data = self.path_generator.generate()
        self._reset_position()
        obstacles = self.current_path_data.get("obstacles")
        obstacle_count = int(len(obstacles)) if obstacles is not None else 0
        segments = self.current_path_data.get("segment_obstacles")
        segment_count = int(len(segments)) if segments is not None else 0
        self._log_event(
            "new_path",
            {
                "path_length": float(self.current_path_data["length"]),
                "obstacle_count": obstacle_count,
                "segment_obstacle_count": segment_count,
            },
        )
        print(f"New path generated: length={self.current_path_data['length']:.1f}m")

    def _reset_position(self):
        """Reset to start position."""
        if self.current_path_data is not None:
            start = self.current_path_data["start"]
            self.scorer = TrajectoryScorer(
                self.current_path_data["path"],
                self.leash_length,
            )
        else:
            start = np.array([0.0, 0.0])

        self.physics.reset(start)
        self.robot_trajectory = []
        self.human_trajectory = []
        self.planned_path = None
        self.frame_count = 0
        self.prev_robot_pos = None
        self._seed_obs_history(self.physics.robot.position, self.physics.human.position)
        if self.policy is not None:
            self.policy.reset()
        # Reset action cache
        self.cached_action_seq = None
        self.cached_action_idx = 0
        self.frames_since_inference = 0
        self.data_step_idx = 0
        self.cached_control = (0.0, 0.0)
        self.current_action = None
        self.current_delta = None
        self._log_event("reset_position", {"robot_pos": self.physics.robot.position.tolist()})

    def _seed_obs_history(self, robot_pos: np.ndarray, human_pos: np.ndarray):
        obs = self._build_obs(robot_pos, human_pos, self.physics.robot.heading)
        self.obs_history.clear()
        for _ in range(self.n_obs_steps):
            self.obs_history.append(obs.copy())
        self.prev_robot_pos = robot_pos.copy()

    def _log_event(self, name: str, extra: Optional[dict] = None):
        if self.log_fp is None:
            return
        payload = {
            "event": name,
            "frame": int(self.frame_count),
            "time_sec": float(self.frame_count * self.sim_dt),
            "data_step": int(self.data_step_idx),
            "data_time_sec": float(self.data_step_idx * self.data_dt),
        }
        if extra:
            payload.update(extra)
        self.log_fp.write(json.dumps(payload, ensure_ascii=True) + "\n")
        self.log_fp.flush()

    def _log_step(
        self,
        frame_idx: int,
        robot_state,
        human_state,
        action: Optional[np.ndarray],
        delta: Optional[np.ndarray],
        forward: float,
        turn: float,
    ):
        if self.log_fp is None:
            return
        if frame_idx % self.log_interval != 0:
            return
        scores = self.scorer.get_scores() if self.scorer else {}
        payload = {
            "event": "step",
            "frame": int(frame_idx),
            "time_sec": float(frame_idx * self.sim_dt),
            "real_time_sec": float(frame_idx / max(1.0, self.fps)),
            "data_step": int(self.data_step_idx),
            "data_time_sec": float(self.data_step_idx * self.data_dt),
            "is_data_step": bool(frame_idx % self.frame_stride == 0),
            "robot_pos": [float(robot_state.position[0]), float(robot_state.position[1])],
            "robot_vel": [float(robot_state.velocity[0]), float(robot_state.velocity[1])],
            "human_pos": [float(human_state.position[0]), float(human_state.position[1])],
            "heading": float(robot_state.heading),
            "forward": float(forward),
            "turn": float(turn),
            "speed_scale": float(self.current_speed_scale),
            "use_policy": bool(self.use_policy),
            "paused": bool(self.paused),
            "scores": scores,
        }
        if self.robot_frame:
            rel = human_state.position - robot_state.position
            cos_h = float(np.cos(robot_state.heading))
            sin_h = float(np.sin(robot_state.heading))
            hx = cos_h * float(rel[0]) + sin_h * float(rel[1])
            hy = -sin_h * float(rel[0]) + cos_h * float(rel[1])
            payload["human_rel"] = [hx, hy]
        if action is not None:
            payload["action"] = [float(action[0]), float(action[1])]
        if delta is not None:
            payload["delta_world"] = [float(delta[0]), float(delta[1])]
        self.log_fp.write(json.dumps(payload, ensure_ascii=True) + "\n")
        self.log_fp.flush()

    def _handle_input(self):
        """Handle keyboard input."""
        for event in self.visualizer.get_events():
            if event.type == pygame.QUIT:
                self.running = False

            self.visualizer.handle_zoom(event)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self._reset_position()
                    print("Position reset")
                elif event.key == pygame.K_n:
                    self._generate_new_path()
                elif event.key == pygame.K_p:
                    if self.policy is not None:
                        self.use_policy = not self.use_policy
                        mode = "policy" if self.use_policy else "manual"
                    else:
                        mode = "manual (no checkpoint)"
                    print(f"Control mode: {mode}")

    def _get_manual_control(self) -> Tuple[float, float]:
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
        return forward, turn

    def _predict_action(self) -> np.ndarray:
        obs_seq = np.stack(self.obs_history, axis=0)
        obs_tensor = torch.from_numpy(obs_seq).to(
            device=self.device, dtype=self.policy.dtype
        )[None, ...]
        obs_dict = {"obs": obs_tensor}
        with torch.no_grad():
            action_dict = self.policy.predict_action(obs_dict)
        action_seq = action_dict["action"].detach().cpu().numpy()[0]
        return action_seq.astype(np.float32)

    def _build_obs(
        self, robot_pos: np.ndarray, human_pos: np.ndarray, heading: float
    ) -> np.ndarray:
        if self.robot_frame:
            # robot-centric base state: [robot_state(2), human_rel(2)]
            human_rel = human_pos - robot_pos
            cos_h = float(np.cos(heading))
            sin_h = float(np.sin(heading))
            hx = cos_h * human_rel[0] + sin_h * human_rel[1]
            hy = -sin_h * human_rel[0] + cos_h * human_rel[1]

            if self.robot_state in ("vel", "velocity"):
                if self.prev_robot_pos is None:
                    vel_world = np.zeros((2,), dtype=np.float32)
                else:
                    vel_world = (robot_pos - self.prev_robot_pos).astype(np.float32) / float(
                        max(1e-6, self.data_dt)
                    )
                vx = cos_h * vel_world[0] + sin_h * vel_world[1]
                vy = -sin_h * vel_world[0] + cos_h * vel_world[1]
                robot_state = np.array([vx, vy], dtype=np.float32)
            else:
                robot_state = np.zeros((2,), dtype=np.float32)
            base = np.concatenate([robot_state, np.array([hx, hy], dtype=np.float32)], axis=0)
        else:
            base = np.concatenate([robot_pos, human_pos], axis=0).astype(np.float32)

        ref_features = (
            self._build_reference_features(robot_pos, heading)
            if self.n_lookahead > 0 and self.current_path_data is not None
            else np.zeros((self.n_lookahead * 2,), dtype=np.float32)
        )
        obstacle_features = self._build_obstacle_features(robot_pos, heading)
        if self.n_lookahead <= 0 or self.current_path_data is None:
            self.lookahead_world = None
        return np.concatenate([base, ref_features, obstacle_features], axis=0).astype(np.float32)

    def _build_reference_features(self, robot_pos: np.ndarray, heading: float) -> np.ndarray:
        ref_path = self.current_path_data["path"]
        if ref_path is None or len(ref_path) == 0:
            self.lookahead_world = None
            return np.zeros((self.n_lookahead * 2,), dtype=np.float32)
        diffs = ref_path - robot_pos
        idx = int(np.argmin(np.sum(diffs * diffs, axis=1)))
        indices = idx + np.arange(self.n_lookahead) * self.lookahead_stride
        indices = np.clip(indices, 0, len(ref_path) - 1)
        points = ref_path[indices]
        self.lookahead_world = points
        rel = points - robot_pos
        cos_h = float(np.cos(heading))
        sin_h = float(np.sin(heading))
        local_x = cos_h * rel[:, 0] + sin_h * rel[:, 1]
        local_y = -sin_h * rel[:, 0] + cos_h * rel[:, 1]
        return np.stack([local_x, local_y], axis=-1).reshape(-1).astype(np.float32)

    def _select_obstacles_for_observation(
        self, robot_pos: np.ndarray
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self.current_path_data is None:
            return None, None

        selected_circles = None
        selected_segments = None

        if self.n_obstacle_circles > 0:
            obstacles = self.current_path_data.get("obstacles")
            if obstacles is not None and len(obstacles) > 0:
                circle_obs = np.asarray(obstacles, dtype=np.float32)
                if circle_obs.ndim == 2 and circle_obs.shape[1] >= 3:
                    centers = circle_obs[:, :2]
                    rel = centers - robot_pos
                    dist_sq = np.sum(rel * rel, axis=1)
                    order = np.argsort(dist_sq)
                    count = min(self.n_obstacle_circles, len(order))
                    if count > 0:
                        selected_circles = circle_obs[order[:count], :3]

        if self.n_obstacle_segments > 0:
            segments = self.current_path_data.get("segment_obstacles")
            if segments is not None and len(segments) > 0:
                seg_obs = np.asarray(segments, dtype=np.float32)
                if seg_obs.ndim == 2 and seg_obs.shape[1] >= 4:
                    seg_obs = seg_obs[:, :4]
                    dist_sq = np.zeros((len(seg_obs),), dtype=np.float32)
                    for i, seg in enumerate(seg_obs):
                        dist_sq[i] = self._point_segment_dist_sq(
                            robot_pos, seg[:2], seg[2:4]
                        )
                    order = np.argsort(dist_sq)
                    count = min(self.n_obstacle_segments, len(order))
                    if count > 0:
                        selected_segments = seg_obs[order[:count], :4]

        return selected_circles, selected_segments

    def _build_obstacle_features(self, robot_pos: np.ndarray, heading: float) -> np.ndarray:
        circle_dim = 3 if self.obstacle_include_radius else 2
        total_dim = self.n_obstacle_circles * circle_dim + self.n_obstacle_segments * 4
        if total_dim == 0:
            return np.zeros((0,), dtype=np.float32)

        feats = np.zeros((total_dim,), dtype=np.float32)
        offset = 0
        if self.current_path_data is None:
            return feats

        selected_circles, selected_segments = self._select_obstacles_for_observation(robot_pos)

        if self.n_obstacle_circles > 0:
            count = 0 if selected_circles is None else len(selected_circles)
            for i in range(self.n_obstacle_circles):
                if i < count:
                    rel_i = selected_circles[i, :2] - robot_pos
                    if self.robot_frame:
                        rel_i = self._rotate_rel(rel_i, heading)
                    feats[offset : offset + 2] = rel_i
                    if self.obstacle_include_radius:
                        feats[offset + 2] = float(selected_circles[i, 2])
                offset += circle_dim

        if self.n_obstacle_segments > 0:
            count = 0 if selected_segments is None else len(selected_segments)
            for i in range(self.n_obstacle_segments):
                if i < count:
                    seg = selected_segments[i]
                    if self.segment_repr == "endpoints":
                        p1 = seg[:2] - robot_pos
                        p2 = seg[2:4] - robot_pos
                        if self.robot_frame:
                            p1 = self._rotate_rel(p1, heading)
                            p2 = self._rotate_rel(p2, heading)
                        feats[offset : offset + 4] = [p1[0], p1[1], p2[0], p2[1]]
                    else:  # closest_dir
                        p1_world = seg[:2].astype(np.float32)
                        p2_world = seg[2:4].astype(np.float32)
                        ab = p2_world - p1_world
                        denom = float(np.dot(ab, ab))
                        if denom < 1e-12:
                            closest = p1_world
                            direction = np.zeros((2,), dtype=np.float32)
                        else:
                            t_proj = float(np.dot(robot_pos - p1_world, ab)) / denom
                            t_proj = float(np.clip(t_proj, 0.0, 1.0))
                            closest = p1_world + t_proj * ab
                            direction = (ab / np.sqrt(denom)).astype(np.float32)
                            if (direction[0] < 0) or (
                                abs(direction[0]) < 1e-6 and direction[1] < 0
                            ):
                                direction = -direction

                        rel = (closest - robot_pos).astype(np.float32)
                        if self.robot_frame:
                            rel = self._rotate_rel(rel, heading)
                            direction = self._rotate_rel(direction, heading)
                        feats[offset : offset + 4] = [
                            float(rel[0]),
                            float(rel[1]),
                            float(direction[0]),
                            float(direction[1]),
                        ]
                offset += 4

        return feats

    def _rotate_rel(self, rel: np.ndarray, heading: float) -> np.ndarray:
        cos_h = float(np.cos(heading))
        sin_h = float(np.sin(heading))
        return np.array(
            [cos_h * rel[0] + sin_h * rel[1], -sin_h * rel[0] + cos_h * rel[1]],
            dtype=np.float32,
        )

    def _point_segment_dist_sq(self, point: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
        ab = p2 - p1
        denom = float(np.dot(ab, ab))
        if denom < 1e-12:
            diff = point - p1
            return float(np.dot(diff, diff))
        t = float(np.dot(point - p1, ab)) / denom
        t = float(np.clip(t, 0.0, 1.0))
        closest = p1 + t * ab
        diff = point - closest
        return float(np.dot(diff, diff))

    def _segments_closest_points_and_dirs(
        self,
        robot_pos: np.ndarray,
        segments: Optional[np.ndarray],
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if segments is None or len(segments) == 0:
            return None, None
        segs = np.asarray(segments, dtype=np.float32)
        if segs.ndim != 2 or segs.shape[1] < 4:
            return None, None
        segs = segs[:, :4]

        closest_points = np.zeros((len(segs), 2), dtype=np.float32)
        directions = np.zeros((len(segs), 2), dtype=np.float32)
        for i, seg in enumerate(segs):
            p1 = seg[:2]
            p2 = seg[2:4]
            ab = p2 - p1
            denom = float(np.dot(ab, ab))
            if denom < 1e-12:
                closest = p1
                direction = np.zeros((2,), dtype=np.float32)
            else:
                t_proj = float(np.dot(robot_pos - p1, ab)) / denom
                t_proj = float(np.clip(t_proj, 0.0, 1.0))
                closest = p1 + t_proj * ab
                direction = (ab / np.sqrt(denom)).astype(np.float32)
                if (direction[0] < 0) or (abs(direction[0]) < 1e-6 and direction[1] < 0):
                    direction = -direction
            closest_points[i] = closest
            directions[i] = direction

        return closest_points, directions

    def _actions_to_path(self, robot_pos: np.ndarray, action_seq: np.ndarray) -> Optional[np.ndarray]:
        if action_seq.size == 0:
            return None
        if self.action_mode == "forward_heading":
            # Rollout using the same control mapping + physics integration as execution,
            # so the visualized planned path matches what would actually happen.
            sim = copy.deepcopy(self.physics)
            sim.robot.position = robot_pos.astype(np.float32).copy()
            points = []
            obstacles = self.current_path_data.get("obstacles") if self.current_path_data else None
            segments = (
                self.current_path_data.get("segment_obstacles") if self.current_path_data else None
            )

            for act in action_seq:
                forward_delta = float(act[0])
                heading_delta = float(act[1])

                turn_speed = float(sim.turn_speed)
                robot_speed = float(sim.robot_speed)
                turn_delta = heading_delta * float(self.turn_gain)
                turn = turn_delta / (turn_speed * self.data_dt) if turn_speed > 0 else 0.0
                forward = forward_delta / (robot_speed * self.data_dt) if robot_speed > 0 else 0.0

                if self.curvature_slowdown and turn_speed > 0:
                    max_turn = turn_speed * self.data_dt
                    if max_turn > 1e-6:
                        ratio = min(1.0, abs(turn_delta) / max_turn)
                        speed_scale = max(
                            float(self.min_speed_scale),
                            1.0 - float(self.curvature_scale) * ratio,
                        )
                        forward *= speed_scale

                forward = float(np.clip(forward, -1.0, 1.0))
                turn = float(np.clip(turn, -1.0, 1.0))
                sim.set_control(forward, turn)

                # One policy action corresponds to one "data step" = frame_stride sim steps.
                for _ in range(int(self.frame_stride)):
                    robot_state, _human_state = sim.step()
                    points.append(robot_state.position.copy())
                    if obstacles is not None or segments is not None:
                        collided, _info = sim.check_collision(
                            obstacles, segment_obstacles=segments
                        )
                        if collided:
                            return np.stack(points, axis=0) if points else None

            return np.stack(points, axis=0) if points else None
        if self.robot_frame and self.action_mode in ("delta", "velocity"):
            cos_h = float(np.cos(self.physics.robot.heading))
            sin_h = float(np.sin(self.physics.robot.heading))
            dx = cos_h * action_seq[:, 0] - sin_h * action_seq[:, 1]
            dy = sin_h * action_seq[:, 0] + cos_h * action_seq[:, 1]
            action_seq = np.stack([dx, dy], axis=-1).astype(np.float32)
        if self.action_mode == "delta":
            points = np.cumsum(
                np.vstack([robot_pos[None, :], action_seq]), axis=0
            )[1:]
        elif self.action_mode == "position":
            points = action_seq
        else:  # velocity
            points = robot_pos[None, :] + np.cumsum(action_seq * self.data_dt, axis=0)
        return points

    def _action_to_delta(self, action: np.ndarray, robot_pos: np.ndarray) -> np.ndarray:
        if self.action_mode == "forward_heading":
            forward = float(action[0])
            cos_h = float(np.cos(self.physics.robot.heading))
            sin_h = float(np.sin(self.physics.robot.heading))
            dx = cos_h * forward
            dy = sin_h * forward
            return np.array([dx, dy], dtype=np.float32)
        if self.action_mode == "delta":
            delta = action
        elif self.action_mode == "position":
            delta = action - robot_pos
        else:
            delta = action * self.data_dt
        if self.robot_frame:
            # model outputs delta in robot frame, convert to world frame for control
            cos_h = float(np.cos(self.physics.robot.heading))
            sin_h = float(np.sin(self.physics.robot.heading))
            dx = cos_h * float(delta[0]) - sin_h * float(delta[1])
            dy = sin_h * float(delta[0]) + cos_h * float(delta[1])
            return np.array([dx, dy], dtype=np.float32)
        return delta

    def _delta_to_control(
        self,
        delta: np.ndarray,
        heading: float,
        dt: Optional[float] = None,
    ) -> Tuple[float, float]:
        if dt is None:
            dt = self.data_dt
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm < 1e-6:
            return 0.0, 0.0
        desired_heading = float(np.arctan2(delta[1], delta[0]))
        heading_error = wrap_angle(desired_heading - heading)
        turn_speed = self.physics.turn_speed
        turn_input = heading_error / (turn_speed * dt) if turn_speed > 0 else 0.0
        turn_input = float(np.clip(turn_input, -1.0, 1.0))

        forward_input = delta_norm / (self.physics.robot_speed * dt)
        forward_input = float(np.clip(forward_input, -1.0, 1.0))
        return forward_input, turn_input

    def _step(self):
        """Advance simulation by one step."""
        forward = 0.0
        turn = 0.0
        action = None
        delta = None
        is_data_step = (self.frame_count % self.frame_stride == 0)

        if not self.paused:
            if self.use_policy and self.policy is not None:
                # Update policy/action at data rate; hold control between data steps.
                if is_data_step:
                    if (self.cached_action_seq is None or 
                        self.cached_action_idx >= len(self.cached_action_seq) or
                        self.frames_since_inference >= self.inference_interval):
                        # Run inference
                        action_seq = self._predict_action()
                        self.cached_action_seq = action_seq
                        self.cached_action_idx = 0
                        self.frames_since_inference = 0
                        self.planned_path = self._actions_to_path(
                            self.physics.robot.position, action_seq
                        )
                        if self.cached_action_seq is not None and len(self.cached_action_seq) > 0:
                            norms = np.linalg.norm(self.cached_action_seq, axis=1)
                            self._log_event(
                                "inference",
                                {
                                    "action_len": int(len(self.cached_action_seq)),
                                    "action_mean_norm": float(np.mean(norms)),
                                    "action_max_norm": float(np.max(norms)),
                                },
                            )
                    else:
                        # Reuse cached actions
                        self.frames_since_inference += 1

                    # Get current action from cached sequence
                    if self.cached_action_seq is not None and len(self.cached_action_seq) > 0:
                        action = self.cached_action_seq[self.cached_action_idx]
                        self.cached_action_idx += 1
                        # Wrap around if needed
                        if self.cached_action_idx >= len(self.cached_action_seq):
                            self.cached_action_idx = len(self.cached_action_seq) - 1
                    else:
                        action = np.zeros(self.action_dim)

                    self.current_action = action
                    if self.action_mode == "forward_heading":
                        forward_delta = float(action[0])
                        heading_delta = float(action[1])
                        turn_speed = float(self.physics.turn_speed)
                        robot_speed = float(self.physics.robot_speed)
                        turn_delta = heading_delta * self.turn_gain
                        turn = turn_delta / (turn_speed * self.data_dt) if turn_speed > 0 else 0.0
                        forward = forward_delta / (robot_speed * self.data_dt) if robot_speed > 0 else 0.0
                        speed_scale = 1.0
                        if self.curvature_slowdown and turn_speed > 0:
                            max_turn = turn_speed * self.data_dt
                            if max_turn > 1e-6:
                                ratio = min(1.0, abs(turn_delta) / max_turn)
                                speed_scale = max(
                                    self.min_speed_scale, 1.0 - self.curvature_scale * ratio
                                )
                                forward *= speed_scale
                        forward = float(np.clip(forward, -1.0, 1.0))
                        turn = float(np.clip(turn, -1.0, 1.0))
                        delta = self._action_to_delta(action, self.physics.robot.position)
                        self.current_speed_scale = speed_scale
                    else:
                        delta = self._action_to_delta(action, self.physics.robot.position)
                        forward, turn = self._delta_to_control(
                            delta, self.physics.robot.heading, dt=self.data_dt
                        )
                        self.current_speed_scale = 1.0
                    self.current_delta = delta
                    self.cached_control = (forward, turn)
                else:
                    forward, turn = self.cached_control
                    action = self.current_action
                    delta = self.current_delta
            else:
                forward, turn = self._get_manual_control()
                self.planned_path = None
                # Reset cache when switching to manual
                self.cached_action_seq = None
                self.cached_action_idx = 0
                self.frames_since_inference = 0
                self.cached_control = (forward, turn)
                self.current_action = None
                self.current_delta = None
                self.current_speed_scale = 1.0

        self.physics.set_control(forward, turn)
        robot_state, human_state = self.physics.step()

        if self._check_collision():
            return self.physics.robot.copy(), self.physics.human.copy()

        self.robot_trajectory.append(robot_state.position.copy())
        self.human_trajectory.append(human_state.position.copy())

        max_trail = 5000
        if len(self.robot_trajectory) > max_trail:
            self.robot_trajectory = self.robot_trajectory[-max_trail:]
            self.human_trajectory = self.human_trajectory[-max_trail:]

        if self.scorer:
            self.scorer.update(robot_state.position, human_state.position)

        if is_data_step:
            obs = self._build_obs(
                robot_state.position, human_state.position, robot_state.heading
            )
            self.obs_history.append(obs)
            self.prev_robot_pos = robot_state.position.copy()
        self._log_step(self.frame_count, robot_state, human_state, action, delta, forward, turn)
        if is_data_step:
            self.data_step_idx += 1
        self.frame_count += 1
        return robot_state, human_state

    def _check_collision(self) -> bool:
        obstacles = self.current_path_data.get("obstacles") if self.current_path_data else None
        segments = self.current_path_data.get("segment_obstacles") if self.current_path_data else None
        collided, info = self.physics.check_collision(obstacles, segment_obstacles=segments)
        if collided:
            who = info.get("who", "agent") if info else "agent"
            idx = info.get("idx") if info else None
            obstacle = info.get("obstacle") if info else None
            obs_type = info.get("type") if info else None
            self._log_event(
                "collision",
                {
                    "who": who,
                    "obstacle_type": obs_type,
                    "obstacle_idx": idx,
                    "obstacle": obstacle,
                    "robot_pos": self.physics.robot.position.tolist(),
                    "human_pos": self.physics.human.position.tolist(),
                },
            )
            label = obs_type or "obstacle"
            print(f"Collision detected ({who}, {label} {idx}), resetting.")
            self._reset_position()
            return True
        return False

    def _render(self, robot_state, human_state, actual_fps: float):
        scores = self.scorer.get_scores() if self.scorer else {}
        mode = "policy" if self.use_policy else "manual"
        if self.paused:
            mode += " (paused)"

        info = {
            "fps": actual_fps,
            "path_length": self.current_path_data["length"] if self.current_path_data else 0,
            "robot_x": robot_state.position[0],
            "robot_y": robot_state.position[1],
            "num_points": self.frame_count,
            "recording": False,
            "scores": scores,
            "mode": mode,
            "robot_radius": self.physics.robot_radius,
            "human_radius": self.physics.human_radius,
            "controls": [
                "P: Policy/Manual",
                "SPACE: Pause",
                "R: Reset",
                "N: New Path",
                "Arrows: Manual control",
                "Scroll: Zoom",
                "ESC: Exit",
            ],
        }

        obs_obstacles, obs_segment_obstacles = self._select_obstacles_for_observation(
            robot_state.position
        )
        seg_closest_pts, seg_dirs = self._segments_closest_points_and_dirs(
            robot_state.position, obs_segment_obstacles
        )
        self.visualizer.render(
            robot_pos=robot_state.position,
            robot_heading=robot_state.heading,
            human_pos=human_state.position,
            reference_path=self.current_path_data["path"] if self.current_path_data else None,
            robot_trajectory=self.robot_trajectory,
            human_trajectory=self.human_trajectory,
            planned_path=self.planned_path,
            lookahead_points=self.lookahead_world,
            obstacles=self.current_path_data.get("obstacles") if self.current_path_data else None,
            segment_obstacles=self.current_path_data.get("segment_obstacles") if self.current_path_data else None,
            obs_obstacles=obs_obstacles,
            obs_segment_obstacles=obs_segment_obstacles,
            obs_segment_closest_points=seg_closest_pts,
            obs_segment_dirs=seg_dirs,
            obstacle_inflation=(self.physics.robot_radius, self.physics.human_radius),
            robot_radius=self.physics.robot_radius,
            human_radius=self.physics.human_radius,
            start_pos=self.current_path_data["start"] if self.current_path_data else None,
            end_pos=self.current_path_data["end"] if self.current_path_data else None,
            leash_tension=self.physics.get_leash_tension(),
            info=info,
        )

    def run(self):
        print("=" * 60)
        print("Guide Dog Robot Planning Tool")
        print("=" * 60)
        print("Controls: P=Policy/Manual | SPACE=Pause | R=Reset | N=NewPath | ESC=Exit")
        print("=" * 60)

        while self.running:
            self._handle_input()
            robot_state, human_state = self._step()
            actual_fps = self.visualizer.tick(self.fps)
            self._render(robot_state, human_state, actual_fps)

        self.visualizer.quit()
        if self.log_fp is not None:
            self._log_event("shutdown")
            self.log_fp.close()
        print("Program exit")


def main():
    parser = argparse.ArgumentParser(description="Guide-follow planning with a trained policy")
    
    # Default checkpoint path - use latest available checkpoint
    default_ckpt = Path(
        "/home/yyf/IROS2026/diffusion_policy/data/outputs/2026.01.20/17.44.20_train_diffusion_unet_lowdim_guide_guide_lowdim/checkpoints/epoch=0270-test_mean_score=0.392.ckpt"
    )
    
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=default_ckpt,
        help="Path to the trained checkpoint (.ckpt). If not found, will run in manual control mode.",
    )
    parser.add_argument("--device", default="auto", help="cpu, cuda:0, or auto")
    parser.add_argument("--no-ema", action="store_true", help="Use non-EMA model")
    parser.add_argument(
        "--action-mode",
        default=None,
        help="delta | forward_heading | position | velocity",
    )
    parser.add_argument(
        "--k-lookahead",
        "--lookahead-stride",
        dest="k_lookahead",
        type=int,
        default=None,
        help="Sample one lookahead point every k path points (e.g., 5)",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=None,
        help="Downsample factor for simulated steps (matches dataset frame_stride)",
    )
    parser.add_argument("--path-length", type=float, default=50.0)
    parser.add_argument("--leash-length", type=float, default=1.5)
    parser.add_argument("--robot-speed", type=float, default=1.5)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--inference-steps", type=int, default=64,
                        help="Number of diffusion inference steps (lower=faster, default=64)")
    parser.add_argument(
        "--turn-gain",
        type=float,
        default=1.2,
        help="Scale heading_delta before converting to turn input.",
    )
    parser.add_argument(
        "--no-curvature-slowdown",
        action="store_true",
        help="Disable curvature-based forward slowdown.",
    )
    parser.add_argument(
        "--curvature-scale",
        type=float,
        default=0.7,
        help="Slowdown strength based on heading_delta ratio.",
    )
    parser.add_argument(
        "--min-speed-scale",
        type=float,
        default=0.25,
        help="Lower bound for curvature slowdown.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "logs",
        help="Directory for planning logs (jsonl).",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="Log every N frames (default: 1).",
    )
    parser.add_argument("--no-log", action="store_true", help="Disable planning logs")
    args = parser.parse_args()

    # Check if checkpoint file exists, but don't exit if it doesn't (will use manual control)
    checkpoint_path = args.ckpt if args.ckpt.exists() else None
    if checkpoint_path is None:
        print(f"Warning: Checkpoint file not found: {args.ckpt}")
        print("Running in manual control mode only (no policy available).")
        print("\nAvailable checkpoints:")
        outputs_dir = Path("/home/yyf/IROS2026/diffusion_policy/data/outputs")
        if outputs_dir.exists():
            for ckpt in sorted(outputs_dir.rglob("*.ckpt")):
                print(f"  {ckpt}")
        print("\nContinuing with manual control...")

    planner = ModelPlanner(
        checkpoint_path=checkpoint_path,
        device=args.device,
        use_ema=not args.no_ema,
        action_mode=args.action_mode,
        k_lookahead=args.k_lookahead,
        frame_stride=args.frame_stride,
        path_length=args.path_length,
        leash_length=args.leash_length,
        robot_speed=args.robot_speed,
        fps=args.fps,
        inference_steps=args.inference_steps,
        turn_gain=args.turn_gain,
        curvature_slowdown=not args.no_curvature_slowdown,
        curvature_scale=args.curvature_scale,
        min_speed_scale=args.min_speed_scale,
        log_path=None if args.no_log else args.log_dir / f"planning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
        log_interval=args.log_interval,
    )
    planner.run()


if __name__ == "__main__":
    main()
