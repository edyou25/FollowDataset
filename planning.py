#!/usr/bin/env python3
"""
Guide Dog Robot Planning Tool (model-based simulation)

Controls:
    P     Toggle policy/manual control
    O     Toggle residual RL (if provided)
    SPACE Pause/Resume
    R     Reset position
    N     Generate new path
    ESC   Exit
    Arrows Manual control (when policy disabled)
"""
from __future__ import annotations

import argparse
import copy
import json
import importlib
import math
import os
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
try:
    import pygame
except ModuleNotFoundError:
    pygame = None  # type: ignore[assignment]
import torch
import dill

# Allow importing project modules when running from the FollowDataset directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FOLLOWDATASET_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(FOLLOWDATASET_DIR) not in sys.path:
    sys.path.insert(0, str(FOLLOWDATASET_DIR))

# Add diffusion_policy submodule to Python path
DIFFUSION_POLICY_DIR = PROJECT_ROOT / "diffusion_policy"
if str(DIFFUSION_POLICY_DIR) not in sys.path:
    sys.path.insert(0, str(DIFFUSION_POLICY_DIR))

from src.path_generator import PathGenerator
from src.physics import PhysicsEngine
from src.rl_policy import ActorCritic
from src.visualizer import Visualizer
from src.scoring import TrajectoryScorer


def _resolve_class(dotted_path: str):
    module_name, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def load_workspace_from_checkpoint(checkpoint_path: Path):
    """
    Load a diffusion_policy workspace from a checkpoint by reading `cfg._target_`.
    This supports both UNet and Transformer workspaces (and any future workspace types)
    without hard-coding the class.
    """
    payload = torch.load(checkpoint_path.open("rb"), pickle_module=dill)
    cfg = payload.get("cfg")

    target = None
    if cfg is not None:
        try:
            target = cfg.get("_target_")
        except Exception:
            target = None
        if not isinstance(target, str) or not target:
            target = getattr(cfg, "_target_", None)

    if not isinstance(target, str) or not target:
        raise ValueError(f"Checkpoint missing cfg._target_: {checkpoint_path}")

    workspace_cls = _resolve_class(target)
    workspace = workspace_cls(cfg)
    # For inference we don't need optimizer state; skip to reduce load time/memory.
    workspace.load_payload(payload, exclude_keys=("optimizer",))
    return workspace


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
        rl_ckpt: Optional[Path] = None,
        rl_device: str = "auto",
        rl_residual_scale: Optional[float] = None,
        rl_boost_margin: float = 0.0,
        rl_boost_gain: float = 0.0,
        rl_boost_max: float = 3.0,
        rl_boost_horizon: Optional[int] = None,
        no_reverse: Optional[bool] = None,
        turn_gain: float = 1.2,
        curvature_slowdown: bool = True,
        curvature_scale: float = 0.7,
        min_speed_scale: float = 0.25,
        log_path: Optional[Path] = None,
        log_interval: int = 1,
        visualizer: Optional[Visualizer] = None,
        create_visualizer: bool = True,
        collision_behavior: str = "reset",
        verbose: bool = True,
    ):
        self.verbose = bool(verbose)
        self.fps = fps
        self.sim_dt = 1.0 / fps
        self.leash_length = leash_length
        self._no_reverse_override = no_reverse
        self.no_reverse = False if no_reverse is None else bool(no_reverse)

        # Initialize modules
        self.path_generator = PathGenerator(target_length=path_length)
        self.visualizer: Optional[Visualizer] = visualizer
        if self.visualizer is None and create_visualizer:
            self.visualizer = Visualizer()

        # Load policy if checkpoint is provided
        self.checkpoint_path = checkpoint_path
        self.device = resolve_device(device)
        self.workspace = None
        self.policy = None
        if checkpoint_path is not None and checkpoint_path.exists():
            self.workspace = load_workspace_from_checkpoint(checkpoint_path)
            if use_ema and getattr(self.workspace, "ema_model", None) is not None:
                self.policy = self.workspace.ema_model
            else:
                self.policy = self.workspace.model
            self.policy.to(self.device)
            self.policy.eval()
        else:
            # No checkpoint - will use manual control only
            if self.verbose:
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

            cfg_obstacle_include_human_clearance = None
            try:
                cfg_obstacle_include_human_clearance = self.workspace.cfg.task.dataset.get(
                    "obstacle_include_human_clearance"
                )
            except Exception:
                cfg_obstacle_include_human_clearance = None

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
            self.obstacle_include_human_clearance = (
                False
                if cfg_obstacle_include_human_clearance is None
                else bool(cfg_obstacle_include_human_clearance)
            )
            self.segment_repr = str(cfg_segment_repr or "endpoints").lower()
            if self.segment_repr not in ("endpoints", "closest_dir"):
                raise ValueError(f"Unsupported segment_repr: {self.segment_repr}")
            circle_dim = 3 if self.obstacle_include_radius else 2
            clearance_dim = (
                (self.n_obstacle_circles + self.n_obstacle_segments)
                if self.obstacle_include_human_clearance
                else 0
            )
            self.obstacle_obs_dim = (
                self.n_obstacle_circles * circle_dim + self.n_obstacle_segments * 4 + clearance_dim
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
            self.obstacle_include_human_clearance = False
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
        self.collision_pause = False
        self.collision_happened = False
        self.collision_info = None
        collision_behavior = str(collision_behavior or "reset").lower()
        if collision_behavior == "freeze":
            collision_behavior = "pause"
        if collision_behavior not in ("reset", "pause"):
            raise ValueError(
                f"Unsupported collision_behavior={collision_behavior!r} (expected 'reset' or 'pause')"
            )
        self.collision_behavior = collision_behavior

        self.robot_trajectory = []
        self.human_trajectory = []
        self.planned_path = None
        self.planned_path_opt = None
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
                if self.verbose:
                    print(
                        f"Set inference steps to {inference_steps} (original: {original_steps}) "
                        "for faster performance"
                    )

        self.residual_policy: Optional[ActorCritic] = None
        self.residual_device = resolve_device(rl_device)
        self.residual_scale = 0.0
        self.use_residual = False
        self.rl_boost_margin = float(rl_boost_margin)
        self.rl_boost_gain = float(rl_boost_gain)
        self.rl_boost_max = float(rl_boost_max)
        self._rl_boost_horizon_from_cli = rl_boost_horizon is not None
        self.rl_boost_horizon = max(1, int(rl_boost_horizon or 1))
        self.residual_clearance_margin: Optional[float] = None
        self.current_residual_action: Optional[np.ndarray] = None
        self.current_base_control: Optional[Tuple[float, float]] = None
        self.current_residual_scale: float = 0.0
        if rl_ckpt is not None:
            rl_ckpt = Path(rl_ckpt)
            if rl_ckpt.exists():
                self._load_residual_policy(rl_ckpt, rl_device=rl_device, residual_scale=rl_residual_scale)
            elif self.verbose:
                print(f"[warn] RL checkpoint not found: {rl_ckpt} (residual disabled)")

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
                    "obstacle_include_human_clearance": bool(
                        getattr(self, "obstacle_include_human_clearance", False)
                    ),
                    "segment_repr": self.segment_repr,
                    "inference_interval": int(self.inference_interval),
                    "turn_gain": float(self.turn_gain),
                    "curvature_slowdown": bool(self.curvature_slowdown),
                    "curvature_scale": float(self.curvature_scale),
                    "min_speed_scale": float(self.min_speed_scale),
                    "rl_ckpt": str(rl_ckpt) if rl_ckpt is not None else None,
                    "rl_residual_scale": float(self.residual_scale),
                    "rl_boost_margin": float(self.rl_boost_margin),
                    "rl_boost_gain": float(self.rl_boost_gain),
                    "rl_boost_max": float(self.rl_boost_max),
                    "rl_boost_horizon": int(self.rl_boost_horizon),
                },
            )

        self._generate_new_path()

    def set_path_data(self, path_data: dict, reset: bool = True):
        """Override the current episode path/obstacles with externally-provided data."""
        self.current_path_data = copy.deepcopy(path_data)
        self._precompute_frenet_cache()
        if reset:
            self._reset_position()

    def _generate_new_path(self):
        """Generate new reference path."""
        self.current_path_data = self.path_generator.generate()
        self._precompute_frenet_cache()
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
        if self.verbose:
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

        self.paused = False
        self.collision_pause = False
        self.physics.reset(start)
        self.robot_trajectory = []
        self.human_trajectory = []
        self.planned_path = None
        self.planned_path_opt = None
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
            "residual_scale": float(self.current_residual_scale),
            "use_policy": bool(self.use_policy),
            "paused": bool(self.paused),
            "scores": scores,
        }
        if self.current_base_control is not None:
            payload["base_control"] = [float(self.current_base_control[0]), float(self.current_base_control[1])]
        if self.current_residual_action is not None:
            payload["residual_action"] = [
                float(self.current_residual_action[0]),
                float(self.current_residual_action[1]),
            ]
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
        if self.visualizer is None:
            return
        for event in self.visualizer.get_events():
            if event.type == pygame.QUIT:
                self.running = False

            self.visualizer.handle_event(event)

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
                    if self.verbose:
                        print(f"Control mode: {mode}")
                elif event.key == pygame.K_o:
                    if self.residual_policy is None:
                        continue
                    self.use_residual = not bool(self.use_residual)
                    if self.verbose:
                        print(f"Residual RL: {'ON' if self.use_residual else 'OFF'}")

    def _get_manual_control(self) -> Tuple[float, float]:
        if pygame is None:
            return 0.0, 0.0
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

    def _load_residual_policy(
        self,
        ckpt_path: Path,
        *,
        rl_device: str = "auto",
        residual_scale: Optional[float] = None,
    ):
        ckpt = torch.load(ckpt_path.open("rb"), map_location="cpu")
        if not isinstance(ckpt, dict):
            raise ValueError(f"Unexpected RL checkpoint format: {type(ckpt)}")
        obs_dim = int(ckpt.get("obs_dim", 0))
        hidden_dim = int(ckpt.get("hidden_dim", 256))
        env_cfg = ckpt.get("env_cfg", {})
        init_log_std = float(env_cfg.get("init_log_std", 0.0)) if isinstance(env_cfg, dict) else 0.0

        if obs_dim <= 0:
            raise ValueError("RL checkpoint missing/invalid obs_dim")
        if "state_dict" not in ckpt:
            raise ValueError("RL checkpoint missing state_dict")

        policy = ActorCritic(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            action_dim=2,
            init_log_std=init_log_std,
        )
        policy.load_state_dict(ckpt["state_dict"])

        self.residual_device = resolve_device(rl_device)
        policy.to(self.residual_device)
        policy.eval()

        scale = float(env_cfg.get("residual_scale", 0.25)) if isinstance(env_cfg, dict) else 0.25
        if residual_scale is not None:
            scale = float(residual_scale)

        self.residual_policy = policy
        self.residual_scale = scale
        self.use_residual = True
        if isinstance(env_cfg, dict):
            if self.rl_boost_margin == 0.0:
                self.rl_boost_margin = float(env_cfg.get("residual_boost_margin", self.rl_boost_margin))
            if self.rl_boost_gain == 0.0:
                self.rl_boost_gain = float(env_cfg.get("residual_boost_gain", self.rl_boost_gain))
            if self.rl_boost_max == 3.0:
                self.rl_boost_max = float(env_cfg.get("residual_boost_max", self.rl_boost_max))
            if not bool(getattr(self, "_rl_boost_horizon_from_cli", False)):
                self.rl_boost_horizon = max(
                    1, int(env_cfg.get("residual_boost_horizon", self.rl_boost_horizon))
                )
            if self._no_reverse_override is None:
                self.no_reverse = bool(env_cfg.get("no_reverse", self.no_reverse))
            reward_cfg = env_cfg.get("reward")
            if isinstance(reward_cfg, dict):
                clearance_margin = reward_cfg.get("clearance_margin")
                if clearance_margin is not None:
                    self.residual_clearance_margin = float(clearance_margin)

        expected_dim = int(self.n_obs_steps) * int(self.obs_dim)
        if expected_dim > 0 and obs_dim not in (expected_dim, expected_dim + 7) and self.verbose:
            print(
                f"[warn] RL obs_dim={obs_dim} != expected {expected_dim} (base) or {expected_dim + 7} (base+aug)"
            )

        base_ckpt = env_cfg.get("base_ckpt") if isinstance(env_cfg, dict) else None
        if base_ckpt and self.checkpoint_path is not None and self.verbose:
            if str(self.checkpoint_path) != str(base_ckpt):
                print(f"[warn] RL was trained with base_ckpt={base_ckpt}, but planning uses ckpt={self.checkpoint_path}")

        if self.verbose:
            print(f"Loaded residual RL: {ckpt_path} | scale={self.residual_scale:.3f} | device={self.residual_device}")

    def _predict_residual_action(self) -> Optional[np.ndarray]:
        if self.residual_policy is None:
            return None
        if len(self.obs_history) == 0:
            return None
        obs_seq = np.stack(self.obs_history, axis=0).astype(np.float32)
        obs_flat = obs_seq.reshape(1, -1)

        # Backward compatible: if RL expects augmented obs, append safety features.
        expected_base_dim = int(self.n_obs_steps) * int(self.obs_dim)
        ckpt_obs_dim = int(getattr(self.residual_policy, "obs_dim", obs_flat.shape[1]))
        if ckpt_obs_dim == expected_base_dim + 7:
            aug = self._build_residual_aug_features()
            obs_flat = np.concatenate([obs_flat, aug.reshape(1, -1)], axis=1).astype(np.float32)
        elif ckpt_obs_dim == expected_base_dim:
            pass
        else:
            # Keep running even if mismatch (helps debugging) by padding/truncating to ckpt dim.
            if self.verbose and not hasattr(self, "_warned_residual_obs_dim"):
                setattr(self, "_warned_residual_obs_dim", True)
                print(
                    f"[warn] residual obs_dim={ckpt_obs_dim} doesn't match base({expected_base_dim}) "
                    "or base+aug(7); padding/truncating observation to fit."
                )
            cur_dim = int(obs_flat.shape[1])
            if ckpt_obs_dim < cur_dim:
                obs_flat = obs_flat[:, :ckpt_obs_dim].astype(np.float32)
            elif ckpt_obs_dim > cur_dim:
                pad = np.zeros((1, ckpt_obs_dim - cur_dim), dtype=np.float32)
                obs_flat = np.concatenate([obs_flat.astype(np.float32), pad], axis=1)
        obs_tensor = torch.from_numpy(obs_flat).to(self.residual_device, dtype=torch.float32)
        action = self.residual_policy.act_deterministic(obs_tensor)[0].detach().cpu().numpy().astype(np.float32)
        return action

    def _build_residual_aug_features(self) -> np.ndarray:
        """
        Must match GuideFollowEnv._build_residual_aug_features() layout (7 dims).
        Uses current state + current_base_control to provide early collision cues.
        """
        if self.current_path_data is None:
            return np.zeros((7,), dtype=np.float32)
        obstacles = self.current_path_data.get("obstacles")
        segments = self.current_path_data.get("segment_obstacles")

        clearance_now = float(
            min(
                self._min_clearance_to_obstacles(
                    self.physics.robot.position,
                    agent_radius=float(self.physics.robot_radius),
                    obstacles=obstacles,
                    segment_obstacles=segments,
                ),
                self._min_clearance_to_obstacles(
                    self.physics.human.position,
                    agent_radius=float(self.physics.human_radius),
                    obstacles=obstacles,
                    segment_obstacles=segments,
                ),
            )
        )
        if self.residual_clearance_margin is not None:
            margin = float(self.residual_clearance_margin)
        else:
            margin = float(self.rl_boost_margin) if self.rl_boost_margin > 0.0 else 0.3
        violation_now = float(max(0.0, margin - clearance_now))

        base_forward = 0.0
        base_turn = 0.0
        clearance_base = float("inf")
        if self.current_base_control is not None:
            base_forward, base_turn = self.current_base_control
            clearance_base = self._predict_min_clearance(float(base_forward), float(base_turn))

        # Most-dangerous obstacle (closest to robot or human), expressed as vector from robot to obstacle point.
        robot_pos = self.physics.robot.position.astype(np.float32)
        human_pos = self.physics.human.position.astype(np.float32)
        heading = float(self.physics.robot.heading)
        rr = float(self.physics.robot_radius)
        hr = float(self.physics.human_radius)

        best_clear = float("inf")
        best_point: Optional[np.ndarray] = None
        agents = (
            (robot_pos, rr),
            (human_pos, hr),
        )

        if obstacles is not None and len(obstacles) > 0:
            for obs in obstacles:
                if isinstance(obs, dict):
                    ox = float(obs.get("x", 0.0))
                    oy = float(obs.get("y", 0.0))
                    radius = float(obs.get("r", 0.0))
                else:
                    ox = float(obs[0])
                    oy = float(obs[1])
                    radius = float(obs[2])
                center = np.array([ox, oy], dtype=np.float32)
                r_obs = float(radius)
                for agent_pos, agent_r in agents:
                    vec_ca = agent_pos - center
                    dist = float(np.linalg.norm(vec_ca))
                    clear = dist - (r_obs + float(agent_r))
                    if clear < best_clear:
                        if dist > 1e-6:
                            dir_ca = (vec_ca / dist).astype(np.float32)
                        else:
                            dir_ca = np.array([1.0, 0.0], dtype=np.float32)
                        boundary = center + dir_ca * r_obs
                        best_clear = clear
                        best_point = boundary

        if segments is not None and len(segments) > 0:
            for seg in segments:
                if isinstance(seg, dict):
                    if "p1" in seg and "p2" in seg:
                        p1 = np.array(seg["p1"], dtype=np.float32)
                        p2 = np.array(seg["p2"], dtype=np.float32)
                    else:
                        p1 = np.array([seg.get("x1", 0.0), seg.get("y1", 0.0)], dtype=np.float32)
                        p2 = np.array([seg.get("x2", 0.0), seg.get("y2", 0.0)], dtype=np.float32)
                else:
                    p1 = np.array([seg[0], seg[1]], dtype=np.float32)
                    p2 = np.array([seg[2], seg[3]], dtype=np.float32)
                ab = p2 - p1
                denom = float(np.dot(ab, ab))
                for agent_pos, agent_r in agents:
                    if denom < 1e-12:
                        closest = p1
                    else:
                        t_proj = float(np.dot(agent_pos - p1, ab)) / denom
                        t_proj = float(np.clip(t_proj, 0.0, 1.0))
                        closest = p1 + t_proj * ab
                    diff = agent_pos - closest
                    dist_sq = float(np.dot(diff, diff))
                    dist = float(np.sqrt(dist_sq))
                    clear = dist - float(agent_r)
                    if clear < best_clear:
                        best_clear = clear
                        best_point = closest

        if best_point is None:
            vec_rf = np.zeros((2,), dtype=np.float32)
        else:
            vec_world = (best_point - robot_pos).astype(np.float32)
            cos_h = float(np.cos(heading))
            sin_h = float(np.sin(heading))
            vx = cos_h * float(vec_world[0]) + sin_h * float(vec_world[1])
            vy = -sin_h * float(vec_world[0]) + cos_h * float(vec_world[1])
            vec_rf = np.clip(np.array([vx, vy], dtype=np.float32), -5.0, 5.0)

        return np.array(
            [
                float(np.clip(clearance_now, -10.0, 10.0)),
                float(np.clip(violation_now, 0.0, 10.0)),
                float(np.clip(clearance_base, -10.0, 10.0)),
                float(np.clip(base_forward, -1.0, 1.0)),
                float(np.clip(base_turn, -1.0, 1.0)),
                float(vec_rf[0]),
                float(vec_rf[1]),
            ],
            dtype=np.float32,
        )

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
        obstacle_features = self._build_obstacle_features(robot_pos, human_pos, heading)
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
        self,
        robot_pos: np.ndarray,
        human_pos: np.ndarray,
        heading: Optional[float] = None,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self.current_path_data is None:
            return None, None

        path = self.current_path_data.get("path")
        path_s = self.current_path_data.get("_path_s")
        if path is None or len(path) == 0 or path_s is None:
            return None, None
        path = np.asarray(path, dtype=np.float32)
        path_s = np.asarray(path_s, dtype=np.float32)
        if len(path_s) != len(path):
            self._precompute_frenet_cache()
            path_s = self.current_path_data.get("_path_s")
            if path_s is None:
                return None, None
            path_s = np.asarray(path_s, dtype=np.float32)

        # Select k nearest *forward* obstacles w.r.t. the human progress along the
        # reference path (instead of the robot), matching training-time features.
        idx_h = self._nearest_path_index(path, human_pos)
        s_human = float(path_s[idx_h])

        selected_circles = None
        selected_segments = None

        if self.n_obstacle_circles > 0:
            obstacles = self.current_path_data.get("obstacles")
            if obstacles is not None and len(obstacles) > 0:
                circle_obs = np.asarray(obstacles, dtype=np.float32)
                if circle_obs.ndim == 2 and circle_obs.shape[1] >= 3:
                    circle_obs = circle_obs[:, :3]
                    circle_s = self.current_path_data.get("_circle_s")
                    if circle_s is None or len(circle_s) != len(circle_obs):
                        self._precompute_frenet_cache()
                        circle_s = self.current_path_data.get("_circle_s")
                    if circle_s is not None and len(circle_s) == len(circle_obs):
                        circle_s = np.asarray(circle_s, dtype=np.float32)
                        delta_s = circle_s - s_human
                        candidates = np.nonzero(delta_s >= 0.0)[0]
                        if len(candidates) > 0:
                            order = candidates[np.argsort(delta_s[candidates])]
                        else:
                            order = np.zeros((0,), dtype=np.int64)
                    else:
                        centers = circle_obs[:, :2]
                        rel = centers - human_pos
                        dist_sq = np.sum(rel * rel, axis=1)
                        order = np.argsort(dist_sq)
                    count = min(self.n_obstacle_circles, int(len(order)))
                    if count > 0:
                        selected_circles = circle_obs[order[:count], :3]

        if self.n_obstacle_segments > 0:
            segments = self.current_path_data.get("segment_obstacles")
            if segments is not None and len(segments) > 0:
                seg_obs = np.asarray(segments, dtype=np.float32)
                if seg_obs.ndim == 2 and seg_obs.shape[1] >= 4:
                    seg_obs = seg_obs[:, :4]
                    seg_s_min = self.current_path_data.get("_segment_s_min")
                    seg_s_max = self.current_path_data.get("_segment_s_max")
                    if (
                        seg_s_min is None
                        or seg_s_max is None
                        or len(seg_s_min) != len(seg_obs)
                        or len(seg_s_max) != len(seg_obs)
                    ):
                        self._precompute_frenet_cache()
                        seg_s_min = self.current_path_data.get("_segment_s_min")
                        seg_s_max = self.current_path_data.get("_segment_s_max")
                    if (
                        seg_s_min is not None
                        and seg_s_max is not None
                        and len(seg_s_min) == len(seg_obs)
                        and len(seg_s_max) == len(seg_obs)
                    ):
                        seg_s_min = np.asarray(seg_s_min, dtype=np.float32)
                        seg_s_max = np.asarray(seg_s_max, dtype=np.float32)
                        valid = seg_s_max >= s_human
                        candidates = np.nonzero(valid)[0]
                        if len(candidates) > 0:
                            delta = np.maximum(0.0, seg_s_min - s_human)
                            order = candidates[np.argsort(delta[candidates])]
                        else:
                            order = np.zeros((0,), dtype=np.int64)
                    else:
                        dist_sq = np.zeros((len(seg_obs),), dtype=np.float32)
                        for i, seg in enumerate(seg_obs):
                            dist_sq[i] = self._point_segment_dist_sq(
                                human_pos, seg[:2], seg[2:4]
                            )
                        order = np.argsort(dist_sq)
                    count = min(self.n_obstacle_segments, int(len(order)))
                    if count > 0:
                        selected_segments = seg_obs[order[:count], :4]

        return selected_circles, selected_segments

    def _precompute_frenet_cache(self):
        if self.current_path_data is None:
            return
        path = self.current_path_data.get("path")
        if path is None or len(path) == 0:
            self.current_path_data["_path_s"] = None
            self.current_path_data["_circle_s"] = None
            self.current_path_data["_segment_s_min"] = None
            self.current_path_data["_segment_s_max"] = None
            return
        path = np.asarray(path, dtype=np.float32)
        path_s = self._compute_path_s(path)
        self.current_path_data["_path_s"] = path_s

        obstacles = self.current_path_data.get("obstacles")
        circle_s = None
        if obstacles is not None and len(obstacles) > 0:
            circle_obs = np.asarray(obstacles, dtype=np.float32)
            if circle_obs.ndim == 2 and circle_obs.shape[1] >= 3:
                circle_obs = circle_obs[:, :3]
                circle_s = np.zeros((len(circle_obs),), dtype=np.float32)
                for i, center in enumerate(circle_obs[:, :2]):
                    idx = self._nearest_path_index(path, center)
                    circle_s[i] = path_s[idx]
        self.current_path_data["_circle_s"] = circle_s

        segments = self.current_path_data.get("segment_obstacles")
        seg_s_min = None
        seg_s_max = None
        if segments is not None and len(segments) > 0:
            seg_obs = np.asarray(segments, dtype=np.float32)
            if seg_obs.ndim == 2 and seg_obs.shape[1] >= 4:
                seg_obs = seg_obs[:, :4]
                s1 = np.zeros((len(seg_obs),), dtype=np.float32)
                s2 = np.zeros((len(seg_obs),), dtype=np.float32)
                for i, seg in enumerate(seg_obs):
                    idx1 = self._nearest_path_index(path, seg[:2])
                    idx2 = self._nearest_path_index(path, seg[2:4])
                    s1[i] = path_s[idx1]
                    s2[i] = path_s[idx2]
                seg_s_min = np.minimum(s1, s2).astype(np.float32)
                seg_s_max = np.maximum(s1, s2).astype(np.float32)
        self.current_path_data["_segment_s_min"] = seg_s_min
        self.current_path_data["_segment_s_max"] = seg_s_max

    def _compute_path_s(self, path: np.ndarray) -> np.ndarray:
        if path is None or len(path) == 0:
            return np.zeros((0,), dtype=np.float32)
        if len(path) == 1:
            return np.zeros((1,), dtype=np.float32)
        diffs = np.diff(path.astype(np.float32), axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1).astype(np.float32)
        s = np.zeros((len(path),), dtype=np.float32)
        s[1:] = np.cumsum(seg_lengths, axis=0).astype(np.float32)
        return s

    def _nearest_path_index(self, path: np.ndarray, point: np.ndarray) -> int:
        diffs = path - point.astype(np.float32)
        dist_sq = np.sum(diffs * diffs, axis=1)
        return int(np.argmin(dist_sq))

    def _build_obstacle_features(
        self, robot_pos: np.ndarray, human_pos: np.ndarray, heading: float
    ) -> np.ndarray:
        circle_dim = 3 if self.obstacle_include_radius else 2
        clearance_dim = (
            (self.n_obstacle_circles + self.n_obstacle_segments)
            if getattr(self, "obstacle_include_human_clearance", False)
            else 0
        )
        total_dim = self.n_obstacle_circles * circle_dim + self.n_obstacle_segments * 4 + clearance_dim
        if total_dim == 0:
            return np.zeros((0,), dtype=np.float32)

        feats = np.zeros((total_dim,), dtype=np.float32)
        offset = 0
        if self.current_path_data is None:
            return feats

        selected_circles, selected_segments = self._select_obstacles_for_observation(
            robot_pos, human_pos, heading
        )
        clearance_offset = self.n_obstacle_circles * circle_dim + self.n_obstacle_segments * 4
        circle_clear_offset = clearance_offset
        seg_clear_offset = circle_clear_offset + self.n_obstacle_circles
        if getattr(self, "obstacle_include_human_clearance", False) and clearance_dim > 0:
            feats[clearance_offset:] = 5.0

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
                    if getattr(self, "obstacle_include_human_clearance", False):
                        d = human_pos - selected_circles[i, :2]
                        dist = float(np.linalg.norm(d))
                        clearance = dist - float(selected_circles[i, 2] + self.physics.human_radius)
                        feats[circle_clear_offset + i] = float(clearance)
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
                    if getattr(self, "obstacle_include_human_clearance", False):
                        dist_sq_h = self._point_segment_dist_sq(human_pos, seg[:2], seg[2:4])
                        clearance = float(np.sqrt(dist_sq_h)) - float(self.physics.human_radius)
                        feats[seg_clear_offset + i] = float(clearance)
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

    def _min_clearance_to_obstacles(
        self,
        point: np.ndarray,
        *,
        agent_radius: float,
        obstacles,
        segment_obstacles,
    ) -> float:
        min_clearance = float("inf")
        if obstacles is not None and len(obstacles) > 0:
            for obs in obstacles:
                if isinstance(obs, dict):
                    ox = float(obs.get("x", 0.0))
                    oy = float(obs.get("y", 0.0))
                    radius = float(obs.get("r", 0.0))
                else:
                    ox = float(obs[0])
                    oy = float(obs[1])
                    radius = float(obs[2])
                dx = float(point[0] - ox)
                dy = float(point[1] - oy)
                dist = math.sqrt(dx * dx + dy * dy)
                clearance = dist - (radius + float(agent_radius))
                if clearance < min_clearance:
                    min_clearance = clearance

        if segment_obstacles is not None and len(segment_obstacles) > 0:
            for seg in segment_obstacles:
                if isinstance(seg, dict):
                    if "p1" in seg and "p2" in seg:
                        p1 = np.array(seg["p1"], dtype=np.float32)
                        p2 = np.array(seg["p2"], dtype=np.float32)
                    else:
                        p1 = np.array([seg.get("x1", 0.0), seg.get("y1", 0.0)], dtype=np.float32)
                        p2 = np.array([seg.get("x2", 0.0), seg.get("y2", 0.0)], dtype=np.float32)
                else:
                    p1 = np.array([seg[0], seg[1]], dtype=np.float32)
                    p2 = np.array([seg[2], seg[3]], dtype=np.float32)
                dist_sq = self._point_segment_dist_sq(point.astype(np.float32), p1, p2)
                dist = math.sqrt(dist_sq)
                clearance = dist - float(agent_radius)
                if clearance < min_clearance:
                    min_clearance = clearance

        return float(min_clearance)

    def _predict_min_clearance(self, forward: float, turn: float) -> float:
        if self.current_path_data is None:
            return float("inf")
        obstacles = self.current_path_data.get("obstacles")
        segments = self.current_path_data.get("segment_obstacles")
        if (obstacles is None or len(obstacles) == 0) and (segments is None or len(segments) == 0):
            return float("inf")

        sim = copy.deepcopy(self.physics)
        sim.set_control(float(forward), float(turn))
        min_clear = float("inf")
        n_steps = int(self.frame_stride) * int(max(1, int(self.rl_boost_horizon)))
        for _ in range(n_steps):
            robot_state, human_state = sim.step()
            min_clear = min(
                min_clear,
                self._min_clearance_to_obstacles(
                    robot_state.position,
                    agent_radius=float(sim.robot_radius),
                    obstacles=obstacles,
                    segment_obstacles=segments,
                ),
                self._min_clearance_to_obstacles(
                    human_state.position,
                    agent_radius=float(sim.human_radius),
                    obstacles=obstacles,
                    segment_obstacles=segments,
                ),
            )
        return float(min_clear)

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

    def _actions_to_path_with_residual(
        self,
        robot_pos: np.ndarray,
        action_seq: np.ndarray,
        residual_action: np.ndarray,
    ) -> Optional[np.ndarray]:
        if action_seq.size == 0:
            return None
        if residual_action is None or len(residual_action) < 2:
            return None

        sim = copy.deepcopy(self.physics)
        sim.robot.position = robot_pos.astype(np.float32).copy()
        points = []
        obstacles = self.current_path_data.get("obstacles") if self.current_path_data else None
        segments = self.current_path_data.get("segment_obstacles") if self.current_path_data else None

        dr0 = float(self.residual_scale) * float(residual_action[0])
        dr1 = float(self.residual_scale) * float(residual_action[1])

        for act in action_seq:
            if self.action_mode == "forward_heading":
                forward_delta = float(act[0])
                heading_delta = float(act[1])

                turn_speed = float(sim.turn_speed)
                robot_speed = float(sim.robot_speed)
                turn_delta = heading_delta * float(self.turn_gain)
                base_turn = turn_delta / (turn_speed * self.data_dt) if turn_speed > 0 else 0.0
                base_forward = forward_delta / (robot_speed * self.data_dt) if robot_speed > 0 else 0.0

                if self.curvature_slowdown and turn_speed > 0:
                    max_turn = turn_speed * self.data_dt
                    if max_turn > 1e-6:
                        ratio = min(1.0, abs(turn_delta) / max_turn)
                        speed_scale = max(
                            float(self.min_speed_scale),
                            1.0 - float(self.curvature_scale) * ratio,
                        )
                        base_forward *= speed_scale

                base_forward = float(np.clip(base_forward, -1.0, 1.0))
                base_turn = float(np.clip(base_turn, -1.0, 1.0))
            else:
                if self.action_mode == "delta":
                    delta = np.asarray(act, dtype=np.float32)
                elif self.action_mode == "position":
                    delta = np.asarray(act, dtype=np.float32) - sim.robot.position.astype(np.float32)
                else:
                    delta = np.asarray(act, dtype=np.float32) * float(self.data_dt)

                if self.robot_frame:
                    cos_h = float(np.cos(sim.robot.heading))
                    sin_h = float(np.sin(sim.robot.heading))
                    dx = cos_h * float(delta[0]) - sin_h * float(delta[1])
                    dy = sin_h * float(delta[0]) + cos_h * float(delta[1])
                    delta = np.array([dx, dy], dtype=np.float32)

                base_forward, base_turn = self._delta_to_control(
                    delta, sim.robot.heading, dt=self.data_dt
                )

            forward = float(np.clip(base_forward + dr0, -1.0, 1.0))
            turn = float(np.clip(base_turn + dr1, -1.0, 1.0))
            sim.set_control(forward, turn)

            for _ in range(int(self.frame_stride)):
                robot_state, _human_state = sim.step()
                points.append(robot_state.position.copy())
                if obstacles is not None or segments is not None:
                    collided, _info = sim.check_collision(obstacles, segment_obstacles=segments)
                    if collided:
                        return np.stack(points, axis=0) if points else None

        return np.stack(points, axis=0) if points else None

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
        self.collision_happened = False
        self.collision_info = None
        if self.paused or self.collision_pause:
            # Freeze simulation state while paused (manual or auto-paused).
            self.physics.set_control(0.0, 0.0)
            return self.physics.robot.copy(), self.physics.human.copy()

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
                        if self.residual_policy is not None:
                            residual_action = self._predict_residual_action()
                            self.current_residual_action = residual_action
                            if residual_action is not None:
                                self.planned_path_opt = self._actions_to_path_with_residual(
                                    self.physics.robot.position, action_seq, residual_action
                                )
                            else:
                                self.planned_path_opt = None
                        else:
                            self.planned_path_opt = None
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
                    self.current_base_control = (float(forward), float(turn))
                    if self.residual_policy is not None:
                        residual_action = self._predict_residual_action()
                        self.current_residual_action = residual_action
                        if self.use_residual and residual_action is not None:
                            residual_scale = float(self.residual_scale)
                            if self.rl_boost_gain > 0.0 and self.rl_boost_margin > 0.0:
                                clearance = self._predict_min_clearance(forward, turn)
                                if math.isfinite(clearance):
                                    violation = max(0.0, float(self.rl_boost_margin) - float(clearance))
                                else:
                                    violation = 0.0
                                boost = 1.0 + float(self.rl_boost_gain) * (violation / float(self.rl_boost_margin))
                                boost = min(float(self.rl_boost_max), max(1.0, float(boost)))
                                residual_scale = residual_scale * boost
                            self.current_residual_scale = float(residual_scale)
                            forward = float(
                                np.clip(
                                    float(forward) + float(residual_scale) * float(residual_action[0]),
                                    -1.0,
                                    1.0,
                                )
                            )
                            turn = float(
                                np.clip(
                                    float(turn) + float(residual_scale) * float(residual_action[1]),
                                    -1.0,
                                    1.0,
                                )
                            )
                        else:
                            self.current_residual_scale = 0.0
                    if self.no_reverse:
                        forward = max(0.0, float(forward))
                    self.current_delta = delta
                    self.cached_control = (forward, turn)
                else:
                    forward, turn = self.cached_control
                    action = self.current_action
                    delta = self.current_delta
            else:
                forward, turn = self._get_manual_control()
                if self.no_reverse:
                    forward = max(0.0, float(forward))
                self.planned_path = None
                self.planned_path_opt = None
                # Reset cache when switching to manual
                self.cached_action_seq = None
                self.cached_action_idx = 0
                self.frames_since_inference = 0
                self.cached_control = (forward, turn)
                self.current_action = None
                self.current_delta = None
                self.current_speed_scale = 1.0

        if self.no_reverse:
            forward = max(0.0, float(forward))
        self.physics.set_control(forward, turn)
        robot_state, human_state = self.physics.step()

        if self._check_collision():
            return self.physics.robot.copy(), self.physics.human.copy()

        if self.current_path_data is not None:
            path = self.current_path_data.get("path")
            path_s = self.current_path_data.get("_path_s")
            if path is not None and len(path) > 0:
                if path_s is None or len(path_s) != len(path):
                    self._precompute_frenet_cache()
                    path_s = self.current_path_data.get("_path_s")
                if path_s is not None and len(path_s) == len(path):
                    path = np.asarray(path, dtype=np.float32)
                    path_s = np.asarray(path_s, dtype=np.float32)
                    idx_h = self._nearest_path_index(path, human_state.position)
                    s_human = float(path_s[idx_h])
                    s_end = float(path_s[-1])
                    if s_human >= s_end - 1e-3:
                        self.paused = True
                        self.physics.set_control(0.0, 0.0)
                        self._log_event(
                            "goal_reached",
                            {
                                "human_pos": human_state.position.tolist(),
                                "s_human": float(s_human),
                                "s_end": float(s_end),
                            },
                        )
                        if self.verbose:
                            print(
                                "Human reached end of path. Simulation paused "
                                "(SPACE to resume, N for new path)."
                            )

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
            self.collision_happened = True
            self.collision_info = info
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
            if self.collision_behavior == "pause":
                if self.verbose:
                    print(f"Collision detected ({who}, {label} {idx}), pausing at collision.")
                self.collision_pause = True
                self.physics.set_control(0.0, 0.0)
                return True
            if self.verbose:
                print(f"Collision detected ({who}, {label} {idx}), resetting.")
            self._reset_position()
            return True
        return False

    def _render(self, robot_state, human_state, actual_fps: float):
        scores = self.scorer.get_scores() if self.scorer else {}
        mode = "policy" if self.use_policy else "manual"
        if self.use_policy and self.residual_policy is not None:
            mode = "policy+rl" if self.use_residual else "policy (base)"
        if self.paused:
            mode += " (paused)"
        elif self.collision_pause:
            mode += " (collision)"

        controls = [
            "P: Policy/Manual",
            "SPACE: Pause",
            "R: Reset",
            "N: New Path",
            "Arrows: Manual control",
            "Scroll: Zoom",
            "ESC: Exit",
        ]
        if self.residual_policy is not None:
            controls.insert(1, "O: Toggle residual RL")

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
            "controls": controls,
        }

        obs_obstacles, obs_segment_obstacles = self._select_obstacles_for_observation(
            robot_state.position, human_state.position, robot_state.heading
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
            planned_path_opt=self.planned_path_opt,
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
        if self.visualizer is None:
            raise RuntimeError(
                "ModelPlanner.run() requires a Visualizer. "
                "Pass create_visualizer=True or provide a visualizer instance."
            )
        print("=" * 60)
        print("Guide Dog Robot Planning Tool")
        print("=" * 60)
        extra = " | O=ResidualRL" if self.residual_policy is not None else ""
        print(f"Controls: P=Policy/Manual{extra} | SPACE=Pause | R=Reset | N=NewPath | ESC=Exit")
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
        "/home/yyf/IROS2026/diffusion_policy/data/outputs/2026.01.21/14.14.46_train_diffusion_unet_lowdim_guide_guide_lowdim/checkpoints/epoch=0090-test_mean_score=0.630.ckpt"
    )
    default_rl_ckpt = Path(
        "/home/yyf/IROS2026/FollowDataset/rl_models/ppo_20260126_145946.pt"
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
        "--rl-ckpt",
        type=Path,
        default=default_rl_ckpt,
        help="Optional PPO residual checkpoint (.pt). Press O to toggle base vs base+RL.",
    )
    parser.add_argument("--rl-device", default="auto", help="cpu, cuda:0, or auto (for residual RL)")
    parser.add_argument(
        "--rl-residual-scale",
        type=float,
        default=None,
        help="Override residual_scale stored in the PPO checkpoint.",
    )
    parser.add_argument(
        "--rl-boost-margin",
        type=float,
        default=0.0,
        help="When >0, boost residual_scale if predicted clearance falls below this margin (m).",
    )
    parser.add_argument(
        "--rl-boost-gain",
        type=float,
        default=0.0,
        help="Boost factor slope for residual_scale near obstacles (0 disables boosting).",
    )
    parser.add_argument(
        "--rl-boost-max",
        type=float,
        default=3.0,
        help="Upper bound on the residual_scale boost multiplier.",
    )
    parser.add_argument(
        "--rl-boost-horizon",
        type=int,
        default=None,
        help="Predict clearance over this many future control steps when boosting (default: load from RL ckpt env_cfg or 1).",
    )
    reverse_group = parser.add_mutually_exclusive_group()
    reverse_group.add_argument("--no-reverse", action="store_true", help="Clamp forward control to >=0 (disallow reversing).")
    reverse_group.add_argument(
        "--allow-reverse",
        action="store_true",
        help="Allow reverse motion (overrides RL checkpoint env_cfg no_reverse).",
    )
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

    no_reverse = None
    if bool(getattr(args, "no_reverse", False)):
        no_reverse = True
    if bool(getattr(args, "allow_reverse", False)):
        no_reverse = False

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
        rl_ckpt=args.rl_ckpt,
        rl_device=args.rl_device,
        rl_residual_scale=args.rl_residual_scale,
        rl_boost_margin=args.rl_boost_margin,
        rl_boost_gain=args.rl_boost_gain,
        rl_boost_max=args.rl_boost_max,
        rl_boost_horizon=args.rl_boost_horizon,
        no_reverse=no_reverse,
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
