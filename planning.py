#!/usr/bin/env python3
from __future__ import annotations
"""
Guide Dog Robot Planning Tool (model-based simulation)

Controls:
    P     Toggle policy/manual control
    SPACE Pause/Resume
    R     Reset position
    N     Generate new path
    O     Add obstacle ahead
    ESC   Exit
    Arrows Manual control (when policy disabled)
"""
import argparse
import copy
import json
import importlib
import re
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pygame
import torch
import dill
from scipy.spatial.transform import Rotation

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
from src.visualizer import Visualizer
from src.data_storage import DataStorage
from src.mid360_storage import Mid360DataStorage
from src.scoring import TrajectoryScorer
from src.safety_filter import QPSafetyFilter
from src.compliance_control import (
    ComplianceControlConfig,
    apply_bre_compliance_control,
    apply_interaction_aware_compliance_control,
)
from src.mid360_gazebo import (
    Mid360GazeboConfig,
    Mid360GazeboSession,
    T_BASE_MID360,
    resolve_mid360_plugin_dir,
    resolve_mid360_plugin_library,
)
from src.vector_map_pointcloud import VectorMapPointCloudConfig, VectorMapPointCloudSimulator
from diffusion_policy.common.guide_mid360 import (
    GuideMid360ObservationConfig,
    encode_mid360_scan_from_pointcloud,
    encode_mid360_scan_from_local_points,
)


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
        # return torch.device("cpu")
    return torch.device(device)


def resolve_default_checkpoint() -> Path:
    outputs_dir = PROJECT_ROOT / "diffusion_policy" / "data" / "outputs"
    fallback = Path(
        "/home/yyf/IROS2026/diffusion_policy/data/outputs/2026.01.21/14.14.46_train_diffusion_unet_lowdim_guide_guide_lowdim/checkpoints/epoch=0090-test_mean_score=0.630.ckpt"
    )
    if not outputs_dir.exists():
        return fallback

    score_pattern = re.compile(r"test_mean_score=([0-9.]+)\.ckpt$")
    mid360_scored: list[tuple[float, float, Path]] = []
    mid360_latest: list[tuple[float, Path]] = []

    for ckpt in outputs_dir.rglob("*.ckpt"):
        path_str = str(ckpt)
        if "guide_mid360" not in path_str:
            continue
        try:
            mtime = ckpt.stat().st_mtime
        except OSError:
            continue
        match = score_pattern.search(ckpt.name)
        if match is not None:
            try:
                score = float(match.group(1))
            except ValueError:
                score = float("-inf")
            mid360_scored.append((score, mtime, ckpt))
        elif ckpt.name == "latest.ckpt":
            mid360_latest.append((mtime, ckpt))

    if mid360_scored:
        mid360_scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return mid360_scored[0][2]
    if mid360_latest:
        mid360_latest.sort(key=lambda item: item[0], reverse=True)
        return mid360_latest[0][1]
    return fallback


def normalize_safety_mode(mode: Optional[str]) -> str:
    mode = str(mode or "off").lower()
    aliases = {
        "off": "off",
        "none": "off",
        "diffusion": "off",
        "qp": "robot_qp",
        "robot": "robot_qp",
        "robot_qp": "robot_qp",
        "human_robot": "human_robot_qp",
        "human_robot_qp": "human_robot_qp",
        "human+robot": "human_robot_qp",
    }
    if mode not in aliases:
        raise ValueError(
            f"Unsupported safety_mode={mode!r} (expected off, robot_qp, or human_robot_qp)"
        )
    return aliases[mode]


def normalize_pointcloud_mode(mode: Optional[str]) -> str:
    mode = str(mode or "auto").lower()
    aliases = {
        "auto": "auto",
        "off": "off",
        "none": "off",
        "mid360": "vector_map",
        "vector_map": "vector_map",
        "gazebo": "gazebo",
        "ros": "gazebo",
        "ros_gazebo": "gazebo",
        "live": "gazebo",
    }
    if mode not in aliases:
        raise ValueError(
            f"Unsupported pointcloud_mode={mode!r} "
            "(expected auto, off, vector_map, or gazebo)"
        )
    return aliases[mode]


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
        corridor_width: float = 2.1,
        obstacle_radius: float = 0.3,
        leash_length: float = 1.5,
        robot_speed: float = 1.0,
        robot_radius: float = 0.3,
        human_radius: float = 0.3,
        fps: int = 20,
        inference_steps: int = 8,
        turn_gain: float = 1.2,
        curvature_slowdown: bool = True,
        curvature_scale: float = 0.7,
        min_speed_scale: float = 0.25,
        log_path: Optional[Path] = None,
        eval_path: Optional[Path] = None,
        log_interval: int = 1,
        collect_enabled: bool = False,
        visualizer: Optional[Visualizer] = None,
        create_visualizer: bool = True,
        collision_behavior: str = "reset",
        safety_mode: str = "off",
        pointcloud_mode: str = "auto",
        mid360_plugin_dir: Optional[str] = None,
        mid360_plugin_lib: Optional[str] = None,
        mid360_downsample: int = 1,
        mid360_gazebo_gui: bool = False,
        mid360_visualize: bool = False,
        safety_margin: float = 0.02,
        safety_alpha: float = 1.0,
        safety_max_constraints: int = 8,
        safety_influence_distance: float = 1.0,
        debug_preview: bool = False,
        debug_preview_limit: int = 5,
        debug_policy: bool = False,
        debug_qp_log: bool = True,
    ):
        self.fps = fps
        self.sim_dt = 1.0 / fps
        self.leash_length = leash_length

        # Initialize modules
        self.path_generator = PathGenerator(
            target_length=path_length,
            corridor_width=corridor_width,
            obstacle_radius=obstacle_radius,
        )
        self.visualizer: Optional[Visualizer] = visualizer
        if self.visualizer is None and create_visualizer:
            self.visualizer = Visualizer()

        # Load policy if checkpoint is provided
        self.device = resolve_device(device)
        self.requested_pointcloud_mode = normalize_pointcloud_mode(pointcloud_mode)
        self.mid360_plugin_dir_arg = mid360_plugin_dir
        self.mid360_plugin_lib_arg = mid360_plugin_lib
        self.mid360_downsample = max(1, int(mid360_downsample))
        self.mid360_gazebo_gui = bool(mid360_gazebo_gui)
        self.mid360_visualize = bool(mid360_visualize)
        self.mid360_gazebo_config: Optional[Mid360GazeboConfig] = None
        self.mid360_session: Optional[Mid360GazeboSession] = None
        self._last_mid360_cloud_seq: Optional[int] = None
        self._mid360_visual_max_points = 4096
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

            cfg_task_name = None
            try:
                cfg_task_name = self.workspace.cfg.task.get("name")
            except Exception:
                cfg_task_name = None

            cfg_dataset_target = None
            try:
                cfg_dataset_target = self.workspace.cfg.task.dataset.get("_target_")
            except Exception:
                cfg_dataset_target = None

            cfg_lidar_num_bins = None
            try:
                cfg_lidar_num_bins = self.workspace.cfg.task.get("lidar_num_bins")
            except Exception:
                cfg_lidar_num_bins = None
            if cfg_lidar_num_bins is None:
                try:
                    cfg_lidar_num_bins = self.workspace.cfg.task.dataset.get("lidar_num_bins")
                except Exception:
                    cfg_lidar_num_bins = None

            def _cfg_dataset_value(key: str, default=None):
                try:
                    value = self.workspace.cfg.task.dataset.get(key)
                except Exception:
                    value = None
                if value is None:
                    return default
                return value

            detected_observation_mode = "mid360" if (
                cfg_lidar_num_bins is not None
                or str(cfg_task_name or "").lower().endswith("mid360")
                or "GuideMid360Dataset" in str(cfg_dataset_target or "")
            ) else "lowdim"
            self.observation_mode = detected_observation_mode
            self.pointcloud_mode = self.requested_pointcloud_mode
            if self.pointcloud_mode == "auto":
                self.pointcloud_mode = (
                    "vector_map" if self.observation_mode == "mid360" else "off"
                )
            if self.observation_mode == "mid360" and self.pointcloud_mode == "off":
                raise ValueError(
                    "pointcloud_mode='off' is incompatible with a guide_mid360 checkpoint. "
                    "Use pointcloud_mode='auto', 'vector_map', or 'gazebo'."
                )

            self.obs_dim = int(self.policy.obs_dim)
            self.action_dim = int(self.policy.action_dim)
            self.n_obs_steps = int(self.policy.n_obs_steps)
            self.n_action_steps = int(self.policy.n_action_steps)
            self.lidar_num_bins = int(cfg_lidar_num_bins or 128)
            self.mid360_obs_config = GuideMid360ObservationConfig(
                num_bins=self.lidar_num_bins,
                min_angle=float(_cfg_dataset_value("lidar_min_angle", -np.pi)),
                max_angle=float(_cfg_dataset_value("lidar_max_angle", np.pi)),
                min_range=float(_cfg_dataset_value("lidar_min_range", 0.2)),
                max_range=float(_cfg_dataset_value("lidar_max_range", 8.0)),
                ground_height=float(_cfg_dataset_value("lidar_ground_height", 0.1)),
                max_height=float(_cfg_dataset_value("lidar_max_height", 2.2)),
                use_world_height=bool(_cfg_dataset_value("lidar_use_world_height", True)),
            )
            self.mid360_simulator = VectorMapPointCloudSimulator(
                VectorMapPointCloudConfig(
                    num_rays=self.lidar_num_bins,
                    min_angle=self.mid360_obs_config.min_angle,
                    max_angle=self.mid360_obs_config.max_angle,
                    min_range=self.mid360_obs_config.min_range,
                    max_range=self.mid360_obs_config.max_range,
                    z_height=1.0,
                )
            )
            if self.observation_mode == "mid360":
                self.n_obstacle_circles = 0
                self.n_obstacle_segments = 0
                self.obstacle_include_radius = False
                self.obstacle_include_human_clearance = False
                self.segment_repr = "closest_dir"
                self.obstacle_obs_dim = 0
                if cfg_n_lookahead is None:
                    extra_obs = self.obs_dim - 4 - self.lidar_num_bins
                    if extra_obs < 0 or extra_obs % 2 != 0:
                        raise ValueError(
                            f"Unsupported obs_dim={self.obs_dim}, expected 4 + 2*N + {self.lidar_num_bins}"
                        )
                    self.n_lookahead = extra_obs // 2
                else:
                    self.n_lookahead = int(cfg_n_lookahead)
                    expected_obs_dim = 4 + 2 * self.n_lookahead + self.lidar_num_bins
                    if self.obs_dim != expected_obs_dim:
                        extra_obs = self.obs_dim - 4 - self.lidar_num_bins
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
            else:
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
            self.observation_mode = "lowdim"
            self.pointcloud_mode = (
                "off" if self.requested_pointcloud_mode == "auto" else self.requested_pointcloud_mode
            )
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
            self.lidar_num_bins = 128
            self.mid360_obs_config = GuideMid360ObservationConfig(
                num_bins=self.lidar_num_bins,
                min_angle=-np.pi,
                max_angle=np.pi,
                min_range=0.2,
                max_range=8.0,
                ground_height=0.1,
                max_height=2.2,
                use_world_height=True,
            )
            self.mid360_simulator = VectorMapPointCloudSimulator(
                VectorMapPointCloudConfig(
                    num_rays=self.lidar_num_bins,
                    min_angle=self.mid360_obs_config.min_angle,
                    max_angle=self.mid360_obs_config.max_angle,
                    min_range=self.mid360_obs_config.min_range,
                    max_range=self.mid360_obs_config.max_range,
                    z_height=1.0,
                )
            )
            self.n_lookahead = 20
            self.lookahead_stride = k_lookahead if k_lookahead is not None else 5
            self.frame_stride = frame_stride if frame_stride is not None else 1
        self.data_dt = self.sim_dt * self.frame_stride
        self.physics = PhysicsEngine(
            leash_length=leash_length,
            robot_speed=robot_speed,
            dt=self.sim_dt,
            robot_radius=robot_radius,
            human_radius=human_radius,
        )

        self.scorer = None
        self.current_path_data = None
        self.running = True
        self.paused = False
        self.bre = False
        self.bre_toggle_times = (4, 5, 9, 10, 14, 15, 19, 20, 24, 25, 29, 30)  # Seconds at which to toggle BRE on/off for testing
        self.bre_toggle_times = (5, 10, 15, 20, 25, 30)  # Seconds at which to toggle BRE on/off for testing
        self.triggered_bre_toggle_times = set()
        self.bre_timer_start_frame = None
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
        self.nominal_planned_path = None
        self.safe_planned_path = None
        self.lookahead_world = None
        self.current_mid360_points_world = None
        self.frame_count = 0
        self.prev_robot_pos = None
        self.data_step_idx = 0
        self.log_fp = None
        self.eval_fp = None
        self.eval_planning_idx = 0
        self.log_interval = max(1, int(log_interval))
        self.collect_enabled = bool(collect_enabled)
        self.recording = False
        self._last_recorded_cloud_seq = None
        self.storage = None
        self.collection_data_dir = FOLLOWDATASET_DIR / "data"
        if self.collect_enabled:
            storage_cls = Mid360DataStorage if self.pointcloud_mode == "gazebo" else DataStorage
            self.storage = storage_cls(base_dir=str(self.collection_data_dir))
            print(
                f"Planning collection enabled: {storage_cls.__name__} "
                f"-> {self.collection_data_dir}"
            )

        self.obs_history = deque(maxlen=self.n_obs_steps)
        
        # Performance optimization: cache actions and reduce inference frequency
        self.cached_action_seq = None
        self.cached_nominal_delta_seq = None
        self.cached_safe_delta_seq = None
        self.cached_safety_info_seq = None
        self.cached_action_idx = 0
        self.inference_interval = max(1, self.n_action_steps // 2)  # Infer every N frames
        self.frames_since_inference = 0
        self.cached_control = (0.0, 0.0)
        self.current_action = None
        self.current_delta = None
        self.latest_nominal_heading_delta = None
        self.turn_gain = float(turn_gain)
        self.curvature_slowdown = bool(curvature_slowdown)
        self.curvature_scale = float(curvature_scale)
        self.min_speed_scale = float(min_speed_scale)
        self.current_speed_scale = 1.0
        self.safety_mode = normalize_safety_mode(safety_mode)
        self.safety_filter = QPSafetyFilter(
            margin=float(safety_margin),
            alpha=float(safety_alpha),
            max_constraints=int(safety_max_constraints),
            influence_distance=float(safety_influence_distance),
        )
        self.safety_stop_clearance = max(0.0, float(safety_margin) * 0.75)
        self.safety_backoff_scales = (1.0, 0.75, 0.5, 0.25, 0.0)
        self.last_safety_stats = {
            "applied": False,
            "modified_steps": 0,
            "total_steps": 0,
            "mean_shift": 0.0,
            "constraint_count": 0,
            "min_clearance": float("inf"),
        }
        self.last_compliance_stats = {
            "applied": False,
            "safety_applied": False,
            "modified_steps": 0,
            "safety_modified_steps": 0,
            "total_steps": 0,
            "mean_shift": 0.0,
            "mean_action_shift": 0.0,
            "constraint_count": 0,
            "min_clearance": float("inf"),
        }
        self.episode_safety_stats = {
            "modified_steps": 0,
            "total_steps": 0,
            "total_shift": 0.0,
            "constraint_count": 0,
            "min_clearance": float("inf"),
        }
        if self.safety_mode != "off" and self.action_mode not in ("forward_heading", "delta", "velocity"):
            print(
                f"[warn] safety_mode={self.safety_mode} is only supported for "
                f"forward_heading/delta/velocity; disabling for action_mode={self.action_mode}"
            )
            self.safety_mode = "off"
        self.debug_preview = bool(debug_preview)
        self.debug_preview_limit = max(1, int(debug_preview_limit))
        self.debug_policy = bool(debug_policy)
        self.debug_qp_log = bool(debug_qp_log)
        self.debug_inference_count = 0

        effective_inference_steps = int(inference_steps)
        if self.observation_mode == "mid360":
            effective_inference_steps = max(effective_inference_steps, 100)
        self.inference_steps = effective_inference_steps
        
        # Reduce inference steps for faster performance
        if self.policy is not None and hasattr(self.policy, 'num_inference_steps'):
            original_steps = self.policy.num_inference_steps
            self.policy.num_inference_steps = self.inference_steps
            if original_steps != self.inference_steps:
                print(
                    f"Set inference steps to {self.inference_steps} "
                    f"(original: {original_steps})"
                )

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
                    "corridor_width": float(self.path_generator.corridor_width),
                    "obstacle_radius": float(self.path_generator.obstacle_radius),
                    "robot_radius": float(self.physics.robot_radius),
                    "human_radius": float(self.physics.human_radius),
                    "action_mode": self.action_mode,
                    "robot_frame": bool(self.robot_frame),
                    "robot_state": self.robot_state,
                    "observation_mode": self.observation_mode,
                    "pointcloud_mode": self.pointcloud_mode,
                    "n_lookahead": int(self.n_lookahead),
                    "lookahead_stride": int(self.lookahead_stride),
                    "lidar_num_bins": int(self.lidar_num_bins),
                    "n_obstacle_circles": int(self.n_obstacle_circles),
                    "n_obstacle_segments": int(self.n_obstacle_segments),
                    "obstacle_obs_dim": int(self.obstacle_obs_dim),
                    "obstacle_include_radius": bool(self.obstacle_include_radius),
                    "obstacle_include_human_clearance": bool(
                        getattr(self, "obstacle_include_human_clearance", False)
                    ),
                    "segment_repr": self.segment_repr,
                    "inference_interval": int(self.inference_interval),
                    "inference_steps": int(self.inference_steps),
                    "turn_gain": float(self.turn_gain),
                    "curvature_slowdown": bool(self.curvature_slowdown),
                    "curvature_scale": float(self.curvature_scale),
                    "min_speed_scale": float(self.min_speed_scale),
                    "safety_mode": self.safety_mode,
                    "safety_margin": float(self.safety_filter.margin),
                    "safety_alpha": float(self.safety_filter.alpha),
                    "safety_max_constraints": int(self.safety_filter.max_constraints),
                    "safety_influence_distance": float(self.safety_filter.influence_distance),
                    "debug_qp_log": bool(self.debug_qp_log),
                },
            )

        if eval_path is not None:
            eval_path = Path(eval_path)
            eval_path.parent.mkdir(parents=True, exist_ok=True)
            self.eval_fp = eval_path.open("w", encoding="utf-8")
            print(f"Planning eval log: {eval_path}")

        self._generate_new_path()

    def set_path_data(self, path_data: dict, reset: bool = True):
        """Override the current episode path/obstacles with externally-provided data."""
        self.current_path_data = copy.deepcopy(path_data)
        self._precompute_frenet_cache()
        self._restart_mid360_gazebo_session_if_needed()
        if reset:
            self._reset_position()

    def _generate_new_path(self):
        """Generate new reference path."""
        self.current_path_data = self.path_generator.generate()
        self._precompute_frenet_cache()
        self._restart_mid360_gazebo_session_if_needed()
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
                "start": self._rounded_list(self.current_path_data.get("start")),
                "end": self._rounded_list(self.current_path_data.get("end")),
                "obstacles": self._serialize_obstacles(obstacles) if self.debug_qp_log else None,
                "segment_obstacles": self._serialize_segments(segments) if self.debug_qp_log else None,
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

        self.paused = False
        self.collision_pause = False
        self.bre = False
        self.triggered_bre_toggle_times.clear()
        self.bre_timer_start_frame = None
        self.physics.reset(start)
        if self.recording:
            self._stop_recording()
        self._last_mid360_cloud_seq = None
        self._sync_mid360_gazebo_session()
        self.robot_trajectory = []
        self.human_trajectory = []
        self.planned_path = None
        self.nominal_planned_path = None
        self.safe_planned_path = None
        self.current_mid360_points_world = None
        self.frame_count = 0
        self.prev_robot_pos = None
        self._seed_obs_history(self.physics.robot.position, self.physics.human.position)
        if self.policy is not None:
            self.policy.reset()
        # Reset action cache
        self._reset_runtime_caches()
        self.data_step_idx = 0
        self.episode_safety_stats = {
            "modified_steps": 0,
            "total_steps": 0,
            "total_shift": 0.0,
            "constraint_count": 0,
            "min_clearance": float("inf"),
        }
        self._log_event("reset_position", {"robot_pos": self.physics.robot.position.tolist()})

    def _update_timed_bre_toggle(self):
        if self.bre_timer_start_frame is None:
            return
        sim_time = float((self.frame_count - self.bre_timer_start_frame) * self.sim_dt)
        for toggle_time in self.bre_toggle_times:
            if toggle_time not in self.triggered_bre_toggle_times and sim_time >= toggle_time:
                self.bre = not self.bre
                self._reset_runtime_caches()
                self.triggered_bre_toggle_times.add(toggle_time)
                print(f"Timed bre toggle at {toggle_time:.1f}s: bre={self.bre}")
                self._log_event(
                    "timed_bre_toggle",
                    {"toggle_time": float(toggle_time), "bre": bool(self.bre)},
                )

    def _seed_obs_history(self, robot_pos: np.ndarray, human_pos: np.ndarray):
        obs = self._build_obs(robot_pos, human_pos, self.physics.robot.heading)
        self.obs_history.clear()
        for _ in range(self.n_obs_steps):
            self.obs_history.append(obs.copy())
        self.prev_robot_pos = robot_pos.copy()

    def _reset_runtime_caches(self):
        self.cached_action_seq = None
        self.cached_nominal_delta_seq = None
        self.cached_safe_delta_seq = None
        self.cached_safety_info_seq = None
        self.cached_action_idx = 0
        self.frames_since_inference = 0
        self.cached_control = (0.0, 0.0)
        self.current_action = None
        self.current_delta = None
        self.latest_nominal_heading_delta = None
        self.current_speed_scale = 1.0
        self.planned_path = None
        self.nominal_planned_path = None
        self.safe_planned_path = None
        self.last_compliance_stats = {
            "applied": False,
            "safety_applied": False,
            "modified_steps": 0,
            "safety_modified_steps": 0,
            "total_steps": 0,
            "mean_shift": 0.0,
            "mean_action_shift": 0.0,
            "constraint_count": 0,
            "min_clearance": float("inf"),
        }

    def _start_recording(self):
        if self.storage is None:
            return
        desired_cls = Mid360DataStorage if self.pointcloud_mode == "gazebo" else DataStorage
        if type(self.storage) is not desired_cls:
            self.storage = desired_cls(base_dir=str(self.collection_data_dir))
        self.recording = True
        self.storage.start_recording()
        if self.bre_timer_start_frame is None:
            self.bre_timer_start_frame = self.frame_count
            print("Timed bre toggle timer started.")
        self._last_recorded_cloud_seq = None
        if self.scorer:
            self.scorer.reset()
        print("Recording started...")

    def _stop_recording(self):
        if self.storage is None:
            return
        self.recording = False
        print(f"Recording paused. Points: {self.storage.get_num_points()}")

    def _record_frame(self, robot_state, human_state):
        if self.storage is None:
            return
        timestamp = float(self.frame_count * self.sim_dt)
        state = 0 if self.bre else 2
        if isinstance(self.storage, Mid360DataStorage):
            if self.mid360_session is None:
                raise RuntimeError("Mid360 Gazebo session is not running.")
            cloud = self.mid360_session.get_pointcloud(
                after_seq=self._last_recorded_cloud_seq,
                wait_timeout=max(self.sim_dt * 1.2, 0.15),
            )
            if cloud is None:
                cloud = {
                    "seq": self._last_recorded_cloud_seq or 0,
                    "stamp": 0.0,
                    "fields": [],
                    "points": np.zeros((0, 0), dtype=np.float32),
                }
            self._last_recorded_cloud_seq = int(cloud.get("seq", 0))
            self.storage.record_frame(
                robot_state.position,
                human_state.position,
                timestamp=timestamp,
                state=state,
                robot_base_pose=self.mid360_session.get_robot_base_pose(robot_state),
                human_base_pose=self.mid360_session.get_human_base_pose(human_state),
                mid360_pose=self.mid360_session.get_mid360_pose(robot_state),
                point_cloud=np.asarray(cloud.get("points", np.zeros((0, 0), dtype=np.float32))),
                point_cloud_timestamp=float(cloud.get("stamp", 0.0)),
                point_cloud_fields=list(cloud.get("fields", [])),
            )
        else:
            self.storage.record_frame(
                robot_state.position,
                human_state.position,
                timestamp=timestamp,
                state=state,
            )

    def _save_episode(self):
        if self.storage is None:
            return
        if self.storage.get_num_points() == 0:
            print("No data to save!")
            return
        scores = self.scorer.get_scores() if self.scorer else {}
        extra_metadata = {"scores": scores, "source": "planning"}
        if self.mid360_session is not None:
            extra_metadata.update(self.mid360_session.metadata())
        episode_dir = self.storage.save_episode(
            reference_path=self.current_path_data["path"],
            start_pos=self.current_path_data["start"],
            end_pos=self.current_path_data["end"],
            obstacles=self.current_path_data.get("obstacles"),
            segment_obstacles=self.current_path_data.get("segment_obstacles"),
            extra_metadata=extra_metadata,
        )
        print(f"Saved planning episode: {episode_dir}")
        self.recording = False
        self.storage.clear()

    def _supported_pointcloud_modes(self) -> list[str]:
        if self.observation_mode == "mid360":
            return ["vector_map", "gazebo"]
        return ["off", "vector_map", "gazebo"]

    def _resolve_mid360_gazebo_config(self) -> Mid360GazeboConfig:
        if self.mid360_gazebo_config is None:
            plugin_dir = resolve_mid360_plugin_dir(self.mid360_plugin_dir_arg)
            plugin_library = resolve_mid360_plugin_library(plugin_dir, self.mid360_plugin_lib_arg)
            self.mid360_gazebo_config = Mid360GazeboConfig(
                plugin_dir=plugin_dir,
                plugin_library_path=plugin_library,
                downsample=self.mid360_downsample,
                update_rate=float(max(1, self.fps)),
                gui=self.mid360_gazebo_gui,
                visualize_laser=self.mid360_visualize,
            )
        return self.mid360_gazebo_config

    def _close_mid360_gazebo_session(self):
        if self.mid360_session is not None:
            try:
                self.mid360_session.close()
            except Exception as exc:
                print(f"[warn] failed to close Mid360 Gazebo session cleanly: {exc}")
            self.mid360_session = None
        self._last_mid360_cloud_seq = None

    def _ensure_mid360_gazebo_session(self):
        if self.mid360_session is not None:
            return
        if self.current_path_data is None:
            return
        config = self._resolve_mid360_gazebo_config()
        self.mid360_session = Mid360GazeboSession(self.current_path_data, config)
        try:
            self.mid360_session.start()
        except Exception:
            self._close_mid360_gazebo_session()
            raise
        self._last_mid360_cloud_seq = None
        self._sync_mid360_gazebo_session()
        print(f"Mid360 Gazebo runtime: {self.mid360_session.runtime_dir}")

    def _restart_mid360_gazebo_session_if_needed(self):
        if self.pointcloud_mode != "gazebo":
            self._close_mid360_gazebo_session()
            return
        self._close_mid360_gazebo_session()
        self._ensure_mid360_gazebo_session()

    def _sync_mid360_gazebo_session(self):
        if self.mid360_session is None:
            return
        self.mid360_session.update_entities(self.physics.robot, self.physics.human)

    def _mid360_pose_from_base(self, robot_pos: np.ndarray, heading: float) -> np.ndarray:
        world_from_base = np.eye(4, dtype=np.float64)
        world_from_base[:3, :3] = Rotation.from_euler("z", float(heading), degrees=False).as_matrix()
        robot_base_z = 0.155
        if self.mid360_gazebo_config is not None:
            robot_base_z = float(self.mid360_gazebo_config.robot_base_z)
        world_from_base[:3, 3] = np.array(
            [float(robot_pos[0]), float(robot_pos[1]), float(robot_base_z)],
            dtype=np.float64,
        )
        world_from_mid360 = world_from_base @ T_BASE_MID360
        quat = Rotation.from_matrix(world_from_mid360[:3, :3]).as_quat()
        return np.array(
            [
                world_from_mid360[0, 3],
                world_from_mid360[1, 3],
                world_from_mid360[2, 3],
                quat[0],
                quat[1],
                quat[2],
                quat[3],
            ],
            dtype=np.float64,
        )

    def _get_mid360_gazebo_cloud(
        self,
        robot_pos: np.ndarray,
        heading: float,
        *,
        wait_timeout: Optional[float] = None,
    ) -> tuple[np.ndarray, list[str], np.ndarray] | None:
        self._ensure_mid360_gazebo_session()
        if self.mid360_session is None:
            return None
        timeout = max(float(self.data_dt) * 1.2, 0.15) if wait_timeout is None else max(float(wait_timeout), 0.0)
        cloud = self.mid360_session.get_pointcloud(
            after_seq=self._last_mid360_cloud_seq,
            wait_timeout=timeout,
        )
        if cloud is None:
            cloud = self.mid360_session.get_pointcloud(wait_timeout=0.0)
        if cloud is None:
            return None
        self._last_mid360_cloud_seq = int(cloud.get("seq", 0))
        frame = np.asarray(cloud.get("points", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32)
        if frame.ndim == 1:
            frame = frame.reshape(1, -1)
        field_names = list(cloud.get("fields", []))
        mid360_pose = self._mid360_pose_from_base(robot_pos, heading)
        return frame, field_names, mid360_pose

    def _update_mid360_pointcloud_from_gazebo(
        self,
        robot_pos: np.ndarray,
        heading: float,
        cloud: tuple[np.ndarray, list[str], np.ndarray] | None = None,
    ) -> np.ndarray:
        if self.mid360_obs_config is None:
            self.current_mid360_points_world = None
            return np.zeros((0, 2), dtype=np.float32)
        if cloud is None:
            cloud = self._get_mid360_gazebo_cloud(robot_pos, heading)
        if cloud is None:
            self.current_mid360_points_world = None
            return np.zeros((0, 2), dtype=np.float32)

        frame, field_names, mid360_pose = cloud
        field_indices = {str(name): idx for idx, name in enumerate(field_names)}
        if "x" not in field_indices or "y" not in field_indices:
            self.current_mid360_points_world = None
            return np.zeros((0, 2), dtype=np.float32)

        ix = field_indices["x"]
        iy = field_indices["y"]
        iz = field_indices.get("z")
        local_xy = frame[:, [ix, iy]].astype(np.float32, copy=False) if len(frame) > 0 else np.zeros((0, 2), dtype=np.float32)
        if len(local_xy) == 0:
            self.current_mid360_points_world = None
            return local_xy

        ranges = np.linalg.norm(local_xy, axis=-1).astype(np.float32, copy=False)
        azimuth = np.arctan2(local_xy[:, 1], local_xy[:, 0]).astype(np.float32, copy=False)
        valid = np.isfinite(local_xy[:, 0]) & np.isfinite(local_xy[:, 1])
        valid &= np.isfinite(ranges) & np.isfinite(azimuth)
        valid &= ranges >= float(self.mid360_obs_config.min_range)
        valid &= ranges <= float(self.mid360_obs_config.max_range)
        angle_width = float(self.mid360_obs_config.max_angle - self.mid360_obs_config.min_angle)
        if angle_width < (2.0 * np.pi - 1e-6):
            valid &= azimuth >= float(self.mid360_obs_config.min_angle)
            valid &= azimuth < float(self.mid360_obs_config.max_angle)

        local_xyz = np.zeros((len(frame), 3), dtype=np.float32)
        local_xyz[:, 0] = frame[:, ix].astype(np.float32, copy=False)
        local_xyz[:, 1] = frame[:, iy].astype(np.float32, copy=False)
        if iz is not None and frame.shape[1] > iz:
            local_xyz[:, 2] = frame[:, iz].astype(np.float32, copy=False)

        if iz is not None and frame.shape[1] > iz:
            if self.mid360_obs_config.use_world_height:
                rot = Rotation.from_quat(mid360_pose[3:7]).as_matrix().astype(np.float32)
                height = local_xyz @ rot[2, :].astype(np.float32) + np.float32(mid360_pose[2])
            else:
                height = local_xyz[:, 2]
            valid &= np.isfinite(height)
            valid &= height >= float(self.mid360_obs_config.ground_height)
            valid &= height <= float(self.mid360_obs_config.max_height)

        if not np.any(valid):
            self.current_mid360_points_world = None
            return np.zeros((0, 2), dtype=np.float32)

        rot = Rotation.from_quat(mid360_pose[3:7]).as_matrix().astype(np.float32)
        world_xyz = local_xyz @ rot.T + np.asarray(mid360_pose[:3], dtype=np.float32)
        world_xy = world_xyz[valid, :2].astype(np.float32, copy=False)
        if len(world_xy) > self._mid360_visual_max_points:
            stride = max(1, len(world_xy) // self._mid360_visual_max_points)
            world_xy = world_xy[::stride]
        self.current_mid360_points_world = world_xy if len(world_xy) > 0 else None
        return local_xy[valid]

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

    def _synchronize_timing_device(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _run_eval_safety_variant(
        self,
        action_seq: np.ndarray,
        safety_mode: str,
    ) -> tuple[np.ndarray, list[dict], dict, float]:
        saved_mode = self.safety_mode
        saved_stats = copy.deepcopy(self.last_safety_stats)
        saved_rng_state = np.random.get_state()
        try:
            self.safety_mode = normalize_safety_mode(safety_mode)
            start = time.perf_counter()
            if self.action_mode == "forward_heading":
                _nominal_deltas, safe_deltas, safety_infos = (
                    self._apply_forward_heading_safety_filter(action_seq)
                )
            else:
                safe_actions = self._apply_safety_filter(action_seq)
                safe_deltas = self._action_seq_to_nominal_delta_seq(safe_actions)
                safety_infos = []
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            stats = copy.deepcopy(self.last_safety_stats)
        finally:
            self.safety_mode = saved_mode
            self.last_safety_stats = saved_stats
            np.random.set_state(saved_rng_state)
        return safe_deltas, safety_infos, stats, elapsed_ms

    def _eval_paths_from_deltas(
        self,
        delta_seq: np.ndarray,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        saved_rng_state = np.random.get_state()
        try:
            sim = copy.deepcopy(self.physics)
            robot_points: list[np.ndarray] = []
            human_points: list[np.ndarray] = []
            for delta in np.asarray(delta_seq, dtype=np.float32):
                forward, turn, _speed_scale = self._delta_to_safe_control(
                    delta,
                    sim.robot.heading,
                    dt=self.data_dt,
                )
                sim.set_control(forward, turn, self.bre)
                for _ in range(int(self.frame_stride)):
                    robot_state, human_state = sim.step()
                    robot_points.append(robot_state.position.copy())
                    human_points.append(human_state.position.copy())
            robot_path = np.stack(robot_points, axis=0) if robot_points else None
            human_path = np.stack(human_points, axis=0) if human_points else None
            return robot_path, human_path
        finally:
            np.random.set_state(saved_rng_state)

    def _serialize_eval_safety_steps(self, safety_infos: list[dict]) -> list[dict]:
        rows = []
        for info in safety_infos:
            min_clearance = float(info.get("min_clearance", float("inf")))
            rows.append(
                {
                    "modified": bool(info.get("modified", False)),
                    "shift": round(float(info.get("shift", 0.0)), 6),
                    "constraint_count": int(info.get("constraint_count", 0)),
                    "min_clearance": (
                        round(min_clearance, 6) if np.isfinite(min_clearance) else "inf"
                    ),
                    "backoff_applied": bool(info.get("backoff_applied", False)),
                    "resolution_stage": str(info.get("resolution_stage", "none")),
                }
            )
        return rows

    def _serialize_eval_safety_stats(self, stats: dict) -> dict:
        min_clearance = float(stats.get("min_clearance", float("inf")))
        return {
            "modified_steps": int(stats.get("modified_steps", 0)),
            "total_steps": int(stats.get("total_steps", 0)),
            "mean_shift": round(float(stats.get("mean_shift", 0.0)), 6),
            "constraint_count": int(stats.get("constraint_count", 0)),
            "min_clearance": (
                round(min_clearance, 6) if np.isfinite(min_clearance) else "inf"
            ),
        }

    def _write_planning_eval(self, action_seq: np.ndarray, diffusion_time_ms: float):
        if self.eval_fp is None:
            return

        action_seq = np.asarray(action_seq, dtype=np.float32)
        saved_rng_state = np.random.get_state()
        try:
            origin_deltas = self._action_seq_to_nominal_delta_seq(action_seq)
        finally:
            np.random.set_state(saved_rng_state)
        origin_robot_path, origin_human_path = self._eval_paths_from_deltas(origin_deltas)

        robot_deltas, robot_infos, robot_stats, robot_qp_time_ms = (
            self._run_eval_safety_variant(action_seq, "robot_qp")
        )
        robot_safe_robot_path, robot_safe_human_path = self._eval_paths_from_deltas(robot_deltas)

        (
            human_robot_deltas,
            human_robot_infos,
            human_robot_stats,
            human_robot_qp_time_ms,
        ) = self._run_eval_safety_variant(action_seq, "human_robot_qp")
        human_robot_safe_robot_path, human_robot_safe_human_path = (
            self._eval_paths_from_deltas(human_robot_deltas)
        )

        diffusion_time_ms = float(diffusion_time_ms)
        obstacles = self.current_path_data.get("obstacles") if self.current_path_data else None
        segments = self.current_path_data.get("segment_obstacles") if self.current_path_data else None
        payload = {
            "event": "planning_eval",
            "planning_index": int(self.eval_planning_idx),
            "frame": int(self.frame_count),
            "time_sec": float(self.frame_count * self.sim_dt),
            "data_step": int(self.data_step_idx),
            "data_time_sec": float(self.data_step_idx * self.data_dt),
            "action_mode": self.action_mode,
            "executed_safety_mode": self.safety_mode,
            "robot_pos": self._rounded_list(self.physics.robot.position, decimals=6),
            "human_pos": self._rounded_list(self.physics.human.position, decimals=6),
            "robot_heading": round(float(self.physics.robot.heading), 6),
            "robot_radius": round(float(self.physics.robot_radius), 6),
            "human_radius": round(float(self.physics.human_radius), 6),
            "obstacles": self._serialize_obstacles(obstacles, decimals=6),
            "segment_obstacles": self._serialize_segments(segments, decimals=6),
            "origin_planning": {
                "diffusion_time_ms": round(diffusion_time_ms, 3),
                "qp_time_ms": 0.0,
                "total_time_ms": round(diffusion_time_ms, 3),
                "action_seq": self._rounded_list(action_seq, decimals=6),
                "delta_seq": self._rounded_list(origin_deltas, decimals=6),
                "path": self._rounded_list(origin_robot_path, decimals=6),
                "robot_path": self._rounded_list(origin_robot_path, decimals=6),
                "human_path": self._rounded_list(origin_human_path, decimals=6),
            },
            "robot_safe_planning": {
                "diffusion_time_ms": round(diffusion_time_ms, 3),
                "qp_time_ms": round(robot_qp_time_ms, 3),
                "total_time_ms": round(diffusion_time_ms + robot_qp_time_ms, 3),
                "delta_seq": self._rounded_list(robot_deltas, decimals=6),
                "path": self._rounded_list(robot_safe_robot_path, decimals=6),
                "robot_path": self._rounded_list(robot_safe_robot_path, decimals=6),
                "human_path": self._rounded_list(robot_safe_human_path, decimals=6),
                "safety_summary": self._serialize_eval_safety_stats(robot_stats),
                "qp_steps": self._serialize_eval_safety_steps(robot_infos),
            },
            "human_robot_safe_planning": {
                "diffusion_time_ms": round(diffusion_time_ms, 3),
                "qp_time_ms": round(human_robot_qp_time_ms, 3),
                "total_time_ms": round(diffusion_time_ms + human_robot_qp_time_ms, 3),
                "delta_seq": self._rounded_list(human_robot_deltas, decimals=6),
                "path": self._rounded_list(human_robot_safe_robot_path, decimals=6),
                "robot_path": self._rounded_list(human_robot_safe_robot_path, decimals=6),
                "human_path": self._rounded_list(human_robot_safe_human_path, decimals=6),
                "safety_summary": self._serialize_eval_safety_stats(human_robot_stats),
                "qp_steps": self._serialize_eval_safety_steps(human_robot_infos),
            },
        }
        self.eval_fp.write(json.dumps(payload, ensure_ascii=True) + "\n")
        self.eval_fp.flush()
        self.eval_planning_idx += 1

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

    def _rounded_list(self, values: Optional[np.ndarray], decimals: int = 4) -> Optional[list]:
        if values is None:
            return None
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0:
            return []
        return np.round(arr, decimals).tolist()

    def _serialize_obstacles(self, obstacles: Optional[np.ndarray], decimals: int = 4) -> list[dict]:
        if obstacles is None:
            return []
        rows: list[dict] = []
        for idx, obs in enumerate(obstacles):
            if isinstance(obs, dict):
                x = float(obs.get("x", 0.0))
                y = float(obs.get("y", 0.0))
                r = float(obs.get("r", 0.0))
            else:
                x = float(obs[0])
                y = float(obs[1])
                r = float(obs[2])
            rows.append(
                {
                    "idx": int(idx),
                    "x": round(x, decimals),
                    "y": round(y, decimals),
                    "r": round(r, decimals),
                }
            )
        return rows

    def _serialize_segments(self, segments: Optional[np.ndarray], decimals: int = 4) -> list[dict]:
        if segments is None:
            return []
        rows: list[dict] = []
        for idx, seg in enumerate(segments):
            if isinstance(seg, dict):
                if "p1" in seg and "p2" in seg:
                    p1 = np.asarray(seg["p1"], dtype=np.float32)
                    p2 = np.asarray(seg["p2"], dtype=np.float32)
                    x1, y1 = float(p1[0]), float(p1[1])
                    x2, y2 = float(p2[0]), float(p2[1])
                else:
                    x1 = float(seg.get("x1", 0.0))
                    y1 = float(seg.get("y1", 0.0))
                    x2 = float(seg.get("x2", 0.0))
                    y2 = float(seg.get("y2", 0.0))
            else:
                x1 = float(seg[0])
                y1 = float(seg[1])
                x2 = float(seg[2])
                y2 = float(seg[3])
            rows.append(
                {
                    "idx": int(idx),
                    "x1": round(x1, decimals),
                    "y1": round(y1, decimals),
                    "x2": round(x2, decimals),
                    "y2": round(y2, decimals),
                }
            )
        return rows

    def _serialize_safety_info(self, safety_info: Optional[dict], decimals: int = 4) -> dict:
        if safety_info is None:
            return {}

        payload = {
            "modified": bool(safety_info.get("modified", False)),
            "shift": round(float(safety_info.get("shift", 0.0)), decimals),
            "constraint_count": int(safety_info.get("constraint_count", 0)),
            "min_clearance": round(float(safety_info.get("min_clearance", float("inf"))), decimals)
            if np.isfinite(float(safety_info.get("min_clearance", float("inf"))))
            else "inf",
        }

        scalar_keys = [
            "nominal_delta_norm",
            "qp_delta_norm",
            "final_delta_norm",
            "robot_heading",
            "qp_total_constraint_count",
            "qp_candidate_count",
            "backoff_scale",
            "stop_clearance_threshold",
        ]
        for key in scalar_keys:
            if key in safety_info and safety_info[key] is not None:
                payload[key] = round(float(safety_info[key]), decimals)

        int_keys = ["qp_constraint_count"]
        for key in int_keys:
            if key in safety_info and safety_info[key] is not None:
                payload[key] = int(safety_info[key])

        bool_keys = [
            "protect_human",
            "qp_modified",
            "qp_ref_feasible",
            "collision_after_qp",
            "backoff_applied",
            "stop_triggered",
        ]
        for key in bool_keys:
            if key in safety_info and safety_info[key] is not None:
                payload[key] = bool(safety_info[key])

        list_keys = [
            "robot_pos",
            "human_pos",
            "nominal_delta",
            "qp_delta",
            "final_delta",
            "nominal_preview_robot_pos",
            "nominal_preview_human_pos",
        ]
        for key in list_keys:
            if key in safety_info:
                payload[key] = self._rounded_list(safety_info.get(key), decimals=decimals)

        text_keys = [
            "resolution_stage",
            "stop_reason",
            "qp_best_candidate_kind",
        ]
        for key in text_keys:
            if key in safety_info and safety_info[key] is not None:
                payload[key] = str(safety_info[key])

        if "qp_best_candidate_constraints" in safety_info:
            payload["qp_best_candidate_constraints"] = [
                int(v) for v in safety_info.get("qp_best_candidate_constraints", [])
            ]

        if "backoff_attempts" in safety_info:
            payload["backoff_attempts"] = [
                {
                    "scale": round(float(item.get("scale", 0.0)), decimals),
                    "collided": bool(item.get("collided", False)),
                    "delta": self._rounded_list(item.get("delta"), decimals=decimals),
                }
                for item in safety_info.get("backoff_attempts", [])
            ]

        if "qp_selected_constraints" in safety_info:
            payload["qp_selected_constraints"] = [
                {
                    "index": int(item.get("index", 0)),
                    "source": str(item.get("source", "")),
                    "clearance": round(float(item.get("clearance", 0.0)), decimals),
                    "predicted_clearance": round(
                        float(item.get("predicted_clearance", 0.0)), decimals
                    ),
                    "h": round(float(item.get("h", 0.0)), decimals),
                    "g": self._rounded_list(item.get("g"), decimals=decimals),
                    "ref_violation": round(float(item.get("ref_violation", 0.0)), decimals),
                }
                for item in safety_info.get("qp_selected_constraints", [])
            ]
        return payload

    def _log_robot_qp_step(
        self,
        action_idx: int,
        action: Optional[np.ndarray],
        delta: Optional[np.ndarray],
        forward: float,
        turn: float,
        speed_scale: float,
        safety_info: Optional[dict],
    ):
        if (
            not self.debug_qp_log
            or self.log_fp is None
            or self.safety_mode != "robot_qp"
            or self.action_mode != "forward_heading"
        ):
            return
        payload = {
            "action_idx": int(action_idx),
            "cached_action_len": int(len(self.cached_action_seq)) if self.cached_action_seq is not None else 0,
            "cached_safe_len": int(len(self.cached_safe_delta_seq)) if self.cached_safe_delta_seq is not None else 0,
            "robot_pos": self._rounded_list(self.physics.robot.position),
            "human_pos": self._rounded_list(self.physics.human.position),
            "robot_heading": round(float(self.physics.robot.heading), 4),
            "action": self._rounded_list(action),
            "executed_delta": self._rounded_list(delta),
            "forward_cmd": round(float(forward), 4),
            "turn_cmd": round(float(turn), 4),
            "speed_scale": round(float(speed_scale), 4),
            "safety": self._serialize_safety_info(safety_info),
        }
        self._log_event("robot_qp_step", payload)

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
                    if self.collect_enabled:
                        if self.recording:
                            self._stop_recording()
                        else:
                            self._start_recording()
                    else:
                        self.paused = not self.paused
                elif event.key == pygame.K_s:
                    if self.collect_enabled:
                        self._save_episode()
                elif event.key == pygame.K_b:
                    self.bre = not self.bre
                    self._reset_runtime_caches()
                    print(f"Interaction state: {self._current_interaction_label()} (bre={self.bre})")
                    self._log_event(
                        "manual_bre_toggle",
                        {
                            "bre": bool(self.bre),
                            "interaction_label": self._current_interaction_label(),
                        },
                    )
                elif event.key == pygame.K_r:
                    self._reset_position()
                    print("Position reset")
                elif event.key == pygame.K_m:
                    self._cycle_safety_mode()
                elif event.key == pygame.K_c:
                    self._cycle_pointcloud_mode()
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

    def _path_heading_delta(self, path: Optional[np.ndarray]) -> Optional[float]:
        if path is None or len(path) < 2:
            return None
        path = np.asarray(path, dtype=np.float32)
        for idx in range(len(path) - 1, 0, -1):
            delta = path[idx] - path[idx - 1]
            if float(np.linalg.norm(delta)) > 1e-6:
                final_heading = float(np.arctan2(delta[1], delta[0]))
                return float(wrap_angle(final_heading - self.physics.robot.heading))
        return None

    def _apply_bre_compliance_control(
        self,
        action_seq: np.ndarray,
        obstacles: Optional[np.ndarray] = None,
        segment_obstacles: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if obstacles is None and self.current_path_data is not None:
            obstacles = self.current_path_data.get("obstacles")
        if segment_obstacles is None and self.current_path_data is not None:
            segment_obstacles = self.current_path_data.get("segment_obstacles")

        result = apply_bre_compliance_control(
            action_seq=action_seq,
            engine=self.physics,
            config=ComplianceControlConfig(
                data_dt=float(self.data_dt),
                sim_dt=float(self.sim_dt),
                frame_stride=int(self.frame_stride),
                turn_gain=float(self.turn_gain),
                safety_mode=self.safety_mode,
                curvature_slowdown=bool(self.curvature_slowdown),
                curvature_scale=float(self.curvature_scale),
                min_speed_scale=float(self.min_speed_scale),
                backoff_scales=tuple(float(v) for v in self.safety_backoff_scales),
                stop_clearance=float(self.safety_stop_clearance),
            ),
            safety_filter=self.safety_filter,
            obstacles=obstacles,
            segment_obstacles=segment_obstacles,
            bre=bool(self.bre),
        )
        self.last_compliance_stats = copy.deepcopy(result.stats)
        return result.actions

    def _current_interaction_label(self) -> str:
        return "leash" if self.bre else "guide"

    def _reset_compliance_stats_for_labels(self, labels: np.ndarray) -> None:
        labels = np.asarray(labels, dtype=object).reshape(-1)
        state_counts: dict[str, int] = {}
        for label in labels:
            key = str(label).strip().lower()
            state_counts[key] = state_counts.get(key, 0) + 1
        compliance_steps = int(
            sum(count for label, count in state_counts.items() if label in ("leash", "tether"))
        )
        self.last_compliance_stats = {
            "applied": True,
            "mode": "interaction_aware",
            "current_label": self._current_interaction_label(),
            "safety_applied": False,
            "modified_steps": 0,
            "safety_modified_steps": 0,
            "total_steps": int(len(labels)),
            "compliance_steps": int(compliance_steps),
            "guide_steps": int(len(labels) - compliance_steps),
            "bre_steps": int(compliance_steps),
            "state_counts": state_counts,
            "mean_shift": 0.0,
            "mean_action_shift": 0.0,
            "constraint_count": 0,
            "min_clearance": float("inf"),
        }

    def _interaction_labels_for_action_seq(self, action_count: int) -> np.ndarray:
        """Predict guide/leash labels over the cached action horizon."""
        action_count = max(0, int(action_count))
        labels = np.full(
            (action_count,),
            self._current_interaction_label(),
            dtype=object,
        )
        if action_count == 0 or self.bre_timer_start_frame is None:
            return labels

        current_time = float((self.frame_count - self.bre_timer_start_frame) * self.sim_dt)
        upcoming_toggles = sorted(
            float(t)
            for t in self.bre_toggle_times
            if t not in self.triggered_bre_toggle_times and float(t) >= current_time
        )
        if not upcoming_toggles:
            return labels

        state = bool(self.bre)
        toggle_idx = 0
        for action_idx in range(action_count):
            action_time = current_time + float(action_idx) * float(self.data_dt)
            while (
                toggle_idx < len(upcoming_toggles)
                and upcoming_toggles[toggle_idx] <= action_time + 1e-9
            ):
                state = not state
                toggle_idx += 1
            labels[action_idx] = "leash" if state else "guide"
        return labels

    def _apply_interaction_aware_compliance_control(
        self,
        action_seq: np.ndarray,
        obstacles: Optional[np.ndarray] = None,
        segment_obstacles: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if obstacles is None and self.current_path_data is not None:
            obstacles = self.current_path_data.get("obstacles")
        if segment_obstacles is None and self.current_path_data is not None:
            segment_obstacles = self.current_path_data.get("segment_obstacles")

        labels = self._interaction_labels_for_action_seq(len(action_seq))
        compliance_mask = np.isin(labels.astype(str), np.array(["leash", "tether"], dtype=object))
        if not np.any(compliance_mask):
            self._reset_compliance_stats_for_labels(labels)
            return np.asarray(action_seq, dtype=np.float32).copy()

        result = apply_interaction_aware_compliance_control(
            action_seq=action_seq,
            interaction_labels=labels,
            engine=self.physics,
            config=ComplianceControlConfig(
                data_dt=float(self.data_dt),
                sim_dt=float(self.sim_dt),
                frame_stride=int(self.frame_stride),
                turn_gain=float(self.turn_gain),
                safety_mode=self.safety_mode,
                curvature_slowdown=bool(self.curvature_slowdown),
                curvature_scale=float(self.curvature_scale),
                min_speed_scale=float(self.min_speed_scale),
                backoff_scales=tuple(float(v) for v in self.safety_backoff_scales),
                stop_clearance=float(self.safety_stop_clearance),
            ),
            safety_filter=self.safety_filter,
            obstacles=obstacles,
            segment_obstacles=segment_obstacles,
            bre=bool(self.bre),
            bre_sequence=compliance_mask,
        )
        self.last_compliance_stats = copy.deepcopy(result.stats)
        self.last_compliance_stats["mode"] = "interaction_aware"
        self.last_compliance_stats["current_label"] = self._current_interaction_label()
        return result.actions

    def _cut_action(self, action_seq: np.ndarray) -> np.ndarray:
        """Stop before the planned robot velocity points behind the human->robot link."""
        action_seq = np.asarray(action_seq, dtype=np.float32)
        if action_seq.ndim != 2 or len(action_seq) == 0:
            return action_seq

        sim = copy.deepcopy(self.physics)
        keep_count = len(action_seq)
        for idx, action in enumerate(action_seq):
            _delta, forward, turn, _speed_scale = self._action_to_execution(
                action,
                sim.robot.position,
                sim.robot.heading,
            )
            sim.set_control(forward, turn, self.bre)
            invalid = False
            for _ in range(int(self.frame_stride)):
                sim.step()
                robot_vel = sim.robot.velocity
                link = sim.robot.position - sim.human.position
                link_angle = float(np.arctan2(link[1], link[0]))
                cos_a = float(np.cos(link_angle))
                sin_a = float(np.sin(link_angle))
                vel_local_x = cos_a * float(robot_vel[0]) + sin_a * float(robot_vel[1])
                if float(np.linalg.norm(link)) > 1e-6 and vel_local_x < 0.0:
                    invalid = True
                    break
            if invalid:
                keep_count = idx
                break

        if keep_count == len(action_seq):
            return action_seq
        if keep_count == 0:
            return np.zeros_like(action_seq[:1])
        return action_seq[:keep_count].copy()

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
        if self.observation_mode == "mid360":
            obs_features = self._build_mid360_features(robot_pos, human_pos, heading)
        else:
            if self.pointcloud_mode in ("vector_map", "gazebo"):
                self._update_mid360_pointcloud(robot_pos, heading)
            else:
                self.current_mid360_points_world = None
            obs_features = self._build_obstacle_features(robot_pos, human_pos, heading)
        if self.n_lookahead <= 0 or self.current_path_data is None:
            self.lookahead_world = None
        return np.concatenate([base, ref_features, obs_features], axis=0).astype(np.float32)

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

    def _simulate_mid360_frame(self, robot_pos: np.ndarray, heading: float) -> np.ndarray:
        if self.current_path_data is None or self.mid360_simulator is None:
            return np.zeros((0, 5), dtype=np.float32)
        obstacles = self.current_path_data.get("obstacles")
        segments = self.current_path_data.get("segment_obstacles")
        return self.mid360_simulator.simulate_frame(
            robot_pos=robot_pos,
            heading=heading,
            obstacles=obstacles,
            segment_obstacles=segments,
        )

    def _update_mid360_pointcloud(
        self,
        robot_pos: np.ndarray,
        heading: float,
        frame: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if self.pointcloud_mode == "gazebo":
            return self._update_mid360_pointcloud_from_gazebo(robot_pos, heading)
        if frame is None:
            frame = self._simulate_mid360_frame(robot_pos, heading)
        if frame is None or len(frame) == 0:
            self.current_mid360_points_world = None
            return np.zeros((0, 2), dtype=np.float32)

        local_xy = np.asarray(frame[:, :2], dtype=np.float32)
        cos_h = float(np.cos(heading))
        sin_h = float(np.sin(heading))
        rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]], dtype=np.float32)
        self.current_mid360_points_world = local_xy @ rot.T + robot_pos.astype(np.float32)
        return local_xy

    def _human_cloud_keep_mask(
        self,
        local_xy: np.ndarray,
        robot_pos: np.ndarray,
        human_pos: np.ndarray,
        heading: float,
    ) -> np.ndarray:
        human_rel = np.asarray(human_pos, dtype=np.float32) - np.asarray(robot_pos, dtype=np.float32)
        cos_h = float(np.cos(heading))
        sin_h = float(np.sin(heading))
        human_local = np.array(
            [
                cos_h * human_rel[0] + sin_h * human_rel[1],
                -sin_h * human_rel[0] + cos_h * human_rel[1],
            ],
            dtype=np.float32,
        )
        distance = np.linalg.norm(local_xy[:, :2] - human_local[None, :], axis=1)
        return distance > float(self.physics.human_radius)

    def _filter_gazebo_cloud_near_human(
        self,
        frame: np.ndarray,
        field_names: list[str],
        mid360_pose: np.ndarray,
        human_pos: np.ndarray,
    ) -> np.ndarray:
        indices = {str(name): idx for idx, name in enumerate(field_names)}
        if len(frame) == 0 or "x" not in indices or "y" not in indices:
            return frame
        local_xyz = np.zeros((len(frame), 3), dtype=np.float32)
        local_xyz[:, 0] = frame[:, indices["x"]]
        local_xyz[:, 1] = frame[:, indices["y"]]
        if "z" in indices:
            local_xyz[:, 2] = frame[:, indices["z"]]
        rot = Rotation.from_quat(mid360_pose[3:7]).as_matrix().astype(np.float32)
        world_xy = (local_xyz @ rot.T + np.asarray(mid360_pose[:3], dtype=np.float32))[:, :2]
        distance = np.linalg.norm(
            world_xy - np.asarray(human_pos, dtype=np.float32)[None, :],
            axis=1,
        )
        return frame[distance > float(self.physics.human_radius)]

    def _build_mid360_features(
        self,
        robot_pos: np.ndarray,
        human_pos: np.ndarray,
        heading: float,
    ) -> np.ndarray:
        if self.current_path_data is None or self.mid360_obs_config is None:
            self.current_mid360_points_world = None
            return np.zeros((self.lidar_num_bins,), dtype=np.float32)

        if self.pointcloud_mode == "gazebo":
            cloud = self._get_mid360_gazebo_cloud(robot_pos, heading)
            if cloud is None:
                self.current_mid360_points_world = None
                return np.full(
                    (self.lidar_num_bins,),
                    self.mid360_obs_config.fill_value,
                    dtype=np.float32,
                )
            frame, field_names, mid360_pose = cloud
            frame = self._filter_gazebo_cloud_near_human(
                frame,
                field_names,
                mid360_pose,
                human_pos,
            )
            cloud = (frame, field_names, mid360_pose)
            self._update_mid360_pointcloud_from_gazebo(
                robot_pos,
                heading,
                cloud=cloud,
            )
            return encode_mid360_scan_from_pointcloud(
                frame=frame,
                field_names=field_names,
                config=self.mid360_obs_config,
                mid360_pose=mid360_pose,
            ).astype(np.float32, copy=False)

        if self.mid360_simulator is None:
            self.current_mid360_points_world = None
            return np.full(
                (self.lidar_num_bins,),
                self.mid360_obs_config.fill_value,
                dtype=np.float32,
            )
        frame = self._simulate_mid360_frame(robot_pos, heading)
        if frame is not None and len(frame) > 0:
            keep = self._human_cloud_keep_mask(
                np.asarray(frame[:, :2], dtype=np.float32),
                robot_pos,
                human_pos,
                heading,
            )
            frame = frame[keep]
        local_xy = self._update_mid360_pointcloud(robot_pos, heading, frame=frame)
        if frame is None or len(frame) == 0:
            return np.full(
                (self.lidar_num_bins,),
                self.mid360_obs_config.fill_value,
                dtype=np.float32,
            )

        ranges = np.asarray(frame[:, 3], dtype=np.float32)
        azimuth = np.asarray(frame[:, 4], dtype=np.float32)
        scan = encode_mid360_scan_from_local_points(
            local_xy=local_xy,
            config=self.mid360_obs_config,
            ranges=ranges,
            azimuth=azimuth,
            height=None,
        )
        return scan.astype(np.float32, copy=False)

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
        if action_seq is None or action_seq.size == 0:
            return None

        pos = np.asarray(robot_pos, dtype=np.float32).copy()
        heading = float(self.physics.robot.heading)
        points: list[np.ndarray] = []
        action_seq = np.asarray(action_seq, dtype=np.float32)
        obstacles = self.current_path_data.get("obstacles") if self.current_path_data else None
        segments = self.current_path_data.get("segment_obstacles") if self.current_path_data else None

        for action_idx, act in enumerate(action_seq):
            step_points, heading, collided = self._rollout_preview_action(
                pos=pos,
                heading=heading,
                action=act,
                obstacles=obstacles,
                segment_obstacles=segments,
            )
            if step_points:
                pos = step_points[-1].copy()
                points.extend(step_points)
            if self.debug_preview and action_idx < self.debug_preview_limit:
                end_pos = pos.tolist()
                print(
                    "[debug preview] "
                    f"idx={action_idx} action={np.round(act, 4).tolist()} "
                    f"end_pos={[round(v, 4) for v in end_pos]} heading={heading:.4f} "
                    f"collided={collided}"
                )
            if collided:
                break

        if not points:
            return None
        return np.stack(points, axis=0)

    def _deltas_to_path(
        self,
        delta_seq: np.ndarray,
        protect_robot: bool = True,
        protect_human: Optional[bool] = None,
    ) -> Optional[np.ndarray]:
        if delta_seq is None or delta_seq.size == 0:
            return None

        delta_seq = np.asarray(delta_seq, dtype=np.float32)
        obstacles = self.current_path_data.get("obstacles") if self.current_path_data else None
        segments = self.current_path_data.get("segment_obstacles") if self.current_path_data else None
        sim = copy.deepcopy(self.physics)
        points: list[np.ndarray] = []
        if protect_human is None:
            protect_human = self._safety_protects_human()

        for delta_idx, delta in enumerate(delta_seq):
            collided, step_points = self._simulate_delta_on_engine(
                sim,
                delta,
                obstacles=obstacles,
                segment_obstacles=segments,
                protect_robot=protect_robot,
                protect_human=protect_human,
            )
            if step_points:
                points.extend(step_points)
            if self.debug_preview and delta_idx < self.debug_preview_limit:
                end_pos = sim.robot.position.tolist()
                print(
                    "[debug preview] "
                    f"idx={delta_idx} safe_delta={np.round(delta, 4).tolist()} "
                    f"end_pos={[round(v, 4) for v in end_pos]} "
                    f"heading={sim.robot.heading:.4f} collided={collided}"
                )
            if collided:
                break

        if not points:
            return None
        return np.stack(points, axis=0)

    def _rollout_preview_action(
        self,
        pos: np.ndarray,
        heading: float,
        action: np.ndarray,
        obstacles: Optional[np.ndarray],
        segment_obstacles: Optional[np.ndarray],
    ) -> tuple[list[np.ndarray], float, bool]:
        pos = np.asarray(pos, dtype=np.float32).copy()
        heading = float(heading)
        action = np.asarray(action, dtype=np.float32)

        if self.action_mode == "forward_heading":
            _delta, forward_input, turn_input, _speed_scale = self._action_to_execution(
                action,
                pos,
                heading,
            )
        else:
            delta = self._action_to_world_delta(action, pos, heading)
            forward_input, turn_input, _speed_scale = self._delta_to_safe_control(
                delta,
                heading,
                dt=self.data_dt,
            )

        step_points: list[np.ndarray] = []
        collided = False
        for _ in range(int(self.frame_stride)):
            heading = wrap_angle(heading + float(turn_input) * float(self.physics.turn_speed) * float(self.sim_dt))
            step_dist = float(forward_input) * float(self.physics.robot_speed) * float(self.sim_dt)
            pos = pos + np.array([
                np.cos(heading) * step_dist,
                np.sin(heading) * step_dist,
            ], dtype=np.float32)
            step_points.append(pos.copy())
            if self._preview_robot_collision(pos, obstacles, segment_obstacles):
                collided = True
                break
        return step_points, heading, collided

    def _preview_robot_collision(
        self,
        robot_pos: np.ndarray,
        obstacles: Optional[np.ndarray],
        segment_obstacles: Optional[np.ndarray],
    ) -> bool:
        robot_pos = np.asarray(robot_pos, dtype=np.float32)
        robot_radius = float(self.physics.robot_radius)

        if obstacles is not None and len(obstacles) > 0:
            obs = np.asarray(obstacles, dtype=np.float32)
            if obs.ndim == 2 and obs.shape[1] >= 3:
                centers = obs[:, :2]
                radii = obs[:, 2]
                dists = np.linalg.norm(centers - robot_pos[None, :], axis=1)
                if np.any(dists <= (radii + robot_radius)):
                    return True

        if segment_obstacles is not None and len(segment_obstacles) > 0:
            segs = np.asarray(segment_obstacles, dtype=np.float32)
            if segs.ndim == 2 and segs.shape[1] >= 4:
                for seg in segs[:, :4]:
                    dist_sq = self._point_segment_dist_sq(robot_pos, seg[:2], seg[2:4])
                    if dist_sq <= robot_radius * robot_radius:
                        return True
        return False

    def _action_to_world_delta(
        self,
        action: np.ndarray,
        robot_pos: np.ndarray,
        heading: float,
    ) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        robot_pos = np.asarray(robot_pos, dtype=np.float32)

        if self.action_mode == "forward_heading":
            forward = float(action[0])
            return np.array(
                [np.cos(heading) * forward, np.sin(heading) * forward],
                dtype=np.float32,
            )

        if self.action_mode == "delta":
            delta = action[:2].astype(np.float32)
        elif self.action_mode == "position":
            delta = action[:2].astype(np.float32) - robot_pos[:2].astype(np.float32)
        else:
            delta = action[:2].astype(np.float32) * float(self.data_dt)

        if self.robot_frame and self.action_mode in ("delta", "velocity"):
            cos_h = float(np.cos(heading))
            sin_h = float(np.sin(heading))
            return np.array(
                [
                    cos_h * float(delta[0]) - sin_h * float(delta[1]),
                    sin_h * float(delta[0]) + cos_h * float(delta[1]),
                ],
                dtype=np.float32,
            )
        return delta.astype(np.float32)

    def _action_to_delta(self, action: np.ndarray, robot_pos: np.ndarray) -> np.ndarray:
        return self._action_to_world_delta(action, robot_pos, self.physics.robot.heading)

    def _world_delta_to_action(
        self,
        delta: np.ndarray,
        robot_pos: np.ndarray,
        heading: float,
    ) -> np.ndarray:
        delta = np.asarray(delta, dtype=np.float32).reshape(2)
        robot_pos = np.asarray(robot_pos, dtype=np.float32).reshape(2)

        if self.action_mode == "forward_heading":
            dist = float(np.linalg.norm(delta))
            if dist < 1e-6:
                return np.zeros((2,), dtype=np.float32)
            desired_heading = float(np.arctan2(delta[1], delta[0]))
            max_heading_delta = float(self.physics.turn_speed * self.data_dt) / max(
                float(self.turn_gain), 1e-6
            )
            heading_delta = wrap_angle(desired_heading - heading) / max(
                float(self.turn_gain), 1e-6
            )
            heading_delta = float(np.clip(heading_delta, -max_heading_delta, max_heading_delta))
            max_forward_delta = float(self.physics.robot_speed * self.data_dt)
            forward_delta = float(np.clip(dist, 0.0, max_forward_delta))
            return np.array([forward_delta, heading_delta], dtype=np.float32)

        local = delta.astype(np.float32)
        if self.robot_frame and self.action_mode in ("delta", "velocity"):
            cos_h = float(np.cos(heading))
            sin_h = float(np.sin(heading))
            local = np.array(
                [
                    cos_h * float(delta[0]) + sin_h * float(delta[1]),
                    -sin_h * float(delta[0]) + cos_h * float(delta[1]),
                ],
                dtype=np.float32,
            )

        if self.action_mode == "delta":
            return local.astype(np.float32)
        if self.action_mode == "velocity":
            return (local / float(max(self.data_dt, 1e-6))).astype(np.float32)
        return (robot_pos + delta).astype(np.float32)

    def _action_to_execution(
        self,
        action: np.ndarray,
        robot_pos: np.ndarray,
        heading: float,
    ) -> tuple[np.ndarray, float, float, float]:
        if self.action_mode == "forward_heading":
            forward_delta = float(action[0])
            heading_delta = float(action[1])
            turn_speed = float(self.physics.turn_speed)
            robot_speed = float(self.physics.robot_speed)
            turn_delta = heading_delta * float(self.turn_gain)
            turn = turn_delta / (turn_speed * self.data_dt) if turn_speed > 0 else 0.0
            forward = forward_delta / (robot_speed * self.data_dt) if robot_speed > 0 else 0.0
            speed_scale = 1.0
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
            delta = self._action_to_world_delta(action, robot_pos, heading)
            return delta, forward, turn, speed_scale

        delta = self._action_to_world_delta(action, robot_pos, heading)
        forward, turn = self._delta_to_control(delta, heading, dt=self.data_dt)
        return delta, forward, turn, 1.0

    def _simulate_action_on_engine(
        self,
        engine: PhysicsEngine,
        action: np.ndarray,
        obstacles: Optional[np.ndarray],
        segment_obstacles: Optional[np.ndarray],
        protect_robot: bool = True,
        protect_human: bool = True,
    ) -> tuple[bool, list[np.ndarray]]:
        _, forward, turn, _speed_scale = self._action_to_execution(
            action,
            engine.robot.position,
            engine.robot.heading,
        )
        engine.set_control(forward, turn, self.bre)
        points: list[np.ndarray] = []
        for _ in range(int(self.frame_stride)):
            robot_state, _human_state = engine.step()
            points.append(robot_state.position.copy())
            if obstacles is not None or segment_obstacles is not None:
                collided, info = engine.check_collision(obstacles, segment_obstacles=segment_obstacles)
                who = info.get("who") if info else None
                relevant = (
                    collided
                    and (
                        (protect_robot and who == "robot")
                        or (protect_human and who == "human")
                    )
                )
                if relevant:
                    return True, points
        return False, points

    def _forward_heading_action_to_nominal_delta(
        self,
        engine: PhysicsEngine,
        action: np.ndarray,
    ) -> tuple[np.ndarray, PhysicsEngine]:
        nominal_engine = copy.deepcopy(engine)
        start_pos = nominal_engine.robot.position.copy()
        self._simulate_action_on_engine(
            nominal_engine,
            action,
            obstacles=None,
            segment_obstacles=None,
            protect_robot=False,
            protect_human=False,
        )
        delta = (nominal_engine.robot.position - start_pos).astype(np.float32)
        return delta, nominal_engine

    def _action_seq_to_nominal_delta_seq(
        self,
        action_seq: np.ndarray,
        engine: Optional[PhysicsEngine] = None,
    ) -> np.ndarray:
        action_seq = np.asarray(action_seq, dtype=np.float32)
        if action_seq.size == 0:
            return np.zeros((0, 2), dtype=np.float32)

        sim = copy.deepcopy(engine if engine is not None else self.physics)
        deltas: list[np.ndarray] = []
        for action in action_seq:
            if self.action_mode == "forward_heading":
                delta, sim = self._forward_heading_action_to_nominal_delta(sim, action)
            else:
                delta = self._action_to_world_delta(action, sim.robot.position, sim.robot.heading)
                self._simulate_delta_on_engine(
                    sim,
                    delta,
                    obstacles=None,
                    segment_obstacles=None,
                    protect_robot=False,
                    protect_human=False,
                )
            deltas.append(delta.astype(np.float32))
        return np.asarray(deltas, dtype=np.float32)

    def _safety_protects_human(self) -> bool:
        return self.safety_mode == "human_robot_qp"

    def _empty_safety_info(self) -> dict:
        return {
            "modified": False,
            "shift": 0.0,
            "constraint_count": 0,
            "min_clearance": float("inf"),
        }

    def _update_episode_safety_stats(self, safety_info: dict):
        self.episode_safety_stats["total_steps"] += 1
        self.episode_safety_stats["total_shift"] += float(safety_info["shift"])
        self.episode_safety_stats["constraint_count"] = max(
            int(self.episode_safety_stats["constraint_count"]),
            int(safety_info["constraint_count"]),
        )
        self.episode_safety_stats["min_clearance"] = min(
            float(self.episode_safety_stats["min_clearance"]),
            float(safety_info["min_clearance"]),
        )
        if safety_info["modified"]:
            self.episode_safety_stats["modified_steps"] += 1

    def _safety_filter_delta(
        self,
        engine: PhysicsEngine,
        nominal_delta: np.ndarray,
        obstacles: Optional[np.ndarray],
        segment_obstacles: Optional[np.ndarray],
        nominal_preview: Optional[PhysicsEngine] = None,
    ) -> tuple[np.ndarray, PhysicsEngine, dict]:
        nominal_delta = np.asarray(nominal_delta, dtype=np.float32).reshape(2)
        info = self._empty_safety_info()
        if self.safety_mode == "off":
            trial_engine = copy.deepcopy(engine)
            self._simulate_delta_on_engine(
                trial_engine,
                nominal_delta,
                obstacles=None,
                segment_obstacles=None,
                protect_robot=False,
                protect_human=False,
            )
            return nominal_delta.astype(np.float32), trial_engine, info

        protect_human = self._safety_protects_human()
        robot_pos = engine.robot.position.copy()
        human_pos = engine.human.position.copy()
        robot_heading = float(engine.robot.heading)
        if nominal_preview is None:
            nominal_preview = copy.deepcopy(engine)
            self._simulate_delta_on_engine(
                nominal_preview,
                nominal_delta,
                obstacles=obstacles,
                segment_obstacles=segment_obstacles,
                protect_robot=False,
                protect_human=False,
            )

        extra_entities = [
            ("robot_future", nominal_preview.robot.position.copy(), self.physics.robot_radius)
        ]
        if protect_human:
            extra_entities.append(
                ("human_future", nominal_preview.human.position.copy(), self.physics.human_radius)
            )

        qp = self.safety_filter.project_delta(
            ref_delta=nominal_delta,
            robot_pos=engine.robot.position,
            robot_radius=self.physics.robot_radius,
            human_pos=engine.human.position,
            human_radius=self.physics.human_radius,
            circle_obstacles=obstacles,
            segment_obstacles=segment_obstacles,
            include_human=protect_human,
            extra_entities=extra_entities,
        )
        qp_delta = qp.delta.astype(np.float32)
        chosen_delta = qp_delta.copy()
        trial_engine = copy.deepcopy(engine)
        collided, _ = self._simulate_delta_on_engine(
            trial_engine,
            chosen_delta,
            obstacles=obstacles,
            segment_obstacles=segment_obstacles,
            protect_robot=True,
            protect_human=protect_human,
        )
        collision_after_qp = bool(collided)
        backoff_attempts: list[dict] = []
        backoff_applied = False
        backoff_scale = 1.0
        resolution_stage = "qp"

        if collided:
            for scale in self.safety_backoff_scales[1:]:
                backoff_delta = (chosen_delta * float(scale)).astype(np.float32)
                backoff_engine = copy.deepcopy(engine)
                collided, _ = self._simulate_delta_on_engine(
                    backoff_engine,
                    backoff_delta,
                    obstacles=obstacles,
                    segment_obstacles=segment_obstacles,
                    protect_robot=True,
                    protect_human=protect_human,
                )
                backoff_attempts.append(
                    {
                        "scale": float(scale),
                        "delta": backoff_delta.astype(np.float32),
                        "collided": bool(collided),
                    }
                )
                if not collided:
                    chosen_delta = backoff_delta
                    trial_engine = backoff_engine
                    backoff_applied = True
                    backoff_scale = float(scale)
                    resolution_stage = "backoff"
                    break

        info = {
            "modified": bool(np.linalg.norm(chosen_delta - nominal_delta) > 1e-5 or qp.modified),
            "shift": float(np.linalg.norm(chosen_delta - nominal_delta)),
            "constraint_count": int(qp.constraint_count),
            "min_clearance": float(qp.min_clearance),
            "robot_pos": robot_pos.astype(np.float32),
            "human_pos": human_pos.astype(np.float32),
            "robot_heading": float(robot_heading),
            "nominal_delta": nominal_delta.astype(np.float32),
            "nominal_delta_norm": float(np.linalg.norm(nominal_delta)),
            "nominal_preview_robot_pos": nominal_preview.robot.position.copy().astype(np.float32),
            "nominal_preview_human_pos": nominal_preview.human.position.copy().astype(np.float32),
            "protect_human": bool(protect_human),
            "qp_delta": qp_delta.astype(np.float32),
            "qp_delta_norm": float(np.linalg.norm(qp_delta)),
            "qp_modified": bool(qp.modified),
            "qp_constraint_count": int(qp.constraint_count),
            "qp_total_constraint_count": int(getattr(qp, "total_constraint_count", qp.constraint_count)),
            "qp_ref_feasible": bool(getattr(qp, "ref_feasible", True)),
            "qp_candidate_count": int(getattr(qp, "candidate_count", 0)),
            "qp_best_candidate_kind": str(getattr(qp, "best_candidate_kind", "unknown")),
            "qp_best_candidate_constraints": list(
                getattr(qp, "best_candidate_constraints", [])
            ),
            "qp_selected_constraints": list(getattr(qp, "selected_constraints", [])),
            "collision_after_qp": bool(collision_after_qp),
            "backoff_attempts": backoff_attempts,
            "backoff_applied": bool(backoff_applied),
            "backoff_scale": float(backoff_scale),
            "stop_triggered": False,
            "stop_clearance_threshold": float(self.safety_stop_clearance),
            "stop_reason": None,
            "resolution_stage": resolution_stage,
        }
        # if info["min_clearance"] < self.safety_stop_clearance:
        #     chosen_delta = np.zeros((2,), dtype=np.float32)
        #     trial_engine = copy.deepcopy(engine)
        #     _collided, _ = self._simulate_delta_on_engine(
        #         trial_engine,
        #         chosen_delta,
        #         obstacles=obstacles,
        #         segment_obstacles=segment_obstacles,
        #         protect_robot=True,
        #         protect_human=protect_human,
        #     )
        #     info["modified"] = True
        #     info["shift"] = float(np.linalg.norm(chosen_delta - nominal_delta))
        #     info["stop_triggered"] = True
        #     info["stop_reason"] = "min_clearance_below_stop_threshold"
        #     info["resolution_stage"] = "stop"
        info["final_delta"] = chosen_delta.astype(np.float32)
        info["final_delta_norm"] = float(np.linalg.norm(chosen_delta))
        return chosen_delta.astype(np.float32), trial_engine, info

    def _safety_filter_action(
        self,
        engine: PhysicsEngine,
        nominal_action: np.ndarray,
        obstacles: Optional[np.ndarray],
        segment_obstacles: Optional[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, PhysicsEngine, dict]:
        nominal_action = np.asarray(nominal_action, dtype=np.float32)
        info = self._empty_safety_info()
        ref_delta = self._action_to_world_delta(
            nominal_action, engine.robot.position, engine.robot.heading
        )
        if self.safety_mode == "off":
            return nominal_action, ref_delta, copy.deepcopy(engine), info

        nominal_preview = copy.deepcopy(engine)
        self._simulate_action_on_engine(
            nominal_preview,
            nominal_action,
            obstacles=obstacles,
            segment_obstacles=segment_obstacles,
            protect_robot=False,
            protect_human=False,
        )
        chosen_delta, trial_engine, info = self._safety_filter_delta(
            engine,
            ref_delta,
            obstacles=obstacles,
            segment_obstacles=segment_obstacles,
            nominal_preview=nominal_preview,
        )
        chosen_action = self._world_delta_to_action(
            chosen_delta,
            engine.robot.position,
            engine.robot.heading,
        )
        return chosen_action.astype(np.float32), chosen_delta.astype(np.float32), trial_engine, info

    def _apply_forward_heading_safety_filter(
        self,
        action_seq: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        action_seq = np.asarray(action_seq, dtype=np.float32)
        stats = {
            "applied": self.safety_mode != "off",
            "modified_steps": 0,
            "total_steps": int(len(action_seq)),
            "mean_shift": 0.0,
            "constraint_count": 0,
            "min_clearance": float("inf"),
        }
        if action_seq.size == 0:
            self.last_safety_stats = stats
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0, 2), dtype=np.float32),
                [],
            )

        obstacles = self.current_path_data.get("obstacles") if self.current_path_data else None
        segments = self.current_path_data.get("segment_obstacles") if self.current_path_data else None
        sim = copy.deepcopy(self.physics)
        nominal_deltas: list[np.ndarray] = []
        safe_deltas: list[np.ndarray] = []
        safety_infos: list[dict] = []
        shifts = []

        for nominal_action in action_seq:
            nominal_delta, nominal_preview = self._forward_heading_action_to_nominal_delta(
                sim, nominal_action
            )
            if self.safety_mode == "off":
                chosen_delta = nominal_delta
                trial_engine = nominal_preview
                info = self._empty_safety_info()
            else:
                chosen_delta, trial_engine, info = self._safety_filter_delta(
                    sim,
                    nominal_delta,
                    obstacles=obstacles,
                    segment_obstacles=segments,
                    nominal_preview=nominal_preview,
                )

            nominal_deltas.append(nominal_delta.astype(np.float32))
            safe_deltas.append(chosen_delta.astype(np.float32))
            safety_infos.append(info)
            stats["constraint_count"] = max(stats["constraint_count"], int(info["constraint_count"]))
            stats["min_clearance"] = min(stats["min_clearance"], float(info["min_clearance"]))
            shifts.append(float(info["shift"]))
            if info["modified"]:
                stats["modified_steps"] += 1
            sim = trial_engine

        if shifts:
            stats["mean_shift"] = float(np.mean(shifts))
        self.last_safety_stats = stats
        return (
            np.asarray(nominal_deltas, dtype=np.float32),
            np.asarray(safe_deltas, dtype=np.float32),
            safety_infos,
        )

    def _apply_safety_filter(self, action_seq: np.ndarray) -> np.ndarray:
        action_seq = np.asarray(action_seq, dtype=np.float32)
        stats = {
            "applied": self.safety_mode != "off",
            "modified_steps": 0,
            "total_steps": int(len(action_seq)),
            "mean_shift": 0.0,
            "constraint_count": 0,
            "min_clearance": float("inf"),
        }
        if (
            self.safety_mode == "off"
            or action_seq.size == 0
            or self.current_path_data is None
        ):
            self.last_safety_stats = stats
            return action_seq

        obstacles = self.current_path_data.get("obstacles")
        segments = self.current_path_data.get("segment_obstacles")
        if (obstacles is None or len(obstacles) == 0) and (segments is None or len(segments) == 0):
            self.last_safety_stats = stats
            return action_seq

        sim = copy.deepcopy(self.physics)
        safe_actions = []
        shifts = []

        for nominal_action in action_seq:
            chosen_action, chosen_delta, trial_engine, info = self._safety_filter_action(
                sim,
                nominal_action,
                obstacles=obstacles,
                segment_obstacles=segments,
            )
            stats["constraint_count"] = max(stats["constraint_count"], int(info["constraint_count"]))
            stats["min_clearance"] = min(stats["min_clearance"], float(info["min_clearance"]))
            safe_actions.append(chosen_action.astype(np.float32))
            sim = trial_engine
            shift = float(info["shift"])
            shifts.append(shift)
            if info["modified"]:
                stats["modified_steps"] += 1

        if shifts:
            stats["mean_shift"] = float(np.mean(shifts))
        self.last_safety_stats = stats
        return np.asarray(safe_actions, dtype=np.float32)

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

    def _delta_to_safe_control(
        self,
        delta: np.ndarray,
        heading: float,
        dt: Optional[float] = None,
    ) -> tuple[float, float, float]:
        if dt is None:
            dt = self.data_dt
        forward_input, turn_input = self._delta_to_control(delta, heading, dt=dt)
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm < 1e-6:
            return 0.0, 0.0, 1.0

        desired_heading = float(np.arctan2(delta[1], delta[0]))
        heading_error = wrap_angle(desired_heading - heading)
        turn_speed = float(self.physics.turn_speed)
        speed_scale = 1.0
        if self.curvature_slowdown and turn_speed > 0:
            max_turn = turn_speed * dt
            if max_turn > 1e-6:
                ratio = min(1.0, abs(heading_error) / max_turn)
                speed_scale = max(
                    float(self.min_speed_scale),
                    1.0 - float(self.curvature_scale) * ratio,
                )
                forward_input *= speed_scale
        return float(forward_input), float(turn_input), float(speed_scale)

    def _simulate_delta_on_engine(
        self,
        engine: PhysicsEngine,
        delta: np.ndarray,
        obstacles: Optional[np.ndarray],
        segment_obstacles: Optional[np.ndarray],
        protect_robot: bool = True,
        protect_human: bool = True,
    ) -> tuple[bool, list[np.ndarray]]:
        forward, turn, _speed_scale = self._delta_to_safe_control(
            delta,
            engine.robot.heading,
            dt=self.data_dt,
        )
        engine.set_control(forward, turn, self.bre)
        points: list[np.ndarray] = []
        for _ in range(int(self.frame_stride)):
            robot_state, _human_state = engine.step()
            points.append(robot_state.position.copy())
            if obstacles is not None or segment_obstacles is not None:
                collided, info = engine.check_collision(obstacles, segment_obstacles=segment_obstacles)
                who = info.get("who") if info else None
                relevant = (
                    collided
                    and (
                        (protect_robot and who == "robot")
                        or (protect_human and who == "human")
                    )
                )
                if relevant:
                    return True, points
        return False, points
    
    def _cycle_safety_mode(self):
        """Cycle safety mode: off -> robot_qp -> human_robot_qp -> off."""
        modes = ["off", "robot_qp", "human_robot_qp"]
        current = normalize_safety_mode(self.safety_mode)
        next_idx = (modes.index(current) + 1) % len(modes)
        new_mode = modes[next_idx]

        # Keep compatibility with action mode constraints
        if new_mode != "off" and self.action_mode not in ("forward_heading", "delta", "velocity"):
            print(
                f"[warn] safety_mode={new_mode} is only supported for "
                f"forward_heading/delta/velocity; fallback to off"
            )
            new_mode = "off"

        self.safety_mode = new_mode

        # Clear caches so the new mode takes effect immediately
        self.cached_action_seq = None
        self.cached_nominal_delta_seq = None
        self.cached_safe_delta_seq = None
        self.cached_safety_info_seq = None
        self.cached_action_idx = 0
        self.frames_since_inference = 0
        self.cached_control = (0.0, 0.0)
        self.current_action = None
        self.current_delta = None
        self.current_speed_scale = 1.0

        # Clear preview paths
        self.planned_path = None
        self.nominal_planned_path = None
        self.safe_planned_path = None

        if self.log_fp is not None:
            self._log_event("safety_mode_changed", {"safety_mode": self.safety_mode})

        print(f"Safety mode: {self.safety_mode}")

    def _set_pointcloud_mode(self, new_mode: str, *, rebuild_obs_history: bool = True):
        new_mode = normalize_pointcloud_mode(new_mode)
        if new_mode == "auto":
            new_mode = "vector_map" if self.observation_mode == "mid360" else "off"
        if self.observation_mode == "mid360" and new_mode == "off":
            raise ValueError("guide_mid360 checkpoint requires pointcloud_mode != 'off'")
        if new_mode == self.pointcloud_mode:
            return
        if new_mode == "gazebo":
            self._ensure_mid360_gazebo_session()
        else:
            self._close_mid360_gazebo_session()
        self.pointcloud_mode = new_mode
        self.current_mid360_points_world = None
        self._last_mid360_cloud_seq = None
        self._reset_runtime_caches()
        if rebuild_obs_history:
            self._seed_obs_history(self.physics.robot.position, self.physics.human.position)
        if self.log_fp is not None:
            self._log_event("pointcloud_mode_changed", {"pointcloud_mode": self.pointcloud_mode})

    def _cycle_pointcloud_mode(self):
        """Cycle point-cloud source during runtime."""
        modes = self._supported_pointcloud_modes()
        current = self.pointcloud_mode
        if current not in modes:
            current = modes[0]
        start_idx = modes.index(current)
        for offset in range(1, len(modes) + 1):
            candidate = modes[(start_idx + offset) % len(modes)]
            try:
                self._set_pointcloud_mode(candidate)
                print(f"Point cloud mode: {self.pointcloud_mode}")
                return
            except Exception as exc:
                print(f"[warn] failed to switch point cloud mode to {candidate}: {exc}")
        print(f"Point cloud mode unchanged: {self.pointcloud_mode}")

    def _step(self):
        """Advance simulation by one step."""
        self.collision_happened = False
        self.collision_info = None
        if self.paused or self.collision_pause:
            # Freeze simulation state while paused (manual or auto-paused).
            self.physics.set_control(0.0, 0.0, False)
            return self.physics.robot.copy(), self.physics.human.copy()

        self._update_timed_bre_toggle()

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
                        diffusion_start = None
                        if self.eval_fp is not None:
                            self._synchronize_timing_device()
                            diffusion_start = time.perf_counter()
                        action_seq = self._predict_action()
                        diffusion_time_ms = 0.0
                        if diffusion_start is not None:
                            self._synchronize_timing_device()
                            diffusion_time_ms = (time.perf_counter() - diffusion_start) * 1000.0
                        raw_nominal_delta_seq = self._action_seq_to_nominal_delta_seq(action_seq)
                        raw_nominal_path = self._deltas_to_path(
                            raw_nominal_delta_seq,
                            protect_robot=False,
                            protect_human=False,
                        )
                        raw_heading_delta = self._path_heading_delta(raw_nominal_path)
                        self._write_planning_eval(action_seq, diffusion_time_ms)
                        if self.action_mode == "forward_heading":
                            obstacles = self.current_path_data.get("obstacles") if self.current_path_data else None
                            segments = (
                                self.current_path_data.get("segment_obstacles")
                                if self.current_path_data
                                else None
                            )
                            action_seq = self._apply_interaction_aware_compliance_control(
                                action_seq,
                                obstacles=obstacles,
                                segment_obstacles=segments,
                            )
                        self.cached_action_seq = action_seq
                        self.cached_action_idx = 0
                        self.frames_since_inference = 0

                        preview_action_seq = None
                        if self.action_mode == "forward_heading" and self.safety_mode != "off":
                            (
                                nominal_delta_seq,
                                safe_delta_seq,
                                safety_info_seq,
                            ) = self._apply_forward_heading_safety_filter(action_seq)
                            self.cached_nominal_delta_seq = nominal_delta_seq
                            self.cached_safe_delta_seq = safe_delta_seq
                            self.cached_safety_info_seq = safety_info_seq
                            self.nominal_planned_path = self._deltas_to_path(
                                nominal_delta_seq,
                                protect_robot=False,
                                protect_human=False,
                            )
                            self.safe_planned_path = self._deltas_to_path(safe_delta_seq)
                            self.planned_path = self.safe_planned_path
                        else:
                            self.cached_nominal_delta_seq = None
                            self.cached_safe_delta_seq = None
                            self.cached_safety_info_seq = None
                            preview_action_seq = self._apply_safety_filter(action_seq)
                            self.nominal_planned_path = self._actions_to_path(
                                self.physics.robot.position, preview_action_seq
                            )
                            self.safe_planned_path = None
                            self.planned_path = self.nominal_planned_path

                        self.latest_nominal_heading_delta = self._path_heading_delta(
                            self.nominal_planned_path
                        )

                        if self.cached_action_seq is not None and len(self.cached_action_seq) > 0:
                            norms = np.linalg.norm(self.cached_action_seq, axis=1)
                            raw_heading = action_seq[:, 1] if action_seq.shape[1] > 1 else np.zeros(
                                (len(action_seq),), dtype=np.float32
                            )
                            log_payload = {
                                "action_len": int(len(self.cached_action_seq)),
                                "action_mean_norm": float(np.mean(norms)),
                                "action_max_norm": float(np.max(norms)),

                                "raw_forward_mean": float(np.mean(action_seq[:, 0])),
                                "raw_forward_std": float(np.std(action_seq[:, 0])),
                                "raw_heading_mean": float(np.mean(raw_heading)),
                                "raw_heading_std": float(np.std(raw_heading)),
                                "raw_heading_maxabs": float(np.max(np.abs(raw_heading))),

                                "raw_first3": np.round(action_seq[:3], 4).tolist(),
                                "raw_action_first3": np.round(action_seq[:3], 4).tolist(),

                                "safety_mode": self.safety_mode,
                                "safety_modified_steps": int(
                                    self.last_safety_stats.get("modified_steps", 0)
                                ),
                                "safety_mean_shift": float(
                                    self.last_safety_stats.get("mean_shift", 0.0)
                                ),
                                "interaction_label": self._current_interaction_label(),
                                "compliance_mode": str(
                                    self.last_compliance_stats.get("mode", "interaction_aware")
                                ),
                                "compliance_steps": int(
                                    self.last_compliance_stats.get("compliance_steps", 0)
                                ),
                                "compliance_guide_steps": int(
                                    self.last_compliance_stats.get("guide_steps", 0)
                                ),
                                "compliance_modified_steps": int(
                                    self.last_compliance_stats.get("modified_steps", 0)
                                ),
                                "compliance_mean_action_shift": float(
                                    self.last_compliance_stats.get("mean_action_shift", 0.0)
                                ),
                                "compliance_state_counts": dict(
                                    self.last_compliance_stats.get("state_counts", {})
                                ),
                            }
                            if self.cached_safe_delta_seq is not None:
                                safe_delta_norms = np.linalg.norm(self.cached_safe_delta_seq, axis=1)
                                log_payload.update(
                                    {
                                        "safe_mean_norm": float(np.mean(safe_delta_norms)),
                                        "safe_max_norm": float(np.max(safe_delta_norms)),
                                        "safe_delta_norm_mean": float(np.mean(safe_delta_norms)),
                                        "safe_delta_norm_max": float(np.max(safe_delta_norms)),
                                        "safe_delta_x_mean": float(np.mean(self.cached_safe_delta_seq[:, 0])),
                                        "safe_delta_y_mean": float(np.mean(self.cached_safe_delta_seq[:, 1])),
                                        "raw_action_seq": np.round(action_seq, 4).tolist(),
                                        "safe_delta_seq": np.round(self.cached_safe_delta_seq, 4).tolist(),
                                        "safe_first3": np.round(self.cached_safe_delta_seq[:3], 4).tolist(),
                                        "safe_delta_first3": np.round(
                                            self.cached_safe_delta_seq[:3], 4
                                        ).tolist(),
                                    }
                                )
                                if self.debug_qp_log and self.cached_safety_info_seq is not None:
                                    qp_min_clearance_seq = [
                                        float(info.get("min_clearance", float("inf")))
                                        for info in self.cached_safety_info_seq
                                    ]
                                    qp_constraint_count_seq = [
                                        int(info.get("constraint_count", 0))
                                        for info in self.cached_safety_info_seq
                                    ]
                                    qp_stop_steps = [
                                        int(idx)
                                        for idx, info in enumerate(self.cached_safety_info_seq)
                                        if bool(info.get("stop_triggered", False))
                                    ]
                                    qp_backoff_steps = [
                                        int(idx)
                                        for idx, info in enumerate(self.cached_safety_info_seq)
                                        if bool(info.get("backoff_applied", False))
                                    ]
                                    log_payload.update(
                                        {
                                            "robot_qp_min_clearance_seq": [
                                                round(v, 4) if np.isfinite(v) else "inf"
                                                for v in qp_min_clearance_seq
                                            ],
                                            "robot_qp_constraint_count_seq": qp_constraint_count_seq,
                                            "robot_qp_stop_steps": qp_stop_steps,
                                            "robot_qp_backoff_steps": qp_backoff_steps,
                                            "robot_qp_debug_seq": [
                                                self._serialize_safety_info(info)
                                                for info in self.cached_safety_info_seq
                                            ],
                                        }
                                    )
                            else:
                                safe_norms = np.linalg.norm(preview_action_seq, axis=1)
                                log_payload.update(
                                    {
                                        "safe_mean_norm": float(np.mean(safe_norms)),
                                        "safe_max_norm": float(np.max(safe_norms)),
                                        "safe_forward_mean": float(np.mean(preview_action_seq[:, 0])),
                                        "safe_forward_std": float(np.std(preview_action_seq[:, 0])),
                                        "safe_heading_mean": float(np.mean(preview_action_seq[:, 1])),
                                        "safe_heading_std": float(np.std(preview_action_seq[:, 1])),
                                        "safe_heading_maxabs": float(
                                            np.max(np.abs(preview_action_seq[:, 1]))
                                        ),
                                        "safe_first3": np.round(preview_action_seq[:3], 4).tolist(),
                                    }
                                )

                            self._log_event("inference", log_payload)
                            if self.debug_policy:
                                self.debug_inference_count += 1
                                preview_count = min(self.debug_preview_limit, len(self.cached_action_seq))
                                preview_actions = np.round(self.cached_action_seq[:preview_count], 4).tolist()
                                if self.cached_safe_delta_seq is not None:
                                    preview_safe_deltas = np.round(
                                        self.cached_safe_delta_seq[:preview_count], 4
                                    ).tolist()
                                    safe_delta_norms = np.linalg.norm(self.cached_safe_delta_seq, axis=1)
                                    print(
                                        "[debug policy] "
                                        f"inference={self.debug_inference_count} "
                                        f"len={len(self.cached_action_seq)} "
                                        f"raw_actions={preview_actions} "
                                        f"safe_deltas={preview_safe_deltas} "
                                        f"raw_heading_mean={float(np.mean(raw_heading)):.4f} "
                                        f"raw_heading_maxabs={float(np.max(np.abs(raw_heading))):.4f} "
                                        f"safe_delta_norm_mean={float(np.mean(safe_delta_norms)):.4f} "
                                        f"safe_delta_norm_max={float(np.max(safe_delta_norms)):.4f}"
                                    )
                                else:
                                    print(
                                        "[debug policy] "
                                        f"inference={self.debug_inference_count} "
                                        f"len={len(self.cached_action_seq)} "
                                        f"mean_norm={float(np.mean(norms)):.4f} "
                                        f"max_norm={float(np.max(norms)):.4f} "
                                        f"actions={preview_actions}"
                                    )
                    else:
                        # Reuse cached actions
                        self.frames_since_inference += 1

                    # Get current action from cached sequence
                    action_idx = 0
                    if self.cached_action_seq is not None and len(self.cached_action_seq) > 0:
                        action_idx = min(self.cached_action_idx, len(self.cached_action_seq) - 1)
                        action = self.cached_action_seq[action_idx]
                        # Hold the last action once the cached sequence is exhausted.
                        self.cached_action_idx = min(action_idx + 1, len(self.cached_action_seq) - 1)
                    else:
                        action = np.zeros(self.action_dim)

                    safety_info = None
                    if self.safety_mode != "off" and self.action_mode == "forward_heading":
                        if self.cached_safe_delta_seq is not None and len(self.cached_safe_delta_seq) > 0:
                            delta_idx = min(action_idx, len(self.cached_safe_delta_seq) - 1)
                            delta = self.cached_safe_delta_seq[delta_idx].astype(np.float32)
                            if (
                                self.cached_safety_info_seq is not None
                                and delta_idx < len(self.cached_safety_info_seq)
                            ):
                                safety_info = self.cached_safety_info_seq[delta_idx]
                            else:
                                safety_info = self._empty_safety_info()
                        else:
                            obstacles = self.current_path_data.get("obstacles") if self.current_path_data else None
                            segments = (
                                self.current_path_data.get("segment_obstacles")
                                if self.current_path_data
                                else None
                            )
                            nominal_delta, nominal_preview = self._forward_heading_action_to_nominal_delta(
                                self.physics, action
                            )
                            delta, _trial_engine, safety_info = self._safety_filter_delta(
                                self.physics,
                                nominal_delta,
                                obstacles=obstacles,
                                segment_obstacles=segments,
                                nominal_preview=nominal_preview,
                            )
                        self._update_episode_safety_stats(safety_info)
                        forward, turn, speed_scale = self._delta_to_safe_control(
                            delta,
                            self.physics.robot.heading,
                        )
                    elif self.safety_mode != "off":
                        obstacles = self.current_path_data.get("obstacles") if self.current_path_data else None
                        segments = (
                            self.current_path_data.get("segment_obstacles")
                            if self.current_path_data
                            else None
                        )
                        safe_action, delta, _trial_engine, safety_info = self._safety_filter_action(
                            self.physics,
                            action,
                            obstacles=obstacles,
                            segment_obstacles=segments,
                        )
                        action = safe_action
                        self._update_episode_safety_stats(safety_info)
                        forward, turn, speed_scale = self._delta_to_safe_control(
                            delta,
                            self.physics.robot.heading,
                        )
                    else:
                        delta, forward, turn, speed_scale = self._action_to_execution(
                            action,
                            self.physics.robot.position,
                            self.physics.robot.heading,
                        )
                    self.current_action = action
                    self.current_speed_scale = speed_scale
                    self.current_delta = delta
                    self.cached_control = (forward, turn)
                    if is_data_step and self.safety_mode == "robot_qp" and self.action_mode == "forward_heading":
                        self._log_robot_qp_step(
                            action_idx=action_idx,
                            action=action,
                            delta=delta,
                            forward=forward,
                            turn=turn,
                            speed_scale=speed_scale,
                            safety_info=safety_info,
                        )
                else:
                    forward, turn = self.cached_control
                    action = self.current_action
                    delta = self.current_delta
            else:
                forward, turn = self._get_manual_control()
                self.planned_path = None
                self.nominal_planned_path = None
                self.safe_planned_path = None
                # Reset cache when switching to manual
                self.cached_action_seq = None
                self.cached_nominal_delta_seq = None
                self.cached_safe_delta_seq = None
                self.cached_safety_info_seq = None
                self.cached_action_idx = 0
                self.frames_since_inference = 0
                self.cached_control = (forward, turn)
                self.current_action = None
                self.current_delta = None
                self.current_speed_scale = 1.0

        self.physics.set_control(forward, turn, self.bre)
        robot_state, human_state = self.physics.step()
        self._sync_mid360_gazebo_session()

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
                        self.physics.set_control(0.0, 0.0, False)
                        self._log_event(
                            "goal_reached",
                            {
                                "human_pos": human_state.position.tolist(),
                                "s_human": float(s_human),
                                "s_end": float(s_end),
                            },
                        )
                        print("Human reached end of path. Simulation paused (SPACE to resume, N for new path).")

        self.robot_trajectory.append(robot_state.position.copy())
        self.human_trajectory.append(human_state.position.copy())
        if self.recording:
            self._record_frame(robot_state, human_state)

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
        # self._log_step(self.frame_count, robot_state, human_state, action, delta, forward, turn)
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
                print(f"Collision detected ({who}, {label} {idx}), pausing at collision.")
                self.collision_pause = True
                self.physics.set_control(0.0, 0.0, False)
                return True
            print(f"Collision detected ({who}, {label} {idx}), resetting.")
            self._reset_position()
            return True
        return False

    def _render(self, robot_state, human_state, actual_fps: float):
        scores = self.scorer.get_scores() if self.scorer else {}
        mode = "policy" if self.use_policy else "manual"
        safety_label = "diffusion" if self.safety_mode == "off" else self.safety_mode
        mode += f" [{safety_label}]"
        mode += f" [{self._current_interaction_label()}]"
        if self.paused:
            mode += " (paused)"
        elif self.collision_pause:
            mode += " (collision)"

        info = {
            "fps": actual_fps,
            "path_length": self.current_path_data["length"] if self.current_path_data else 0,
            "robot_x": robot_state.position[0],
            "robot_y": robot_state.position[1],
            "num_points": self.storage.get_num_points() if self.storage is not None else self.frame_count,
            "recording": self.recording,
            "scores": scores,
            "mode": mode,
            "safety_mode": self.safety_mode,
            "pointcloud_mode": self.pointcloud_mode,
            "interaction_label": self._current_interaction_label(),
            "compliance_steps": int(self.last_compliance_stats.get("compliance_steps", 0)),
            "compliance_total_steps": int(self.last_compliance_stats.get("total_steps", 0)),
            "robot_radius": self.physics.robot_radius,
            "human_radius": self.physics.human_radius,
            "nominal_heading_delta": self.latest_nominal_heading_delta,
            "controls": [
                "P: Policy/Manual",
                "C: PointCloud",
                "SPACE: Record/Pause" if self.collect_enabled else "SPACE: Pause",
                "S: Save episode" if self.collect_enabled else "M: Safety mode",
                "B: Guide/Tether",
                "R: Reset",
                "N: New Path",
                "Arrows: Manual control",
                "Scroll: Zoom",
                "ESC: Exit",
            ],
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
            nominal_planned_path=self.nominal_planned_path,
            safe_planned_path=self.safe_planned_path,
            lookahead_points=self.lookahead_world,
            point_cloud=self.current_mid360_points_world,
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
        print("Controls: P=Policy/Manual | C=PointCloud | SPACE=Pause | R=Reset | N=NewPath | ESC=Exit")
        print("=" * 60)

        while self.running:
            self._handle_input()
            robot_state, human_state = self._step()
            actual_fps = self.visualizer.tick(self.fps)
            self._render(robot_state, human_state, actual_fps)

        self._close_mid360_gazebo_session()
        self.visualizer.quit()
        if self.log_fp is not None:
            total_steps = max(1, int(self.episode_safety_stats["total_steps"]))
            self._log_event(
                "shutdown",
                {
                    "episode_safety_stats": {
                        "modified_steps": int(self.episode_safety_stats["modified_steps"]),
                        "total_steps": int(self.episode_safety_stats["total_steps"]),
                        "mean_shift": round(
                            float(self.episode_safety_stats["total_shift"]) / float(total_steps),
                            4,
                        ),
                        "constraint_count": int(self.episode_safety_stats["constraint_count"]),
                        "min_clearance": round(
                            float(self.episode_safety_stats["min_clearance"]), 4
                        )
                        if np.isfinite(float(self.episode_safety_stats["min_clearance"]))
                        else "inf",
                    }
                },
            )
            self.log_fp.close()
        if self.eval_fp is not None:
            self.eval_fp.close()
        print("Program exit")


def main():
    parser = argparse.ArgumentParser(description="Guide-follow planning with a trained policy")
    
    # Default checkpoint path - prefer the best available guide_mid360 checkpoint.
    default_ckpt = resolve_default_checkpoint()
    
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
    parser.add_argument(
        "--pointcloud-mode",
        default="auto",
        help="auto | off | vector_map | gazebo. `mid360` is kept as an alias of vector_map.",
    )
    parser.add_argument(
        "--mid360-plugin-dir",
        type=str,
        default=None,
        help="Path to the Mid360_simulation_plugin repository for Gazebo point clouds.",
    )
    parser.add_argument(
        "--mid360-plugin-lib",
        type=str,
        default=None,
        help="Path to liblivox_laser_simulation.so for Gazebo point clouds.",
    )
    parser.add_argument(
        "--mid360-downsample",
        type=int,
        default=1,
        help="Downsample factor passed to the Mid360 Gazebo plugin.",
    )
    parser.add_argument(
        "--mid360-gazebo-gui",
        action="store_true",
        help="Launch Gazebo with GUI when pointcloud_mode=gazebo.",
    )
    parser.add_argument(
        "--mid360-visualize",
        action="store_true",
        help="Enable laser ray visualization in Gazebo when pointcloud_mode=gazebo.",
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
    parser.add_argument(
        "-e",
        "--eval",
        action="store_true",
        help="Write each origin/robot-safe/human-robot-safe planning result and timing to JSONL.",
    )
    parser.add_argument(
        "-c",
        "--collect",
        action="store_true",
        help="Enable collect-style episode recording. SPACE toggles recording; S saves.",
    )
    parser.add_argument(
        "--safety-mode",
        default="human_robot_qp",
        help="off | robot_qp | human_robot_qp. Apply online safety filtering before execution.",
    )
    parser.add_argument(
        "--safety-margin",
        type=float,
        default=0.2,
        help="Extra safety margin in meters added around obstacles for QP filtering.",
    )
    parser.add_argument(
        "--safety-alpha",
        type=float,
        default=1.0,
        help="Barrier gain for QP halfspace constraints.",
    )
    parser.add_argument(
        "--safety-max-constraints",
        type=int,
        default=8,
        help="Maximum nearby obstacle constraints kept in each 2D QP projection.",
    )
    parser.add_argument(
        "--safety-influence-distance",
        type=float,
        default=2.0,
        help="Only obstacles within this clearance band are included in the QP.",
    )
    parser.add_argument(
        "--debug-preview",
        action="store_true",
        help="Print preview-rollout diagnostics from _actions_to_path.",
    )
    parser.add_argument(
        "--debug-policy",
        action="store_true",
        help="Print predicted action summaries after each inference.",
    )
    parser.add_argument(
        "--no-debug-qp-log",
        action="store_true",
        help="Disable detailed robot_qp diagnostics in the jsonl log.",
    )
    parser.add_argument(
        "--debug-preview-limit",
        type=int,
        default=5,
        help="Maximum number of actions printed for each debug preview/policy dump.",
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

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    planner = ModelPlanner(
        checkpoint_path=checkpoint_path,
        device=args.device,
        use_ema=not args.no_ema,
        action_mode=args.action_mode,
        k_lookahead=args.k_lookahead,
        frame_stride=args.frame_stride,
        pointcloud_mode=args.pointcloud_mode,
        mid360_plugin_dir=args.mid360_plugin_dir,
        mid360_plugin_lib=args.mid360_plugin_lib,
        mid360_downsample=args.mid360_downsample,
        mid360_gazebo_gui=args.mid360_gazebo_gui,
        mid360_visualize=args.mid360_visualize,
        path_length=args.path_length,
        leash_length=args.leash_length,
        robot_speed=args.robot_speed,
        fps=args.fps,
        inference_steps=args.inference_steps,
        turn_gain=args.turn_gain,
        curvature_slowdown=not args.no_curvature_slowdown,
        curvature_scale=args.curvature_scale,
        min_speed_scale=args.min_speed_scale,
        log_path=None if args.no_log else args.log_dir / f"planning_{run_timestamp}.jsonl",
        eval_path=(
            args.log_dir / f"planning_eval_{run_timestamp}.jsonl"
            if args.eval
            else None
        ),
        log_interval=args.log_interval,
        collect_enabled=args.collect,
        safety_mode=args.safety_mode,
        safety_margin=args.safety_margin,
        safety_alpha=args.safety_alpha,
        safety_max_constraints=args.safety_max_constraints,
        safety_influence_distance=args.safety_influence_distance,
        debug_preview=args.debug_preview,
        debug_preview_limit=args.debug_preview_limit,
        debug_policy=args.debug_policy,
        debug_qp_log=not args.no_debug_qp_log,
    )
    planner.run()


if __name__ == "__main__":
    main()
