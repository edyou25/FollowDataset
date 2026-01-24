#!/usr/bin/env python3
"""
RL learning entrypoint that reuses FollowDataset/planning.py simulation (headless).

This trains a simple actor-critic policy (PPO) to output low-level controls:
    action = [forward, turn] in [-1, 1]^2

Example:
    python3 FollowDataset/learning.py train --total-steps 200000 --save-dir FollowDataset/rl_models
    python3 FollowDataset/learning.py train --total-steps 200000 --wandb --wandb-mode offline
    python3 FollowDataset/learning.py train --total-steps 200000 --robust-avoidance --wandb --wandb-mode offline
    python3 FollowDataset/learning.py eval --ckpt FollowDataset/rl_models/ppo_*.pt --episodes 10

Fine-tune (residual RL) on top of an existing diffusion_policy checkpoint:
    python3 FollowDataset/learning.py train \\
        --base-ckpt /home/yyf/IROS2026/diffusion_policy/data/outputs/2026.01.21/14.14.46_train_diffusion_unet_lowdim_guide_guide_lowdim/checkpoints/epoch=0090-test_mean_score=0.630.ckpt \\
        --base-device cuda:0 \\
        --base-inference-steps 64 \\
        --residual-scale 0.10 \\
        --init-log-std -2.0 \\
        --device cuda:0 \\
        --total-steps 200000
    python3 FollowDataset/learning.py train \\
        --base-ckpt /home/yyf/IROS2026/diffusion_policy/data/outputs/2026.01.21/15.00.42_train_diffusion_transformer_lowdim_guide_guide_lowdim/checkpoints/latest.ckpt \\
        --base-device cuda:0 \\
        --base-inference-steps 64 \\
        --residual-scale 0.10 \\
        --init-log-std -2.0 \\
        --device cuda:0 \\
        --total-steps 200000
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Local imports (run from repo root or FollowDataset/)
import sys

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from planning import ModelPlanner
from src.path_generator import PathGenerator
from src.rl_policy import ActorCritic


@dataclass
class RewardConfig:
    progress_w: float = 1.0
    deviation_w: float = 0.3
    robot_deviation_w: float = 0.0
    tension_w: float = 0.1
    action_w: float = 0.01
    clearance_w: float = 0.0
    clearance_margin: float = 0.2
    collision_penalty: float = 10.0
    goal_bonus: float = 10.0
    target_tension: float = 0.7


class GuideFollowEnv:
    """
    A headless env wrapper around ModelPlanner's simulation utilities.

    Step frequency is the "data step" (every frame_stride physics steps).
    """

    def __init__(
        self,
        *,
        base_ckpt: Optional[Path] = None,
        base_device: str = "auto",
        base_use_ema: bool = True,
        base_inference_steps: int = 8,
        residual_scale: float = 0.25,
        residual_boost_margin: float = 0.0,
        residual_boost_gain: float = 0.0,
        residual_boost_max: float = 3.0,
        residual_boost_horizon: int = 1,
        path_length: float = 50.0,
        leash_length: float = 1.5,
        robot_speed: float = 1.5,
        fps: int = 20,
        frame_stride: int = 5,
        n_obs_steps: int = 1,
        n_lookahead: int = 10,
        lookahead_stride: int = 5,
        n_obstacle_circles: int = 14,
        n_obstacle_segments: int = 12,
        max_steps: int = 300,
        seed: Optional[int] = None,
        reward: Optional[RewardConfig] = None,
    ):
        self.reward = reward or RewardConfig()
        self.max_steps = int(max_steps)
        self._base_seed = None if seed is None else int(seed)
        self._episode_idx = 0
        self._step_idx = 0
        self.base_ckpt = None if base_ckpt is None else Path(base_ckpt)
        self.uses_base_policy = self.base_ckpt is not None
        self.residual_scale = float(residual_scale)
        self.residual_boost_margin = float(residual_boost_margin)
        self.residual_boost_gain = float(residual_boost_gain)
        self.residual_boost_max = float(residual_boost_max)
        self.residual_boost_horizon = max(1, int(residual_boost_horizon))
        self._base_action_seq: Optional[np.ndarray] = None
        self._base_action_idx = 0
        self._base_steps_since_inference = 0
        self._base_inference_interval = 1

        # Use ModelPlanner as a convenient container for physics + obs building.
        if self.base_ckpt is not None and not self.base_ckpt.exists():
            raise FileNotFoundError(f"base_ckpt not found: {self.base_ckpt}")

        self.planner = ModelPlanner(
            checkpoint_path=self.base_ckpt,
            device=base_device if self.base_ckpt is not None else "cpu",
            use_ema=bool(base_use_ema),
            action_mode=None if self.base_ckpt is not None else "forward_heading",
            k_lookahead=None if self.base_ckpt is not None else lookahead_stride,
            frame_stride=None if self.base_ckpt is not None else int(frame_stride),
            path_length=float(path_length),
            leash_length=float(leash_length),
            robot_speed=float(robot_speed),
            fps=int(fps),
            inference_steps=int(base_inference_steps) if self.base_ckpt is not None else 1,
            create_visualizer=False,
            collision_behavior="reset",
            verbose=False,
        )

        if self.base_ckpt is not None:
            if self.planner.policy is None:
                raise RuntimeError(f"Failed to load base policy from: {self.base_ckpt}")
        else:
            # Override observation configuration to include obstacles, matching PathGenerator defaults.
            self.planner.n_obstacle_circles = int(n_obstacle_circles)
            self.planner.n_obstacle_segments = int(n_obstacle_segments)
            self.planner.obstacle_include_radius = True
            self.planner.obstacle_include_human_clearance = False
            self.planner.segment_repr = "endpoints"
            self.planner.n_lookahead = int(n_lookahead)
            self.planner.lookahead_stride = max(1, int(lookahead_stride))

            if int(n_obs_steps) != int(self.planner.n_obs_steps):
                from collections import deque

                self.planner.n_obs_steps = int(n_obs_steps)
                self.planner.obs_history = deque(maxlen=self.planner.n_obs_steps)

        self.path_generator = PathGenerator(target_length=float(path_length))
        self._path_s: Optional[np.ndarray] = None
        self._path_points: Optional[np.ndarray] = None
        self._s_end: float = 0.0
        self._prev_s: float = 0.0
        if self.uses_base_policy:
            self._base_inference_interval = max(1, int(getattr(self.planner, "n_action_steps", 2)) // 2)

    @property
    def obs_dim(self) -> int:
        # Flatten obs_history: (n_obs_steps, obs_dim) -> (n_obs_steps * obs_dim,)
        obs = self._get_obs_vector()
        return int(obs.shape[0])

    def _maybe_seed(self):
        if self._base_seed is None:
            return
        # NOTE: PathGenerator uses global np.random; this is "good enough" for a single-env trainer.
        np.random.seed(self._base_seed + int(self._episode_idx))

    def reset(self) -> np.ndarray:
        self._maybe_seed()
        self._episode_idx += 1
        self._step_idx = 0
        self._base_action_seq = None
        self._base_action_idx = 0
        self._base_steps_since_inference = 0

        path_data = self.path_generator.generate()
        self.planner.set_path_data(path_data, reset=True)

        # Cache path arclength for progress reward.
        self._path_points = np.asarray(self.planner.current_path_data["path"], dtype=np.float32)
        path_s = self.planner.current_path_data.get("_path_s")
        if path_s is None or len(path_s) != len(self._path_points):
            self.planner._precompute_frenet_cache()
            path_s = self.planner.current_path_data.get("_path_s")
        self._path_s = np.asarray(path_s, dtype=np.float32) if path_s is not None else None
        self._s_end = float(self._path_s[-1]) if self._path_s is not None and len(self._path_s) > 0 else 0.0

        # Initialize progress tracker using current human position.
        self._prev_s, _ = self._compute_progress()
        return self._get_obs_vector()

    def _nearest_path_index(self, point: np.ndarray) -> int:
        assert self._path_points is not None and len(self._path_points) > 0
        diffs = self._path_points - point.astype(np.float32)
        dist_sq = np.sum(diffs * diffs, axis=1)
        return int(np.argmin(dist_sq))

    def _compute_progress(self) -> Tuple[float, float]:
        """Returns (s_human, deviation_m)."""
        if self._path_points is None or self._path_s is None or len(self._path_s) == 0:
            return 0.0, 0.0
        human_pos = self.planner.physics.human.position.astype(np.float32)
        idx = self._nearest_path_index(human_pos)
        s = float(self._path_s[idx])
        deviation = float(np.linalg.norm(self._path_points[idx] - human_pos))
        return s, deviation

    @staticmethod
    def _point_segment_dist_sq(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom < 1e-12:
            diff = point - a
            return float(np.dot(diff, diff))
        t = float(np.dot(point - a, ab)) / denom
        t = float(np.clip(t, 0.0, 1.0))
        closest = a + t * ab
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

    def _predict_min_clearance(self, forward: float, turn: float, *, obstacles, segments) -> float:
        if (obstacles is None or len(obstacles) == 0) and (segments is None or len(segments) == 0):
            return float("inf")
        sim = copy.deepcopy(self.planner.physics)
        sim.set_control(float(forward), float(turn))
        min_clear = float("inf")
        n_steps = int(self.planner.frame_stride) * int(max(1, int(self.residual_boost_horizon)))
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

    def _get_obs_vector(self) -> np.ndarray:
        # Use last observation in history; keep history for compatibility if n_obs_steps > 1.
        if len(self.planner.obs_history) == 0:
            obs = self.planner._build_obs(
                self.planner.physics.robot.position,
                self.planner.physics.human.position,
                self.planner.physics.robot.heading,
            )
            self.planner._seed_obs_history(self.planner.physics.robot.position, self.planner.physics.human.position)
        obs_seq = np.stack(self.planner.obs_history, axis=0).astype(np.float32)
        return obs_seq.reshape(-1)

    def _pop_base_action(self) -> np.ndarray:
        if not self.uses_base_policy:
            raise RuntimeError("Base policy is not enabled")
        if (
            self._base_action_seq is None
            or self._base_action_idx >= len(self._base_action_seq)
            or self._base_steps_since_inference >= int(self._base_inference_interval)
        ):
            self._base_action_seq = self.planner._predict_action()
            self._base_action_idx = 0
            self._base_steps_since_inference = 0
        else:
            self._base_steps_since_inference += 1

        if self._base_action_seq is None or len(self._base_action_seq) == 0:
            return np.zeros((2,), dtype=np.float32)
        action = self._base_action_seq[self._base_action_idx]
        self._base_action_idx += 1
        if self._base_action_idx >= len(self._base_action_seq):
            self._base_action_idx = len(self._base_action_seq) - 1
        return np.asarray(action, dtype=np.float32)

    def _base_action_to_control(self, action: np.ndarray) -> Tuple[float, float]:
        if self.planner.action_mode == "forward_heading":
            forward_delta = float(action[0])
            heading_delta = float(action[1])
            turn_speed = float(self.planner.physics.turn_speed)
            robot_speed = float(self.planner.physics.robot_speed)
            turn_delta = heading_delta * float(self.planner.turn_gain)
            turn = turn_delta / (turn_speed * self.planner.data_dt) if turn_speed > 0 else 0.0
            forward = forward_delta / (robot_speed * self.planner.data_dt) if robot_speed > 0 else 0.0
            if self.planner.curvature_slowdown and turn_speed > 0:
                max_turn = turn_speed * self.planner.data_dt
                if max_turn > 1e-6:
                    ratio = min(1.0, abs(turn_delta) / max_turn)
                    speed_scale = max(
                        float(self.planner.min_speed_scale),
                        1.0 - float(self.planner.curvature_scale) * ratio,
                    )
                    forward *= speed_scale
            forward = float(np.clip(forward, -1.0, 1.0))
            turn = float(np.clip(turn, -1.0, 1.0))
            return forward, turn

        delta = self.planner._action_to_delta(action, self.planner.physics.robot.position)
        forward, turn = self.planner._delta_to_control(
            delta, self.planner.physics.robot.heading, dt=self.planner.data_dt
        )
        return float(forward), float(turn)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict[str, Any]]:
        a0 = float(np.clip(float(action[0]), -1.0, 1.0))
        a1 = float(np.clip(float(action[1]), -1.0, 1.0))

        obstacles = self.planner.current_path_data.get("obstacles") if self.planner.current_path_data else None
        segments = (
            self.planner.current_path_data.get("segment_obstacles") if self.planner.current_path_data else None
        )

        base_forward = 0.0
        base_turn = 0.0
        residual_scale_eff = 0.0
        if self.uses_base_policy:
            base_action = self._pop_base_action()
            base_forward, base_turn = self._base_action_to_control(base_action)
            residual_scale_eff = float(self.residual_scale)
            if self.residual_boost_gain > 0.0 and self.residual_boost_margin > 0.0:
                clearance = self._predict_min_clearance(base_forward, base_turn, obstacles=obstacles, segments=segments)
                violation = (
                    max(0.0, float(self.residual_boost_margin) - float(clearance))
                    if math.isfinite(clearance)
                    else 0.0
                )
                boost = 1.0 + float(self.residual_boost_gain) * (
                    violation / float(self.residual_boost_margin)
                )
                boost = min(float(self.residual_boost_max), max(1.0, float(boost)))
                residual_scale_eff = residual_scale_eff * boost
            d_forward = float(residual_scale_eff) * a0
            d_turn = float(residual_scale_eff) * a1
            forward = float(np.clip(base_forward + d_forward, -1.0, 1.0))
            turn = float(np.clip(base_turn + d_turn, -1.0, 1.0))
            action_cost = d_forward * d_forward + d_turn * d_turn
        else:
            forward = a0
            turn = a1
            action_cost = forward * forward + turn * turn

        collided = False
        collision_info = None

        min_clearance_robot = float("inf")
        min_clearance_human = float("inf")
        for _ in range(int(self.planner.frame_stride)):
            self.planner.physics.set_control(forward, turn)
            robot_state, human_state = self.planner.physics.step()
            self.planner.robot_trajectory.append(robot_state.position.copy())
            self.planner.human_trajectory.append(human_state.position.copy())
            if self.planner.scorer is not None:
                self.planner.scorer.update(robot_state.position, human_state.position)
            self.planner.frame_count += 1

            # Track minimum clearance within this control window (useful for safety shaping).
            min_clearance_robot = min(
                min_clearance_robot,
                self._min_clearance_to_obstacles(
                    robot_state.position,
                    agent_radius=float(self.planner.physics.robot_radius),
                    obstacles=obstacles,
                    segment_obstacles=segments,
                ),
            )
            min_clearance_human = min(
                min_clearance_human,
                self._min_clearance_to_obstacles(
                    human_state.position,
                    agent_radius=float(self.planner.physics.human_radius),
                    obstacles=obstacles,
                    segment_obstacles=segments,
                ),
            )

            collided, collision_info = self.planner.physics.check_collision(
                obstacles, segment_obstacles=segments
            )
            if collided:
                break

        # Update observation history at data rate.
        robot_pos = self.planner.physics.robot.position
        human_pos = self.planner.physics.human.position
        heading = self.planner.physics.robot.heading
        obs = self.planner._build_obs(robot_pos, human_pos, heading)
        self.planner.obs_history.append(obs)
        self.planner.prev_robot_pos = robot_pos.copy()
        self.planner.data_step_idx += 1

        # Reward terms.
        s_human, deviation = self._compute_progress()
        progress = max(0.0, s_human - float(self._prev_s))
        self._prev_s = float(s_human)

        leash_dist = float(np.linalg.norm(robot_pos - human_pos))
        tension = leash_dist / float(self.planner.leash_length)
        tension_err = abs(tension - float(self.reward.target_tension))

        reward = 0.0
        reward += float(self.reward.progress_w) * progress
        reward -= float(self.reward.deviation_w) * deviation
        robot_deviation = 0.0
        if self._path_points is not None and len(self._path_points) > 0:
            idx_r = self._nearest_path_index(robot_pos.astype(np.float32))
            robot_deviation = float(np.linalg.norm(self._path_points[idx_r] - robot_pos.astype(np.float32)))
        reward -= float(self.reward.robot_deviation_w) * float(robot_deviation)
        reward -= float(self.reward.tension_w) * tension_err
        reward -= float(self.reward.action_w) * float(action_cost)

        # Penalize being too close to obstacles/walls (dense safety shaping).
        clearance_min = min(min_clearance_robot, min_clearance_human)
        clearance_margin = float(self.reward.clearance_margin)
        clearance_violation = max(0.0, clearance_margin - float(clearance_min))
        clearance_penalty = 0.0
        if not collided and float(self.reward.clearance_w) > 0.0 and clearance_margin > 0.0:
            clearance_penalty = float(self.reward.clearance_w) * (clearance_violation**2)
            reward -= clearance_penalty

        done = False
        done_reason = None

        if collided:
            done = True
            done_reason = "collision"
            reward -= float(self.reward.collision_penalty)

        # Goal reached (by human progress).
        if not done and self._path_s is not None and s_human >= self._s_end - 1e-3:
            done = True
            done_reason = "goal"
            reward += float(self.reward.goal_bonus)

        self._step_idx += 1
        if not done and self._step_idx >= self.max_steps:
            done = True
            done_reason = "timeout"

        info = {
            "progress": float(progress),
            "s_human": float(s_human),
            "s_end": float(self._s_end),
            "deviation": float(deviation),
            "robot_deviation": float(robot_deviation),
            "tension": float(tension),
            "base_forward": float(base_forward),
            "base_turn": float(base_turn),
            "control_forward": float(forward),
            "control_turn": float(turn),
            "residual_scale_eff": float(residual_scale_eff),
            "clearance_robot": float(np.clip(min_clearance_robot, -10.0, 10.0)),
            "clearance_human": float(np.clip(min_clearance_human, -10.0, 10.0)),
            "clearance_min": float(np.clip(clearance_min, -10.0, 10.0)),
            "clearance_violation": float(clearance_violation),
            "clearance_penalty": float(clearance_penalty),
            "collided": bool(collided),
            "done_reason": done_reason,
            "collision_info": collision_info,
            "step": int(self._step_idx),
        }
        return self._get_obs_vector(), float(reward), bool(done), info


def resolve_device(device: str) -> torch.device:
    device = str(device or "auto")
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def write_jsonl(fp, payload: dict):
    fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
    fp.flush()


def build_env_cfg(
    args: argparse.Namespace,
    env: "GuideFollowEnv",
    reward_cfg: RewardConfig,
    *,
    init_log_std: float,
) -> dict:
    return {
        "base_ckpt": str(args.base_ckpt) if args.base_ckpt is not None else None,
        "base_use_ema": bool(not args.base_no_ema),
        "base_inference_steps": int(args.base_inference_steps),
        "residual_scale": float(args.residual_scale),
        "residual_boost_margin": float(args.residual_boost_margin),
        "residual_boost_gain": float(args.residual_boost_gain),
        "residual_boost_max": float(args.residual_boost_max),
        "residual_boost_horizon": int(args.residual_boost_horizon),
        "init_log_std": float(init_log_std),
        "path_length": float(args.path_length),
        "leash_length": float(args.leash_length),
        "robot_speed": float(args.robot_speed),
        "fps": int(args.fps),
        "frame_stride": int(env.planner.frame_stride),
        "n_obs_steps": int(env.planner.n_obs_steps),
        "n_lookahead": int(env.planner.n_lookahead),
        "k_lookahead": int(env.planner.lookahead_stride),
        "n_obstacle_circles": int(env.planner.n_obstacle_circles),
        "n_obstacle_segments": int(env.planner.n_obstacle_segments),
        "max_steps": int(args.max_steps),
        "reward": reward_cfg.__dict__,
    }


def maybe_init_wandb(
    args: argparse.Namespace,
    *,
    run_id: str,
    wandb_dir: Path,
    config: dict,
):
    if not bool(getattr(args, "wandb", False)):
        return None
    try:
        import wandb  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "wandb is not available. Install it (e.g., `pip install wandb`) or rerun without `--wandb`."
        ) from exc

    tags = None
    if getattr(args, "wandb_tags", None):
        tags = [t for t in str(args.wandb_tags).split(",") if t.strip()]

    wandb_dir = Path(wandb_dir)
    wandb_dir.mkdir(parents=True, exist_ok=True)
    return wandb.init(
        dir=str(wandb_dir),
        project=getattr(args, "wandb_project", None),
        entity=getattr(args, "wandb_entity", None),
        name=getattr(args, "wandb_name", None) or run_id,
        group=getattr(args, "wandb_group", None),
        tags=tags,
        mode=getattr(args, "wandb_mode", "offline"),
        config=config,
    )


def train(args: argparse.Namespace) -> Path:
    device = resolve_device(args.device)
    init_log_std = args.init_log_std
    if init_log_std is None:
        init_log_std = -2.0 if args.base_ckpt is not None else 0.0

    if getattr(args, "robust_avoidance", False):
        args.rw_collision = max(float(args.rw_collision), 100.0)
        args.rw_clearance = max(float(args.rw_clearance), 5.0)
        args.clearance_margin = max(float(args.clearance_margin), 0.3)
        args.rw_robot_deviation = max(float(args.rw_robot_deviation), 0.5)
        args.rw_action = min(float(args.rw_action), 0.005)
        args.residual_scale = max(float(args.residual_scale), 0.25)
        args.residual_boost_margin = max(float(args.residual_boost_margin), float(args.clearance_margin))
        args.residual_boost_gain = max(float(args.residual_boost_gain), 5.0)
        args.residual_boost_max = max(float(args.residual_boost_max), 3.0)
        args.residual_boost_horizon = max(int(args.residual_boost_horizon), 3)
        args.ent_coef = max(float(args.ent_coef), 0.01)

    reward_cfg = RewardConfig(
        progress_w=args.rw_progress,
        deviation_w=args.rw_deviation,
        robot_deviation_w=args.rw_robot_deviation,
        tension_w=args.rw_tension,
        action_w=args.rw_action,
        clearance_w=args.rw_clearance,
        clearance_margin=args.clearance_margin,
        collision_penalty=args.rw_collision,
        goal_bonus=args.rw_goal,
        target_tension=args.target_tension,
    )
    env = GuideFollowEnv(
        base_ckpt=args.base_ckpt,
        base_device=args.base_device,
        base_use_ema=not args.base_no_ema,
        base_inference_steps=args.base_inference_steps,
        residual_scale=args.residual_scale,
        residual_boost_margin=args.residual_boost_margin,
        residual_boost_gain=args.residual_boost_gain,
        residual_boost_max=args.residual_boost_max,
        residual_boost_horizon=args.residual_boost_horizon,
        path_length=args.path_length,
        leash_length=args.leash_length,
        robot_speed=args.robot_speed,
        fps=args.fps,
        frame_stride=args.frame_stride,
        n_obs_steps=args.n_obs_steps,
        n_lookahead=args.n_lookahead,
        lookahead_stride=args.k_lookahead,
        n_obstacle_circles=args.n_obstacle_circles,
        n_obstacle_segments=args.n_obstacle_segments,
        max_steps=args.max_steps,
        seed=args.seed,
        reward=reward_cfg,
    )

    policy = ActorCritic(
        env.obs_dim,
        hidden_dim=args.hidden_dim,
        action_dim=2,
        init_log_std=float(init_log_std),
    ).to(device)
    if env.uses_base_policy and args.zero_init_residual:
        last = policy.actor[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)

    obs = env.reset()
    obs_t = torch.from_numpy(obs).to(device=device, dtype=torch.float32)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = save_dir / f"ppo_{run_id}.pt"

    log_dir = None if args.log_dir is None else Path(args.log_dir)
    log_fp = None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_fp = (log_dir / f"learning_{run_id}.jsonl").open("w", encoding="utf-8")

    env_cfg = build_env_cfg(args, env, reward_cfg, init_log_std=float(init_log_std))
    default_wandb_dir = save_dir / "logs"
    wandb_dir = Path(args.wandb_dir) if getattr(args, "wandb_dir", None) is not None else default_wandb_dir
    wandb_run = maybe_init_wandb(
        args,
        run_id=run_id,
        wandb_dir=wandb_dir,
        config={
            "algo": "ppo",
            "run_id": run_id,
            "ckpt_path": str(ckpt_path),
            "device": str(device),
            "total_steps": int(args.total_steps),
            "rollout_steps": int(args.rollout_steps),
            "update_epochs": int(args.update_epochs),
            "minibatch_size": int(args.minibatch_size),
            "gamma": float(args.gamma),
            "gae_lambda": float(args.gae_lambda),
            "clip_coef": float(args.clip_coef),
            "ent_coef": float(args.ent_coef),
            "vf_coef": float(args.vf_coef),
            "max_grad_norm": float(args.max_grad_norm),
            "lr": float(args.lr),
            "hidden_dim": int(args.hidden_dim),
            "env_cfg": env_cfg,
        },
    )

    # Storage for one rollout.
    rollout_steps = int(args.rollout_steps)
    obs_buf = np.zeros((rollout_steps, env.obs_dim), dtype=np.float32)
    actions_buf = np.zeros((rollout_steps, 2), dtype=np.float32)
    logp_buf = np.zeros((rollout_steps,), dtype=np.float32)
    rewards_buf = np.zeros((rollout_steps,), dtype=np.float32)
    dones_buf = np.zeros((rollout_steps,), dtype=np.float32)
    values_buf = np.zeros((rollout_steps,), dtype=np.float32)

    global_step = 0
    episode_return = 0.0
    episode_len = 0
    episode_idx = 0
    start_time = time.time()

    num_updates = max(1, int(args.total_steps) // rollout_steps)

    for update in range(num_updates):
        for t in range(rollout_steps):
            obs_buf[t] = obs_t.detach().cpu().numpy()
            with torch.no_grad():
                action_t, logp_t, _entropy_t, value_t = policy.get_action_and_value(obs_t[None, :])
            action_np = action_t[0].detach().cpu().numpy().astype(np.float32)

            next_obs, reward, done, info = env.step(action_np)

            actions_buf[t] = action_np
            logp_buf[t] = float(logp_t[0].detach().cpu().item())
            rewards_buf[t] = float(reward)
            dones_buf[t] = 1.0 if done else 0.0
            values_buf[t] = float(value_t[0].detach().cpu().item())

            episode_return += float(reward)
            episode_len += 1
            global_step += 1

            if done:
                episode_idx += 1
                if args.print_interval > 0 and episode_idx % int(args.print_interval) == 0:
                    elapsed = max(1e-6, time.time() - start_time)
                    sps = int(global_step / elapsed)
                    print(
                        f"ep={episode_idx} return={episode_return:.2f} len={episode_len} "
                        f"done={info.get('done_reason')} "
                        f"s={float(info.get('s_human', 0.0)):.1f}/{float(info.get('s_end', 0.0)):.1f} "
                        f"step={global_step} sps={sps}"
                    )
                if wandb_run is not None:
                    done_reason = str(info.get("done_reason") or "")
                    wandb_run.log(
                        {
                            "episode/return": float(episode_return),
                            "episode/len": int(episode_len),
                            "episode/done_reason": done_reason,
                            "episode/done_collision": 1.0 if done_reason == "collision" else 0.0,
                            "episode/done_goal": 1.0 if done_reason == "goal" else 0.0,
                            "episode/done_timeout": 1.0 if done_reason == "timeout" else 0.0,
                            "episode/s_end": float(info.get("s_end", 0.0)),
                            "episode/s_human": float(info.get("s_human", 0.0)),
                            "safety/clearance_min": float(info.get("clearance_min", 0.0)),
                            "safety/clearance_violation": float(info.get("clearance_violation", 0.0)),
                        },
                        step=int(global_step),
                    )
                if log_fp is not None:
                    write_jsonl(
                        log_fp,
                        {
                            "event": "episode_end",
                            "episode": int(episode_idx),
                            "global_step": int(global_step),
                            "return": float(episode_return),
                            "len": int(episode_len),
                            "done_reason": info.get("done_reason"),
                            "s_end": float(info.get("s_end", 0.0)),
                            "s_human": float(info.get("s_human", 0.0)),
                            "clearance_min": float(info.get("clearance_min", 0.0)),
                            "clearance_violation": float(info.get("clearance_violation", 0.0)),
                        },
                    )
                episode_return = 0.0
                episode_len = 0
                next_obs = env.reset()

            obs_t = torch.from_numpy(next_obs).to(device=device, dtype=torch.float32)

        with torch.no_grad():
            _last_action, _last_logp, _last_entropy, last_value = policy.get_action_and_value(
                obs_t[None, :]
            )
            last_value = float(last_value[0].detach().cpu().item())

        # GAE-Lambda advantage.
        advantages = np.zeros_like(rewards_buf, dtype=np.float32)
        lastgaelam = 0.0
        for t in reversed(range(rollout_steps)):
            next_nonterminal = 1.0 - dones_buf[t]
            next_value = last_value if t == rollout_steps - 1 else float(values_buf[t + 1])
            delta = rewards_buf[t] + args.gamma * next_value * next_nonterminal - values_buf[t]
            lastgaelam = delta + args.gamma * args.gae_lambda * next_nonterminal * lastgaelam
            advantages[t] = lastgaelam
        returns = advantages + values_buf

        # Normalize advantages.
        adv_mean = float(np.mean(advantages))
        adv_std = float(np.std(advantages) + 1e-8)
        advantages = (advantages - adv_mean) / adv_std

        b_obs = torch.from_numpy(obs_buf).to(device=device, dtype=torch.float32)
        b_actions = torch.from_numpy(actions_buf).to(device=device, dtype=torch.float32)
        b_logp = torch.from_numpy(logp_buf).to(device=device, dtype=torch.float32)
        b_adv = torch.from_numpy(advantages).to(device=device, dtype=torch.float32)
        b_returns = torch.from_numpy(returns).to(device=device, dtype=torch.float32)
        b_values = torch.from_numpy(values_buf).to(device=device, dtype=torch.float32)

        batch_size = rollout_steps
        minibatch_size = max(1, int(args.minibatch_size))
        n_minibatch = max(1, batch_size // minibatch_size)

        indices = np.arange(batch_size)
        pg_losses = []
        v_losses = []
        entropies = []
        approx_kls = []
        clipfracs = []
        for _epoch in range(int(args.update_epochs)):
            np.random.shuffle(indices)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_idx = indices[start:end]
                mb_obs = b_obs[mb_idx]
                mb_actions = b_actions[mb_idx]
                mb_logp_old = b_logp[mb_idx]
                mb_adv = b_adv[mb_idx]
                mb_returns = b_returns[mb_idx]
                mb_values_old = b_values[mb_idx]

                _a, logp, entropy, value = policy.get_action_and_value(mb_obs, action=mb_actions)
                ratio = torch.exp(logp - mb_logp_old)
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
                pg_loss = torch.mean(torch.max(pg_loss1, pg_loss2))

                v_loss = 0.5 * torch.mean((value - mb_returns) ** 2)
                entropy_mean = torch.mean(entropy)

                loss = pg_loss + args.vf_coef * v_loss - args.ent_coef * entropy_mean

                with torch.no_grad():
                    approx_kl = torch.mean(mb_logp_old - logp)
                    clipfrac = torch.mean((torch.abs(ratio - 1.0) > args.clip_coef).float())
                pg_losses.append(float(pg_loss.detach().cpu().item()))
                v_losses.append(float(v_loss.detach().cpu().item()))
                entropies.append(float(entropy_mean.detach().cpu().item()))
                approx_kls.append(float(approx_kl.detach().cpu().item()))
                clipfracs.append(float(clipfrac.detach().cpu().item()))

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()

        if args.save_every > 0 and (update + 1) % int(args.save_every) == 0:
            torch.save(
                {
                    "algo": "ppo",
                    "run_id": run_id,
                    "global_step": int(global_step),
                    "obs_dim": int(env.obs_dim),
                    "action_dim": 2,
                    "hidden_dim": int(args.hidden_dim),
                    "state_dict": policy.state_dict(),
                    "env_cfg": env_cfg,
                },
                ckpt_path,
            )

        elapsed = max(1e-6, time.time() - start_time)
        sps = int(global_step / elapsed)
        if wandb_run is not None:
            var_y = float(np.var(returns))
            explained_var = 0.0 if var_y < 1e-8 else float(1.0 - np.var(returns - values_buf) / var_y)
            wandb_run.log(
                {
                    "train/update": int(update + 1),
                    "train/global_step": int(global_step),
                    "time/sps": int(sps),
                    "rollout/reward_mean": float(np.mean(rewards_buf)),
                    "rollout/value_mean": float(np.mean(values_buf)),
                    "rollout/adv_mean": float(adv_mean),
                    "rollout/adv_std": float(adv_std),
                    "loss/policy": float(np.mean(pg_losses)) if pg_losses else 0.0,
                    "loss/value": float(np.mean(v_losses)) if v_losses else 0.0,
                    "loss/entropy": float(np.mean(entropies)) if entropies else 0.0,
                    "stats/approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
                    "stats/clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
                    "stats/explained_variance": float(explained_var),
                },
                step=int(global_step),
            )
        if log_fp is not None:
            write_jsonl(
                log_fp,
                {
                    "event": "update_end",
                    "update": int(update + 1),
                    "global_step": int(global_step),
                    "sps": int(sps),
                },
            )

    torch.save(
        {
            "algo": "ppo",
            "run_id": run_id,
            "global_step": int(global_step),
            "obs_dim": int(env.obs_dim),
            "action_dim": 2,
            "hidden_dim": int(args.hidden_dim),
            "state_dict": policy.state_dict(),
            "env_cfg": env_cfg,
        },
        ckpt_path,
    )
    if wandb_run is not None:
        wandb_run.summary["ckpt_path"] = str(ckpt_path)
        wandb_run.summary["global_step"] = int(global_step)
        wandb_run.finish()
    if log_fp is not None:
        log_fp.close()
    print(f"Saved: {ckpt_path}")
    return ckpt_path


@torch.no_grad()
def evaluate(args: argparse.Namespace):
    ckpt = torch.load(Path(args.ckpt).open("rb"), map_location="cpu")
    env_cfg = ckpt.get("env_cfg", {})
    reward_cfg = RewardConfig(**env_cfg.get("reward", {}))

    base_ckpt = args.base_ckpt
    if base_ckpt is None:
        base_ckpt_str = env_cfg.get("base_ckpt")
        base_ckpt = Path(base_ckpt_str) if base_ckpt_str else None
    base_use_ema = bool(env_cfg.get("base_use_ema", True))
    if args.base_no_ema:
        base_use_ema = False
    base_inference_steps = env_cfg.get("base_inference_steps", 8)
    if args.base_inference_steps is not None:
        base_inference_steps = int(args.base_inference_steps)
    residual_scale = env_cfg.get("residual_scale", 0.25)
    if args.residual_scale is not None:
        residual_scale = float(args.residual_scale)

    residual_boost_margin = float(env_cfg.get("residual_boost_margin", 0.0))
    residual_boost_gain = float(env_cfg.get("residual_boost_gain", 0.0))
    residual_boost_max = float(env_cfg.get("residual_boost_max", 3.0))
    residual_boost_horizon = int(env_cfg.get("residual_boost_horizon", 1))

    env = GuideFollowEnv(
        base_ckpt=base_ckpt,
        base_device=args.base_device,
        base_use_ema=base_use_ema,
        base_inference_steps=int(base_inference_steps),
        residual_scale=float(residual_scale),
        residual_boost_margin=float(residual_boost_margin),
        residual_boost_gain=float(residual_boost_gain),
        residual_boost_max=float(residual_boost_max),
        residual_boost_horizon=int(residual_boost_horizon),
        path_length=env_cfg.get("path_length", 50.0),
        leash_length=env_cfg.get("leash_length", 1.5),
        robot_speed=env_cfg.get("robot_speed", 1.5),
        fps=env_cfg.get("fps", 20),
        frame_stride=env_cfg.get("frame_stride", 5),
        n_obs_steps=env_cfg.get("n_obs_steps", 1),
        n_lookahead=env_cfg.get("n_lookahead", 10),
        lookahead_stride=env_cfg.get("k_lookahead", 5),
        n_obstacle_circles=env_cfg.get("n_obstacle_circles", 14),
        n_obstacle_segments=env_cfg.get("n_obstacle_segments", 12),
        max_steps=env_cfg.get("max_steps", 300),
        seed=args.seed,
        reward=reward_cfg,
    )

    device = resolve_device(args.device)
    policy = ActorCritic(
        int(ckpt["obs_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        action_dim=2,
        init_log_std=float(env_cfg.get("init_log_std", 0.0)),
    ).to(device)
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()

    returns = []
    successes = 0
    collisions = 0
    for _ep in range(int(args.episodes)):
        obs = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            obs_t = torch.from_numpy(obs).to(device=device, dtype=torch.float32)[None, :]
            mu = policy.actor(obs_t)
            action = torch.tanh(mu)[0].detach().cpu().numpy().astype(np.float32)
            obs, reward, done, info = env.step(action)
            ep_ret += float(reward)
            if done:
                returns.append(ep_ret)
                if info.get("done_reason") == "goal":
                    successes += 1
                if info.get("done_reason") == "collision":
                    collisions += 1

    mean_ret = float(np.mean(returns)) if returns else 0.0
    print(
        f"episodes={len(returns)} mean_return={mean_ret:.2f} "
        f"success={successes}/{len(returns)} collision={collisions}/{len(returns)}"
    )


@torch.no_grad()
def evaluate_base(args: argparse.Namespace):
    env = GuideFollowEnv(
        base_ckpt=Path(args.base_ckpt),
        base_device=args.base_device,
        base_use_ema=not bool(args.base_no_ema),
        base_inference_steps=int(args.base_inference_steps),
        residual_scale=0.0,
        seed=int(args.seed),
    )

    returns = []
    successes = 0
    collisions = 0
    timeouts = 0
    for _ep in range(int(args.episodes)):
        obs = env.reset()
        done = False
        ep_ret = 0.0
        info = {}
        while not done:
            obs, reward, done, info = env.step(np.zeros((2,), dtype=np.float32))
            ep_ret += float(reward)
        returns.append(ep_ret)
        reason = info.get("done_reason")
        if reason == "goal":
            successes += 1
        elif reason == "collision":
            collisions += 1
        elif reason == "timeout":
            timeouts += 1

    mean_ret = float(np.mean(returns)) if returns else 0.0
    print(
        f"episodes={len(returns)} mean_return={mean_ret:.2f} "
        f"success={successes}/{len(returns)} collision={collisions}/{len(returns)} timeout={timeouts}/{len(returns)}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RL learning (PPO) on planning simulator")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--total-steps", type=int, default=200_000)
    p_train.add_argument("--rollout-steps", type=int, default=2048)
    p_train.add_argument("--update-epochs", type=int, default=10)
    p_train.add_argument("--minibatch-size", type=int, default=256)
    p_train.add_argument("--gamma", type=float, default=0.99)
    p_train.add_argument("--gae-lambda", type=float, default=0.95)
    p_train.add_argument("--clip-coef", type=float, default=0.2)
    p_train.add_argument("--ent-coef", type=float, default=0.0)
    p_train.add_argument("--vf-coef", type=float, default=0.5)
    p_train.add_argument("--max-grad-norm", type=float, default=0.5)
    p_train.add_argument("--lr", type=float, default=3e-4)
    p_train.add_argument("--hidden-dim", type=int, default=256)
    p_train.add_argument(
        "--init-log-std",
        type=float,
        default=None,
        help="Initial log std for Gaussian policy (smaller is less exploratory).",
    )
    p_train.add_argument("--device", default="auto")
    p_train.add_argument("--seed", type=int, default=None)

    # Base policy (residual RL)
    p_train.add_argument(
        "--base-ckpt",
        type=Path,
        default=None,
        help="Diffusion_policy checkpoint to use as baseline (enables residual RL).",
    )
    p_train.add_argument("--base-device", default="auto")
    p_train.add_argument("--base-no-ema", action="store_true", help="Use non-EMA model for base policy")
    p_train.add_argument("--base-inference-steps", type=int, default=64)
    p_train.add_argument("--residual-scale", type=float, default=0.10)
    p_train.add_argument(
        "--residual-boost-margin",
        type=float,
        default=0.0,
        help="When >0, boost residual_scale if predicted clearance falls below this margin (m).",
    )
    p_train.add_argument(
        "--residual-boost-gain",
        type=float,
        default=0.0,
        help="Boost factor slope for residual_scale near obstacles (0 disables boosting).",
    )
    p_train.add_argument(
        "--residual-boost-max",
        type=float,
        default=3.0,
        help="Upper bound on the residual_scale boost multiplier.",
    )
    p_train.add_argument(
        "--residual-boost-horizon",
        type=int,
        default=1,
        help="Predict clearance over this many future control steps when boosting (>=1).",
    )
    p_train.add_argument(
        "--robust-avoidance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply a safety-focused preset for residual RL (stronger collision/clearance + more authority).",
    )
    p_train.add_argument(
        "--zero-init-residual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Zero-initialize actor output layer so residual starts at 0.",
    )

    # env params
    p_train.add_argument("--path-length", type=float, default=50.0)
    p_train.add_argument("--leash-length", type=float, default=1.5)
    p_train.add_argument("--robot-speed", type=float, default=1.5)
    p_train.add_argument("--fps", type=int, default=20)
    p_train.add_argument("--frame-stride", type=int, default=5)
    p_train.add_argument("--n-obs-steps", type=int, default=1)
    p_train.add_argument("--n-lookahead", type=int, default=10)
    p_train.add_argument("--k-lookahead", type=int, default=5)
    p_train.add_argument("--n-obstacle-circles", type=int, default=14)
    p_train.add_argument("--n-obstacle-segments", type=int, default=12)
    p_train.add_argument("--max-steps", type=int, default=300)

    # rewards
    p_train.add_argument("--rw-progress", type=float, default=1.0)
    p_train.add_argument("--rw-deviation", type=float, default=0.3)
    p_train.add_argument(
        "--rw-robot-deviation",
        type=float,
        default=0.0,
        help="Penalty weight for robot deviation from reference path centerline.",
    )
    p_train.add_argument("--rw-tension", type=float, default=0.1)
    p_train.add_argument("--rw-action", type=float, default=0.01)
    p_train.add_argument(
        "--rw-clearance",
        type=float,
        default=0.0,
        help="Penalty weight when robot/human gets closer than --clearance-margin to obstacles/walls.",
    )
    p_train.add_argument(
        "--clearance-margin",
        type=float,
        default=0.2,
        help="Safety margin (m) beyond collision boundary for clearance penalty.",
    )
    p_train.add_argument("--rw-collision", type=float, default=10.0)
    p_train.add_argument("--rw-goal", type=float, default=10.0)
    p_train.add_argument("--target-tension", type=float, default=0.7)

    # io
    p_train.add_argument("--save-dir", type=Path, default=THIS_DIR / "rl_models")
    p_train.add_argument("--save-every", type=int, default=0, help="Save every N updates (0=only final)")
    p_train.add_argument("--log-dir", type=Path, default=THIS_DIR / "logs")
    p_train.add_argument("--print-interval", type=int, default=5)
    p_train.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Log training metrics to Weights & Biases (wandb).",
    )
    p_train.add_argument("--wandb-project", default="followdataset-rl")
    p_train.add_argument("--wandb-entity", default=None)
    p_train.add_argument("--wandb-name", default=None)
    p_train.add_argument("--wandb-group", default=None)
    p_train.add_argument("--wandb-tags", default=None, help="Comma-separated tags, e.g. 'ppo,residual'.")
    p_train.add_argument("--wandb-mode", default="offline", choices=["online", "offline", "disabled"])
    p_train.add_argument(
        "--wandb-dir",
        type=Path,
        default=None,
        help="Directory to store wandb run files (default: <save-dir>/logs).",
    )

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--ckpt", type=Path, required=True)
    p_eval.add_argument("--episodes", type=int, default=10)
    p_eval.add_argument("--device", default="auto")
    p_eval.add_argument("--seed", type=int, default=0)
    p_eval.add_argument("--base-ckpt", type=Path, default=None, help="Override base checkpoint path")
    p_eval.add_argument("--base-device", default="auto")
    p_eval.add_argument("--base-no-ema", action="store_true")
    p_eval.add_argument("--base-inference-steps", type=int, default=None)
    p_eval.add_argument("--residual-scale", type=float, default=None)

    p_base = sub.add_parser("eval-base")
    p_base.add_argument("--base-ckpt", type=Path, required=True)
    p_base.add_argument("--base-device", default="auto")
    p_base.add_argument("--base-no-ema", action="store_true")
    p_base.add_argument("--base-inference-steps", type=int, default=64)
    p_base.add_argument("--episodes", type=int, default=10)
    p_base.add_argument("--seed", type=int, default=0)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.cmd == "train":
        train(args)
        return
    if args.cmd == "eval":
        evaluate(args)
        return
    if args.cmd == "eval-base":
        evaluate_base(args)
        return
    raise RuntimeError(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()
