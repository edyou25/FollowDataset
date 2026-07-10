from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 120,
    }
)


FOLLOWDATASET_DIR = Path(__file__).resolve().parents[1]
if str(FOLLOWDATASET_DIR) not in sys.path:
    sys.path.insert(0, str(FOLLOWDATASET_DIR))

from src.compliance_control import (  # noqa: E402
    ComplianceControlConfig,
    apply_bre_compliance_control,
    apply_interaction_aware_compliance_control,
)
from src.path_generator import PathGenerator  # noqa: E402
from src.physics import PhysicsEngine  # noqa: E402
from src.safety_filter import QPSafetyFilter  # noqa: E402


ARTIFACT_DIR = (
    Path(__file__).resolve().parent
    / "artifacts"
    / "compliance_full_benchmark"
)

MODE_ORDER = ("diffusion", "diffusion_qp", "safe_compliance", "ours")
MODE_LABELS = {
    "diffusion": "diffusion",
    "diffusion_qp": "diffusion+qp",
    "safe_compliance": "safe-compliance",
    "ours": "ours",
}
MODE_COLORS = {
    "diffusion": "#7F1D1D",
    "diffusion_qp": "#B45309",
    "safe_compliance": "#581C87",
    "ours": "#1D4ED8",
}


@dataclass(frozen=True)
class BenchmarkConfig:
    num_cases: int = 48
    seed: int = 202607
    fps: int = 20
    frame_stride: int = 5
    path_length: float = 12.0
    corridor_width: float = 2.20
    obstacle_radius: float = 0.14
    robot_radius: float = 0.22
    human_radius: float = 0.22
    leash_length: float = 1.0
    robot_speed: float = 1.0
    raw_forward_delta: float = 0.18
    max_actions: int = 420
    lookahead_points: int = 16
    goal_fraction: float = 0.88
    tether_target_ratio: float = 0.28
    safety_margin: float = 0.11

    @property
    def sim_dt(self) -> float:
        return 1.0 / float(self.fps)

    @property
    def data_dt(self) -> float:
        return float(self.frame_stride) / float(self.fps)


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    label: str
    path_length: float
    num_segments: int
    corridor_width: float
    obstacle_radius: float
    obstacle_jitter: float
    tether_ratio: float


SCENARIO_SPECS = (
    ScenarioSpec(
        name="regular",
        label="regular",
        path_length=12.0,
        num_segments=4,
        corridor_width=2.35,
        obstacle_radius=0.13,
        obstacle_jitter=0.16,
        tether_ratio=0.24,
    ),
    ScenarioSpec(
        name="cluttered",
        label="cluttered",
        path_length=12.0,
        num_segments=4,
        corridor_width=2.05,
        obstacle_radius=0.16,
        obstacle_jitter=0.22,
        tether_ratio=0.28,
    ),
    ScenarioSpec(
        name="narrow_turns",
        label="narrow turns",
        path_length=13.0,
        num_segments=5,
        corridor_width=2.15,
        obstacle_radius=0.13,
        obstacle_jitter=0.18,
        tether_ratio=0.32,
    ),
    ScenarioSpec(
        name="long_generalization",
        label="long path",
        path_length=14.0,
        num_segments=5,
        corridor_width=2.45,
        obstacle_radius=0.14,
        obstacle_jitter=0.24,
        tether_ratio=0.30,
    ),
)


def scenario_for_case(case_idx: int) -> ScenarioSpec:
    return SCENARIO_SPECS[int(case_idx) % len(SCENARIO_SPECS)]


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def serialize(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return "inf" if value > 0 else "-inf"
    return value


def make_engine(path_data: dict, config: BenchmarkConfig) -> PhysicsEngine:
    engine = PhysicsEngine(
        leash_length=config.leash_length,
        robot_speed=config.robot_speed,
        dt=config.sim_dt,
        robot_radius=config.robot_radius,
        human_radius=config.human_radius,
    )
    start = np.asarray(path_data["start"], dtype=float)
    engine.reset(start)
    first_heading = path_heading_at(path_data["path"], 0)
    engine.robot.heading = first_heading
    engine.human.position = start - np.array(
        [np.cos(first_heading), np.sin(first_heading)],
        dtype=float,
    ) * config.leash_length * 0.8
    engine.human.heading = first_heading
    engine.random_angle = 0.0
    return engine


def make_compliance_config(config: BenchmarkConfig, safety_mode: str) -> ComplianceControlConfig:
    return ComplianceControlConfig(
        data_dt=config.data_dt,
        sim_dt=config.sim_dt,
        frame_stride=config.frame_stride,
        turn_gain=1.0,
        safety_mode=safety_mode,
        curvature_slowdown=False,
        backoff_scales=(1.0, 0.75, 0.5, 0.25, 0.0),
        stop_clearance=0.01,
    )


def path_length(path: np.ndarray) -> float:
    path = np.asarray(path, dtype=np.float32)
    if len(path) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


def path_s(path: np.ndarray) -> np.ndarray:
    path = np.asarray(path, dtype=np.float32)
    if len(path) == 0:
        return np.zeros((0,), dtype=np.float32)
    ds = np.linalg.norm(np.diff(path, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(ds)]).astype(np.float32)


def nearest_path_index(path: np.ndarray, position: np.ndarray) -> int:
    diffs = np.asarray(path, dtype=np.float32) - np.asarray(position, dtype=np.float32)
    return int(np.argmin(np.linalg.norm(diffs, axis=1)))


def path_heading_at(path: np.ndarray, idx: int) -> float:
    path = np.asarray(path, dtype=np.float32)
    idx = int(np.clip(idx, 0, max(0, len(path) - 2)))
    delta = path[idx + 1] - path[idx]
    if float(np.linalg.norm(delta)) < 1e-6 and idx > 0:
        delta = path[idx] - path[idx - 1]
    return float(np.arctan2(delta[1], delta[0]))


def point_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    point = np.asarray(point, dtype=np.float32)
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return float(np.linalg.norm(point - a))
    t = float(np.dot(point - a, ab) / denom)
    t = float(np.clip(t, 0.0, 1.0))
    closest = a + t * ab
    return float(np.linalg.norm(point - closest))


def min_obstacle_clearance(
    path: np.ndarray,
    radius: float,
    circle_obstacles: np.ndarray,
    segment_obstacles: np.ndarray,
) -> float:
    points = np.asarray(path, dtype=np.float32)
    clearances: list[float] = []
    circles = np.asarray(circle_obstacles, dtype=np.float32)
    if len(points) and circles.ndim == 2 and circles.shape[1] >= 3:
        for obs in circles:
            clearances.append(
                float(np.min(np.linalg.norm(points - obs[:2][None, :], axis=1) - radius - obs[2]))
            )
    segments = np.asarray(segment_obstacles, dtype=np.float32)
    if len(points) and segments.ndim == 2 and segments.shape[1] >= 4:
        for seg in segments:
            distances = [
                point_segment_distance(point, seg[:2], seg[2:4])
                for point in points
            ]
            clearances.append(float(np.min(distances) - radius))
    return min(clearances) if clearances else float("inf")


def path_deviation(path: np.ndarray, reference_path: np.ndarray) -> float:
    points = np.asarray(path, dtype=np.float32)
    ref = np.asarray(reference_path, dtype=np.float32)
    if len(points) == 0 or len(ref) == 0:
        return float("inf")
    deviations = []
    for point in points:
        deviations.append(float(np.min(np.linalg.norm(ref - point[None, :], axis=1))))
    return float(np.mean(deviations))


def generate_path_case(case_idx: int, config: BenchmarkConfig) -> dict:
    np.random.seed(config.seed + int(case_idx) * 17)
    scenario = scenario_for_case(case_idx)
    generator = PathGenerator(
        target_length=scenario.path_length,
        num_segments=scenario.num_segments,
        corridor_width=scenario.corridor_width,
        obstacle_radius=scenario.obstacle_radius,
        obstacle_jitter=scenario.obstacle_jitter,
    )
    path_data = generator.generate()
    path_data["scenario"] = {
        "name": scenario.name,
        "label": scenario.label,
        "path_length": scenario.path_length,
        "num_segments": scenario.num_segments,
        "corridor_width": scenario.corridor_width,
        "obstacle_radius": scenario.obstacle_radius,
        "obstacle_jitter": scenario.obstacle_jitter,
        "tether_ratio": scenario.tether_ratio,
    }
    return path_data


def generate_raw_actions(path_data: dict, config: BenchmarkConfig) -> np.ndarray:
    path = np.asarray(path_data["path"], dtype=np.float32)
    engine = make_engine(path_data, config)
    actions: list[np.ndarray] = []

    for _ in range(config.max_actions):
        action = pure_pursuit_action(engine, path, config)
        actions.append(action)

        forward = float(action[0]) / (config.robot_speed * config.data_dt)
        turn = float(action[1]) / (engine.turn_speed * config.data_dt)
        engine.set_control(float(np.clip(forward, -1.0, 1.0)), float(np.clip(turn, -1.0, 1.0)), False)
        for _substep in range(config.frame_stride):
            engine.step()

    return np.asarray(actions, dtype=np.float32)


def pure_pursuit_action(
    engine: PhysicsEngine,
    reference_path: np.ndarray,
    config: BenchmarkConfig,
) -> np.ndarray:
    path = np.asarray(reference_path, dtype=np.float32)
    max_heading_delta = float(engine.turn_speed * config.data_dt)
    idx = nearest_path_index(path, engine.robot.position)
    target_idx = min(len(path) - 1, idx + int(config.lookahead_points))
    target = path[target_idx]
    to_target = target - engine.robot.position.astype(np.float32)
    if float(np.linalg.norm(to_target)) < 1e-5:
        desired_heading = path_heading_at(path, idx)
    else:
        desired_heading = float(np.arctan2(to_target[1], to_target[0]))
    heading_delta = float(
        np.clip(
            wrap_angle(desired_heading - float(engine.robot.heading)),
            -max_heading_delta,
            max_heading_delta,
        )
    )
    return np.array([config.raw_forward_delta, heading_delta], dtype=np.float32)


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if radius <= 0 or not np.any(mask):
        return mask.copy()
    out = mask.copy()
    indices = np.flatnonzero(mask)
    for idx in indices:
        start = max(0, int(idx) - int(radius))
        end = min(len(mask), int(idx) + int(radius) + 1)
        out[start:end] = True
    return out


def generate_interaction_labels(
    actions: np.ndarray,
    case_idx: int,
    config: BenchmarkConfig,
) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    rng = np.random.default_rng(config.seed + 1000 + int(case_idx))
    scenario = scenario_for_case(case_idx)
    target_count = int(round(scenario.tether_ratio * len(actions)))
    interval = max(14, int(round(0.055 * len(actions))))
    leash_mask = np.zeros((len(actions),), dtype=bool)

    turn_scores = np.abs(actions[:, 1])
    ranked_turns = list(np.argsort(turn_scores)[::-1])
    centers: list[int] = []
    for idx in ranked_turns:
        if turn_scores[idx] < 0.04:
            break
        if all(abs(int(idx) - center) > interval for center in centers):
            centers.append(int(idx))
        if len(centers) >= 3:
            break
    while len(centers) < 4:
        frac = (len(centers) + 1) / 5.0
        centers.append(int(round(frac * len(actions) + rng.integers(-8, 9))))

    for center in centers:
        if int(np.sum(leash_mask)) >= target_count:
            break
        start = int(np.clip(center - interval // 2, 0, len(actions)))
        end = int(np.clip(start + interval, 0, len(actions)))
        leash_mask[start:end] = True

    while int(np.sum(leash_mask)) < target_count:
        remaining = target_count - int(np.sum(leash_mask))
        length = min(interval, remaining)
        start = int(rng.integers(0, max(1, len(actions) - length)))
        leash_mask[start:start + length] = True

    true_indices = np.flatnonzero(leash_mask)
    if len(true_indices) > target_count:
        keep = np.sort(rng.choice(true_indices, size=target_count, replace=False))
        leash_mask[:] = False
        leash_mask[keep] = True

    labels = np.where(leash_mask, "leash", "guide").astype(object)
    return labels


def combine_step_stats(accum: dict, step_stats: dict, action_shift: float) -> None:
    accum["total_steps"] += int(step_stats.get("total_steps", 1))
    accum["modified_steps"] += int(action_shift > 1e-5)
    accum["safety_modified_steps"] += int(step_stats.get("safety_modified_steps", 0))
    accum["compliance_steps"] += int(step_stats.get("compliance_steps", 1))
    accum["guide_steps"] += int(step_stats.get("guide_steps", 0))
    accum["constraint_count"] = max(
        int(accum["constraint_count"]),
        int(step_stats.get("constraint_count", 0)),
    )
    accum["min_clearance"] = min(
        float(accum["min_clearance"]),
        float(step_stats.get("min_clearance", float("inf"))),
    )
    accum["action_shift_sum"] += float(action_shift)


def finalize_step_stats(accum: dict, mode: str) -> dict:
    total_steps = max(1, int(accum["total_steps"]))
    min_clearance = float(accum["min_clearance"])
    return {
        "applied": mode != "diffusion",
        "mode": mode,
        "total_steps": int(accum["total_steps"]),
        "modified_steps": int(accum["modified_steps"]),
        "safety_modified_steps": int(accum["safety_modified_steps"]),
        "compliance_steps": int(accum["compliance_steps"]),
        "guide_steps": int(accum["guide_steps"]),
        "constraint_count": int(accum["constraint_count"]),
        "min_clearance": min_clearance,
        "mean_action_shift": float(accum["action_shift_sum"] / float(total_steps)),
    }


def rollout_policy(
    path_data: dict,
    labels: np.ndarray,
    config: BenchmarkConfig,
    mode: str,
    seed: int,
) -> dict:
    np.random.seed(int(seed))
    engine = make_engine(path_data, config)
    ref_path = np.asarray(path_data["path"], dtype=np.float32)
    s_values = path_s(ref_path)
    goal_s = float(s_values[-1]) * float(config.goal_fraction)
    obstacles = np.asarray(path_data.get("obstacles", np.zeros((0, 3))), dtype=np.float32)
    segments = np.asarray(path_data.get("segment_obstacles", np.zeros((0, 4))), dtype=np.float32)
    labels = np.asarray(labels, dtype=object)
    qp = QPSafetyFilter(
        margin=config.safety_margin,
        alpha=1.0,
        max_constraints=10,
        influence_distance=0.9,
    )
    compliance_config = make_compliance_config(config, "human_robot_qp")

    robot_path = [engine.robot.position.copy()]
    human_path = [engine.human.position.copy()]
    selected_actions: list[np.ndarray] = []
    raw_actions: list[np.ndarray] = []
    finish = None
    first_hit = None
    total_substeps = 0
    stats_accum = {
        "total_steps": 0,
        "modified_steps": 0,
        "safety_modified_steps": 0,
        "compliance_steps": 0,
        "guide_steps": 0,
        "constraint_count": 0,
        "min_clearance": float("inf"),
        "action_shift_sum": 0.0,
    }

    for action_idx in range(config.max_actions):
        label = str(labels[min(action_idx, len(labels) - 1)]).lower()
        bre = label in ("leash", "tether")
        raw_action = pure_pursuit_action(engine, ref_path, config)
        action = raw_action.copy()
        raw_actions.append(raw_action.copy())

        if mode == "diffusion_qp":
            result = apply_interaction_aware_compliance_control(
                raw_action[None, :],
                np.asarray([label], dtype=object),
                engine,
                compliance_config,
                safety_filter=qp,
                obstacles=obstacles,
                segment_obstacles=segments,
                bre=bre,
                bre_sequence=np.asarray([bre], dtype=bool),
                compliance_labels=(),
                guide_safety=True,
            )
            action = result.actions[0].astype(np.float32)
            combine_step_stats(
                stats_accum,
                result.stats,
                float(np.linalg.norm(action - raw_action)),
            )
        elif mode == "safe_compliance":
            result = apply_bre_compliance_control(
                raw_action[None, :],
                engine,
                compliance_config,
                safety_filter=qp,
                obstacles=obstacles,
                segment_obstacles=segments,
                bre=True,
            )
            action = result.actions[0].astype(np.float32)
            combine_step_stats(
                stats_accum,
                result.stats,
                float(np.linalg.norm(action - raw_action)),
            )
        elif mode == "ours":
            result = apply_interaction_aware_compliance_control(
                raw_action[None, :],
                np.asarray([label], dtype=object),
                engine,
                compliance_config,
                safety_filter=qp,
                obstacles=obstacles,
                segment_obstacles=segments,
                bre=bre,
                bre_sequence=np.asarray([bre], dtype=bool),
            )
            action = result.actions[0].astype(np.float32)
            combine_step_stats(
                stats_accum,
                result.stats,
                float(np.linalg.norm(action - raw_action)),
            )
        elif mode == "diffusion":
            stats_accum["total_steps"] += 1
            stats_accum["guide_steps"] += int(not bre)
        else:
            raise ValueError(f"Unsupported benchmark mode: {mode}")

        selected_actions.append(action.copy())
        forward = float(action[0]) / (config.robot_speed * config.data_dt)
        turn = float(action[1]) / (engine.turn_speed * config.data_dt)
        engine.set_control(float(np.clip(forward, -1.0, 1.0)), float(np.clip(turn, -1.0, 1.0)), bre)
        for substep_idx in range(config.frame_stride):
            robot_state, human_state = engine.step()
            total_substeps += 1
            robot_path.append(robot_state.position.copy())
            human_path.append(human_state.position.copy())

            collided, info = engine.check_collision(obstacles, segment_obstacles=segments)
            if collided and first_hit is None:
                first_hit = {
                    "action_idx": int(action_idx),
                    "substep_idx": int(substep_idx),
                    "time_sec": float(total_substeps * config.sim_dt),
                    "who": str(info.get("who", "unknown")) if info else "unknown",
                    "type": str(info.get("type", "unknown")) if info else "unknown",
                    "idx": int(info.get("idx", -1)) if info else -1,
                }
                break

            human_idx = nearest_path_index(ref_path, human_state.position)
            if finish is None and float(s_values[human_idx]) >= goal_s:
                finish = {
                    "action_idx": int(action_idx),
                    "substep_idx": int(substep_idx),
                    "time_sec": float(total_substeps * config.sim_dt),
                    "human_s": float(s_values[human_idx]),
                }
                break
        if first_hit is not None or finish is not None:
            break

    robot_arr = np.asarray(robot_path, dtype=np.float32)
    human_arr = np.asarray(human_path, dtype=np.float32)
    reached = finish is not None
    collision = first_hit is not None
    return {
        "robot_path": robot_arr,
        "human_path": human_arr,
        "actions": np.asarray(selected_actions, dtype=np.float32),
        "raw_actions": np.asarray(raw_actions, dtype=np.float32),
        "stats": finalize_step_stats(stats_accum, mode),
        "finish": finish,
        "first_hit": first_hit,
        "reached_goal": bool(reached),
        "collision": bool(collision),
        "success": bool(reached and not collision),
        "duration_sec": float(total_substeps * config.sim_dt),
        "completion_time_sec": None if finish is None else float(finish["time_sec"]),
        "robot_min_clearance": min_obstacle_clearance(
            robot_arr,
            config.robot_radius,
            obstacles,
            segments,
        ),
        "human_min_clearance": min_obstacle_clearance(
            human_arr,
            config.human_radius,
            obstacles,
            segments,
        ),
        "human_path_deviation": path_deviation(human_arr, ref_path),
        "robot_path_length": path_length(robot_arr),
        "human_path_length": path_length(human_arr),
    }


def summarize_case(
    case_idx: int,
    path_data: dict,
    labels: np.ndarray,
    rollouts: dict[str, dict],
    config: BenchmarkConfig,
) -> dict:
    label_counts = {
        "guide": int(np.sum(np.asarray(labels, dtype=object) == "guide")),
        "leash": int(np.sum(np.asarray(labels, dtype=object) == "leash")),
    }
    case = {
        "case_idx": int(case_idx),
        "scenario": serialize(path_data.get("scenario", {})),
        "path_length": float(path_data["length"]),
        "obstacle_count": int(len(path_data.get("obstacles", []))),
        "segment_count": int(len(path_data.get("segment_obstacles", []))),
        "label_counts": label_counts,
        "tether_ratio": float(label_counts["leash"] / max(1, len(labels))),
        "modes": {},
    }
    for mode, rollout in rollouts.items():
        completion = rollout["completion_time_sec"]
        action_seq = np.asarray(rollout["actions"], dtype=np.float32)
        mode_stats = rollout.get("stats", {})
        case["modes"][mode] = {
            "success": bool(rollout["success"]),
            "reached_goal": bool(rollout["reached_goal"]),
            "collision": bool(rollout["collision"]),
            "completion_time_sec": None if completion is None else float(completion),
            "duration_sec": float(rollout["duration_sec"]),
            "robot_min_clearance": float(rollout["robot_min_clearance"]),
            "human_min_clearance": float(rollout["human_min_clearance"]),
            "human_path_deviation": float(rollout["human_path_deviation"]),
            "robot_path_length": float(rollout["robot_path_length"]),
            "human_path_length": float(rollout["human_path_length"]),
            "mean_forward_action": float(np.mean(action_seq[:, 0])),
            "modified_steps": int(mode_stats.get("modified_steps", 0)),
            "compliance_steps": int(mode_stats.get("compliance_steps", 0)),
            "stats": serialize(mode_stats),
        }
    return case


def flatten_case_rows(cases: list[dict]) -> list[dict]:
    rows = []
    for case in cases:
        for mode, item in case["modes"].items():
            rows.append(
                {
                    "case_idx": int(case["case_idx"]),
                    "scenario": str(case.get("scenario", {}).get("name", "unknown")),
                    "scenario_label": str(case.get("scenario", {}).get("label", "unknown")),
                    "mode": mode,
                    "path_length": float(case["path_length"]),
                    "tether_ratio": float(case["tether_ratio"]),
                    "success": int(item["success"]),
                    "reached_goal": int(item["reached_goal"]),
                    "collision": int(item["collision"]),
                    "completion_time_sec": (
                        "" if item["completion_time_sec"] is None else float(item["completion_time_sec"])
                    ),
                    "duration_sec": float(item["duration_sec"]),
                    "robot_min_clearance": float(item["robot_min_clearance"]),
                    "human_min_clearance": float(item["human_min_clearance"]),
                    "human_path_deviation": float(item["human_path_deviation"]),
                    "mean_forward_action": float(item["mean_forward_action"]),
                    "modified_steps": int(item["modified_steps"]),
                    "compliance_steps": int(item["compliance_steps"]),
                }
            )
    return rows


def aggregate_mode_items(items: list[dict]) -> dict:
    success = np.asarray([item["success"] for item in items], dtype=bool)
    reached = np.asarray([item["reached_goal"] for item in items], dtype=bool)
    collisions = np.asarray([item["collision"] for item in items], dtype=bool)
    completion_times = np.asarray(
        [
            item["completion_time_sec"]
            for item in items
            if item["completion_time_sec"] is not None and item["success"]
        ],
        dtype=float,
    )
    durations = np.asarray([item["duration_sec"] for item in items], dtype=float)
    return {
        "success_rate": float(np.mean(success)),
        "reach_rate": float(np.mean(reached)),
        "collision_rate": float(np.mean(collisions)),
        "mean_completion_time_sec": (
            None if len(completion_times) == 0 else float(np.mean(completion_times))
        ),
        "median_completion_time_sec": (
            None if len(completion_times) == 0 else float(np.median(completion_times))
        ),
        "mean_duration_sec": float(np.mean(durations)),
        "mean_human_path_deviation": float(np.mean([item["human_path_deviation"] for item in items])),
        "mean_human_min_clearance": float(np.mean([item["human_min_clearance"] for item in items])),
        "mean_forward_action": float(np.mean([item["mean_forward_action"] for item in items])),
        "mean_modified_steps": float(np.mean([item["modified_steps"] for item in items])),
        "mean_compliance_steps": float(np.mean([item["compliance_steps"] for item in items])),
    }


def aggregate_cases(cases: list[dict], config: BenchmarkConfig) -> dict:
    aggregate = {
        "num_cases": int(len(cases)),
        "config": serialize(config.__dict__),
        "scenario_specs": [serialize(spec.__dict__) for spec in SCENARIO_SPECS],
        "modes": {},
        "scenarios": {},
    }
    for mode in MODE_ORDER:
        items = [case["modes"][mode] for case in cases]
        aggregate["modes"][mode] = aggregate_mode_items(items)

    for spec in SCENARIO_SPECS:
        scenario_cases = [
            case
            for case in cases
            if case.get("scenario", {}).get("name") == spec.name
        ]
        if not scenario_cases:
            continue
        aggregate["scenarios"][spec.name] = {
            "label": spec.label,
            "num_cases": int(len(scenario_cases)),
            "modes": {
                mode: aggregate_mode_items([case["modes"][mode] for case in scenario_cases])
                for mode in MODE_ORDER
            },
        }
    return aggregate


def plot_aggregate(aggregate: dict, output_path: Path) -> None:
    metrics = [
        ("success_rate", "Success rate"),
        ("collision_rate", "Collision rate"),
        ("mean_completion_time_sec", "Completion time [s]"),
        ("mean_compliance_steps", "Compliance steps"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 6.4), constrained_layout=True)
    for ax, (key, ylabel) in zip(axes.ravel(), metrics):
        values = []
        for mode in MODE_ORDER:
            value = aggregate["modes"][mode].get(key)
            values.append(np.nan if value is None else float(value))
        ax.bar(
            [MODE_LABELS[mode] for mode in MODE_ORDER],
            values,
            color=[MODE_COLORS[mode] for mode in MODE_ORDER],
            alpha=0.88,
        )
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        ax.tick_params(axis="x", labelsize=9)
        if key.endswith("_rate"):
            ax.set_ylim(0.0, 1.05)
        for idx, value in enumerate(values):
            if np.isfinite(value):
                text = f"{value:.2f}" if key.endswith("_rate") else f"{value:.1f}"
                ax.text(idx, value, text, ha="center", va="bottom", fontsize=9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_scenario_breakdown(aggregate: dict, output_path: Path) -> None:
    metrics = [
        ("success_rate", "Success rate"),
        ("collision_rate", "Collision rate"),
        ("mean_completion_time_sec", "Completion time [s]"),
        ("mean_compliance_steps", "Compliance steps"),
    ]
    scenario_items = [
        (spec.name, aggregate["scenarios"][spec.name]["label"])
        for spec in SCENARIO_SPECS
        if spec.name in aggregate.get("scenarios", {})
    ]
    x = np.arange(len(scenario_items), dtype=float)
    width = 0.18
    offsets = (np.arange(len(MODE_ORDER), dtype=float) - (len(MODE_ORDER) - 1) / 2.0) * width

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 7.0), constrained_layout=True)
    for ax, (key, ylabel) in zip(axes.ravel(), metrics):
        for mode_idx, mode in enumerate(MODE_ORDER):
            values = []
            for scenario_name, _scenario_label in scenario_items:
                value = aggregate["scenarios"][scenario_name]["modes"][mode].get(key)
                values.append(np.nan if value is None else float(value))
            ax.bar(
                x + offsets[mode_idx],
                values,
                width=width,
                color=MODE_COLORS[mode],
                alpha=0.88,
                label=MODE_LABELS[mode],
            )
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([label for _name, label in scenario_items])
        ax.grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        if key.endswith("_rate"):
            ax.set_ylim(0.0, 1.05)
    axes.ravel()[1].legend(frameon=False, ncol=2, loc="upper right")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_representative_case(
    path_data: dict,
    rollouts: dict[str, dict],
    output_path: Path,
    config: BenchmarkConfig,
    entity: str,
) -> None:
    entity = str(entity).lower()
    if entity not in ("robot", "human"):
        raise ValueError(f"Unsupported representative entity: {entity!r}")

    fig, ax = plt.subplots(figsize=(11.0, 5.8), constrained_layout=True)
    path = np.asarray(path_data["path"], dtype=np.float32)
    obstacles = np.asarray(path_data.get("obstacles", []), dtype=np.float32)
    segments = np.asarray(path_data.get("segment_obstacles", []), dtype=np.float32)
    ax.plot(path[:, 0], path[:, 1], "--", color="#111827", linewidth=1.2, label="reference")
    for seg_idx, seg in enumerate(segments):
        ax.plot(
            [seg[0], seg[2]],
            [seg[1], seg[3]],
            color="#9CA3AF",
            linewidth=1.2,
            label="corridor wall" if seg_idx == 0 else None,
        )
    for obs_idx, obs in enumerate(obstacles):
        ax.add_patch(
            plt.Circle(
                obs[:2],
                obs[2],
                color="#7C2D12",
                alpha=0.18,
                label="circle obstacle" if obs_idx == 0 else None,
            )
        )
    collision_label_added = False
    for mode in MODE_ORDER:
        entity_path = rollouts[mode][f"{entity}_path"]
        ax.plot(
            entity_path[:, 0],
            entity_path[:, 1],
            color=MODE_COLORS[mode],
            linewidth=2.1,
            label=MODE_LABELS[mode],
        )
        hit = rollouts[mode]["first_hit"]
        if hit is not None:
            hit_idx = min(
                len(entity_path) - 1,
                int(hit["action_idx"]) * config.frame_stride + int(hit["substep_idx"]) + 1,
            )
            ax.scatter(
                [entity_path[hit_idx, 0]],
                [entity_path[hit_idx, 1]],
                marker="x",
                s=90,
                color=MODE_COLORS[mode],
                linewidths=2.0,
                label="collision event" if not collision_label_added else None,
            )
            collision_label_added = True
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, color="#E5E7EB", linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_labeled_human_path(
    ax: plt.Axes,
    human_path: np.ndarray,
    labels: np.ndarray,
    config: BenchmarkConfig,
) -> None:
    colors = {"guide": "#2563EB", "leash": "#EA580C", "tether": "#EA580C"}
    names = {"guide": "human path: guide", "leash": "human path: leash", "tether": "human path: leash"}
    used: set[str] = set()
    point_labels = []
    for point_idx in range(len(human_path)):
        action_idx = int(np.clip((point_idx - 1) // config.frame_stride, 0, len(labels) - 1))
        point_labels.append(str(labels[action_idx]).lower())

    start = 0
    while start < len(human_path) - 1:
        label = point_labels[start]
        end = start + 1
        while end < len(human_path) and point_labels[end] == label:
            end += 1
        ax.plot(
            human_path[start:end, 0],
            human_path[start:end, 1],
            color=colors.get(label, "#6B7280"),
            linewidth=2.1,
            alpha=0.88,
            label=names.get(label, label) if label not in used else None,
        )
        used.add(label)
        start = max(end - 1, start + 1)


def paper_scene_case_indices(config: BenchmarkConfig, count: int = 5) -> tuple[int, ...]:
    if config.num_cases <= 0:
        return ()
    if config.num_cases <= count:
        return tuple(range(config.num_cases))
    scenario_seed_indices = list(range(min(len(SCENARIO_SPECS), config.num_cases)))
    extra_needed = max(0, count - len(scenario_seed_indices))
    extra_indices = (
        np.linspace(
            len(scenario_seed_indices),
            config.num_cases - 1,
            num=extra_needed,
            dtype=int,
        ).tolist()
        if extra_needed
        else []
    )
    indices: list[int] = []
    for index in scenario_seed_indices + extra_indices:
        if int(index) not in indices:
            indices.append(int(index))
        if len(indices) == count:
            break
    return tuple(indices)


def plot_scene_environment(
    ax: plt.Axes,
    path_data: dict,
    *,
    reference_alpha: float = 0.45,
) -> list[np.ndarray]:
    path = np.asarray(path_data["path"], dtype=np.float32)
    obstacles = np.asarray(path_data.get("obstacles", []), dtype=np.float32)
    segments = np.asarray(path_data.get("segment_obstacles", []), dtype=np.float32)

    plotted_points = [path]
    ax.plot(
        path[:, 0],
        path[:, 1],
        color="#111827",
        linestyle="--",
        linewidth=0.75,
        alpha=reference_alpha,
        zorder=1,
    )
    for seg in segments:
        ax.plot(
            [seg[0], seg[2]],
            [seg[1], seg[3]],
            color="#9CA3AF",
            linewidth=0.75,
            alpha=0.90,
            zorder=0,
        )
        plotted_points.append(np.asarray(seg, dtype=np.float32).reshape(2, 2))
    for obs in obstacles:
        ax.add_patch(
            plt.Circle(
                obs[:2],
                obs[2],
                facecolor="#B45309",
                edgecolor="#7C2D12",
                linewidth=0.35,
                alpha=0.30,
                zorder=2,
            )
        )
        plotted_points.append(obs[:2][None, :])
    return plotted_points


def plot_labeled_reference_path(
    ax: plt.Axes,
    path: np.ndarray,
    labels: np.ndarray,
) -> None:
    colors = {"guide": "#2563EB", "leash": "#EA580C", "tether": "#EA580C"}
    path = np.asarray(path, dtype=np.float32)
    labels = np.asarray(labels, dtype=object)
    if len(path) < 2 or len(labels) == 0:
        return
    segment_count = len(path) - 1
    for segment_idx in range(segment_count):
        label_idx = int(round(segment_idx * (len(labels) - 1) / max(1, segment_count - 1)))
        label = str(labels[label_idx]).lower()
        ax.plot(
            path[segment_idx:segment_idx + 2, 0],
            path[segment_idx:segment_idx + 2, 1],
            color=colors.get(label, "#6B7280"),
            linewidth=1.65,
            alpha=0.92,
            solid_capstyle="round",
            zorder=3,
        )


def format_dense_scene_axis(
    ax: plt.Axes,
    point_sets: list[np.ndarray],
) -> None:
    finite_sets = [
        np.asarray(points, dtype=float).reshape(-1, 2)
        for points in point_sets
        if np.asarray(points).size
    ]
    if finite_sets:
        points = np.vstack(finite_sets)
        points = points[np.isfinite(points).all(axis=1)]
    else:
        points = np.empty((0, 2), dtype=float)
    if len(points):
        pad = 0.55
        ax.set_xlim(float(points[:, 0].min() - pad), float(points[:, 0].max() + pad))
        ax.set_ylim(float(points[:, 1].min() - pad), float(points[:, 1].max() + pad))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, color="#E5E7EB", linewidth=0.35)
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)
        spine.set_color("#D1D5DB")


def plot_paper_scene_grid(
    examples: list[dict],
    output_path: Path,
    config: BenchmarkConfig,
) -> None:
    if not examples:
        return
    examples = examples[:5]
    col_count = len(examples)
    fig, axes = plt.subplots(
        2,
        col_count,
        figsize=(2.22 * col_count, 4.75),
        constrained_layout=False,
    )
    if col_count == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for col_idx, example in enumerate(examples):
        path_data = example["path_data"]
        labels = np.asarray(example["labels"], dtype=object)
        path = np.asarray(path_data["path"], dtype=np.float32)

        train_ax = axes[0, col_idx]
        train_points = plot_scene_environment(train_ax, path_data, reference_alpha=0.28)
        plot_labeled_reference_path(train_ax, path, labels)
        train_ax.scatter(path[0, 0], path[0, 1], s=12, color="#059669", zorder=5)
        train_ax.scatter(path[-1, 0], path[-1, 1], s=15, marker="*", color="#111827", zorder=5)
        format_dense_scene_axis(
            train_ax,
            train_points,
        )

        exp_ax = axes[1, col_idx]
        rollout = example["rollout"]
        exp_points = plot_scene_environment(exp_ax, path_data, reference_alpha=0.20)
        robot_path = np.asarray(rollout["robot_path"], dtype=np.float32)
        human_path = np.asarray(rollout["human_path"], dtype=np.float32)
        exp_ax.plot(
            robot_path[:, 0],
            robot_path[:, 1],
            color="#111827",
            linewidth=0.95,
            alpha=0.62,
            zorder=3,
        )
        plot_labeled_human_path(exp_ax, human_path, labels, config)
        exp_ax.scatter(human_path[0, 0], human_path[0, 1], s=12, color="#059669", zorder=5)
        exp_ax.scatter(human_path[-1, 0], human_path[-1, 1], s=15, marker="*", color="#111827", zorder=5)
        exp_points.extend([robot_path, human_path])
        format_dense_scene_axis(
            exp_ax,
            exp_points,
        )

    axes[0, 0].set_ylabel("training\nscenes", fontsize=9)
    axes[1, 0].set_ylabel("experiment\nrollouts", fontsize=9)
    legend_handles = [
        Line2D([0], [0], color="#2563EB", linewidth=2.0, label="guide label / human path"),
        Line2D([0], [0], color="#EA580C", linewidth=2.0, label="leash label / human path"),
        Line2D([0], [0], color="#111827", linewidth=1.0, alpha=0.65, label="robot path"),
        Line2D([0], [0], color="#111827", linestyle="--", linewidth=0.9, alpha=0.45, label="reference path"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#B45309", alpha=0.55, label="obstacle"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=7.7,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.subplots_adjust(left=0.045, right=0.995, top=0.985, bottom=0.12, wspace=0.06, hspace=0.20)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def shade_interaction_spans(
    ax: plt.Axes,
    labels: np.ndarray,
    config: BenchmarkConfig,
    duration_sec: float,
) -> None:
    colors = {"guide": "#DBEAFE", "leash": "#FFEDD5", "tether": "#FFEDD5"}
    start = 0
    while start < len(labels):
        label = str(labels[start]).lower()
        end = start + 1
        while end < len(labels) and str(labels[end]).lower() == label:
            end += 1
        x0 = float(start * config.data_dt)
        x1 = min(float(end * config.data_dt), duration_sec)
        if x0 < duration_sec:
            ax.axvspan(x0, x1, color=colors.get(label, "#F3F4F6"), alpha=0.55, linewidth=0)
        start = end


def plot_interaction_aware_collision_diagnosis(
    config: BenchmarkConfig,
    output_path: Path,
) -> dict:
    diagnosis_config = replace(config, safety_margin=0.04)
    for case_idx in range(config.num_cases):
        path_data = generate_path_case(case_idx, diagnosis_config)
        seed_actions = generate_raw_actions(path_data, diagnosis_config)
        labels = generate_interaction_labels(seed_actions, case_idx, diagnosis_config)
        rollout = rollout_policy(
            path_data,
            labels,
            diagnosis_config,
            mode="ours",
            seed=diagnosis_config.seed + case_idx * 97,
        )
        if rollout["collision"]:
            break
    else:
        raise AssertionError("Expected the low-margin diagnosis run to contain one ours collision")

    hit = rollout["first_hit"]
    if hit is None or hit.get("type") != "circle":
        raise AssertionError("Expected a circle-obstacle collision for the diagnosis plot")

    hit_action_idx = int(hit["action_idx"])
    hit_path_idx = min(
        len(rollout["human_path"]) - 1,
        hit_action_idx * diagnosis_config.frame_stride + int(hit["substep_idx"]) + 1,
    )
    obstacles = np.asarray(path_data.get("obstacles", []), dtype=np.float32)
    segments = np.asarray(path_data.get("segment_obstacles", []), dtype=np.float32)
    hit_obstacle = obstacles[int(hit["idx"])]
    human_path = np.asarray(rollout["human_path"], dtype=np.float32)
    robot_path = np.asarray(rollout["robot_path"], dtype=np.float32)
    time = np.arange(len(human_path), dtype=np.float32) * diagnosis_config.sim_dt
    clearance = (
        np.linalg.norm(human_path - hit_obstacle[:2][None, :], axis=1)
        - diagnosis_config.human_radius
        - float(hit_obstacle[2])
    )

    fig, (ax_path, ax_clearance) = plt.subplots(
        1,
        2,
        figsize=(12.0, 5.4),
        constrained_layout=True,
    )

    ref_path = np.asarray(path_data["path"], dtype=np.float32)
    ax_path.plot(ref_path[:, 0], ref_path[:, 1], "--", color="#111827", linewidth=1.0, label="reference")
    for seg_idx, seg in enumerate(segments):
        ax_path.plot(
            [seg[0], seg[2]],
            [seg[1], seg[3]],
            color="#9CA3AF",
            linewidth=1.0,
            label="corridor wall" if seg_idx == 0 else None,
        )
    for obs_idx, obs in enumerate(obstacles):
        is_hit = obs_idx == int(hit["idx"])
        ax_path.add_patch(
            plt.Circle(
                obs[:2],
                obs[2],
                facecolor="#FCA5A5" if is_hit else "#D6D3D1",
                edgecolor="#DC2626" if is_hit else "#A8A29E",
                linewidth=1.5 if is_hit else 0.8,
                alpha=0.60 if is_hit else 0.35,
                label="hit obstacle" if is_hit else ("circle obstacle" if obs_idx == 0 else None),
            )
        )
    ax_path.plot(robot_path[:, 0], robot_path[:, 1], color="#111827", linewidth=1.5, label="robot path")
    plot_labeled_human_path(ax_path, human_path, labels, diagnosis_config)
    ax_path.scatter(
        [human_path[hit_path_idx, 0]],
        [human_path[hit_path_idx, 1]],
        marker="x",
        s=120,
        color="#DC2626",
        linewidths=2.5,
        label="human collision",
        zorder=10,
    )
    ax_path.add_patch(
        plt.Circle(
            human_path[hit_path_idx],
            diagnosis_config.human_radius,
            fill=False,
            edgecolor="#DC2626",
            linestyle="--",
            linewidth=1.3,
        )
    )
    local_start = max(0, hit_path_idx - 110)
    local_end = min(len(human_path), hit_path_idx + 45)
    local_points = np.vstack(
        [
            human_path[local_start:local_end],
            robot_path[local_start:local_end],
            hit_obstacle[:2][None, :],
        ]
    )
    pad = 0.85
    ax_path.set_xlim(float(np.min(local_points[:, 0]) - pad), float(np.max(local_points[:, 0]) + pad))
    ax_path.set_ylim(float(np.min(local_points[:, 1]) - pad), float(np.max(local_points[:, 1]) + pad))
    ax_path.set_aspect("equal", adjustable="box")
    ax_path.set_xlabel("x [m]")
    ax_path.set_ylabel("y [m]")
    ax_path.grid(True, color="#E5E7EB", linewidth=0.8)
    ax_path.legend(frameon=False, fontsize=8, loc="best")

    shade_interaction_spans(ax_clearance, labels, diagnosis_config, float(time[-1]))
    ax_clearance.plot(time, clearance, color="#991B1B", linewidth=2.0, label="human clearance to hit obstacle")
    ax_clearance.axhline(0.0, color="#111827", linewidth=1.0, label="physical collision boundary")
    ax_clearance.axhline(
        diagnosis_config.safety_margin,
        color="#DC2626",
        linestyle="--",
        linewidth=1.1,
        label="old safety margin 0.04 m",
    )
    ax_clearance.axhline(
        config.safety_margin,
        color="#16A34A",
        linestyle="--",
        linewidth=1.1,
        label=f"current safety margin {config.safety_margin:.2f} m",
    )
    transition_indices = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    for transition_idx in transition_indices:
        if abs(int(transition_idx) - hit_action_idx) < 18:
            ax_clearance.axvline(
                float(transition_idx * diagnosis_config.data_dt),
                color="#6B7280",
                linestyle=":",
                linewidth=1.0,
            )
    ax_clearance.axvline(float(hit["time_sec"]), color="#DC2626", linewidth=1.4, label="collision time")
    ax_clearance.set_xlim(max(0.0, float(hit["time_sec"]) - 4.5), float(hit["time_sec"]) + 0.8)
    ax_clearance.set_ylim(min(-0.05, float(np.min(clearance[local_start:local_end])) - 0.02), 0.35)
    ax_clearance.set_xlabel("time [s]")
    ax_clearance.set_ylabel("clearance [m]")
    ax_clearance.grid(True, color="#E5E7EB", linewidth=0.8)
    ax_clearance.legend(frameon=False, fontsize=8, loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return {
        "case_idx": int(case_idx),
        "safety_margin": float(diagnosis_config.safety_margin),
        "current_safety_margin": float(config.safety_margin),
        "first_hit": serialize(hit),
        "hit_action_label": str(labels[hit_action_idx]),
        "hit_clearance_m": float(clearance[hit_path_idx]),
    }


def run_full_benchmark(config: BenchmarkConfig = BenchmarkConfig()) -> dict:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    cases: list[dict] = []
    representative = None
    paper_scene_indices = set(paper_scene_case_indices(config, count=5))
    paper_scene_examples: list[dict] = []

    for case_idx in range(config.num_cases):
        path_data = generate_path_case(case_idx, config)
        seed_actions = generate_raw_actions(path_data, config)
        labels = generate_interaction_labels(seed_actions, case_idx, config)
        rollouts = {
            mode: rollout_policy(
                path_data,
                labels,
                config,
                mode=mode,
                seed=config.seed + case_idx * 97,
            )
            for mode in MODE_ORDER
        }
        case = summarize_case(
            case_idx=case_idx,
            path_data=path_data,
            labels=labels,
            rollouts=rollouts,
            config=config,
        )
        cases.append(case)
        if case_idx in paper_scene_indices:
            paper_scene_examples.append(
                {
                    "case_idx": int(case_idx),
                    "path_data": path_data,
                    "labels": labels.copy(),
                    "rollout": rollouts["ours"],
                }
            )
        if (
            representative is None
            or (
                not representative[1]["diffusion"]["collision"]
                and rollouts["diffusion"]["collision"]
                and rollouts["ours"]["success"]
            )
        ):
            representative = (path_data, rollouts)

    aggregate = aggregate_cases(cases, config)
    rows = flatten_case_rows(cases)

    summary_path = ARTIFACT_DIR / "compliance_full_benchmark_summary.json"
    csv_path = ARTIFACT_DIR / "compliance_full_benchmark_cases.csv"
    aggregate_plot_path = ARTIFACT_DIR / "compliance_full_benchmark_aggregate.png"
    scenario_plot_path = ARTIFACT_DIR / "compliance_full_benchmark_scenarios.png"
    representative_robot_plot_path = ARTIFACT_DIR / "compliance_full_benchmark_representative_robot.png"
    representative_human_plot_path = ARTIFACT_DIR / "compliance_full_benchmark_representative_human.png"
    paper_scene_grid_path = ARTIFACT_DIR / "paper_training_experiment_scene_grid.png"
    diagnosis_plot_path = ARTIFACT_DIR / "ours_collision_diagnosis.png"
    data_path = ARTIFACT_DIR / "compliance_full_benchmark_data.npz"
    diagnosis = plot_interaction_aware_collision_diagnosis(config, diagnosis_plot_path)

    result = {
        "aggregate": aggregate,
        "cases": cases,
        "diagnosis": diagnosis,
        "artifacts": {
            "summary": str(summary_path),
            "csv": str(csv_path),
            "aggregate_plot": str(aggregate_plot_path),
            "scenario_plot": str(scenario_plot_path),
            "representative_robot_plot": str(representative_robot_plot_path),
            "representative_human_plot": str(representative_human_plot_path),
            "paper_scene_grid": str(paper_scene_grid_path),
            "collision_diagnosis_plot": str(diagnosis_plot_path),
            "data": str(data_path),
        },
    }
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(serialize(result), fp, indent=2)
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    plot_aggregate(aggregate, aggregate_plot_path)
    plot_scenario_breakdown(aggregate, scenario_plot_path)
    plot_paper_scene_grid(paper_scene_examples, paper_scene_grid_path, config)
    if representative is not None:
        plot_representative_case(
            path_data=representative[0],
            rollouts=representative[1],
            output_path=representative_robot_plot_path,
            config=config,
            entity="robot",
        )
        plot_representative_case(
            path_data=representative[0],
            rollouts=representative[1],
            output_path=representative_human_plot_path,
            config=config,
            entity="human",
        )

    metric_matrix = np.asarray(
        [
            [
                aggregate["modes"][mode]["success_rate"],
                aggregate["modes"][mode]["collision_rate"],
                np.nan
                if aggregate["modes"][mode]["mean_completion_time_sec"] is None
                else aggregate["modes"][mode]["mean_completion_time_sec"],
                aggregate["modes"][mode]["mean_compliance_steps"],
            ]
            for mode in MODE_ORDER
        ],
        dtype=np.float32,
    )
    scenario_names = [
        spec.name for spec in SCENARIO_SPECS if spec.name in aggregate.get("scenarios", {})
    ]
    metric_keys = ["success_rate", "collision_rate", "mean_completion_time_sec", "mean_compliance_steps"]
    scenario_metric_tensor = np.asarray(
        [
            [
                [
                    np.nan
                    if aggregate["scenarios"][scenario_name]["modes"][mode][metric_key] is None
                    else aggregate["scenarios"][scenario_name]["modes"][mode][metric_key]
                    for metric_key in metric_keys
                ]
                for mode in MODE_ORDER
            ]
            for scenario_name in scenario_names
        ],
        dtype=np.float32,
    )
    np.savez(
        data_path,
        modes=np.asarray(MODE_ORDER, dtype=object),
        labels=np.asarray([MODE_LABELS[mode] for mode in MODE_ORDER], dtype=object),
        metrics=np.asarray(metric_keys, dtype=object),
        metric_matrix=metric_matrix,
        scenarios=np.asarray(scenario_names, dtype=object),
        scenario_labels=np.asarray(
            [aggregate["scenarios"][name]["label"] for name in scenario_names],
            dtype=object,
        ),
        scenario_metric_tensor=scenario_metric_tensor,
    )
    return result


def assert_full_benchmark_result(result: dict) -> None:
    aggregate = result["aggregate"]["modes"]
    diffusion = aggregate["diffusion"]
    diffusion_qp = aggregate["diffusion_qp"]
    safe_compliance = aggregate["safe_compliance"]
    ours = aggregate["ours"]

    assert result["aggregate"]["num_cases"] >= 48
    assert len(result["aggregate"].get("scenarios", {})) >= 4
    for scenario in result["aggregate"]["scenarios"].values():
        assert scenario["num_cases"] >= 8
        assert scenario["modes"]["ours"]["collision_rate"] == 0.0
    assert diffusion_qp["collision_rate"] <= diffusion["collision_rate"]
    assert diffusion["collision_rate"] >= 0.25
    assert safe_compliance["collision_rate"] == 0.0
    assert ours["collision_rate"] == 0.0
    assert ours["success_rate"] >= 0.95
    assert ours["success_rate"] >= diffusion["success_rate"]
    assert ours["success_rate"] >= safe_compliance["success_rate"] - 0.10
    assert ours["collision_rate"] <= diffusion["collision_rate"]
    assert ours["mean_completion_time_sec"] is not None
    assert safe_compliance["mean_completion_time_sec"] is not None
    assert ours["mean_completion_time_sec"] < safe_compliance["mean_completion_time_sec"] * 0.80
    assert ours["mean_compliance_steps"] < safe_compliance["mean_compliance_steps"] * 0.70
    assert diffusion["mean_compliance_steps"] == 0.0
    assert diffusion_qp["mean_compliance_steps"] == 0.0

    for artifact in result["artifacts"].values():
        assert Path(artifact).exists()


def test_compliance_full_benchmark_visualization() -> None:
    result = run_full_benchmark()
    assert_full_benchmark_result(result)


if __name__ == "__main__":
    output = run_full_benchmark()
    assert_full_benchmark_result(output)
    print(json.dumps(output["artifacts"], indent=2))
