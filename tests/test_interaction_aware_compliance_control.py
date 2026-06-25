from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FOLLOWDATASET_DIR = Path(__file__).resolve().parents[1]
if str(FOLLOWDATASET_DIR) not in sys.path:
    sys.path.insert(0, str(FOLLOWDATASET_DIR))

from src.compliance_control import (  # noqa: E402
    ComplianceControlConfig,
    apply_bre_compliance_control,
    apply_interaction_aware_compliance_control,
)
from src.physics import PhysicsEngine  # noqa: E402
from src.safety_filter import QPSafetyFilter  # noqa: E402


SEGMENTATION_PATH = Path(
    "/home/yyf/Downloads/292_transforming_a_quadruped_into_-Supplementary Material/"
    "interaction-data/guidedog-mocap-dataprocessing-web2/segmentation.py"
)
ARTIFACT_DIR = (
    Path(__file__).resolve().parent
    / "artifacts"
    / "interaction_aware_compliance"
)


@dataclass(frozen=True)
class InteractionAwareScenario:
    frame_count: int
    fps: int
    frame_stride: int
    robot_radius: float
    human_radius: float
    leash_length: float
    robot_speed: float
    goal_x: float
    raw_actions: np.ndarray


def load_segmentation_module():
    spec = importlib.util.spec_from_file_location(
        "guidedog_interaction_segmentation",
        SEGMENTATION_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import segmentation model: {SEGMENTATION_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_long_mocap_frame() -> tuple[pd.DataFrame, np.ndarray]:
    frame_count = 360
    tether_intervals = ((55, 85), (120, 145), (210, 245), (285, 315))

    tether_target = np.zeros((frame_count,), dtype=float)
    for start, end in tether_intervals:
        tether_target[start:end] = 1.0

    transition_window = 25
    radius = transition_window // 2
    kernel = np.hanning(transition_window)
    kernel /= float(np.sum(kernel))
    tether_alpha = np.convolve(
        np.pad(tether_target, (radius, radius), mode="edge"),
        kernel,
        mode="valid",
    )
    truth = np.where(tether_alpha >= 0.5, "leash", "guide").astype(object)

    progress = np.zeros((frame_count,), dtype=float)
    for idx in range(1, frame_count):
        progress[idx] = progress[idx - 1] + (18.0 - 6.0 * tether_alpha[idx])

    frame_idx = np.arange(frame_count, dtype=float)
    center_y = 18.0 * np.sin(frame_idx / 17.0)
    guide_robot_lead = 300.0
    tether_robot_lead = -300.0
    lead = (1.0 - tether_alpha) * guide_robot_lead + tether_alpha * tether_robot_lead
    lateral = 18.0 * np.sin(frame_idx / 19.0 + tether_alpha * np.pi)
    robot = np.column_stack(
        (
            progress + 0.5 * lead,
            center_y + 0.5 * lateral,
        )
    )
    human = np.column_stack(
        (
            progress - 0.5 * lead,
            center_y - 0.5 * lateral,
        )
    )

    frame = pd.DataFrame(
        {
            "robot_x": robot[:, 0],
            "robot_y": robot[:, 1],
            "robot_orientation": 0.0,
            "robot_occlusion": False,
            "human_x": human[:, 0],
            "human_y": human[:, 1],
            "human_rotation": 0.0,
            "human_occlusion": False,
            "state": truth.tolist(),
        }
    )
    return frame, np.asarray(truth, dtype=object)


def decode_interaction_labels() -> tuple[pd.DataFrame, dict]:
    segmentation = load_segmentation_module()
    frame, truth = build_long_mocap_frame()
    feature_config = segmentation.FeatureConfig(
        smoothing_window=5,
        lead_distance_mm=120.0,
        link_rate_mm_per_frame=1.0,
        speed_balance_mm_per_frame=1.0,
        causal_max_lag_frames=6,
        causal_window=9,
        evidence_threshold=0.0,
    )
    features = segmentation.extract_interaction_features(frame, feature_config)
    trajectory = segmentation.Trajectory(
        csv_path=Path("synthetic_interaction_aware.csv"),
        frame=frame,
        features=features,
    )
    model = segmentation.train_drag_threshold_model(
        [trajectory],
        segmentation.HMMConfig(min_state_duration=6),
        feature_config,
        threshold_percentile=40.0,
    )
    labeled = segmentation.decode_trajectory(frame, features, model)
    labels = labeled["semantic_label"].to_numpy(dtype=object)
    metadata = {
        "model_path": str(SEGMENTATION_PATH),
        "classifier_kind": str(model.classifier_kind),
        "score_threshold": float(model.score_threshold),
        "score_threshold_percentile": float(model.score_threshold_percentile),
        "label_counts": {
            label: int(np.sum(labels == label))
            for label in ("guide", "leash", "unknown")
        },
        "truth_counts": {
            label: int(np.sum(truth == label))
            for label in ("guide", "leash")
        },
        "state_accuracy": float(np.mean(labels == truth)),
    }
    labeled["synthetic_truth"] = truth
    return labeled, metadata


def build_scenario(frame_count: int) -> InteractionAwareScenario:
    return InteractionAwareScenario(
        frame_count=int(frame_count),
        fps=20,
        frame_stride=5,
        robot_radius=0.22,
        human_radius=0.22,
        leash_length=1.0,
        robot_speed=1.0,
        goal_x=12.0,
        raw_actions=np.tile(
            np.array([0.18, 0.0], dtype=np.float32),
            (int(frame_count), 1),
        ),
    )


def make_engine(scenario: InteractionAwareScenario) -> PhysicsEngine:
    engine = PhysicsEngine(
        leash_length=scenario.leash_length,
        robot_speed=scenario.robot_speed,
        dt=1.0 / float(scenario.fps),
        robot_radius=scenario.robot_radius,
        human_radius=scenario.human_radius,
    )
    engine.reset(np.array([0.0, 0.0], dtype=float))
    engine.human.position = np.array([-0.8, 0.0], dtype=float)
    engine.bre = True
    engine.random_angle = 0.0
    return engine


def make_config(scenario: InteractionAwareScenario) -> ComplianceControlConfig:
    return ComplianceControlConfig(
        data_dt=float(scenario.frame_stride) / float(scenario.fps),
        sim_dt=1.0 / float(scenario.fps),
        frame_stride=int(scenario.frame_stride),
        turn_gain=1.0,
        safety_mode="off",
        curvature_slowdown=False,
    )


def rollout_actions(
    scenario: InteractionAwareScenario,
    actions: np.ndarray,
) -> dict:
    config = make_config(scenario)
    engine = make_engine(scenario)
    robot_path = [engine.robot.position.copy()]
    human_path = [engine.human.position.copy()]
    finish = None

    for action_idx, action in enumerate(np.asarray(actions, dtype=np.float32)):
        forward = float(action[0]) / (engine.robot_speed * config.data_dt)
        turn = float(action[1]) / (engine.turn_speed * config.data_dt)
        engine.set_control(
            float(np.clip(forward, -1.0, 1.0)),
            float(np.clip(turn, -1.0, 1.0)),
            True,
        )
        for substep_idx in range(int(config.frame_stride)):
            robot_state, human_state = engine.step()
            robot_path.append(robot_state.position.copy())
            human_path.append(human_state.position.copy())
            if finish is None and float(robot_state.position[0]) >= float(scenario.goal_x):
                finish = {
                    "action_idx": int(action_idx),
                    "substep_idx": int(substep_idx),
                    "time_sec": float(
                        (action_idx * config.frame_stride + substep_idx + 1)
                        * config.sim_dt
                    ),
                    "robot_x": float(robot_state.position[0]),
                }

    robot_path_arr = np.asarray(robot_path, dtype=np.float32)
    human_path_arr = np.asarray(human_path, dtype=np.float32)
    return {
        "robot_path": robot_path_arr,
        "human_path": human_path_arr,
        "finish": finish,
        "reached_goal": finish is not None,
        "final_robot_x": float(robot_path_arr[-1, 0]),
        "final_human_x": float(human_path_arr[-1, 0]),
        "robot_path_length": path_length(robot_path_arr),
        "human_path_length": path_length(human_path_arr),
    }


def path_length(path: np.ndarray) -> float:
    if len(path) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


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


def summarize_mode(actions: np.ndarray, rollout: dict, stats: dict | None = None) -> dict:
    finish = rollout["finish"]
    return {
        "reached_goal": bool(rollout["reached_goal"]),
        "completion_time_sec": None if finish is None else float(finish["time_sec"]),
        "completion_action_idx": None if finish is None else int(finish["action_idx"]),
        "final_robot_x": float(rollout["final_robot_x"]),
        "final_human_x": float(rollout["final_human_x"]),
        "robot_path_length": float(rollout["robot_path_length"]),
        "human_path_length": float(rollout["human_path_length"]),
        "mean_forward_action": float(np.mean(actions[:, 0])),
        "min_forward_action": float(np.min(actions[:, 0])),
        "max_forward_action": float(np.max(actions[:, 0])),
        "stats": serialize(stats or {}),
    }


def plot_interaction_aware_demo(
    labeled: pd.DataFrame,
    scenario: InteractionAwareScenario,
    rollouts: dict[str, dict],
    actions: dict[str, np.ndarray],
    summary: dict,
    segmentation_path: Path,
    progress_action_path: Path,
) -> None:
    label_colors = {"guide": "#2563EB", "leash": "#DC2626", "unknown": "#9CA3AF"}

    fig, ax = plt.subplots(figsize=(10.8, 2.2))
    ax.set_title("Segmentation: Guide vs Tether", fontsize=12, fontweight="bold")
    ax.plot(
        labeled["robot_x"].to_numpy(dtype=float) / 1000.0,
        labeled["robot_y"].to_numpy(dtype=float) / 1000.0,
        color="#111827",
        linewidth=1.1,
        alpha=0.36,
        label="robot mocap",
    )
    human_x = labeled["human_x"].to_numpy(dtype=float) / 1000.0
    human_y = labeled["human_y"].to_numpy(dtype=float) / 1000.0
    ax.plot(
        human_x,
        human_y,
        color="#6B7280",
        linewidth=1.1,
        alpha=0.36,
        label="human mocap",
    )
    labels = labeled["semantic_label"].to_numpy(dtype=object)
    for label in ("guide", "leash"):
        mask = labels == label
        ax.scatter(
            human_x[mask],
            human_y[mask],
            s=9,
            color=label_colors[label],
            alpha=0.76,
            label="tether" if label == "leash" else "guide",
        )
    ax.set_xlabel("mocap x [m]")
    ax.set_ylabel("mocap y [m]")
    ax.set_aspect("5", adjustable="box")
    ax.set_ylim(-.08, .08)
    ax.grid(True, color="#E5E7EB", linewidth=0.8)
    ax.legend(
        frameon=False,
        fontsize=8,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
    )
    segmentation_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(segmentation_path, dpi=180, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.4, 7.0),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.0, 1.25]},
    )
    config = make_config(scenario)

    ax = axes[0]
    ax.set_title("Robot Progress", fontsize=12, fontweight="bold")
    for mode, color, label in (
        ("raw_policy", "#7F1D1D", "raw"),
        ("full_time_compliance", "#581C87", "full-time compliance"),
        ("interaction_aware", "#1D4ED8", "interaction-aware"),
    ):
        robot_path = rollouts[mode]["robot_path"]
        time_axis = np.arange(len(robot_path), dtype=float) * config.sim_dt
        ax.plot(
            time_axis,
            robot_path[:, 0],
            color=color,
            linewidth=2.0,
            label=label,
        )
        finish = summary["modes"][mode]["completion_time_sec"]
        if finish is not None:
            ax.axvline(float(finish), color=color, linestyle=":", linewidth=1.2)
    ax.axhline(scenario.goal_x, color="#111827", linestyle="--", linewidth=1.0, label="goal")
    ax.set_ylabel("robot x [m]")
    ax.grid(True, color="#E5E7EB", linewidth=0.8)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.set_title("Forward Action Gating (Smoothed)", fontsize=12, fontweight="bold")
    action_time = np.arange(scenario.frame_count, dtype=float) * config.data_dt
    tether_span_labeled = False
    for start, end in state_runs(labels):
        if labels[start] == "leash":
            ax.axvspan(
                float(start) * config.data_dt,
                float(end) * config.data_dt,
                color="#FCA5A5",
                alpha=0.24,
                linewidth=0,
                label="tether interval" if not tether_span_labeled else None,
            )
            tether_span_labeled = True
    ax.plot(
        action_time,
        smooth_forward_delta(actions["raw_policy"][:, 0]),
        color="#7F1D1D",
        linewidth=1.8,
        label="raw",
    )
    ax.plot(
        action_time,
        smooth_forward_delta(actions["full_time_compliance"][:, 0]),
        color="#581C87",
        linewidth=1.8,
        label="full-time",
    )
    ax.plot(
        action_time,
        smooth_forward_delta(actions["interaction_aware"][:, 0]),
        color="#1D4ED8",
        linewidth=1.8,
        label="interaction-aware",
    )
    ax.set_xlim(0.0, float(scenario.frame_count) * config.data_dt)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("forward delta [m]")
    ax.grid(True, color="#E5E7EB", linewidth=0.8)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Interaction-Aware Compliance Timing and Gating", fontsize=14)
    progress_action_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(progress_action_path, dpi=180)
    plt.close(fig)


def state_runs(labels: np.ndarray) -> list[tuple[int, int]]:
    runs = []
    start = 0
    for idx in range(1, len(labels) + 1):
        if idx == len(labels) or labels[idx] != labels[start]:
            runs.append((start, idx))
            start = idx
    return runs


def smooth_forward_delta(values: np.ndarray, window: int = 15, sigma: float = 3.0) -> np.ndarray:
    series = np.asarray(values, dtype=float).reshape(-1)
    if window <= 1 or len(series) < 3:
        return series.copy()
    window = min(int(window), len(series))
    if window % 2 == 0:
        window -= 1
    if window <= 1:
        return series.copy()
    radius = window // 2
    padded = np.pad(series, (radius, radius), mode="edge")
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / max(float(sigma), 1e-6)) ** 2)
    kernel /= float(np.sum(kernel))
    return np.convolve(padded, kernel, mode="valid")


def mocap_step_summary(frame: pd.DataFrame) -> dict:
    summary = {}
    for entity in ("robot", "human"):
        path = frame[[f"{entity}_x", f"{entity}_y"]].to_numpy(dtype=float) / 1000.0
        steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
        summary[f"max_{entity}_step_m"] = float(np.max(steps))
        summary[f"p95_{entity}_step_m"] = float(np.percentile(steps, 95))
    return summary


def run_interaction_aware_demo() -> dict:
    labeled, segmentation_metadata = decode_interaction_labels()
    labels = labeled["semantic_label"].to_numpy(dtype=object)
    scenario = build_scenario(len(labels))
    config = make_config(scenario)
    qp = QPSafetyFilter()

    full_time = apply_bre_compliance_control(
        scenario.raw_actions,
        make_engine(scenario),
        config,
        safety_filter=qp,
        obstacles=None,
        segment_obstacles=None,
        bre=True,
    )
    interaction_aware = apply_interaction_aware_compliance_control(
        scenario.raw_actions,
        labels,
        make_engine(scenario),
        config,
        safety_filter=qp,
        obstacles=None,
        segment_obstacles=None,
        bre=True,
    )

    actions = {
        "raw_policy": scenario.raw_actions.copy(),
        "full_time_compliance": full_time.actions,
        "interaction_aware": interaction_aware.actions,
    }
    stats = {
        "raw_policy": {},
        "full_time_compliance": full_time.stats,
        "interaction_aware": interaction_aware.stats,
    }
    rollouts = {
        name: rollout_actions(scenario, action_seq)
        for name, action_seq in actions.items()
    }
    summary = {
        "scenario": {
            "frame_count": int(scenario.frame_count),
            "fps": int(scenario.fps),
            "frame_stride": int(scenario.frame_stride),
            "goal_x": float(scenario.goal_x),
            "robot_speed": float(scenario.robot_speed),
        },
        "segmentation": segmentation_metadata,
        "mocap_continuity": mocap_step_summary(labeled),
        "modes": {
            name: summarize_mode(action_seq, rollouts[name], stats.get(name))
            for name, action_seq in actions.items()
        },
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    segmentation_plot_path = ARTIFACT_DIR / "interaction_aware_segmentation.png"
    progress_action_plot_path = ARTIFACT_DIR / "interaction_aware_progress_and_gating.png"
    summary_path = ARTIFACT_DIR / "interaction_aware_compliance_control_summary.json"
    data_path = ARTIFACT_DIR / "interaction_aware_compliance_control_data.npz"

    plot_interaction_aware_demo(
        labeled=labeled,
        scenario=scenario,
        rollouts=rollouts,
        actions=actions,
        summary=summary,
        segmentation_path=segmentation_plot_path,
        progress_action_path=progress_action_plot_path,
    )
    summary["artifacts"] = {
        "segmentation_plot": str(segmentation_plot_path),
        "progress_action_plot": str(progress_action_plot_path),
        "summary": str(summary_path),
        "data": str(data_path),
    }
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(serialize(summary), fp, indent=2)
    np.savez(
        data_path,
        semantic_label=labels,
        synthetic_truth=labeled["synthetic_truth"].to_numpy(dtype=object),
        drag_score=labeled["drag_score"].to_numpy(dtype=float),
        raw_policy_actions=actions["raw_policy"],
        full_time_compliance_actions=actions["full_time_compliance"],
        interaction_aware_actions=actions["interaction_aware"],
        raw_policy_robot_path=rollouts["raw_policy"]["robot_path"],
        full_time_compliance_robot_path=rollouts["full_time_compliance"]["robot_path"],
        interaction_aware_robot_path=rollouts["interaction_aware"]["robot_path"],
        raw_policy_human_path=rollouts["raw_policy"]["human_path"],
        full_time_compliance_human_path=rollouts["full_time_compliance"]["human_path"],
        interaction_aware_human_path=rollouts["interaction_aware"]["human_path"],
    )
    return summary


def assert_interaction_aware_summary(summary: dict) -> None:
    segmentation = summary["segmentation"]
    assert segmentation["classifier_kind"] == "drag-threshold"
    assert segmentation["state_accuracy"] >= 0.90
    assert segmentation["label_counts"]["guide"] > 0
    assert segmentation["label_counts"]["leash"] > 0
    assert summary["mocap_continuity"]["max_human_step_m"] < 0.06
    assert summary["mocap_continuity"]["max_robot_step_m"] < 0.06

    modes = summary["modes"]
    raw = modes["raw_policy"]
    full_time = modes["full_time_compliance"]
    aware = modes["interaction_aware"]
    assert raw["reached_goal"]
    assert full_time["reached_goal"]
    assert aware["reached_goal"]
    assert raw["completion_time_sec"] < aware["completion_time_sec"]
    assert aware["completion_time_sec"] < 0.45 * full_time["completion_time_sec"]
    assert aware["final_robot_x"] > full_time["final_robot_x"] + 10.0
    assert aware["mean_forward_action"] > full_time["mean_forward_action"] * 2.0

    aware_stats = aware["stats"]
    assert aware_stats["compliance_steps"] == segmentation["label_counts"]["leash"]
    assert aware_stats["guide_steps"] == segmentation["label_counts"]["guide"]
    assert aware_stats["compliance_steps"] < full_time["stats"]["modified_steps"]
    assert aware_stats["modified_steps"] == aware_stats["compliance_steps"]

    for artifact in summary["artifacts"].values():
        assert Path(artifact).exists()


def test_interaction_aware_compliance_control_visualization() -> None:
    summary = run_interaction_aware_demo()
    assert_interaction_aware_summary(summary)


if __name__ == "__main__":
    result = run_interaction_aware_demo()
    assert_interaction_aware_summary(result)
    print(json.dumps(result["artifacts"], indent=2))
