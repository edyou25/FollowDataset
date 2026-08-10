from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


FOLLOWDATASET_DIR = Path(__file__).resolve().parents[1]
if str(FOLLOWDATASET_DIR) not in sys.path:
    sys.path.insert(0, str(FOLLOWDATASET_DIR))

from src.compliance_control import (  # noqa: E402
    ComplianceControlConfig,
    apply_bre_compliance_control,
)
from src.physics import PhysicsEngine  # noqa: E402
from src.safety_filter import QPSafetyFilter  # noqa: E402
from tests.plot_styles import (  # noqa: E402
    COLLISION_COLOR,
    GRID_COLOR,
    HUMAN_COLOR,
    OBSTACLE_EDGE_COLOR,
    PANEL_FACE_COLOR,
    REFERENCE_COLOR,
    ROBOT_COLOR,
    SAFETY_PROJECTION_COLOR,
    SAFETY_RING_COLOR,
    SPINE_COLOR,
)


ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts" / "compliance_control"


@dataclass(frozen=True)
class ComplianceScenario:
    robot_start: np.ndarray
    human_start: np.ndarray
    robot_velocity: np.ndarray
    human_velocity: np.ndarray
    robot_radius: float
    human_radius: float
    leash_length: float
    robot_speed: float
    fps: int
    frame_stride: int
    circle_obstacles: np.ndarray
    segment_obstacles: np.ndarray
    reference_path: np.ndarray
    raw_actions: np.ndarray


def build_compliance_collision_scenario() -> ComplianceScenario:
    """Human-leading lateral motion bends compliance into a front obstacle."""
    reference_x = np.linspace(0.0, 1.45, 120, dtype=np.float32)
    reference_path = np.stack(
        [reference_x, np.zeros((len(reference_x),), dtype=np.float32)],
        axis=1,
    )
    raw_actions = np.tile(np.array([0.05, 0.0], dtype=np.float32), (36, 1))
    return ComplianceScenario(
        robot_start=np.array([0.0, 0.0], dtype=np.float32),
        human_start=np.array([-0.82, 0.0], dtype=np.float32),
        robot_velocity=np.array([0.0, 0.6], dtype=np.float32),
        human_velocity=np.array([0.0, -0.4], dtype=np.float32),
        robot_radius=0.22,
        human_radius=0.22,
        leash_length=1.0,
        robot_speed=1.0,
        fps=20,
        frame_stride=5,
        circle_obstacles=np.array([[0.78, -0.12, 0.22]], dtype=np.float32),
        segment_obstacles=np.zeros((0, 4), dtype=np.float32),
        reference_path=reference_path.astype(np.float32),
        raw_actions=raw_actions,
    )


def make_engine(scenario: ComplianceScenario) -> PhysicsEngine:
    engine = PhysicsEngine(
        leash_length=scenario.leash_length,
        robot_speed=scenario.robot_speed,
        dt=1.0 / float(scenario.fps),
        robot_radius=scenario.robot_radius,
        human_radius=scenario.human_radius,
    )
    engine.reset(scenario.robot_start.astype(float))
    engine.human.position = scenario.human_start.astype(float).copy()
    engine.robot.velocity = scenario.robot_velocity.astype(float).copy()
    engine.human.velocity = scenario.human_velocity.astype(float).copy()
    engine.bre = True
    engine.random_angle = 0.0
    return engine


def make_config(scenario: ComplianceScenario, safety_mode: str) -> ComplianceControlConfig:
    return ComplianceControlConfig(
        data_dt=float(scenario.frame_stride) / float(scenario.fps),
        sim_dt=1.0 / float(scenario.fps),
        frame_stride=int(scenario.frame_stride),
        turn_gain=1.0,
        safety_mode=safety_mode,
        curvature_slowdown=False,
    )


def rollout_actions(
    scenario: ComplianceScenario,
    actions: np.ndarray,
) -> dict:
    config = make_config(scenario, "off")
    engine = make_engine(scenario)
    robot_path = [engine.robot.position.copy()]
    human_path = [engine.human.position.copy()]
    first_hit = None

    for action_idx, action in enumerate(np.asarray(actions, dtype=np.float32)):
        forward = float(action[0]) / (engine.robot_speed * config.data_dt)
        turn = float(action[1]) / (engine.turn_speed * config.data_dt)
        engine.set_control(float(np.clip(forward, -1.0, 1.0)), float(np.clip(turn, -1.0, 1.0)), True)
        for substep_idx in range(int(config.frame_stride)):
            robot_state, human_state = engine.step()
            robot_path.append(robot_state.position.copy())
            human_path.append(human_state.position.copy())
            collided, info = engine.check_collision(
                scenario.circle_obstacles,
                segment_obstacles=scenario.segment_obstacles,
            )
            if collided and first_hit is None:
                first_hit = {
                    "action_idx": int(action_idx),
                    "substep_idx": int(substep_idx),
                    "who": str(info.get("who", "unknown")) if info else "unknown",
                    "type": str(info.get("type", "unknown")) if info else "unknown",
                    "obstacle_idx": int(info.get("idx", -1)) if info else -1,
                    "robot_pos": robot_state.position.astype(float).tolist(),
                    "human_pos": human_state.position.astype(float).tolist(),
                }

    return {
        "robot_path": np.asarray(robot_path, dtype=np.float32),
        "human_path": np.asarray(human_path, dtype=np.float32),
        "first_hit": first_hit,
    }


def circle_min_clearance(
    path: np.ndarray,
    radius: float,
    obstacles: np.ndarray,
) -> float:
    path = np.asarray(path, dtype=np.float32)
    obstacles = np.asarray(obstacles, dtype=np.float32)
    if len(path) == 0 or len(obstacles) == 0:
        return float("inf")
    clearances = []
    for obs in obstacles:
        clearances.append(np.linalg.norm(path - obs[:2][None, :], axis=1) - radius - float(obs[2]))
    return float(np.min(np.concatenate(clearances, axis=0)))


def path_length(path: np.ndarray) -> float:
    path = np.asarray(path, dtype=np.float32)
    if len(path) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


def serialize(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return "inf" if value > 0 else "-inf"
    return value


def summarize_mode(
    scenario: ComplianceScenario,
    result: dict,
    actions: np.ndarray,
    stats: dict | None = None,
    infos: list[dict] | None = None,
) -> dict:
    robot_path = result["robot_path"]
    human_path = result["human_path"]
    robot_min_clearance = circle_min_clearance(
        robot_path,
        scenario.robot_radius,
        scenario.circle_obstacles,
    )
    human_min_clearance = circle_min_clearance(
        human_path,
        scenario.human_radius,
        scenario.circle_obstacles,
    )
    projected_steps = [
        int(idx)
        for idx, info in enumerate(infos or [])
        if bool(info.get("action_projection_applied", False))
    ]
    return {
        "collision": result["first_hit"] is not None,
        "first_hit": result["first_hit"],
        "robot_min_clearance": float(robot_min_clearance),
        "human_min_clearance": float(human_min_clearance),
        "robot_path_length": path_length(robot_path),
        "human_path_length": path_length(human_path),
        "robot_final": robot_path[-1].astype(float).tolist(),
        "human_final": human_path[-1].astype(float).tolist(),
        "action_forward_min": float(np.min(actions[:, 0])),
        "action_forward_max": float(np.max(actions[:, 0])),
        "action_heading_abs_max": float(np.max(np.abs(actions[:, 1]))),
        "projected_steps": projected_steps,
        "stats": serialize(stats or {}),
    }


def plot_compliance_scenario(
    scenario: ComplianceScenario,
    rollouts: dict[str, dict],
    actions: dict[str, np.ndarray],
    summary: dict,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
    panels = [
        ("raw_policy", "Raw Policy"),
        ("compliance_no_safety", "Compliance Only"),
        ("compliance_safe", "Compliance + Safety"),
    ]
    for ax, (mode, _title) in zip(axes, panels):
        ax.set_facecolor(PANEL_FACE_COLOR)
        ax.plot(
            scenario.reference_path[:, 0],
            scenario.reference_path[:, 1],
            color=REFERENCE_COLOR,
            linewidth=1.2,
            linestyle="--",
            label="reference",
            zorder=1,
        )
        for obs_idx, obs in enumerate(scenario.circle_obstacles):
            ax.add_patch(
                plt.Circle(
                    obs[:2],
                    obs[2],
                    color=OBSTACLE_EDGE_COLOR,
                    alpha=0.30,
                    label="obstacle" if obs_idx == 0 else None,
                    zorder=1,
                )
            )
            ax.add_patch(
                plt.Circle(
                    obs[:2],
                    obs[2] + scenario.robot_radius,
                    fill=False,
                    linestyle=":",
                    color=SAFETY_RING_COLOR,
                    linewidth=1.2,
                    label="robot collision radius" if obs_idx == 0 else None,
                    zorder=1,
                )
            )
        result = rollouts[mode]
        robot_path = result["robot_path"]
        human_path = result["human_path"]
        ax.plot(
            robot_path[:, 0],
            robot_path[:, 1],
            color=ROBOT_COLOR,
            linewidth=2.2,
            label="robot",
            zorder=4,
        )
        ax.plot(
            human_path[:, 0],
            human_path[:, 1],
            color=HUMAN_COLOR,
            linewidth=2.2,
            label="human",
            zorder=4,
        )
        ax.scatter(
            [robot_path[0, 0]],
            [robot_path[0, 1]],
            s=80,
            c=ROBOT_COLOR,
            edgecolors="white",
            linewidths=1.0,
            zorder=5,
        )
        ax.scatter(
            [human_path[0, 0]],
            [human_path[0, 1]],
            s=80,
            c=HUMAN_COLOR,
            edgecolors="white",
            linewidths=1.0,
            zorder=5,
        )
        hit = result["first_hit"]
        if hit is not None:
            hit_pos = robot_path[min(len(robot_path) - 1, hit["action_idx"] * scenario.frame_stride + hit["substep_idx"] + 1)]
            ax.scatter(
                [hit_pos[0]],
                [hit_pos[1]],
                s=110,
                c=COLLISION_COLOR,
                marker="x",
                linewidths=2.4,
                label="first collision",
                zorder=6,
            )

        projected_steps = summary["modes"][mode].get("projected_steps", [])
        if projected_steps:
            indices = [
                min(len(robot_path) - 1, step * scenario.frame_stride)
                for step in projected_steps
            ]
            projected_points = robot_path[indices]
            ax.scatter(
                projected_points[:, 0],
                projected_points[:, 1],
                s=42,
                c=SAFETY_PROJECTION_COLOR,
                marker="D",
                label="safety projection",
                zorder=6,
            )

        mode_summary = summary["modes"][mode]
        status = (
            f"collision: {mode_summary['collision']}\n"
            f"robot clearance: {mode_summary['robot_min_clearance']:.3f} m\n"
            f"heading max: {mode_summary['action_heading_abs_max']:.3f} rad"
        )
        ax.text(
            0.02,
            0.98,
            status,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": SPINE_COLOR, "alpha": 0.92},
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1.0, 1.45)
        ax.set_ylim(-0.75, 1.05)
        ax.grid(True, color=GRID_COLOR, linewidth=0.8)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

    legend_items = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            legend_items.setdefault(label, handle)
    fig.legend(
        list(legend_items.values()),
        list(legend_items.keys()),
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=9,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def run_compliance_control_demo() -> dict:
    scenario = build_compliance_collision_scenario()
    qp = QPSafetyFilter(
        margin=0.04,
        alpha=1.0,
        max_constraints=12,
        influence_distance=1.0,
    )
    actions: dict[str, np.ndarray] = {
        "raw_policy": scenario.raw_actions.copy(),
    }
    infos: dict[str, list[dict]] = {
        "raw_policy": [],
    }
    stats: dict[str, dict] = {
        "raw_policy": {},
    }

    no_safety = apply_bre_compliance_control(
        scenario.raw_actions,
        make_engine(scenario),
        make_config(scenario, "off"),
        safety_filter=qp,
        obstacles=scenario.circle_obstacles,
        segment_obstacles=scenario.segment_obstacles,
        bre=True,
    )
    actions["compliance_no_safety"] = no_safety.actions
    infos["compliance_no_safety"] = no_safety.infos
    stats["compliance_no_safety"] = no_safety.stats

    safe = apply_bre_compliance_control(
        scenario.raw_actions,
        make_engine(scenario),
        make_config(scenario, "human_robot_qp"),
        safety_filter=qp,
        obstacles=scenario.circle_obstacles,
        segment_obstacles=scenario.segment_obstacles,
        bre=True,
    )
    actions["compliance_safe"] = safe.actions
    infos["compliance_safe"] = safe.infos
    stats["compliance_safe"] = safe.stats

    rollouts = {
        name: rollout_actions(scenario, action_seq)
        for name, action_seq in actions.items()
    }
    summary = {
        "scenario": {
            "robot_start": scenario.robot_start.astype(float).tolist(),
            "human_start": scenario.human_start.astype(float).tolist(),
            "robot_radius": float(scenario.robot_radius),
            "human_radius": float(scenario.human_radius),
            "leash_length": float(scenario.leash_length),
            "circle_obstacles": scenario.circle_obstacles.astype(float).tolist(),
            "frame_stride": int(scenario.frame_stride),
            "fps": int(scenario.fps),
        },
        "modes": {
            name: summarize_mode(
                scenario,
                rollouts[name],
                action_seq,
                stats=stats.get(name),
                infos=infos.get(name),
            )
            for name, action_seq in actions.items()
        },
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = ARTIFACT_DIR / "compliance_control_safety.png"
    summary_path = ARTIFACT_DIR / "compliance_control_safety_summary.json"
    data_path = ARTIFACT_DIR / "compliance_control_safety_data.npz"

    plot_compliance_scenario(
        scenario=scenario,
        rollouts=rollouts,
        actions=actions,
        summary=summary,
        output_path=plot_path,
    )
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(serialize(summary), fp, indent=2)
    np.savez(
        data_path,
        raw_policy_actions=actions["raw_policy"],
        compliance_no_safety_actions=actions["compliance_no_safety"],
        compliance_safe_actions=actions["compliance_safe"],
        raw_policy_robot_path=rollouts["raw_policy"]["robot_path"],
        raw_policy_human_path=rollouts["raw_policy"]["human_path"],
        compliance_no_safety_robot_path=rollouts["compliance_no_safety"]["robot_path"],
        compliance_no_safety_human_path=rollouts["compliance_no_safety"]["human_path"],
        compliance_safe_robot_path=rollouts["compliance_safe"]["robot_path"],
        compliance_safe_human_path=rollouts["compliance_safe"]["human_path"],
        circle_obstacles=scenario.circle_obstacles,
        reference_path=scenario.reference_path,
    )
    summary["artifacts"] = {
        "plot": str(plot_path),
        "summary": str(summary_path),
        "data": str(data_path),
    }
    return summary


def assert_compliance_control_summary(summary: dict) -> None:
    modes = summary["modes"]
    assert modes["raw_policy"]["collision"]
    assert modes["compliance_no_safety"]["collision"]
    assert not modes["compliance_safe"]["collision"]
    assert modes["raw_policy"]["robot_min_clearance"] < 0.0
    assert modes["compliance_no_safety"]["robot_min_clearance"] < 0.0
    assert modes["compliance_safe"]["robot_min_clearance"] > 0.02
    assert modes["compliance_no_safety"]["action_heading_abs_max"] > 0.08
    assert modes["compliance_safe"]["action_heading_abs_max"] > modes["compliance_no_safety"]["action_heading_abs_max"]

    safe_stats = modes["compliance_safe"]["stats"]
    assert safe_stats["safety_applied"]
    assert safe_stats["constraint_count"] >= 1
    assert safe_stats["safety_modified_steps"] >= 1
    assert modes["compliance_safe"]["projected_steps"]
    assert modes["compliance_safe"]["projected_steps"][0] <= 10

    for artifact in summary["artifacts"].values():
        assert Path(artifact).exists()


def test_compliance_control_safety_visualization() -> None:
    summary = run_compliance_control_demo()
    assert_compliance_control_summary(summary)


def test_preserve_heading_compliance_only_slows_forward_motion() -> None:
    engine = PhysicsEngine(robot_speed=1.5, dt=0.05)
    engine.reset(np.array([0.0, 0.0], dtype=np.float32))
    actions = np.array(
        [
            [0.30, 0.38],
            [-0.12, -0.31],
        ],
        dtype=np.float32,
    )
    config = ComplianceControlConfig(
        data_dt=0.25,
        sim_dt=0.05,
        frame_stride=5,
        safety_mode="off",
        heading_control=False,
        forward_only_slowdown=True,
        preserve_heading=True,
    )

    result = apply_bre_compliance_control(actions, engine, config, bre=True)

    assert np.allclose(result.actions[:, 1], 0.0)
    assert np.all(result.actions[:, 0] <= 0.075 + 1e-6)
    assert np.all(result.actions[:, 0] >= 0.0)


if __name__ == "__main__":
    result = run_compliance_control_demo()
    assert_compliance_control_summary(result)
    print(json.dumps(result["artifacts"], indent=2))
