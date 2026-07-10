"""Interaction compliance control with optional QP safety projection."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.physics import PhysicsEngine
from src.safety_filter import QPSafetyFilter


def _normalize_safety_mode(mode: str) -> str:
    mode = str(mode or "off").lower()
    aliases = {
        "off": "off",
        "none": "off",
        "robot": "robot_qp",
        "robot_qp": "robot_qp",
        "human_robot": "human_robot_qp",
        "human_robot_qp": "human_robot_qp",
        "human+robot": "human_robot_qp",
    }
    if mode not in aliases:
        raise ValueError(f"Unsupported compliance safety mode: {mode!r}")
    return aliases[mode]


def _wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _has_obstacles(
    obstacles: Optional[np.ndarray],
    segment_obstacles: Optional[np.ndarray],
) -> bool:
    return (
        obstacles is not None
        and len(obstacles) > 0
    ) or (
        segment_obstacles is not None
        and len(segment_obstacles) > 0
    )


@dataclass(frozen=True)
class ComplianceControlConfig:
    data_dt: float
    sim_dt: float
    frame_stride: int
    turn_gain: float = 1.0
    safety_mode: str = "human_robot_qp"
    min_forward_scale: float = 0.05
    max_forward_scale: float = 0.20
    heading_clip_scale: float = 0.35
    heading_damping: float = 0.8
    heading_mix: float = 0.7
    heading_control: bool = True
    forward_only_slowdown: bool = False
    preserve_heading: bool = False
    curvature_slowdown: bool = True
    curvature_scale: float = 0.7
    min_speed_scale: float = 0.25
    backoff_scales: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25, 0.0)
    stop_clearance: float = 0.0


@dataclass
class ComplianceControlResult:
    actions: np.ndarray
    nominal_deltas: np.ndarray
    safe_deltas: np.ndarray
    infos: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def _empty_info() -> dict:
    return {
        "modified": False,
        "shift": 0.0,
        "constraint_count": 0,
        "min_clearance": float("inf"),
        "safety_applied": False,
    }


def _delta_to_safe_control(
    engine: PhysicsEngine,
    delta: np.ndarray,
    config: ComplianceControlConfig,
) -> tuple[float, float, float]:
    delta = np.asarray(delta, dtype=np.float32).reshape(2)
    delta_norm = float(np.linalg.norm(delta))
    if delta_norm < 1e-6:
        return 0.0, 0.0, 1.0

    desired_heading = float(np.arctan2(delta[1], delta[0]))
    heading_error = _wrap_angle(desired_heading - float(engine.robot.heading))
    turn_speed = float(engine.turn_speed)
    turn_input = heading_error / (turn_speed * config.data_dt) if turn_speed > 0 else 0.0
    turn_input = float(np.clip(turn_input, -1.0, 1.0))

    robot_speed = float(engine.robot_speed)
    forward_input = delta_norm / (robot_speed * config.data_dt) if robot_speed > 0 else 0.0
    forward_input = float(np.clip(forward_input, -1.0, 1.0))

    speed_scale = 1.0
    if config.curvature_slowdown and turn_speed > 0:
        max_turn = turn_speed * config.data_dt
        if max_turn > 1e-6:
            ratio = min(1.0, abs(heading_error) / max_turn)
            speed_scale = max(
                float(config.min_speed_scale),
                1.0 - float(config.curvature_scale) * ratio,
            )
            forward_input *= speed_scale
    return float(forward_input), float(turn_input), float(speed_scale)


def _forward_heading_action_to_world_delta(
    action: np.ndarray,
    heading: float,
) -> np.ndarray:
    forward = float(action[0])
    return np.array(
        [np.cos(heading) * forward, np.sin(heading) * forward],
        dtype=np.float32,
    )


def _forward_heading_action_to_control(
    engine: PhysicsEngine,
    action: np.ndarray,
    config: ComplianceControlConfig,
) -> tuple[np.ndarray, float, float, float]:
    action = np.asarray(action, dtype=np.float32).reshape(2)
    forward_delta = float(action[0])
    heading_delta = float(action[1])
    turn_speed = float(engine.turn_speed)
    robot_speed = float(engine.robot_speed)
    turn_delta = heading_delta * float(config.turn_gain)
    turn = turn_delta / (turn_speed * config.data_dt) if turn_speed > 0 else 0.0
    forward = forward_delta / (robot_speed * config.data_dt) if robot_speed > 0 else 0.0
    speed_scale = 1.0
    if config.curvature_slowdown and turn_speed > 0:
        max_turn = turn_speed * config.data_dt
        if max_turn > 1e-6:
            ratio = min(1.0, abs(turn_delta) / max_turn)
            speed_scale = max(
                float(config.min_speed_scale),
                1.0 - float(config.curvature_scale) * ratio,
            )
            forward *= speed_scale
    forward = float(np.clip(forward, -1.0, 1.0))
    turn = float(np.clip(turn, -1.0, 1.0))
    delta = _forward_heading_action_to_world_delta(action, float(engine.robot.heading))
    return delta, forward, turn, speed_scale


def _world_delta_to_forward_heading_action(
    engine: PhysicsEngine,
    delta: np.ndarray,
    config: ComplianceControlConfig,
) -> np.ndarray:
    delta = np.asarray(delta, dtype=np.float32).reshape(2)
    dist = float(np.linalg.norm(delta))
    if dist < 1e-6:
        return np.zeros((2,), dtype=np.float32)

    desired_heading = float(np.arctan2(delta[1], delta[0]))
    max_heading_delta = float(engine.turn_speed * config.data_dt) / max(
        float(config.turn_gain), 1e-6
    )
    heading_delta = _wrap_angle(desired_heading - float(engine.robot.heading)) / max(
        float(config.turn_gain), 1e-6
    )
    heading_delta = float(np.clip(heading_delta, -max_heading_delta, max_heading_delta))
    max_forward_delta = float(engine.robot_speed * config.data_dt)
    forward_delta = float(np.clip(dist, 0.0, max_forward_delta))
    return np.array([forward_delta, heading_delta], dtype=np.float32)


def _simulate_action_on_engine(
    engine: PhysicsEngine,
    action: np.ndarray,
    config: ComplianceControlConfig,
    bre: bool,
    obstacles: Optional[np.ndarray] = None,
    segment_obstacles: Optional[np.ndarray] = None,
    protect_robot: bool = True,
    protect_human: bool = True,
) -> tuple[bool, list[np.ndarray]]:
    _delta, forward, turn, _speed_scale = _forward_heading_action_to_control(
        engine, action, config
    )
    engine.set_control(forward, turn, bre)
    points: list[np.ndarray] = []
    for _ in range(int(config.frame_stride)):
        robot_state, _human_state = engine.step()
        points.append(robot_state.position.copy())
        if obstacles is not None or segment_obstacles is not None:
            collided, info = engine.check_collision(
                obstacles,
                segment_obstacles=segment_obstacles,
            )
            who = info.get("who") if info else None
            relevant = collided and (
                (protect_robot and who == "robot")
                or (protect_human and who == "human")
            )
            if relevant:
                return True, points
    return False, points


def _simulate_delta_on_engine(
    engine: PhysicsEngine,
    delta: np.ndarray,
    config: ComplianceControlConfig,
    bre: bool,
    obstacles: Optional[np.ndarray] = None,
    segment_obstacles: Optional[np.ndarray] = None,
    protect_robot: bool = True,
    protect_human: bool = True,
) -> tuple[bool, list[np.ndarray]]:
    forward, turn, _speed_scale = _delta_to_safe_control(engine, delta, config)
    engine.set_control(forward, turn, bre)
    points: list[np.ndarray] = []
    for _ in range(int(config.frame_stride)):
        robot_state, _human_state = engine.step()
        points.append(robot_state.position.copy())
        if obstacles is not None or segment_obstacles is not None:
            collided, info = engine.check_collision(
                obstacles,
                segment_obstacles=segment_obstacles,
            )
            who = info.get("who") if info else None
            relevant = collided and (
                (protect_robot and who == "robot")
                or (protect_human and who == "human")
            )
            if relevant:
                return True, points
    return False, points


def _forward_heading_action_to_nominal_delta(
    engine: PhysicsEngine,
    action: np.ndarray,
    config: ComplianceControlConfig,
    bre: bool,
) -> tuple[np.ndarray, PhysicsEngine]:
    nominal_engine = copy.deepcopy(engine)
    start_pos = nominal_engine.robot.position.copy()
    _simulate_action_on_engine(
        nominal_engine,
        action,
        config,
        bre,
        obstacles=None,
        segment_obstacles=None,
        protect_robot=False,
        protect_human=False,
    )
    delta = (nominal_engine.robot.position - start_pos).astype(np.float32)
    return delta, nominal_engine


def _safety_filter_delta(
    engine: PhysicsEngine,
    nominal_delta: np.ndarray,
    nominal_preview: PhysicsEngine,
    config: ComplianceControlConfig,
    bre: bool,
    safety_filter: QPSafetyFilter,
    obstacles: Optional[np.ndarray],
    segment_obstacles: Optional[np.ndarray],
) -> tuple[np.ndarray, PhysicsEngine, dict]:
    safety_mode = _normalize_safety_mode(config.safety_mode)
    nominal_delta = np.asarray(nominal_delta, dtype=np.float32).reshape(2)
    if safety_mode == "off" or not _has_obstacles(obstacles, segment_obstacles):
        return nominal_delta.astype(np.float32), nominal_preview, _empty_info()

    protect_human = safety_mode == "human_robot_qp"
    extra_entities = [
        ("robot_future", nominal_preview.robot.position.copy(), engine.robot_radius),
    ]
    if protect_human:
        extra_entities.append(
            ("human_future", nominal_preview.human.position.copy(), engine.human_radius)
        )

    qp = safety_filter.project_delta(
        ref_delta=nominal_delta,
        robot_pos=engine.robot.position,
        robot_radius=engine.robot_radius,
        human_pos=engine.human.position,
        human_radius=engine.human_radius,
        circle_obstacles=obstacles,
        segment_obstacles=segment_obstacles,
        include_human=protect_human,
        extra_entities=extra_entities,
    )
    qp_delta = qp.delta.astype(np.float32)
    chosen_delta = qp_delta.copy()
    trial_engine = copy.deepcopy(engine)
    collided, _ = _simulate_delta_on_engine(
        trial_engine,
        chosen_delta,
        config,
        bre,
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
        for scale in config.backoff_scales[1:]:
            backoff_delta = (chosen_delta * float(scale)).astype(np.float32)
            backoff_engine = copy.deepcopy(engine)
            collided, _ = _simulate_delta_on_engine(
                backoff_engine,
                backoff_delta,
                config,
                bre,
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
        "protect_human": bool(protect_human),
        "safety_applied": True,
        "nominal_delta": nominal_delta.astype(np.float32),
        "nominal_delta_norm": float(np.linalg.norm(nominal_delta)),
        "qp_delta": qp_delta.astype(np.float32),
        "qp_delta_norm": float(np.linalg.norm(qp_delta)),
        "qp_modified": bool(qp.modified),
        "qp_constraint_count": int(qp.constraint_count),
        "qp_total_constraint_count": int(getattr(qp, "total_constraint_count", qp.constraint_count)),
        "qp_ref_feasible": bool(getattr(qp, "ref_feasible", True)),
        "qp_candidate_count": int(getattr(qp, "candidate_count", 0)),
        "qp_best_candidate_kind": str(getattr(qp, "best_candidate_kind", "unknown")),
        "qp_selected_constraints": list(getattr(qp, "selected_constraints", [])),
        "collision_after_qp": bool(collision_after_qp),
        "backoff_attempts": backoff_attempts,
        "backoff_applied": bool(backoff_applied),
        "backoff_scale": float(backoff_scale),
        "resolution_stage": resolution_stage,
        "final_delta": chosen_delta.astype(np.float32),
        "final_delta_norm": float(np.linalg.norm(chosen_delta)),
    }
    return chosen_delta.astype(np.float32), trial_engine, info


def apply_bre_compliance_control(
    action_seq: np.ndarray,
    engine: PhysicsEngine,
    config: ComplianceControlConfig,
    safety_filter: Optional[QPSafetyFilter] = None,
    obstacles: Optional[np.ndarray] = None,
    segment_obstacles: Optional[np.ndarray] = None,
    bre: bool = True,
) -> ComplianceControlResult:
    """Apply BRE compliance damping and project each previewed step onto safety constraints."""
    actions = np.asarray(action_seq, dtype=np.float32).copy()
    if actions.ndim != 2 or actions.shape[1] < 2:
        raise ValueError("forward-heading compliance expects action_seq with shape (N, 2)")

    safety_mode = _normalize_safety_mode(config.safety_mode)
    safety_filter = safety_filter or QPSafetyFilter()
    sim = copy.deepcopy(engine)
    min_forward = float(engine.robot_speed * config.data_dt * config.min_forward_scale)
    max_forward = float(engine.robot_speed * config.data_dt * config.max_forward_scale)
    max_heading_delta = (
        float(engine.turn_speed * config.data_dt)
        / max(float(config.turn_gain), 1e-6)
        * float(config.heading_clip_scale)
    )

    nominal_deltas: list[np.ndarray] = []
    safe_deltas: list[np.ndarray] = []
    infos: list[dict] = []
    shifts: list[float] = []
    action_shifts: list[float] = []
    stats = {
        "applied": True,
        "safety_applied": safety_mode != "off" and _has_obstacles(obstacles, segment_obstacles),
        "modified_steps": 0,
        "safety_modified_steps": 0,
        "total_steps": int(len(actions)),
        "mean_shift": 0.0,
        "mean_action_shift": 0.0,
        "constraint_count": 0,
        "min_clearance": float("inf"),
    }

    for action in actions:
        before_action = action.copy()
        link = sim.robot.position - sim.human.position
        link_length = float(np.linalg.norm(link))
        if config.preserve_heading:
            action[1] = 0.0
        elif config.heading_control and link_length > 1e-6:
            link_dir = link / link_length
            lateral_dir = np.array([-link_dir[1], link_dir[0]], dtype=np.float32)
            relative_velocity = sim.robot.velocity - sim.human.velocity
            lateral_velocity = float(np.dot(relative_velocity, lateral_dir))
            link_angular_velocity = lateral_velocity / link_length
            damping_turn = -float(config.heading_damping) * link_angular_velocity * float(config.data_dt)
            action[1] = (
                float(config.heading_mix) * float(action[1])
                + damping_turn / max(float(config.turn_gain), 1e-6)
            )
            action[1] = float(np.clip(action[1], -max_heading_delta, max_heading_delta))

        if config.forward_only_slowdown:
            forward = float(action[0])
            action[0] = float(np.clip(forward, 0.0, max_forward))
        else:
            action[0] = float(np.clip(action[0], min_forward, max_forward))
        nominal_delta, nominal_preview = _forward_heading_action_to_nominal_delta(
            sim,
            action,
            config,
            bre,
        )
        action_collision_before_qp = False
        if stats["safety_applied"]:
            protect_human = safety_mode == "human_robot_qp"
            action_check_engine = copy.deepcopy(sim)
            action_collision_before_qp, _ = _simulate_action_on_engine(
                action_check_engine,
                action,
                config,
                bre,
                obstacles=obstacles,
                segment_obstacles=segment_obstacles,
                protect_robot=True,
                protect_human=protect_human,
            )
        chosen_delta, trial_engine, info = _safety_filter_delta(
            sim,
            nominal_delta,
            nominal_preview,
            config,
            bre,
            safety_filter,
            obstacles,
            segment_obstacles,
        )
        info["action_collision_before_qp"] = bool(action_collision_before_qp)
        project_action = bool(
            info.get("modified", False)
            or info.get("collision_after_qp", False)
            or info.get("backoff_applied", False)
            or action_collision_before_qp
        )
        if info.get("safety_applied", False) and project_action:
            safe_action = _world_delta_to_forward_heading_action(sim, chosen_delta, config)
            safe_action[0] = float(np.clip(safe_action[0], 0.0, max_forward))
            action[:] = safe_action
            info["modified"] = True
            info["action_projection_applied"] = True
            sim = trial_engine
        else:
            info["action_projection_applied"] = False
            sim = nominal_preview

        action_shift = float(np.linalg.norm(action - before_action))
        action_shifts.append(action_shift)
        nominal_deltas.append(nominal_delta.astype(np.float32))
        safe_deltas.append(chosen_delta.astype(np.float32))
        infos.append(info)
        shifts.append(float(info.get("shift", 0.0)))
        stats["constraint_count"] = max(
            int(stats["constraint_count"]),
            int(info.get("constraint_count", 0)),
        )
        stats["min_clearance"] = min(
            float(stats["min_clearance"]),
            float(info.get("min_clearance", float("inf"))),
        )
        if action_shift > 1e-5:
            stats["modified_steps"] += 1
        if bool(info.get("modified", False)):
            stats["safety_modified_steps"] += 1

    if shifts:
        stats["mean_shift"] = float(np.mean(shifts))
    if action_shifts:
        stats["mean_action_shift"] = float(np.mean(action_shifts))

    return ComplianceControlResult(
        actions=actions.astype(np.float32),
        nominal_deltas=np.asarray(nominal_deltas, dtype=np.float32),
        safe_deltas=np.asarray(safe_deltas, dtype=np.float32),
        infos=infos,
        stats=stats,
    )


def apply_interaction_aware_compliance_control(
    action_seq: np.ndarray,
    interaction_labels: np.ndarray,
    engine: PhysicsEngine,
    config: ComplianceControlConfig,
    safety_filter: Optional[QPSafetyFilter] = None,
    obstacles: Optional[np.ndarray] = None,
    segment_obstacles: Optional[np.ndarray] = None,
    bre: bool = True,
    bre_sequence: Optional[np.ndarray] = None,
    compliance_labels: tuple[str, ...] = ("leash", "tether"),
    guide_safety: bool = True,
) -> ComplianceControlResult:
    """Apply compliance only on tether/leash states while keeping safety projection active."""
    raw_actions = np.asarray(action_seq, dtype=np.float32)
    labels = np.asarray(interaction_labels, dtype=object)
    if raw_actions.ndim != 2 or raw_actions.shape[1] < 2:
        raise ValueError("interaction-aware compliance expects action_seq with shape (N, 2)")
    if labels.shape[0] != raw_actions.shape[0]:
        raise ValueError("interaction_labels must have one label per action")
    bre_flags = None
    if bre_sequence is not None:
        bre_flags = np.asarray(bre_sequence, dtype=bool).reshape(-1)
        if bre_flags.shape[0] != raw_actions.shape[0]:
            raise ValueError("bre_sequence must have one flag per action")

    normalized_compliance_labels = {str(label).strip().lower() for label in compliance_labels}
    sim = copy.deepcopy(engine)
    selected_actions: list[np.ndarray] = []
    nominal_deltas: list[np.ndarray] = []
    safe_deltas: list[np.ndarray] = []
    infos: list[dict] = []
    action_shifts: list[float] = []
    safety_shifts: list[float] = []
    state_counts: dict[str, int] = {}
    compliance_steps = 0

    safety_filter = safety_filter or QPSafetyFilter()
    for idx, raw_action in enumerate(raw_actions):
        label = str(labels[idx]).strip().lower()
        step_bre = bool(bre_flags[idx]) if bre_flags is not None else bool(bre)
        state_counts[label] = state_counts.get(label, 0) + 1
        if label in normalized_compliance_labels:
            step_result = apply_bre_compliance_control(
                raw_action[None, :],
                sim,
                config,
                safety_filter=safety_filter,
                obstacles=obstacles,
                segment_obstacles=segment_obstacles,
                bre=step_bre,
            )
            selected_action = step_result.actions[0].astype(np.float32)
            info = dict(step_result.infos[0] if step_result.infos else _empty_info())
            nominal_delta = (
                step_result.nominal_deltas[0].astype(np.float32)
                if len(step_result.nominal_deltas)
                else _forward_heading_action_to_world_delta(raw_action, sim.robot.heading)
            )
            safe_delta = (
                step_result.safe_deltas[0].astype(np.float32)
                if len(step_result.safe_deltas)
                else nominal_delta.copy()
            )
            compliance_steps += 1
        else:
            selected_action = raw_action.astype(np.float32).copy()
            nominal_delta, nominal_preview = _forward_heading_action_to_nominal_delta(
                sim,
                selected_action,
                config,
                step_bre,
            )
            safe_delta = nominal_delta.copy()
            info = _empty_info()
            if guide_safety:
                safety_mode = _normalize_safety_mode(config.safety_mode)
                if safety_mode != "off" and _has_obstacles(obstacles, segment_obstacles):
                    protect_human = safety_mode == "human_robot_qp"
                    action_check_engine = copy.deepcopy(sim)
                    action_collision_before_qp, _ = _simulate_action_on_engine(
                        action_check_engine,
                        selected_action,
                        config,
                        step_bre,
                        obstacles=obstacles,
                        segment_obstacles=segment_obstacles,
                        protect_robot=True,
                        protect_human=protect_human,
                    )
                    chosen_delta, _trial_engine, info = _safety_filter_delta(
                        sim,
                        nominal_delta,
                        nominal_preview,
                        config,
                        step_bre,
                        safety_filter,
                        obstacles,
                        segment_obstacles,
                    )
                    info["action_collision_before_qp"] = bool(action_collision_before_qp)
                    project_action = bool(
                        info.get("modified", False)
                        or info.get("collision_after_qp", False)
                        or info.get("backoff_applied", False)
                        or action_collision_before_qp
                    )
                    if info.get("safety_applied", False) and project_action:
                        selected_action = _world_delta_to_forward_heading_action(
                            sim,
                            chosen_delta,
                            config,
                        )
                        max_forward_delta = float(sim.robot_speed * config.data_dt)
                        selected_action[0] = float(
                            np.clip(selected_action[0], 0.0, max_forward_delta)
                        )
                        info["modified"] = True
                        info["action_projection_applied"] = True
                    else:
                        info["action_projection_applied"] = False
                    safe_delta = chosen_delta.astype(np.float32)

        info["interaction_label"] = label
        info["compliance_enabled"] = bool(label in normalized_compliance_labels)
        info["bre"] = bool(step_bre)
        selected_actions.append(selected_action.astype(np.float32))
        nominal_deltas.append(nominal_delta.astype(np.float32))
        safe_deltas.append(safe_delta.astype(np.float32))
        infos.append(info)
        action_shifts.append(float(np.linalg.norm(selected_action - raw_action)))
        safety_shifts.append(float(info.get("shift", 0.0)))
        _simulate_action_on_engine(
            sim,
            selected_action,
            config,
            step_bre,
            obstacles=None,
            segment_obstacles=None,
            protect_robot=False,
            protect_human=False,
        )

    stats = {
        "applied": True,
        "total_steps": int(len(raw_actions)),
        "compliance_steps": int(compliance_steps),
        "guide_steps": int(len(raw_actions) - compliance_steps),
        "bre_steps": (
            int(np.sum(bre_flags))
            if bre_flags is not None
            else int(bool(bre)) * int(len(raw_actions))
        ),
        "state_counts": state_counts,
        "mean_action_shift": float(np.mean(action_shifts)) if action_shifts else 0.0,
        "mean_shift": float(np.mean(safety_shifts)) if safety_shifts else 0.0,
        "modified_steps": int(sum(shift > 1e-5 for shift in action_shifts)),
        "safety_modified_steps": int(sum(bool(info.get("modified", False)) for info in infos)),
        "constraint_count": int(
            max([int(info.get("constraint_count", 0)) for info in infos], default=0)
        ),
        "min_clearance": float(
            min(
                [float(info.get("min_clearance", float("inf"))) for info in infos],
                default=float("inf"),
            )
        ),
    }
    return ComplianceControlResult(
        actions=np.asarray(selected_actions, dtype=np.float32),
        nominal_deltas=np.asarray(nominal_deltas, dtype=np.float32),
        safe_deltas=np.asarray(safe_deltas, dtype=np.float32),
        infos=infos,
        stats=stats,
    )
