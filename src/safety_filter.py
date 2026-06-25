"""QP-style safety filtering for 2D waypoint deltas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np


@dataclass
class SafetyConstraint:
    g: np.ndarray
    h: float
    clearance: float
    predicted_clearance: float
    source: str


@dataclass
class SafetyProjectionResult:
    delta: np.ndarray
    modified: bool
    constraint_count: int
    min_clearance: float
    total_constraint_count: int = 0
    ref_feasible: bool = True
    candidate_count: int = 0
    best_candidate_kind: str = "ref"
    best_candidate_constraints: list[int] = field(default_factory=list)
    selected_constraints: list[dict] = field(default_factory=list)


def _as_circle_array(obstacles: Optional[np.ndarray]) -> np.ndarray:
    if obstacles is None:
        return np.zeros((0, 3), dtype=np.float32)
    rows = []
    for obs in obstacles:
        if isinstance(obs, dict):
            rows.append(
                [float(obs.get("x", 0.0)), float(obs.get("y", 0.0)), float(obs.get("r", 0.0))]
            )
        else:
            rows.append([float(obs[0]), float(obs[1]), float(obs[2])])
    if not rows:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def _as_segment_array(segments: Optional[np.ndarray]) -> np.ndarray:
    if segments is None:
        return np.zeros((0, 4), dtype=np.float32)
    rows = []
    for seg in segments:
        if isinstance(seg, dict):
            if "p1" in seg and "p2" in seg:
                p1 = seg["p1"]
                p2 = seg["p2"]
                rows.append([float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1])])
            else:
                rows.append(
                    [
                        float(seg.get("x1", 0.0)),
                        float(seg.get("y1", 0.0)),
                        float(seg.get("x2", 0.0)),
                        float(seg.get("y2", 0.0)),
                    ]
                )
        else:
            rows.append([float(seg[0]), float(seg[1]), float(seg[2]), float(seg[3])])
    if not rows:
        return np.zeros((0, 4), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def _point_segment_closest(point: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    ab = p2 - p1
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return p1.copy()
    t = float(np.dot(point - p1, ab)) / denom
    t = float(np.clip(t, 0.0, 1.0))
    return p1 + t * ab


def _normalize(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm > 1e-9:
        return (vec / norm).astype(np.float32)
    fb_norm = float(np.linalg.norm(fallback))
    if fb_norm > 1e-9:
        return (fallback / fb_norm).astype(np.float32)
    return np.array([1.0, 0.0], dtype=np.float32)


def _project_halfspace_qp(
    ref_delta: np.ndarray,
    constraints: Iterable[SafetyConstraint],
    tol: float = 1e-6,
) -> tuple[np.ndarray, dict]:
    ref_delta = np.asarray(ref_delta, dtype=np.float32).reshape(2)
    constraints = list(constraints)
    if not constraints:
        return ref_delta, {
            "ref_feasible": True,
            "candidate_count": 1,
            "best_candidate_kind": "ref",
            "best_candidate_constraints": [],
        }

    def feasible(delta: np.ndarray) -> bool:
        for item in constraints:
            if float(np.dot(item.g, delta)) > item.h + tol:
                return False
        return True

    candidates: list[dict] = []
    ref_feasible = feasible(ref_delta)
    if ref_feasible:
        candidates.append(
            {
                "delta": ref_delta.astype(np.float32),
                "kind": "ref",
                "constraint_indices": [],
            }
        )

    for idx, item in enumerate(constraints):
        g = item.g
        denom = float(np.dot(g, g))
        if denom < 1e-9:
            continue
        violation = float(np.dot(g, ref_delta) - item.h)
        cand = ref_delta - max(0.0, violation) * g / denom
        if feasible(cand):
            candidates.append(
                {
                    "delta": cand.astype(np.float32),
                    "kind": "single",
                    "constraint_indices": [int(idx)],
                }
            )

    for i in range(len(constraints)):
        gi = constraints[i].g
        hi = constraints[i].h
        for j in range(i + 1, len(constraints)):
            gj = constraints[j].g
            hj = constraints[j].h
            A = np.stack([gi, gj], axis=0)
            det = float(np.linalg.det(A))
            if abs(det) < 1e-9:
                continue
            cand = np.linalg.solve(A, np.array([hi, hj], dtype=np.float32)).astype(np.float32)
            if feasible(cand):
                candidates.append(
                    {
                        "delta": cand,
                        "kind": "pair",
                        "constraint_indices": [int(i), int(j)],
                    }
                )

    if not candidates:
        return np.zeros((2,), dtype=np.float32), {
            "ref_feasible": bool(ref_feasible),
            "candidate_count": 0,
            "best_candidate_kind": "zero_fallback",
            "best_candidate_constraints": [],
        }

    best = min(
        candidates,
        key=lambda item: float(np.sum((item["delta"] - ref_delta) ** 2)),
    )
    return np.asarray(best["delta"], dtype=np.float32), {
        "ref_feasible": bool(ref_feasible),
        "candidate_count": int(len(candidates)),
        "best_candidate_kind": str(best["kind"]),
        "best_candidate_constraints": [int(v) for v in best["constraint_indices"]],
    }


class QPSafetyFilter:
    """Project nominal 2D displacements onto linearized obstacle-safe halfspaces."""

    def __init__(
        self,
        margin: float = 0.2,
        alpha: float = 1.0,
        max_constraints: int = 8,
        influence_distance: float = 0.1,
    ):
        self.margin = float(margin)
        self.alpha = float(alpha)
        self.max_constraints = max(1, int(max_constraints))
        self.influence_distance = float(influence_distance)

    def _circle_constraint(
        self,
        point: np.ndarray,
        ref_delta: np.ndarray,
        center: np.ndarray,
        radius: float,
        tag: str,
    ) -> SafetyConstraint:
        rel = point - center
        grad = _normalize(rel, fallback=-ref_delta)
        dist = float(np.linalg.norm(rel))
        clearance = dist - float(radius) - self.margin
        predicted_clearance = clearance + float(np.dot(grad, ref_delta))
        return SafetyConstraint(
            g=-grad.astype(np.float32),
            h=float(self.alpha * clearance),
            clearance=float(clearance),
            predicted_clearance=float(predicted_clearance),
            source=tag,
        )

    def _segment_constraint(
        self,
        point: np.ndarray,
        ref_delta: np.ndarray,
        p1: np.ndarray,
        p2: np.ndarray,
        radius: float,
        tag: str,
    ) -> SafetyConstraint:
        closest = _point_segment_closest(point, p1, p2)
        rel = point - closest
        seg_dir = p2 - p1
        seg_normal = np.array([-seg_dir[1], seg_dir[0]], dtype=np.float32)
        grad = _normalize(rel, fallback=seg_normal if np.linalg.norm(seg_normal) > 1e-9 else -ref_delta)
        dist = float(np.linalg.norm(rel))
        clearance = dist - float(radius) - self.margin
        predicted_clearance = clearance + float(np.dot(grad, ref_delta))
        return SafetyConstraint(
            g=-grad.astype(np.float32),
            h=float(self.alpha * clearance),
            clearance=float(clearance),
            predicted_clearance=float(predicted_clearance),
            source=tag,
        )

    def project_delta(
        self,
        ref_delta: np.ndarray,
        robot_pos: np.ndarray,
        robot_radius: float,
        human_pos: Optional[np.ndarray] = None,
        human_radius: Optional[float] = None,
        circle_obstacles: Optional[np.ndarray] = None,
        segment_obstacles: Optional[np.ndarray] = None,
        include_human: bool = True,
        extra_entities: Optional[list[tuple[str, np.ndarray, float]]] = None,
    ) -> SafetyProjectionResult:
        ref_delta = np.asarray(ref_delta, dtype=np.float32).reshape(2)
        robot_pos = np.asarray(robot_pos, dtype=np.float32).reshape(2)
        entities = [("robot", robot_pos, float(robot_radius))]
        if include_human and human_pos is not None and human_radius is not None:
            entities.append(
                ("human", np.asarray(human_pos, dtype=np.float32).reshape(2), float(human_radius))
            )
        if extra_entities:
            for name, pos, radius in extra_entities:
                entities.append(
                    (
                        str(name),
                        np.asarray(pos, dtype=np.float32).reshape(2),
                        float(radius),
                    )
                )

        constraints: list[SafetyConstraint] = []
        circles = _as_circle_array(circle_obstacles)
        segments = _as_segment_array(segment_obstacles)

        for name, point, radius in entities:
            for idx, obs in enumerate(circles):
                center = obs[:2]
                constraint = self._circle_constraint(
                    point=point,
                    ref_delta=ref_delta,
                    center=center,
                    radius=float(obs[2]) + radius,
                    tag=f"{name}:circle:{idx}",
                )
                if (
                    constraint.clearance <= self.influence_distance
                    or constraint.predicted_clearance <= self.influence_distance
                ):
                    constraints.append(constraint)

            for idx, seg in enumerate(segments):
                p1 = seg[:2]
                p2 = seg[2:4]
                constraint = self._segment_constraint(
                    point=point,
                    ref_delta=ref_delta,
                    p1=p1,
                    p2=p2,
                    radius=radius,
                    tag=f"{name}:segment:{idx}",
                )
                if (
                    constraint.clearance <= self.influence_distance
                    or constraint.predicted_clearance <= self.influence_distance
                ):
                    constraints.append(constraint)

        if not constraints:
            return SafetyProjectionResult(
                delta=ref_delta,
                modified=False,
                constraint_count=0,
                min_clearance=float("inf"),
                total_constraint_count=0,
                ref_feasible=True,
                candidate_count=1,
                best_candidate_kind="ref",
                best_candidate_constraints=[],
                selected_constraints=[],
            )

        constraints.sort(key=lambda item: (item.predicted_clearance, item.clearance))
        selected = constraints[: self.max_constraints]
        projected, qp_debug = _project_halfspace_qp(ref_delta, selected)
        return SafetyProjectionResult(
            delta=projected.astype(np.float32),
            modified=bool(np.linalg.norm(projected - ref_delta) > 1e-5),
            constraint_count=len(selected),
            min_clearance=float(min(item.clearance for item in selected)),
            total_constraint_count=int(len(constraints)),
            ref_feasible=bool(qp_debug["ref_feasible"]),
            candidate_count=int(qp_debug["candidate_count"]),
            best_candidate_kind=str(qp_debug["best_candidate_kind"]),
            best_candidate_constraints=[
                int(v) for v in qp_debug["best_candidate_constraints"]
            ],
            selected_constraints=[
                {
                    "index": int(idx),
                    "source": str(item.source),
                    "clearance": float(item.clearance),
                    "predicted_clearance": float(item.predicted_clearance),
                    "h": float(item.h),
                    "g": [float(item.g[0]), float(item.g[1])],
                    "ref_violation": float(np.dot(item.g, ref_delta) - item.h),
                }
                for idx, item in enumerate(selected)
            ],
        )
