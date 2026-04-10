"""Helpers for adding interactive obstacles at runtime."""

from __future__ import annotations

from typing import Optional

import numpy as np


def _normalize_circle_obstacles(obstacles: Optional[np.ndarray]) -> np.ndarray:
    if obstacles is None:
        return np.zeros((0, 3), dtype=np.float32)

    normalized = []
    for obs in obstacles:
        if isinstance(obs, dict):
            normalized.append(
                [
                    float(obs.get("x", 0.0)),
                    float(obs.get("y", 0.0)),
                    float(obs.get("r", 0.0)),
                ]
            )
        else:
            normalized.append([float(obs[0]), float(obs[1]), float(obs[2])])

    if not normalized:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(normalized, dtype=np.float32)


def append_forward_circle_obstacle(
    obstacles: Optional[np.ndarray],
    robot_pos: np.ndarray,
    heading: float,
    radius: float,
    base_distance: float = 2.0,
    clearance: float = 0.15,
    step_distance: Optional[float] = None,
    max_attempts: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Append a circle obstacle along the robot's forward center line."""
    robot_pos = np.asarray(robot_pos, dtype=np.float32).reshape(2)
    radius = float(radius)
    base_distance = float(base_distance)
    clearance = float(clearance)
    step_distance = (
        max(radius * 2.5, 0.75)
        if step_distance is None
        else max(float(step_distance), radius * 2.0)
    )

    direction = np.array([np.cos(heading), np.sin(heading)], dtype=np.float32)
    current = _normalize_circle_obstacles(obstacles)

    candidate = robot_pos + direction * base_distance
    for attempt in range(max(1, int(max_attempts))):
        candidate = robot_pos + direction * (base_distance + attempt * step_distance)
        if len(current) == 0:
            break

        deltas = current[:, :2] - candidate[None, :]
        min_allowed = current[:, 2] + radius + clearance
        if np.all(np.sum(deltas * deltas, axis=1) > (min_allowed * min_allowed)):
            break

    new_obstacle = np.array([[candidate[0], candidate[1], radius]], dtype=np.float32)
    if len(current) == 0:
        return new_obstacle, candidate
    return np.concatenate([current, new_obstacle], axis=0), candidate
