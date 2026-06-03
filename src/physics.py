"""
Physics model: robot motion + rigid leash interaction with a human.

Key changes compared with the original implementation:
1. The leash is modeled as a rigid unilateral constraint, not a spring with hard position projection.
2. The robot is dynamic: commanded velocity is converted into drive force, so leash reaction can slow it down.
3. Human behavior modes are supported: follow / stop / side_pull / custom.
4. Leash force is estimated from the constraint impulse, so sudden stops and acceleration shocks are visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


EPS = 1e-9


def _vec2(x: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return a float64 2D vector copy."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.shape != (2,):
        raise ValueError(f"Expected shape (2,), got {arr.shape}")
    return arr.copy()


def _clip_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    """Clip vector magnitude without changing direction."""
    if max_norm <= 0.0:
        return np.zeros_like(v)
    norm = float(np.linalg.norm(v))
    if norm <= max_norm or norm < EPS:
        return v
    return v / norm * max_norm


@dataclass
class RobotState:
    """Robot state in the world frame."""

    position: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    heading: float = 0.0

    def copy(self) -> "RobotState":
        return RobotState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            heading=float(self.heading),
        )


@dataclass
class HumanState:
    """Human state in the world frame."""

    position: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))

    def copy(self) -> "HumanState":
        return HumanState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
        )


class PhysicsEngine:
    """Physics engine for robot-guided walking with a rigid leash.

    The leash is unilateral:
    - If distance < leash_length, it is slack and applies no force.
    - If distance >= leash_length and the two bodies are moving apart, an impulse constraint
      removes the separating velocity.
    - A small position correction only removes numerical drift; it is not used as the main physics.
    """

    def __init__(
        self,
        leash_length: float = 1.5,
        robot_speed: float = 1.5,
        turn_speed: float = 1.5,
        dt: float = 0.02,
        robot_radius: float = 0.3,
        human_radius: float = 0.3,
        # Dynamics
        robot_mass: float = 18.0,
        human_mass: float = 65.0,
        robot_drive_gain: float = 160.0,
        robot_max_drive_force: float = 220.0,
        robot_drag_coeff: float = 8.0,
        human_drag_coeff: float = 18.0,
        human_max_force: float = 180.0,
        human_max_speed: float = 1.8,
        # Human behavior gains
        human_follow_gain: float = 90.0,
        human_brake_gain: float = 220.0,
        human_pull_gain: float = 160.0,
        # Constraint solver
        constraint_beta: float = 0.15,
        max_position_correction: float = 0.05,
        # Backward-compatible arguments from the old code
        human_drag: Optional[float] = None,
        leash_stiffness: Optional[float] = None,
    ) -> None:
        if leash_length <= 0.0:
            raise ValueError("leash_length must be positive")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if robot_mass <= 0.0 or human_mass <= 0.0:
            raise ValueError("robot_mass and human_mass must be positive")

        self.leash_length = float(leash_length)
        self.robot_speed = float(robot_speed)
        self.turn_speed = float(turn_speed)
        self.dt = float(dt)
        self.robot_radius = float(robot_radius)
        self.human_radius = float(human_radius)

        self.robot_mass = float(robot_mass)
        self.human_mass = float(human_mass)
        self.robot_drive_gain = float(robot_drive_gain)
        self.robot_max_drive_force = float(robot_max_drive_force)
        self.robot_drag_coeff = float(robot_drag_coeff)
        self.human_drag_coeff = float(human_drag_coeff)
        self.human_max_force = float(human_max_force)
        self.human_max_speed = float(human_max_speed)

        self.human_follow_gain = float(human_follow_gain)
        self.human_brake_gain = float(human_brake_gain)
        self.human_pull_gain = float(human_pull_gain)

        self.constraint_beta = float(np.clip(constraint_beta, 0.0, 1.0))
        self.max_position_correction = float(max_position_correction)

        # Kept only so old constructors will still run. Rigid leash does not use stiffness.
        self.legacy_human_drag = human_drag
        self.legacy_leash_stiffness = leash_stiffness

        self.robot = RobotState()
        self.human = HumanState()

        self.forward_input = 0.0
        self.turn_input = 0.0

        self.human_mode = "follow"  # follow | stop | side_pull | custom
        self.human_desired_velocity = np.zeros(2, dtype=np.float64)
        self.human_side_velocity = np.zeros(2, dtype=np.float64)

        self.last_leash_force = np.zeros(2, dtype=np.float64)  # force on human, N
        self.last_leash_force_magnitude = 0.0
        self.last_robot_drive_force = np.zeros(2, dtype=np.float64)
        self.last_human_intent_force = np.zeros(2, dtype=np.float64)

    def reset(self, start_position: Optional[np.ndarray] = None, heading: float = 0.0) -> None:
        """Reset robot and human states."""
        start = np.zeros(2, dtype=np.float64) if start_position is None else _vec2(start_position)

        self.robot = RobotState(
            position=start.copy(),
            velocity=np.zeros(2, dtype=np.float64),
            heading=float(heading),
        )

        # Human starts behind the robot with slack in the leash.
        backward = np.array([-np.cos(heading), -np.sin(heading)], dtype=np.float64)
        self.human = HumanState(
            position=start + backward * self.leash_length * 0.8,
            velocity=np.zeros(2, dtype=np.float64),
        )

        self.forward_input = 0.0
        self.turn_input = 0.0
        self.human_mode = "follow"
        self.human_desired_velocity[:] = 0.0
        self.human_side_velocity[:] = 0.0
        self._clear_force_logs()

    def set_control(self, forward: float, turn: float) -> None:
        """Set normalized robot command in [-1, 1]."""
        self.forward_input = float(np.clip(forward, -1.0, 1.0))
        self.turn_input = float(np.clip(turn, -1.0, 1.0))

    def set_human_mode(
        self,
        mode: str,
        desired_velocity: Optional[np.ndarray] = None,
        side_velocity: Optional[np.ndarray] = None,
    ) -> None:
        """Set human behavior mode.

        Args:
            mode: "follow", "stop", "side_pull", or "custom".
            desired_velocity: used by "custom" mode.
            side_velocity: used by "side_pull" mode.
        """
        valid_modes = {"follow", "stop", "side_pull", "custom"}
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {sorted(valid_modes)}, got {mode!r}")
        self.human_mode = mode
        if desired_velocity is not None:
            self.human_desired_velocity = _vec2(desired_velocity)
        if side_velocity is not None:
            self.human_side_velocity = _vec2(side_velocity)

    def step(self) -> tuple[RobotState, HumanState]:
        """Advance simulation by one time step."""
        self._clear_force_logs()

        # Heading is controlled directly; translational motion is dynamic.
        self.robot.heading += self.turn_input * self.turn_speed * self.dt
        self.robot.heading = float(np.arctan2(np.sin(self.robot.heading), np.cos(self.robot.heading)))

        robot_force = self._compute_robot_drive_force()
        human_force = self._compute_human_intent_force()
        robot_drag = -self.robot_drag_coeff * self.robot.velocity
        human_drag = -self.human_drag_coeff * self.human.velocity

        self.last_robot_drive_force = robot_force.copy()
        self.last_human_intent_force = human_force.copy()

        # Semi-implicit Euler: update velocities first.
        self.robot.velocity += (robot_force + robot_drag) / self.robot_mass * self.dt
        self.human.velocity += (human_force + human_drag) / self.human_mass * self.dt
        self.human.velocity = _clip_norm(self.human.velocity, self.human_max_speed)

        # Predict positions.
        self.robot.position += self.robot.velocity * self.dt
        self.human.position += self.human.velocity * self.dt

        # Apply rigid leash constraint. This is where leash shock/force is produced.
        self._apply_rigid_leash_constraint()

        return self.robot.copy(), self.human.copy()

    def _compute_robot_drive_force(self) -> np.ndarray:
        direction = np.array(
            [np.cos(self.robot.heading), np.sin(self.robot.heading)],
            dtype=np.float64,
        )
        target_velocity = direction * self.forward_input * self.robot_speed
        force = self.robot_drive_gain * (target_velocity - self.robot.velocity)
        return _clip_norm(force, self.robot_max_drive_force)

    def _compute_human_intent_force(self) -> np.ndarray:
        if self.human_mode == "follow":
            desired_velocity = self.robot.velocity.copy()
            gain = self.human_follow_gain
        elif self.human_mode == "stop":
            desired_velocity = np.zeros(2, dtype=np.float64)
            gain = self.human_brake_gain
        elif self.human_mode == "side_pull":
            desired_velocity = self.human_side_velocity.copy()
            gain = self.human_pull_gain
        elif self.human_mode == "custom":
            desired_velocity = self.human_desired_velocity.copy()
            gain = self.human_pull_gain
        else:  # Should never happen because set_human_mode validates this.
            desired_velocity = np.zeros(2, dtype=np.float64)
            gain = self.human_brake_gain

        force = gain * (desired_velocity - self.human.velocity)
        return _clip_norm(force, self.human_max_force)

    def _apply_rigid_leash_constraint(self) -> None:
        """Apply unilateral rigid leash constraint using impulse resolution."""
        r = self.human.position - self.robot.position
        distance = float(np.linalg.norm(r))
        if distance < EPS:
            return

        # Slack leash: no tension.
        if distance < self.leash_length:
            return

        n = r / distance  # robot -> human
        inv_m_h = 1.0 / self.human_mass
        inv_m_r = 1.0 / self.robot_mass
        inv_mass_sum = inv_m_h + inv_m_r

        # Relative velocity along the rope. Positive means distance is increasing.
        separating_speed = float(np.dot(self.human.velocity - self.robot.velocity, n))
        position_error = max(0.0, distance - self.leash_length)
        bias_speed = self.constraint_beta * position_error / self.dt

        # Impulse only pulls; it cannot push.
        remove_speed = separating_speed + bias_speed
        if remove_speed > 0.0:
            impulse = remove_speed / inv_mass_sum

            self.human.velocity -= impulse * inv_m_h * n
            self.robot.velocity += impulse * inv_m_r * n

            force_on_human = -impulse / self.dt * n
            self.last_leash_force = force_on_human
            self.last_leash_force_magnitude = float(np.linalg.norm(force_on_human))

        # Small position correction for numerical drift only.
        self._correct_leash_position_drift()

    def _correct_leash_position_drift(self) -> None:
        r = self.human.position - self.robot.position
        distance = float(np.linalg.norm(r))
        if distance <= self.leash_length or distance < EPS:
            return

        n = r / distance
        error = min(distance - self.leash_length, self.max_position_correction)

        inv_m_h = 1.0 / self.human_mass
        inv_m_r = 1.0 / self.robot_mass
        correction = error / (inv_m_h + inv_m_r)

        self.human.position -= correction * inv_m_h * n
        self.robot.position += correction * inv_m_r * n

    def _clear_force_logs(self) -> None:
        self.last_leash_force = np.zeros(2, dtype=np.float64)
        self.last_leash_force_magnitude = 0.0
        self.last_robot_drive_force = np.zeros(2, dtype=np.float64)
        self.last_human_intent_force = np.zeros(2, dtype=np.float64)

    def get_leash_tension(self) -> float:
        """Return a normalized tautness proxy in [0, 1]."""
        distance = float(np.linalg.norm(self.human.position - self.robot.position))
        return float(np.clip(distance / self.leash_length, 0.0, 1.0))

    def get_leash_force(self) -> tuple[float, np.ndarray]:
        """Return previous-step leash force estimate in Newtons, acting on the human."""
        return self.last_leash_force_magnitude, self.last_leash_force.copy()

    def get_debug_forces(self) -> dict[str, np.ndarray | float]:
        """Return debug force logs for plotting or reward design."""
        return {
            "leash_force": self.last_leash_force.copy(),
            "leash_force_magnitude": self.last_leash_force_magnitude,
            "robot_drive_force": self.last_robot_drive_force.copy(),
            "human_intent_force": self.last_human_intent_force.copy(),
        }

    def check_collision(
        self,
        obstacles,
        segment_obstacles=None,
        robot_radius: Optional[float] = None,
        human_radius: Optional[float] = None,
    ):
        """Check collision between robot/human and circle or segment obstacles."""
        if (obstacles is None or len(obstacles) == 0) and (
            segment_obstacles is None or len(segment_obstacles) == 0
        ):
            return False, None

        robot_radius = self.robot_radius if robot_radius is None else float(robot_radius)
        human_radius = self.human_radius if human_radius is None else float(human_radius)

        robot_pos = self.robot.position
        human_pos = self.human.position

        if obstacles is not None:
            for idx, obs in enumerate(obstacles):
                if isinstance(obs, dict):
                    ox = float(obs.get("x", 0.0))
                    oy = float(obs.get("y", 0.0))
                    radius = float(obs.get("r", 0.0))
                else:
                    ox = float(obs[0])
                    oy = float(obs[1])
                    radius = float(obs[2])

                if np.sum((robot_pos - np.array([ox, oy])) ** 2) <= (radius + robot_radius) ** 2:
                    return True, {
                        "type": "circle",
                        "who": "robot",
                        "idx": int(idx),
                        "obstacle": [ox, oy, radius],
                    }

                if np.sum((human_pos - np.array([ox, oy])) ** 2) <= (radius + human_radius) ** 2:
                    return True, {
                        "type": "circle",
                        "who": "human",
                        "idx": int(idx),
                        "obstacle": [ox, oy, radius],
                    }

        if segment_obstacles is not None:
            for idx, seg in enumerate(segment_obstacles):
                p1, p2 = self._parse_segment(seg)
                if self._point_segment_dist_sq(robot_pos, p1, p2) <= robot_radius**2:
                    return True, {
                        "type": "segment",
                        "who": "robot",
                        "idx": int(idx),
                        "obstacle": [float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1])],
                    }
                if self._point_segment_dist_sq(human_pos, p1, p2) <= human_radius**2:
                    return True, {
                        "type": "segment",
                        "who": "human",
                        "idx": int(idx),
                        "obstacle": [float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1])],
                    }

        return False, None

    @staticmethod
    def _parse_segment(seg) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(seg, dict):
            if "p1" in seg and "p2" in seg:
                return _vec2(seg["p1"]), _vec2(seg["p2"])
            return _vec2([seg.get("x1", 0.0), seg.get("y1", 0.0)]), _vec2(
                [seg.get("x2", 0.0), seg.get("y2", 0.0)]
            )
        return _vec2([seg[0], seg[1]]), _vec2([seg[2], seg[3]])

    @staticmethod
    def _point_segment_dist_sq(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom < EPS:
            diff = point - a
            return float(np.dot(diff, diff))
        t = float(np.dot(point - a, ab) / denom)
        t = float(np.clip(t, 0.0, 1.0))
        closest = a + t * ab
        diff = point - closest
        return float(np.dot(diff, diff))


if __name__ == "__main__":
    engine = PhysicsEngine()
    engine.reset(np.array([0.0, 0.0]))
    engine.set_control(forward=1.0, turn=0.0)

    for i in range(160):
        if i == 60:
            engine.set_human_mode("stop")
        robot, human = engine.step()
        if i % 20 == 0:
            force_mag, _ = engine.get_leash_force()
            dist = np.linalg.norm(human.position - robot.position)
            print(
                f"Step {i:03d} | "
                f"Robot={robot.position.round(3)} | "
                f"Human={human.position.round(3)} | "
                f"dist={dist:.3f} | "
                f"leash_force={force_mag:.2f} N | "
                f"mode={engine.human_mode}"
            )