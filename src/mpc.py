"""
Sampling-based MPC (CEM) for guide-follow simulation.

This module is intentionally standalone (numpy-only) so it can be reused by
planning/eval tools without pulling in policy code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def wrap_angle_np(angle: np.ndarray) -> np.ndarray:
    """Vectorized wrap to [-pi, pi] for numpy arrays."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _point_segment_dist_sq(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute squared distance from points to line segments.

    Args:
        points: (N,2)
        a: (S,2) segment start
        b: (S,2) segment end

    Returns:
        dist_sq: (N,S)
    """
    points = np.asarray(points, dtype=np.float32)
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    ab = (b - a).astype(np.float32)  # (S,2)
    ab_sq = np.sum(ab * ab, axis=1).astype(np.float32)  # (S,)
    denom = np.maximum(ab_sq[None, :], 1e-12)
    pa = points[:, None, :] - a[None, :, :]
    t = np.sum(pa * ab[None, :, :], axis=2) / denom
    t = np.clip(t, 0.0, 1.0).astype(np.float32)
    closest = a[None, :, :] + t[:, :, None] * ab[None, :, :]
    diff = points[:, None, :] - closest
    return np.sum(diff * diff, axis=2).astype(np.float32)


def _compute_path_s(path: np.ndarray) -> np.ndarray:
    path = np.asarray(path, dtype=np.float32)
    if len(path) == 0:
        return np.zeros((0,), dtype=np.float32)
    if len(path) == 1:
        return np.zeros((1,), dtype=np.float32)
    diffs = np.diff(path, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1).astype(np.float32)
    s = np.zeros((len(path),), dtype=np.float32)
    s[1:] = np.cumsum(seg_lengths, axis=0).astype(np.float32)
    return s


def _compute_path_heading(path: np.ndarray) -> np.ndarray:
    path = np.asarray(path, dtype=np.float32)
    if len(path) < 2:
        return np.zeros((len(path),), dtype=np.float32)
    diffs = np.diff(path, axis=0)
    headings = np.arctan2(diffs[:, 1], diffs[:, 0]).astype(np.float32)
    return np.concatenate([headings, headings[-1:]], axis=0).astype(np.float32)


def _normalize_obstacles(
    obstacles: Optional[object],
    segments: Optional[object],
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    circles_arr = None
    segs_arr = None
    if obstacles is not None:
        arr = np.asarray(obstacles, dtype=np.float32)
        if arr.size != 0:
            if arr.ndim >= 2 and arr.shape[-1] >= 3:
                circles_arr = arr.reshape(-1, arr.shape[-1])[:, :3]
            else:
                circles_arr = arr.reshape(-1, 3)[:, :3]
            if circles_arr.size == 0:
                circles_arr = None
    if segments is not None:
        arr = np.asarray(segments, dtype=np.float32)
        if arr.size != 0:
            if arr.ndim >= 2 and arr.shape[-1] >= 4:
                segs_arr = arr.reshape(-1, arr.shape[-1])[:, :4]
            else:
                segs_arr = arr.reshape(-1, 4)[:, :4]
            if segs_arr.size == 0:
                segs_arr = None
    return circles_arr, segs_arr


def _min_clearance(
    points: np.ndarray,
    *,
    circles: Optional[np.ndarray],
    segments: Optional[np.ndarray],
    agent_radius: float,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    n = int(points.shape[0])
    min_clear = np.full((n,), np.inf, dtype=np.float32)

    if circles is not None and len(circles) > 0:
        centers = circles[:, :2].astype(np.float32)
        radii = circles[:, 2].astype(np.float32)
        diff = points[:, None, :] - centers[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2) + 1e-12).astype(np.float32)
        clear = dist - (radii[None, :] + float(agent_radius))
        min_clear = np.minimum(min_clear, np.min(clear, axis=1))

    if segments is not None and len(segments) > 0:
        a = segments[:, :2].astype(np.float32)
        b = segments[:, 2:4].astype(np.float32)
        dist_sq = _point_segment_dist_sq(points, a, b)
        dist = np.sqrt(dist_sq + 1e-12).astype(np.float32)
        clear = dist - float(agent_radius)
        min_clear = np.minimum(min_clear, np.min(clear, axis=1))

    return min_clear.astype(np.float32)


@dataclass
class CEMMPCConfig:
    horizon: int = 8
    samples: int = 256
    iters: int = 2
    elites: int = 16
    forward: float = 0.8
    forward_std: float = 0.3
    turn_std: float = 0.8
    track: str = "human"  # "human" | "robot"

    w_track: float = 10.0
    w_heading: float = 1.0
    w_u: float = 0.1
    w_smooth: float = 0.05
    w_progress: float = 2.0

    # Safety shaping / constraints (robot + human, circles + segments).
    clearance_margin: float = 0.2
    w_clearance: float = 200.0
    w_collision: float = 1.0e5

    allow_reverse: bool = False


class CEMMPC:
    def __init__(
        self,
        *,
        config: CEMMPCConfig,
        sim_dt: float,
        frame_stride: int,
        robot_speed: float,
        turn_speed: float,
        leash_length: float,
        human_drag: float,
        robot_radius: float,
        human_radius: float,
        seed: int = 0,
    ):
        self.cfg = config
        self.sim_dt = float(sim_dt)
        self.frame_stride = max(1, int(frame_stride))
        self.robot_speed = float(robot_speed)
        self.turn_speed = float(turn_speed)
        self.leash_length = float(leash_length)
        self.human_drag = float(human_drag)
        self.robot_radius = float(robot_radius)
        self.human_radius = float(human_radius)
        self._rng = np.random.default_rng(int(seed))
        self._mean: Optional[np.ndarray] = None  # (H,2)

    def reset(self):
        self._mean = None

    def _rollout(
        self,
        u: np.ndarray,
        *,
        robot_pos: np.ndarray,
        robot_heading: float,
        robot_vel: np.ndarray,
        human_pos: np.ndarray,
        human_vel: np.ndarray,
        circles: Optional[np.ndarray],
        segments: Optional[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Rollout robot + human leash physics for candidate control sequences.

        Returns:
            robot_pos_hist: (N,H,2) sampled at data rate (after each action window)
            robot_heading_hist: (N,H)
            human_pos_hist: (N,H,2)
            min_clear_robot: (N,) minimum clearance over rollout
            min_clear_human: (N,)
        """
        u = np.asarray(u, dtype=np.float32)
        n, horizon, _ = u.shape

        r_pos = np.repeat(np.asarray(robot_pos, dtype=np.float32)[None, :], n, axis=0)
        r_vel = np.repeat(np.asarray(robot_vel, dtype=np.float32)[None, :], n, axis=0)
        r_heading = np.full((n,), float(robot_heading), dtype=np.float32)

        h_pos = np.repeat(np.asarray(human_pos, dtype=np.float32)[None, :], n, axis=0)
        h_vel = np.repeat(np.asarray(human_vel, dtype=np.float32)[None, :], n, axis=0)

        robot_pos_hist = np.zeros((n, horizon, 2), dtype=np.float32)
        robot_heading_hist = np.zeros((n, horizon), dtype=np.float32)
        human_pos_hist = np.zeros((n, horizon, 2), dtype=np.float32)

        min_clear_r = np.full((n,), np.inf, dtype=np.float32)
        min_clear_h = np.full((n,), np.inf, dtype=np.float32)

        dt = float(self.sim_dt)
        leash_len_sq = float(self.leash_length * self.leash_length)

        for t in range(horizon):
            forward = u[:, t, 0].astype(np.float32)
            turn = u[:, t, 1].astype(np.float32)
            for _ in range(int(self.frame_stride)):
                # Robot update (matches PhysicsEngine.step()).
                r_heading = r_heading + turn * float(self.turn_speed) * dt
                dir_x = np.cos(r_heading).astype(np.float32)
                dir_y = np.sin(r_heading).astype(np.float32)
                target_vx = dir_x * forward * float(self.robot_speed)
                target_vy = dir_y * forward * float(self.robot_speed)
                r_vel[:, 0] = r_vel[:, 0] * 0.8 + target_vx * 0.2
                r_vel[:, 1] = r_vel[:, 1] * 0.8 + target_vy * 0.2
                r_pos[:, 0] = r_pos[:, 0] + r_vel[:, 0] * dt
                r_pos[:, 1] = r_pos[:, 1] + r_vel[:, 1] * dt

                # Human update (matches PhysicsEngine._update_human()).
                rel = h_pos - r_pos
                dist_sq = np.sum(rel * rel, axis=1).astype(np.float32)
                mask = dist_sq > 1e-12  # distance > 1e-6
                if np.any(mask):
                    idx = np.where(mask)[0]
                    dist = np.sqrt(dist_sq[idx]).astype(np.float32)
                    leash_dir = rel[idx] / dist[:, None]

                    taut = dist > float(self.leash_length)
                    if np.any(taut):
                        taut_idx = idx[taut]
                        pull_distance = (dist[taut] - float(self.leash_length)).astype(np.float32)
                        pull_force = -leash_dir[taut] * pull_distance[:, None] * 10.0
                        h_vel[taut_idx] = h_vel[taut_idx] + pull_force * dt

                    h_vel[idx] = h_vel[idx] * float(self.human_drag)
                    h_pos[idx] = h_pos[idx] + h_vel[idx] * dt

                    # Hard constraint: ensure within leash length.
                    rel2 = h_pos[idx] - r_pos[idx]
                    dist2_sq = np.sum(rel2 * rel2, axis=1).astype(np.float32)
                    taut2 = dist2_sq > leash_len_sq
                    if np.any(taut2):
                        idx2 = idx[taut2]
                        dist2 = np.sqrt(dist2_sq[taut2]).astype(np.float32)
                        leash_dir2 = rel2[taut2] / dist2[:, None]
                        h_pos[idx2] = r_pos[idx2] + leash_dir2 * float(self.leash_length)

                if circles is not None or segments is not None:
                    min_clear_r = np.minimum(
                        min_clear_r,
                        _min_clearance(
                            r_pos, circles=circles, segments=segments, agent_radius=float(self.robot_radius)
                        ),
                    )
                    min_clear_h = np.minimum(
                        min_clear_h,
                        _min_clearance(
                            h_pos, circles=circles, segments=segments, agent_radius=float(self.human_radius)
                        ),
                    )

            robot_pos_hist[:, t, :] = r_pos
            robot_heading_hist[:, t] = r_heading
            human_pos_hist[:, t, :] = h_pos

        return robot_pos_hist, robot_heading_hist, human_pos_hist, min_clear_r, min_clear_h

    def _track_cost(
        self,
        *,
        pos_track: np.ndarray,
        heading_hist: np.ndarray,
        path: np.ndarray,
        path_s: np.ndarray,
        path_heading: np.ndarray,
        s0: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute tracking + heading + progress terms based on positions (robot or human).

        Returns:
            track_sum: (N,) sum of squared distance to path
            heading_sum: (N,) sum of squared heading error (robot heading vs tangent at closest point)
            progress_sum: (N,) sum of forward progress along path_s
        """
        pos_track = np.asarray(pos_track, dtype=np.float32)
        heading_hist = np.asarray(heading_hist, dtype=np.float32)
        path = np.asarray(path, dtype=np.float32)
        path_s = np.asarray(path_s, dtype=np.float32)
        path_heading = np.asarray(path_heading, dtype=np.float32)

        n, horizon, _ = pos_track.shape
        track_cost_sum = np.zeros((n,), dtype=np.float32)
        heading_cost_sum = np.zeros((n,), dtype=np.float32)
        progress_sum = np.zeros((n,), dtype=np.float32)

        prev_s = np.full((n,), float(s0), dtype=np.float32)
        for t in range(horizon):
            p = pos_track[:, t, :]  # (n,2)
            diff = path[None, :, :] - p[:, None, :]
            dist_sq = np.sum(diff * diff, axis=2).astype(np.float32)  # (n,P)
            idx = np.argmin(dist_sq, axis=1).astype(np.int32)
            track_cost_sum += dist_sq[np.arange(n), idx]

            s_t = path_s[idx].astype(np.float32)
            progress_sum += np.maximum(0.0, s_t - prev_s).astype(np.float32)
            prev_s = s_t

            ref_h = path_heading[idx].astype(np.float32)
            err = wrap_angle_np(ref_h - heading_hist[:, t]).astype(np.float32)
            heading_cost_sum += (err * err).astype(np.float32)

        return track_cost_sum, heading_cost_sum, progress_sum

    def solve(
        self,
        *,
        robot_pos: np.ndarray,
        robot_heading: float,
        robot_vel: np.ndarray,
        human_pos: np.ndarray,
        human_vel: np.ndarray,
        path: np.ndarray,
        path_s: Optional[np.ndarray] = None,
        obstacles: Optional[object] = None,
        segments: Optional[object] = None,
    ) -> tuple[float, float, Optional[np.ndarray]]:
        """
        Solve MPC and return (forward, turn, planned_robot_path).

        planned_robot_path: (H+1,2) including current robot position.
        """
        cfg = self.cfg
        path = np.asarray(path, dtype=np.float32)
        if len(path) == 0:
            return 0.0, 0.0, None
        if path_s is None or len(path_s) != len(path):
            path_s = _compute_path_s(path)
        path_s = np.asarray(path_s, dtype=np.float32)
        path_heading = _compute_path_heading(path)

        circles, segs = _normalize_obstacles(obstacles, segments)

        track = str(cfg.track or "human").strip().lower()
        track_is_human = track == "human"

        point_for_s0 = np.asarray(human_pos if track_is_human else robot_pos, dtype=np.float32)
        diff0 = path - point_for_s0[None, :]
        idx0 = int(np.argmin(np.sum(diff0 * diff0, axis=1)))
        s0 = float(path_s[idx0]) if len(path_s) > 0 else 0.0

        horizon = max(1, int(cfg.horizon))
        samples = max(8, int(cfg.samples))
        iters = max(1, int(cfg.iters))
        elites = int(max(1, min(int(cfg.elites), samples)))

        if self._mean is None or self._mean.shape != (horizon, 2):
            mean = np.zeros((horizon, 2), dtype=np.float32)
            mean[:, 0] = float(cfg.forward)
            mean[:, 1] = 0.0
        else:
            mean = self._mean.astype(np.float32).copy()

        std = np.zeros((horizon, 2), dtype=np.float32)
        std[:, 0] = float(cfg.forward_std)
        std[:, 1] = float(cfg.turn_std)

        best_seq = None
        best_cost = None
        best_robot_hist = None

        for _ in range(iters):
            noise = self._rng.normal(size=(samples, horizon, 2)).astype(np.float32)
            u = mean[None, :, :] + noise * std[None, :, :]
            if cfg.allow_reverse:
                u[:, :, 0] = np.clip(u[:, :, 0], -1.0, 1.0)
            else:
                u[:, :, 0] = np.clip(u[:, :, 0], 0.0, 1.0)
            u[:, :, 1] = np.clip(u[:, :, 1], -1.0, 1.0)

            r_hist, r_heading_hist, h_hist, min_clear_r, min_clear_h = self._rollout(
                u,
                robot_pos=robot_pos,
                robot_heading=robot_heading,
                robot_vel=robot_vel,
                human_pos=human_pos,
                human_vel=human_vel,
                circles=circles,
                segments=segs,
            )

            pos_track = h_hist if track_is_human else r_hist
            track_sum, heading_sum, progress_sum = self._track_cost(
                pos_track=pos_track,
                heading_hist=r_heading_hist,
                path=path,
                path_s=path_s,
                path_heading=path_heading,
                s0=s0,
            )

            u_cost = np.sum(u[:, :, 0] * u[:, :, 0] + u[:, :, 1] * u[:, :, 1], axis=1).astype(np.float32)
            smooth_cost = np.zeros((samples,), dtype=np.float32)
            if horizon > 1:
                dturn = np.diff(u[:, :, 1], axis=1)
                smooth_cost = np.sum(dturn * dturn, axis=1).astype(np.float32)

            # Collision constraints (robot + human, circles + segments).
            collided_r = (min_clear_r < 0.0).astype(np.float32)
            collided_h = (min_clear_h < 0.0).astype(np.float32)
            collision_cost = float(cfg.w_collision) * (collided_r + collided_h)

            margin = float(cfg.clearance_margin)
            clearance_violation_r = np.maximum(0.0, margin - min_clear_r).astype(np.float32)
            clearance_violation_h = np.maximum(0.0, margin - min_clear_h).astype(np.float32)
            clearance_cost = float(cfg.w_clearance) * (
                clearance_violation_r * clearance_violation_r + clearance_violation_h * clearance_violation_h
            )

            costs = (
                float(cfg.w_track) * track_sum
                + float(cfg.w_heading) * heading_sum
                + float(cfg.w_u) * u_cost
                + float(cfg.w_smooth) * smooth_cost
                - float(cfg.w_progress) * progress_sum
                + collision_cost
                + clearance_cost
            ).astype(np.float32)
            costs = np.where(np.isfinite(costs), costs, 1e9).astype(np.float32)

            i_best = int(np.argmin(costs))
            c_best = float(costs[i_best])
            if best_cost is None or c_best < float(best_cost):
                best_cost = c_best
                best_seq = u[i_best].copy()
                best_robot_hist = r_hist[i_best].copy()

            elite_idx = np.argpartition(costs, elites - 1)[:elites]
            elite = u[elite_idx]
            mean = elite.mean(axis=0).astype(np.float32)
            std = (elite.std(axis=0).astype(np.float32) + 1e-3)

        if best_seq is None:
            return 0.0, 0.0, None

        self._mean = np.concatenate([best_seq[1:], best_seq[-1:]], axis=0).astype(np.float32)
        planned = (
            np.concatenate([np.asarray(robot_pos, dtype=np.float32)[None, :], best_robot_hist], axis=0).astype(np.float32)
            if best_robot_hist is not None
            else None
        )
        return float(best_seq[0, 0]), float(best_seq[0, 1]), planned
