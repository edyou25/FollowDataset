from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation


FOLLOWDATASET_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = FOLLOWDATASET_DIR.parent
DIFFUSION_POLICY_DIR = WORKSPACE_ROOT / "diffusion_policy"
for path in (FOLLOWDATASET_DIR, DIFFUSION_POLICY_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from diffusion_policy.common.guide_mid360 import (  # noqa: E402
    GuideMid360ObservationConfig,
    encode_mid360_scan_from_pointcloud,
)
from src.safety_filter import QPSafetyFilter  # noqa: E402


ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts" / "safety_filter_qp"


@dataclass(frozen=True)
class Scenario:
    robot_start: np.ndarray
    human_start: np.ndarray
    robot_radius: float
    human_radius: float
    circle_obstacles: np.ndarray
    segment_obstacles: np.ndarray
    reference_path: np.ndarray
    raw_deltas: np.ndarray


def build_understeering_raw_deltas() -> tuple[np.ndarray, np.ndarray]:
    """Nominal front-pull path that starts to bypass the obstacle but returns too early."""
    num_points = 32
    x = np.linspace(0.0, 4.2, num_points, dtype=np.float32)
    y = (
        0.34
        * np.exp(-0.5 * ((x - 1.25) / 0.42) ** 2)
    ).astype(np.float32)
    y = y - y[0]
    y[-5:] = np.linspace(float(y[-5]), 0.0, 5, dtype=np.float32)
    raw_path = np.stack([x, y], axis=1).astype(np.float32)
    return raw_path, np.diff(raw_path, axis=0).astype(np.float32)


def build_ideal_collision_scenario() -> Scenario:
    """Front robot pulls a rear human around a front obstacle."""
    _, raw_deltas = build_understeering_raw_deltas()
    reference_path = np.stack(
        [
            np.linspace(0.0, 4.2, 100, dtype=np.float32),
            np.zeros((100,), dtype=np.float32),
        ],
        axis=1,
    )
    return Scenario(
        robot_start=np.array([0.0, 0.0], dtype=np.float32),
        human_start=np.array([-0.78, 0.0], dtype=np.float32),
        robot_radius=0.25,
        human_radius=0.25,
        circle_obstacles=np.array(
            [
                [1.25, 0.0, 0.28],
            ],
            dtype=np.float32,
        ),
        segment_obstacles=np.array(
            [
                [-0.40, 1.05, 4.40, 1.05],
                [-0.40, -0.85, 4.40, -0.85],
            ],
            dtype=np.float32,
        ),
        reference_path=reference_path,
        raw_deltas=raw_deltas,
    )


def rollout_qp_mode(
    scenario: Scenario,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Roll out raw / robot-only / human-aware QP in a deterministic 2D unit-test model."""
    if mode not in {"raw_diffusion", "robot_qp", "human_robot_qp"}:
        raise ValueError(f"unsupported mode: {mode}")

    qp = QPSafetyFilter(
        margin=0.06,
        alpha=1.0,
        max_constraints=12,
        influence_distance=1.4,
    )
    robot_pos = scenario.robot_start.astype(np.float32).copy()
    human_pos = scenario.human_start.astype(np.float32).copy()
    robot_path = [robot_pos.copy()]
    human_path = [human_pos.copy()]
    chosen_deltas = []
    infos = []

    for ref_delta in scenario.raw_deltas:
        ref_delta = ref_delta.astype(np.float32)
        if mode == "raw_diffusion":
            chosen_delta = ref_delta
            info = {
                "modified": False,
                "shift": 0.0,
                "constraint_count": 0,
                "min_clearance": float("inf"),
            }
        else:
            protect_human = mode == "human_robot_qp"
            extra_entities = [
                ("robot_future", robot_pos + ref_delta, scenario.robot_radius),
            ]
            if protect_human:
                extra_entities.append(
                    ("human_future", human_pos + ref_delta, scenario.human_radius)
                )
            result = qp.project_delta(
                ref_delta=ref_delta,
                robot_pos=robot_pos,
                robot_radius=scenario.robot_radius,
                human_pos=human_pos,
                human_radius=scenario.human_radius,
                circle_obstacles=scenario.circle_obstacles,
                segment_obstacles=scenario.segment_obstacles,
                include_human=protect_human,
                extra_entities=extra_entities,
            )
            chosen_delta = result.delta.astype(np.float32)
            info = {
                "modified": bool(result.modified),
                "shift": float(np.linalg.norm(chosen_delta - ref_delta)),
                "constraint_count": int(result.constraint_count),
                "min_clearance": float(result.min_clearance),
                "selected_constraints": list(result.selected_constraints),
            }

        robot_pos = robot_pos + chosen_delta
        human_pos = human_pos + chosen_delta
        robot_path.append(robot_pos.copy())
        human_path.append(human_pos.copy())
        chosen_deltas.append(chosen_delta.copy())
        infos.append(info)

    return (
        np.asarray(robot_path, dtype=np.float32),
        np.asarray(human_path, dtype=np.float32),
        np.asarray(chosen_deltas, dtype=np.float32),
        infos,
    )


def circle_path_collisions(
    path: np.ndarray,
    radius: float,
    circle_obstacles: np.ndarray,
) -> list[dict]:
    hits = []
    for point_idx, point in enumerate(np.asarray(path, dtype=np.float32)):
        for obs_idx, obs in enumerate(np.asarray(circle_obstacles, dtype=np.float32)):
            clearance = float(np.linalg.norm(point - obs[:2]) - radius - obs[2])
            if clearance <= 0.0:
                hits.append(
                    {
                        "point_idx": int(point_idx),
                        "obstacle_idx": int(obs_idx),
                        "clearance": clearance,
                        "point": point.astype(float).tolist(),
                    }
                )
    return hits


def segment_path_collisions(
    path: np.ndarray,
    radius: float,
    segment_obstacles: np.ndarray,
) -> list[dict]:
    hits = []
    for point_idx, point in enumerate(np.asarray(path, dtype=np.float32)):
        for obs_idx, seg in enumerate(np.asarray(segment_obstacles, dtype=np.float32)):
            p1 = seg[:2]
            p2 = seg[2:4]
            ab = p2 - p1
            denom = float(np.dot(ab, ab))
            if denom < 1e-9:
                closest = p1
            else:
                t = float(np.dot(point - p1, ab)) / denom
                closest = p1 + np.clip(t, 0.0, 1.0) * ab
            clearance = float(np.linalg.norm(point - closest) - radius)
            if clearance <= 0.0:
                hits.append(
                    {
                        "point_idx": int(point_idx),
                        "obstacle_idx": int(obs_idx),
                        "clearance": clearance,
                        "point": point.astype(float).tolist(),
                    }
                )
    return hits


def scan_to_world_points(
    scan: np.ndarray,
    config: GuideMid360ObservationConfig,
    mid360_pose: np.ndarray,
) -> np.ndarray:
    scan = np.asarray(scan, dtype=np.float32)
    valid = np.isfinite(scan)
    valid &= scan >= float(config.min_range)
    valid &= scan < float(config.fill_value) - 1e-6
    if not np.any(valid):
        return np.zeros((0, 2), dtype=np.float32)
    idx = np.nonzero(valid)[0].astype(np.float32)
    width = float(config.max_angle - config.min_angle)
    azimuth = float(config.min_angle) + (idx + 0.5) * width / float(config.num_bins)
    local_xy = np.stack(
        [scan[valid] * np.cos(azimuth), scan[valid] * np.sin(azimuth)],
        axis=1,
    ).astype(np.float32)
    local_xyz = np.zeros((len(local_xy), 3), dtype=np.float32)
    local_xyz[:, :2] = local_xy
    pose = np.asarray(mid360_pose, dtype=np.float64)
    rot = Rotation.from_quat(pose[3:7]).as_matrix().astype(np.float32)
    world_xyz = local_xyz @ rot.T + pose[:3].astype(np.float32)[None, :]
    return world_xyz[:, :2].astype(np.float32)


def frame_to_world_xy(
    frame: np.ndarray,
    field_names: list[str],
    mid360_pose: np.ndarray,
) -> np.ndarray:
    indices = {str(name): idx for idx, name in enumerate(field_names)}
    if len(frame) == 0 or "x" not in indices or "y" not in indices:
        return np.zeros((0, 2), dtype=np.float32)
    local_xyz = np.zeros((len(frame), 3), dtype=np.float32)
    local_xyz[:, 0] = frame[:, indices["x"]]
    local_xyz[:, 1] = frame[:, indices["y"]]
    if "z" in indices:
        local_xyz[:, 2] = frame[:, indices["z"]]
    pose = np.asarray(mid360_pose, dtype=np.float64)
    rot = Rotation.from_quat(pose[3:7]).as_matrix().astype(np.float32)
    world_xyz = local_xyz @ rot.T + pose[:3].astype(np.float32)[None, :]
    return world_xyz[:, :2].astype(np.float32)


def filter_gazebo_cloud_near_human(
    frame: np.ndarray,
    field_names: list[str],
    mid360_pose: np.ndarray,
    human_pos: np.ndarray,
    human_radius: float,
) -> np.ndarray:
    world_xy = frame_to_world_xy(frame, field_names, mid360_pose)
    if len(world_xy) == 0:
        return frame
    keep = np.linalg.norm(
        world_xy - np.asarray(human_pos, dtype=np.float32)[None, :],
        axis=1,
    ) > float(human_radius)
    return frame[keep]


def filter_frame_for_mid360_observation(
    frame: np.ndarray,
    field_names: list[str],
    config: GuideMid360ObservationConfig,
    mid360_pose: np.ndarray,
) -> np.ndarray:
    indices = {str(name): idx for idx, name in enumerate(field_names)}
    if len(frame) == 0 or "x" not in indices or "y" not in indices:
        return frame

    ix = indices["x"]
    iy = indices["y"]
    iz = indices.get("z")
    local_xy = frame[:, [ix, iy]].astype(np.float32, copy=False)
    ranges = np.linalg.norm(local_xy, axis=1).astype(np.float32, copy=False)
    azimuth = np.arctan2(local_xy[:, 1], local_xy[:, 0]).astype(np.float32, copy=False)
    valid = np.isfinite(local_xy[:, 0]) & np.isfinite(local_xy[:, 1])
    valid &= np.isfinite(ranges) & np.isfinite(azimuth)
    valid &= ranges >= float(config.min_range)
    valid &= ranges <= float(config.max_range)
    angle_width = float(config.max_angle - config.min_angle)
    if angle_width < (2.0 * np.pi - 1e-6):
        valid &= azimuth >= float(config.min_angle)
        valid &= azimuth < float(config.max_angle)

    if iz is not None and frame.shape[1] > iz:
        local_xyz = np.zeros((len(frame), 3), dtype=np.float32)
        local_xyz[:, 0] = frame[:, ix]
        local_xyz[:, 1] = frame[:, iy]
        local_xyz[:, 2] = frame[:, iz]
        if config.use_world_height:
            pose = np.asarray(mid360_pose, dtype=np.float64)
            rot = Rotation.from_quat(pose[3:7]).as_matrix().astype(np.float32)
            height = local_xyz @ rot[2, :].astype(np.float32) + np.float32(pose[2])
        else:
            height = local_xyz[:, 2]
        valid &= np.isfinite(height)
        valid &= height >= float(config.ground_height)
        valid &= height <= float(config.max_height)

    return frame[valid]


def render_mid360_plugin_pointcloud(
    scenario: Scenario,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray, str]:
    """Render point cloud through the real Mid-360 Gazebo plugin. This is intentionally required."""
    from src.mid360_gazebo import (  # noqa: WPS433
        Mid360GazeboConfig,
        Mid360GazeboSession,
        resolve_mid360_plugin_dir,
        resolve_mid360_plugin_library,
    )

    plugin_dir = resolve_mid360_plugin_dir(None)
    plugin_library = resolve_mid360_plugin_library(plugin_dir, None)
    if plugin_library is None:
        raise RuntimeError(
            "Cannot find liblivox_laser_simulation.so. "
            "Set MID360_PLUGIN_LIBRARY or pass a build under MID360_PLUGIN_DIR."
        )

    config = Mid360GazeboConfig(
        plugin_dir=plugin_dir,
        plugin_library_path=plugin_library,
        gui=False,
        visualize_laser=False,
        samples=20000,
        downsample=2,
        update_rate=10.0,
        wait_for_first_cloud_sec=20.0,
        keep_runtime_artifacts=True,
    )
    path_data = {
        "path": scenario.reference_path,
        "waypoints": scenario.reference_path[::20],
        "start": scenario.robot_start,
        "end": scenario.reference_path[-1],
        "obstacles": scenario.circle_obstacles,
        "segment_obstacles": scenario.segment_obstacles,
    }
    obs_config = GuideMid360ObservationConfig(
        num_bins=128,
        min_angle=-np.pi,
        max_angle=np.pi,
        min_range=0.2,
        max_range=6.0,
        ground_height=0.1,
        max_height=2.2,
        use_world_height=True,
    )

    session = Mid360GazeboSession(path_data=path_data, config=config)
    try:
        session.start()
        warmup_cloud = session.get_pointcloud(wait_timeout=0.0)
        last_seq = int(warmup_cloud.get("seq", 0)) if warmup_cloud is not None else None
        robot_state = SimpleNamespace(
            position=scenario.robot_start.astype(np.float64),
            heading=0.0,
        )
        human_state = SimpleNamespace(position=scenario.human_start.astype(np.float64))
        session.update_entities(robot_state, human_state)
        cloud = session.get_pointcloud(after_seq=last_seq, wait_timeout=3.0)
        if cloud is None:
            session.update_entities(robot_state, human_state)
            cloud = session.get_pointcloud(after_seq=last_seq, wait_timeout=5.0)
        if cloud is None:
            raise RuntimeError("Mid360 Gazebo plugin returned no PointCloud2 frame.")
        raw_frame = np.asarray(cloud.get("points", np.zeros((0, 0))), dtype=np.float32)
        if raw_frame.ndim == 1:
            raw_frame = raw_frame.reshape(1, -1)
        field_names = list(cloud.get("fields", []))
        mid360_pose = session.get_mid360_pose(robot_state)
        filtered_frame = filter_gazebo_cloud_near_human(
            raw_frame,
            field_names,
            mid360_pose,
            scenario.human_start,
            scenario.human_radius,
        )
        observation_frame = filter_frame_for_mid360_observation(
            filtered_frame,
            field_names,
            obs_config,
            mid360_pose,
        )
        scan = encode_mid360_scan_from_pointcloud(
            frame=observation_frame,
            field_names=field_names,
            config=obs_config,
            mid360_pose=mid360_pose,
        )
        raw_world_xy = frame_to_world_xy(raw_frame, field_names, mid360_pose)
        obs_world_xy = frame_to_world_xy(observation_frame, field_names, mid360_pose)
        obs_bins = int(np.sum(scan < obs_config.fill_value - 1e-6))
        status = (
            f"mid360_gazebo_plugin: {len(raw_frame)} raw points, "
            f"{len(filtered_frame)} after human mask, "
            f"{len(observation_frame)} observation points, "
            f"{obs_bins} observation bins"
        )
        return raw_frame, field_names, raw_world_xy, scan, obs_world_xy, status
    finally:
        session.close()


def summarize_rollouts(
    scenario: Scenario,
    rollouts: dict[str, dict],
    pointcloud_status: str,
) -> dict:
    summary = {
        "scenario": {
            "robot_start": scenario.robot_start.astype(float).tolist(),
            "human_start": scenario.human_start.astype(float).tolist(),
            "robot_radius": float(scenario.robot_radius),
            "human_radius": float(scenario.human_radius),
            "circle_obstacles": scenario.circle_obstacles.astype(float).tolist(),
            "segment_obstacles": scenario.segment_obstacles.astype(float).tolist(),
        },
        "pointcloud_source": pointcloud_status,
        "modes": {},
    }
    for name, result in rollouts.items():
        robot_circle_hits = circle_path_collisions(
            result["robot_path"],
            scenario.robot_radius,
            scenario.circle_obstacles,
        )
        human_circle_hits = circle_path_collisions(
            result["human_path"],
            scenario.human_radius,
            scenario.circle_obstacles,
        )
        robot_segment_hits = segment_path_collisions(
            result["robot_path"],
            scenario.robot_radius,
            scenario.segment_obstacles,
        )
        human_segment_hits = segment_path_collisions(
            result["human_path"],
            scenario.human_radius,
            scenario.segment_obstacles,
        )
        infos = result["infos"]
        summary["modes"][name] = {
            "robot_collision": bool(robot_circle_hits or robot_segment_hits),
            "human_collision": bool(human_circle_hits or human_segment_hits),
            "robot_hits": robot_circle_hits + robot_segment_hits,
            "human_hits": human_circle_hits + human_segment_hits,
            "robot_circle_hits": robot_circle_hits,
            "human_circle_hits": human_circle_hits,
            "robot_segment_hits": robot_segment_hits,
            "human_segment_hits": human_segment_hits,
            "modified_steps": int(sum(bool(item["modified"]) for item in infos)),
            "mean_shift": float(np.mean([item["shift"] for item in infos])) if infos else 0.0,
            "max_constraints": int(max([item["constraint_count"] for item in infos], default=0)),
        }
    return summary


def compute_rollout_geometry(rollouts: dict[str, dict]) -> dict:
    geometry = {}
    for name, result in rollouts.items():
        robot_path = np.asarray(result["robot_path"], dtype=np.float32)
        human_path = np.asarray(result["human_path"], dtype=np.float32)
        leash_delta = robot_path - human_path
        geometry[name] = {
            "front_dx_min": float(np.min(leash_delta[:, 0])),
            "front_dx_max": float(np.max(leash_delta[:, 0])),
            "lateral_leash_abs_max": float(np.max(np.abs(leash_delta[:, 1]))),
            "robot_final_y": float(robot_path[-1, 1]),
            "human_final_y": float(human_path[-1, 1]),
            "robot_max_y": float(np.max(robot_path[:, 1])),
            "human_max_y": float(np.max(human_path[:, 1])),
        }
    return geometry


def compute_pointcloud_geometry(
    raw_cloud_world: np.ndarray,
    obs_cloud_world: np.ndarray,
    scenario: Scenario,
) -> dict:
    raw_cloud_world = np.asarray(raw_cloud_world, dtype=np.float32)
    obs_cloud_world = np.asarray(obs_cloud_world, dtype=np.float32)
    obstacle = np.asarray(scenario.circle_obstacles[0], dtype=np.float32)

    def count(points: np.ndarray, mask: np.ndarray) -> int:
        return int(np.sum(mask)) if len(points) > 0 else 0

    raw = raw_cloud_world
    obs = obs_cloud_world
    raw_dist_to_obstacle = np.linalg.norm(raw - obstacle[:2][None, :], axis=1)
    obs_dist_to_obstacle = np.linalg.norm(obs - obstacle[:2][None, :], axis=1)
    obstacle_annulus_min = max(0.0, float(obstacle[2]) - 0.08)
    obstacle_annulus_max = float(obstacle[2]) + 0.16
    raw_upper = (raw[:, 1] > 0.95) & (raw[:, 0] > 0.0) & (raw[:, 0] < 4.1)
    raw_lower = (raw[:, 1] < -0.75) & (raw[:, 0] > 0.0) & (raw[:, 0] < 4.1)
    raw_front = (
        (raw_dist_to_obstacle >= obstacle_annulus_min)
        & (raw_dist_to_obstacle <= obstacle_annulus_max)
    )
    obs_upper = (obs[:, 1] > 0.85) & (obs[:, 0] > 0.0) & (obs[:, 0] < 4.1)
    obs_lower = (obs[:, 1] < -0.70) & (obs[:, 0] > 0.0) & (obs[:, 0] < 4.1)
    obs_front = (
        (obs_dist_to_obstacle >= obstacle_annulus_min)
        & (obs_dist_to_obstacle <= obstacle_annulus_max)
    )

    return {
        "raw_count": int(len(raw)),
        "observation_count": int(len(obs)),
        "raw_upper_wall_points": count(raw, raw_upper),
        "raw_lower_wall_points": count(raw, raw_lower),
        "raw_front_obstacle_points": count(raw, raw_front),
        "observation_upper_wall_points": count(obs, obs_upper),
        "observation_lower_wall_points": count(obs, obs_lower),
        "observation_front_obstacle_points": count(obs, obs_front),
        "raw_bounds_min": raw.min(axis=0).astype(float).tolist() if len(raw) else [0.0, 0.0],
        "raw_bounds_max": raw.max(axis=0).astype(float).tolist() if len(raw) else [0.0, 0.0],
        "observation_bounds_min": obs.min(axis=0).astype(float).tolist() if len(obs) else [0.0, 0.0],
        "observation_bounds_max": obs.max(axis=0).astype(float).tolist() if len(obs) else [0.0, 0.0],
    }


def plot_scenario(
    scenario: Scenario,
    rollouts: dict[str, dict],
    raw_cloud_world: np.ndarray,
    obs_cloud_world: np.ndarray,
    summary: dict,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), constrained_layout=True)
    panels = [
        ("raw_diffusion", "Raw Diffusion"),
        ("robot_qp", "Robot QP"),
        ("human_robot_qp", "Human-Aware QP"),
    ]
    colors = {
        "raw_diffusion": ("#991B1B", "#EA580C"),
        "robot_qp": ("#581C87", "#16A34A"),
        "human_robot_qp": ("#1D4ED8", "#0F766E"),
    }
    raw_stride = max(1, int(np.ceil(len(raw_cloud_world) / 6000.0))) if len(raw_cloud_world) else 1
    obs_stride = max(1, int(np.ceil(len(obs_cloud_world) / 3500.0))) if len(obs_cloud_world) else 1
    raw_cloud_for_plot = raw_cloud_world[::raw_stride]
    obs_cloud_for_plot = obs_cloud_world[::obs_stride]
    for ax, (mode, _title) in zip(axes, panels):
        ax.set_facecolor("#F8FAFC")
        ax.plot(
            scenario.reference_path[:, 0],
            scenario.reference_path[:, 1],
            color="#6B7280",
            linewidth=1.2,
            linestyle="--",
            label="reference centerline",
            zorder=1,
        )
        for seg_idx, seg in enumerate(scenario.segment_obstacles):
            ax.plot(
                [seg[0], seg[2]],
                [seg[1], seg[3]],
                color="#F59E0B",
                linewidth=3.0,
                label="vector map walls" if seg_idx == 0 else None,
                zorder=1,
            )
        for obs_idx, obs in enumerate(scenario.circle_obstacles):
            ax.add_patch(
                plt.Circle(
                    obs[:2],
                    obs[2],
                    color="#7C2D12",
                    alpha=0.30,
                    label="vector map circles" if obs_idx == 0 else None,
                    zorder=1,
                )
            )
            ax.add_patch(
                plt.Circle(
                    obs[:2],
                    obs[2] + scenario.human_radius,
                    fill=False,
                    linestyle=":",
                    color="#9CA3AF",
                    linewidth=1.1,
                    zorder=1,
                )
            )

        if len(raw_cloud_for_plot) > 0:
            ax.scatter(
                raw_cloud_for_plot[:, 0],
                raw_cloud_for_plot[:, 1],
                s=3,
                c="#EF4444",
                alpha=0.22,
                label="raw Mid360 point cloud",
                zorder=2,
            )
        if len(obs_cloud_for_plot) > 0:
            ax.scatter(
                obs_cloud_for_plot[:, 0],
                obs_cloud_for_plot[:, 1],
                s=6,
                c="#2563EB",
                alpha=0.58,
                label="processed observation cloud",
                zorder=3,
            )

        robot_color, human_color = colors[mode]
        result = rollouts[mode]
        robot_path = result["robot_path"]
        human_path = result["human_path"]
        ax.plot(
            [robot_path[0, 0], human_path[0, 0]],
            [robot_path[0, 1], human_path[0, 1]],
            color="#111827",
            linewidth=1.4,
            linestyle="-.",
            alpha=0.85,
            label="initial leash",
            zorder=4,
        )
        ax.plot(
            robot_path[:, 0],
            robot_path[:, 1],
            color=robot_color,
            linewidth=2.4,
            marker="o",
            markersize=3.2,
            label="robot path points",
            zorder=4,
        )
        ax.plot(
            human_path[:, 0],
            human_path[:, 1],
            color=human_color,
            linewidth=2.4,
            marker="s",
            markersize=3.0,
            label="human path points",
            zorder=4,
        )
        ax.scatter(
            [scenario.robot_start[0]],
            [scenario.robot_start[1]],
            s=90,
            c=robot_color,
            edgecolors="white",
            linewidths=1.0,
            zorder=5,
        )
        ax.text(
            float(scenario.robot_start[0]) + 0.04,
            float(scenario.robot_start[1]) + 0.08,
            "robot front",
            fontsize=8,
            color=robot_color,
            zorder=6,
        )
        ax.scatter(
            [scenario.human_start[0]],
            [scenario.human_start[1]],
            s=90,
            c=human_color,
            edgecolors="white",
            linewidths=1.0,
            zorder=5,
        )
        ax.text(
            float(scenario.human_start[0]) + 0.04,
            float(scenario.human_start[1]) - 0.16,
            "human rear",
            fontsize=8,
            color=human_color,
            zorder=6,
        )
        mode_summary = summary["modes"][mode]
        status = (
            f"robot collision: {mode_summary['robot_collision']}\n"
            f"human collision: {mode_summary['human_collision']}\n"
            f"modified steps: {mode_summary['modified_steps']}"
        )
        ax.text(
            0.02,
            0.98,
            status,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#D1D5DB", "alpha": 0.92},
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1.05, 4.45)
        ax.set_ylim(-1.10, 1.25)
        ax.grid(True, color="#E5E7EB", linewidth=0.8)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def run_safety_filter_demo() -> dict:
    scenario = build_ideal_collision_scenario()
    rollouts = {}
    for mode in ("raw_diffusion", "robot_qp", "human_robot_qp"):
        robot_path, human_path, deltas, infos = rollout_qp_mode(scenario, mode)
        rollouts[mode] = {
            "robot_path": robot_path,
            "human_path": human_path,
            "deltas": deltas,
            "infos": infos,
        }

    raw_frame, field_names, raw_cloud_world, scan, obs_cloud_world, pointcloud_status = (
        render_mid360_plugin_pointcloud(scenario)
    )
    summary = summarize_rollouts(scenario, rollouts, pointcloud_status)
    summary["mid360_plugin_raw_points"] = int(len(raw_frame))
    summary["mid360_plugin_observation_bins"] = int(np.sum(scan < 6.0 - 1e-6))
    summary["rollout_geometry"] = compute_rollout_geometry(rollouts)
    summary["pointcloud_geometry"] = compute_pointcloud_geometry(
        raw_cloud_world,
        obs_cloud_world,
        scenario,
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = ARTIFACT_DIR / "safety_filter_qp_modes_mid360_plugin.png"
    summary_path = ARTIFACT_DIR / "safety_filter_qp_modes_mid360_plugin_summary.json"
    data_path = ARTIFACT_DIR / "safety_filter_qp_modes_mid360_plugin_data.npz"

    plot_scenario(
        scenario=scenario,
        rollouts=rollouts,
        raw_cloud_world=raw_cloud_world,
        obs_cloud_world=obs_cloud_world,
        summary=summary,
        output_path=plot_path,
    )
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    np.savez(
        data_path,
        raw_pointcloud_frame=raw_frame,
        raw_pointcloud_fields=np.asarray(field_names, dtype=object),
        raw_pointcloud_world=raw_cloud_world,
        observation_scan=scan,
        observation_cloud_world=obs_cloud_world,
        raw_robot_path=rollouts["raw_diffusion"]["robot_path"],
        raw_human_path=rollouts["raw_diffusion"]["human_path"],
        robot_qp_robot_path=rollouts["robot_qp"]["robot_path"],
        robot_qp_human_path=rollouts["robot_qp"]["human_path"],
        human_qp_robot_path=rollouts["human_robot_qp"]["robot_path"],
        human_qp_human_path=rollouts["human_robot_qp"]["human_path"],
        circle_obstacles=scenario.circle_obstacles,
        segment_obstacles=scenario.segment_obstacles,
        reference_path=scenario.reference_path,
    )
    summary["artifacts"] = {
        "plot": str(plot_path),
        "summary": str(summary_path),
        "data": str(data_path),
    }
    return summary


def assert_safety_filter_summary(summary: dict) -> None:
    modes = summary["modes"]
    assert modes["raw_diffusion"]["robot_collision"]
    assert modes["raw_diffusion"]["human_collision"]
    assert not modes["robot_qp"]["robot_collision"]
    assert modes["robot_qp"]["human_collision"]
    assert not modes["human_robot_qp"]["robot_collision"]
    assert not modes["human_robot_qp"]["human_collision"]
    assert modes["robot_qp"]["modified_steps"] > 0
    assert modes["human_robot_qp"]["modified_steps"] > 0
    assert modes["robot_qp"]["human_hits"]
    assert not modes["human_robot_qp"]["human_hits"]

    rollout_geometry = summary["rollout_geometry"]
    for mode_geometry in rollout_geometry.values():
        assert mode_geometry["front_dx_min"] > 0.70
        assert mode_geometry["front_dx_max"] < 0.86
        assert mode_geometry["lateral_leash_abs_max"] < 1e-4
    assert (
        rollout_geometry["human_robot_qp"]["human_final_y"]
        > rollout_geometry["robot_qp"]["human_final_y"] + 0.10
    )

    pointcloud_geometry = summary["pointcloud_geometry"]
    assert pointcloud_geometry["raw_count"] >= 5000
    assert summary["mid360_plugin_observation_bins"] >= 40
    assert pointcloud_geometry["observation_count"] >= 3000
    assert pointcloud_geometry["raw_upper_wall_points"] >= 500
    assert pointcloud_geometry["raw_lower_wall_points"] >= 500
    assert pointcloud_geometry["raw_front_obstacle_points"] >= 100
    assert pointcloud_geometry["observation_upper_wall_points"] >= 500
    assert pointcloud_geometry["observation_lower_wall_points"] >= 500
    assert pointcloud_geometry["observation_front_obstacle_points"] >= 100

    for artifact in summary["artifacts"].values():
        assert Path(artifact).exists()


def test_safety_filter_qp_modes_visualization() -> None:
    summary = run_safety_filter_demo()
    assert_safety_filter_summary(summary)


if __name__ == "__main__":
    result = run_safety_filter_demo()
    assert_safety_filter_summary(result)
    print(json.dumps(result["artifacts"], indent=2))
