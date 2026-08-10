#!/usr/bin/env python3
from __future__ import annotations

"""Offline/static real-world benchmark for the guide-dog robot pipeline.

The script reads a rosbag directly, without starting ``rosbag play``,
``dr_spaam_ros`` or ``running.py`` as ROS nodes. Each LaserScan frame passes
through the same stateful detector/tracker, the downstream human Kalman filter,
the causal guide/tether classifier and four planning variants:

    diffusion, diffusion+qp, safe-compliance and ours.

The benchmark is static/open-loop. It ONLY runs the perception/interaction/
planning algorithms and serializes raw per-frame results to one JSON file.
Plotting, aggregation and paper-oriented statistics are intentionally handled by
the separate ``analyze_benchmark.py`` script.

The policy checkpoint still expects planning-path features. In this real-world
benchmark, those features and all path-based metrics use only the robot trajectory
recorded on ``/odom`` in the rosbag. ``PathGenerator`` is never used.
"""

import argparse
import copy
import importlib.machinery
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


MODE_ORDER = ("diffusion", "diffusion_qp", "safe_compliance", "ours")
MODE_LABELS = {
    "diffusion": "diffusion",
    "diffusion_qp": "diffusion+qp",
    "safe_compliance": "safe-compliance",
    "ours": "ours",
}
MODE_COLORS = {
    "diffusion": "tab:gray",
    "diffusion_qp": "tab:blue",
    "safe_compliance": "tab:orange",
    "ours": "tab:green",
}


@dataclass
class BagInspection:
    bag_start: float
    bag_end: float
    odom_samples: np.ndarray
    scan_stamps: np.ndarray
    first_scan: Any
    first_cloud: Any
    first_odom: Any
    topic_types: dict[str, str]

    @property
    def duration(self) -> float:
        return max(0.0, float(self.bag_end - self.bag_start))

    @property
    def median_scan_dt(self) -> float:
        if len(self.scan_stamps) < 2:
            return float("nan")
        diffs = np.diff(self.scan_stamps)
        diffs = diffs[np.isfinite(diffs) & (diffs > 1e-6)]
        return float(np.median(diffs)) if len(diffs) else float("nan")


class BagClock:
    def __init__(self, initial_time: float = 0.0) -> None:
        self._time = float(initial_time)

    def set(self, value: float) -> None:
        self._time = float(value)

    def now(self) -> float:
        return float(self._time)


@dataclass
class OfflineInterfaceSeed:
    odom_position: np.ndarray
    odom_yaw: float
    odom_stamp: float
    scan_xyz: np.ndarray
    cloud_xyz: np.ndarray
    human_local_xy: np.ndarray
    detector_to_base_xy: np.ndarray
    detector_to_base_yaw: float


class OfflineRosInterface:
    """Drop-in replacement for running.LiveRobotRosInterface.

    It exposes the same read methods used by ModelPlanner, but all values are
    updated synchronously by the bag reader. Human positions enter in the
    detector convention (x-forward, y-right) and are transformed to odom/world
    with the current recorded odometry pose.
    """

    def __init__(
        self,
        *,
        clock: BagClock,
        seed: OfflineInterfaceSeed,
        odom_topic: str,
        pointcloud_topic: str,
        laser_scan_topic: str,
        range_source: str,
        human_detections_topic: str,
        cmd_vel_topic: str,
        max_linear_speed: float,
        max_angular_speed: float,
        odom_timeout: float,
        pointcloud_timeout: float,
        human_detection_timeout: float,
        human_detector_frame: str,
        human_world_frame: str,
        human_detector_y_axis: str,
        human_tf_timeout: float,
        enabled: bool,
    ) -> None:
        del cmd_vel_topic, odom_timeout, pointcloud_timeout, human_tf_timeout
        self.clock = clock
        self.odom_topic = str(odom_topic)
        self.pointcloud_topic = str(pointcloud_topic)
        self.laser_scan_topic = str(laser_scan_topic)
        self.range_source = str(range_source)
        self.human_detections_topic = str(human_detections_topic)
        self.human_detection_timeout = max(0.0, float(human_detection_timeout))
        self.human_detector_frame = str(human_detector_frame or "laser")
        self.human_world_frame = str(human_world_frame or "odom")
        self.human_detector_y_axis = str(human_detector_y_axis or "right").lower()
        self.max_linear_speed = float(max_linear_speed)
        self.max_angular_speed = float(max_angular_speed)
        self.enabled = bool(enabled)

        self._odom_position = np.asarray(seed.odom_position, dtype=np.float32).reshape(2)
        self._odom_yaw = float(seed.odom_yaw)
        self._odom_stamp = float(seed.odom_stamp)
        self._odom_message_stamp = float(seed.odom_stamp)
        self._odom_replay_epoch = 0
        self._odom_rewind_count = 0
        self._scan_xyz = np.asarray(seed.scan_xyz, dtype=np.float32).reshape(-1, 3)
        self._cloud_xyz = np.asarray(seed.cloud_xyz, dtype=np.float32).reshape(-1, 3)
        self._scan_seq = 1 if len(self._scan_xyz) else 0
        self._cloud_seq = 1 if len(self._cloud_xyz) else 0
        self._scan_stamp = float(seed.odom_stamp)
        self._cloud_stamp = float(seed.odom_stamp)
        self._human_local_xy = np.asarray(seed.human_local_xy, dtype=np.float32).reshape(-1, 2)
        self._detector_to_base_xy = np.asarray(
            seed.detector_to_base_xy, dtype=np.float32
        ).reshape(2)
        self._detector_to_base_yaw = float(seed.detector_to_base_yaw)
        self._human_seq = 1 if len(self._human_local_xy) else 0
        self._human_receive_stamp = float(seed.odom_stamp)
        self._human_message_stamp = float(seed.odom_stamp)
        self._human_rewind_count = 0
        self._last_control = (0.0, 0.0)

    def wait_until_ready(self, timeout: float) -> None:
        del timeout
        if self._scan_seq <= 0 and self._cloud_seq <= 0:
            raise RuntimeError("Offline interface seed has no range frame")

    def update_odom(self, position: np.ndarray, yaw: float, stamp: float) -> None:
        stamp = float(stamp)
        if self._odom_message_stamp > 0.0 and stamp < self._odom_message_stamp - 0.25:
            self._odom_replay_epoch += 1
            self._odom_rewind_count += 1
        self._odom_position = np.asarray(position, dtype=np.float32).reshape(2)
        self._odom_yaw = float(yaw)
        self._odom_stamp = stamp
        self._odom_message_stamp = stamp

    def update_scan(self, xyz: np.ndarray, stamp: float) -> None:
        self._scan_xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
        self._scan_stamp = float(stamp)
        self._scan_seq += 1

    def update_cloud(self, xyz: np.ndarray, stamp: float) -> None:
        self._cloud_xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
        self._cloud_stamp = float(stamp)
        self._cloud_seq += 1

    def update_human_local(self, local_xy: np.ndarray, stamp: float) -> None:
        stamp = float(stamp)
        if self._human_message_stamp > 0.0 and stamp < self._human_message_stamp - 0.25:
            self._human_rewind_count += 1
        self._human_local_xy = np.asarray(local_xy, dtype=np.float32).reshape(-1, 2)
        self._human_receive_stamp = stamp
        self._human_message_stamp = stamp
        self._human_seq += 1

    def set_range_source(self, source: str) -> None:
        self.range_source = str(source)

    def robot_pose(self) -> tuple[np.ndarray, float]:
        return self._odom_position.copy(), float(self._odom_yaw)

    def odom_replay_status(self) -> dict[str, Any]:
        return {
            "message_stamp": float(self._odom_message_stamp),
            "replay_epoch": int(self._odom_replay_epoch),
            "rewind_count": int(self._odom_rewind_count),
        }

    def human_detections_world(self) -> Optional[tuple[np.ndarray, int, float]]:
        age = float(self.clock.now() - self._human_receive_stamp)
        if self._human_seq <= 0 or age > self.human_detection_timeout:
            return None
        if len(self._human_local_xy) == 0:
            return np.zeros((0, 2), dtype=np.float32), int(self._human_seq), float(self._human_receive_stamp)

        local = self._human_local_xy.astype(np.float32, copy=True)
        # Detector file uses x-forward/y-right; ROS base convention is y-left.
        if self.human_detector_y_axis == "right":
            local[:, 1] *= -1.0
        # Optional static detector-frame -> robot-base extrinsic. The default
        # is identity, matching the online odometry-pose fallback.
        ce = float(np.cos(self._detector_to_base_yaw))
        se = float(np.sin(self._detector_to_base_yaw))
        base = np.empty_like(local)
        base[:, 0] = self._detector_to_base_xy[0] + ce * local[:, 0] - se * local[:, 1]
        base[:, 1] = self._detector_to_base_xy[1] + se * local[:, 0] + ce * local[:, 1]

        c = float(np.cos(self._odom_yaw))
        s = float(np.sin(self._odom_yaw))
        world = np.empty_like(base)
        world[:, 0] = self._odom_position[0] + c * base[:, 0] - s * base[:, 1]
        world[:, 1] = self._odom_position[1] + s * base[:, 0] + c * base[:, 1]
        return world, int(self._human_seq), float(self._human_receive_stamp)

    def human_detection_status(self) -> dict[str, Any]:
        age = (
            float(self.clock.now() - self._human_receive_stamp)
            if self._human_seq > 0
            else float("inf")
        )
        return {
            "seq": int(self._human_seq),
            "count": int(len(self._human_local_xy)),
            "age": age,
            "message_stamp": float(self._human_message_stamp),
            "receive_stamp": float(self._human_receive_stamp),
            "source_frame": self.human_detector_frame,
            "target_frame": self.human_world_frame,
            "world_cache_seq": int(self._human_seq),
            "transform_error": None,
            "transform_mode": "offline_odom",
            "human_rewind_count": int(self._human_rewind_count),
        }

    def pointcloud(self) -> tuple[np.ndarray, list[str], int]:
        if self.range_source in ("point_cloud", "pointcloud", "cloud", "livox"):
            if self._cloud_seq > 0 and len(self._cloud_xyz):
                return self._cloud_xyz.copy(), ["x", "y", "z"], int(self._cloud_seq)
        return self._scan_xyz.copy(), ["x", "y", "z"], int(self._scan_seq)

    def range_source_status(self) -> dict[str, Any]:
        if self.range_source in ("point_cloud", "pointcloud", "cloud", "livox") and self._cloud_seq > 0:
            return {
                "source": "point_cloud",
                "topic": self.pointcloud_topic,
                "seq": int(self._cloud_seq),
                "age": float(self.clock.now() - self._cloud_stamp),
                "count": int(len(self._cloud_xyz)),
            }
        return {
            "source": "laser_scan",
            "topic": self.laser_scan_topic,
            "seq": int(self._scan_seq),
            "age": float(self.clock.now() - self._scan_stamp),
            "count": int(len(self._scan_xyz)),
        }

    def assert_fresh(self) -> None:
        return

    def publish_control(self, forward: float, turn: float) -> None:
        self._last_control = (float(forward), float(turn))

    def stop(self) -> None:
        self._last_control = (0.0, 0.0)


@dataclass
class VariantResult:
    mode: str
    actions: np.ndarray
    deltas: np.ndarray
    robot_path: np.ndarray
    human_path: np.ndarray
    projected_collision: bool
    collision_who: str
    collision_step: int
    robot_min_clearance: float
    human_min_clearance: float
    mean_action_shift: float
    mean_delta_shift: float
    safety_modified_steps: int
    safety_constraint_count: int
    safety_min_clearance: float
    robot_horizon_displacement: float
    human_horizon_displacement: float
    path_deviation: float
    runtime_ms: float
    compliance_steps: int


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def load_source_module(name: str, path: Path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    if spec is None:
        raise ImportError(f"Cannot create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


def message_stamp(message: Any, bag_time: Any) -> float:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is not None and hasattr(stamp, "to_sec"):
        value = float(stamp.to_sec())
        if value > 0.0:
            return value
    return float(bag_time.to_sec() if hasattr(bag_time, "to_sec") else bag_time)


def quaternion_to_yaw(q: Any) -> float:
    return float(
        np.arctan2(
            2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y)),
            1.0 - 2.0 * (float(q.y) ** 2 + float(q.z) ** 2),
        )
    )


def odom_to_xy_yaw(message: Any) -> tuple[np.ndarray, float]:
    pose = message.pose.pose
    position = np.array([float(pose.position.x), float(pose.position.y)], dtype=np.float32)
    return position, quaternion_to_yaw(pose.orientation)


def laser_scan_to_xyz(message: Any) -> np.ndarray:
    ranges = np.asarray(message.ranges, dtype=np.float32)
    if ranges.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    angles = float(message.angle_min) + np.arange(ranges.size, dtype=np.float32) * float(message.angle_increment)
    keep = np.isfinite(ranges)
    range_min = float(getattr(message, "range_min", 0.0))
    range_max = float(getattr(message, "range_max", float("inf")))
    if np.isfinite(range_min):
        keep &= ranges >= range_min
    if np.isfinite(range_max) and range_max > 0.0:
        keep &= ranges <= range_max
    valid_ranges = ranges[keep]
    valid_angles = angles[keep]
    if len(valid_ranges) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.column_stack(
        [
            valid_ranges * np.cos(valid_angles),
            valid_ranges * np.sin(valid_angles),
            np.zeros_like(valid_ranges),
        ]
    ).astype(np.float32, copy=False)


def pointcloud_to_xyz(message: Any) -> np.ndarray:
    message_type = str(getattr(message, "_type", ""))
    if message_type == "sensor_msgs/PointCloud2":
        from sensor_msgs import point_cloud2

        values = np.fromiter(
            (
                value
                for point in point_cloud2.read_points(
                    message,
                    field_names=("x", "y", "z"),
                    skip_nans=True,
                )
                for value in point
            ),
            dtype=np.float32,
        )
        return values.reshape(-1, 3) if values.size else np.zeros((0, 3), dtype=np.float32)

    points = getattr(message, "points", ())
    if points:
        values = np.fromiter(
            (value for point in points for value in (point.x, point.y, point.z)),
            dtype=np.float32,
            count=3 * len(points),
        )
        return values.reshape(-1, 3)
    return np.zeros((0, 3), dtype=np.float32)


def inspect_bag(
    bag_path: Path,
    *,
    odom_topic: str,
    scan_topic: str,
    cloud_topic: str,
) -> BagInspection:
    import rosbag

    odom_rows: list[list[float]] = []
    scan_stamps: list[float] = []
    first_scan = None
    first_cloud = None
    first_odom = None

    with rosbag.Bag(str(bag_path), "r") as bag:
        info = bag.get_type_and_topic_info()
        topic_types = {
            str(topic): str(topic_info.msg_type)
            for topic, topic_info in info.topics.items()
        }
        bag_start = float(bag.get_start_time())
        bag_end = float(bag.get_end_time())
        topics = [odom_topic, scan_topic]
        if cloud_topic:
            topics.append(cloud_topic)
        for topic, message, bag_time in bag.read_messages(topics=topics):
            stamp = message_stamp(message, bag_time)
            if topic == odom_topic:
                position, yaw = odom_to_xy_yaw(message)
                odom_rows.append([stamp, float(position[0]), float(position[1]), yaw])
                if first_odom is None:
                    first_odom = message
            elif topic == scan_topic:
                scan_stamps.append(stamp)
                if first_scan is None:
                    first_scan = message
            elif topic == cloud_topic and first_cloud is None:
                first_cloud = message

    if first_odom is None:
        raise RuntimeError(f"No odometry messages found on {odom_topic}")
    if first_scan is None:
        raise RuntimeError(f"No LaserScan messages found on {scan_topic}")
    return BagInspection(
        bag_start=bag_start,
        bag_end=bag_end,
        odom_samples=np.asarray(odom_rows, dtype=np.float64),
        scan_stamps=np.asarray(scan_stamps, dtype=np.float64),
        first_scan=first_scan,
        first_cloud=first_cloud,
        first_odom=first_odom,
        topic_types=topic_types,
    )


def moving_average_path(path: np.ndarray, window: int) -> np.ndarray:
    path = np.asarray(path, dtype=np.float32).reshape(-1, 2)
    window = max(1, int(window))
    if window <= 1 or len(path) < 3:
        return path.copy()
    if window % 2 == 0:
        window += 1
    radius = window // 2
    padded = np.pad(path, ((radius, radius), (0, 0)), mode="edge")
    kernel = np.ones((window,), dtype=np.float32) / float(window)
    smoothed = np.column_stack(
        [
            np.convolve(padded[:, axis], kernel, mode="valid")
            for axis in range(2)
        ]
    ).astype(np.float32)
    smoothed[0] = path[0]
    smoothed[-1] = path[-1]
    return smoothed


def resample_polyline(path: np.ndarray, spacing: float) -> np.ndarray:
    path = np.asarray(path, dtype=np.float32).reshape(-1, 2)
    if len(path) < 2:
        raise ValueError("Reference path requires at least two points")
    keep = np.ones((len(path),), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-4
    path = path[keep]
    if len(path) < 2:
        raise ValueError("Recorded odometry path has no measurable motion")
    ds = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(ds)]).astype(np.float64)
    total = float(s[-1])
    spacing = max(0.01, float(spacing))
    targets = np.arange(0.0, total, spacing, dtype=np.float64)
    if len(targets) == 0 or targets[-1] < total:
        targets = np.concatenate([targets, [total]])
    x = np.interp(targets, s, path[:, 0])
    y = np.interp(targets, s, path[:, 1])
    return np.column_stack([x, y]).astype(np.float32)


def load_recorded_robot_path(
    inspection: BagInspection,
    *,
    spacing: float,
    smoothing_window: int,
) -> np.ndarray:
    """Build the only path used by the real-world benchmark.

    The path is reconstructed from the robot odometry recorded in the rosbag,
    then lightly smoothed and arc-length resampled. No PathGenerator path, CSV
    reference path, or synthetic scenario path is accepted here.
    """
    path = inspection.odom_samples[:, 1:3].astype(np.float32)
    path = moving_average_path(path, smoothing_window)
    return resample_polyline(path, spacing)


def path_data_from_reference(path: np.ndarray) -> dict[str, Any]:
    path = np.asarray(path, dtype=np.float32).reshape(-1, 2)
    length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    return {
        "path": path,
        "start": path[0].copy(),
        "end": path[-1].copy(),
        "length": length,
        "obstacles": np.zeros((0, 3), dtype=np.float32),
        "segment_obstacles": np.zeros((0, 4), dtype=np.float32),
    }


def discover_detector_script(explicit: Optional[Path], running_script: Path) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    local_candidates = [
        running_script.parent / "detection.py",
        running_script.parent / "dr_spaam_ros.py",
        running_script.parent / "dr_spaam_detection.py",
    ]
    for candidate in local_candidates:
        if candidate.is_file() and "SingleUserTracker" in candidate.read_text(encoding="utf-8", errors="ignore"):
            return candidate.resolve()
    try:
        import rospkg

        package_root = Path(rospkg.RosPack().get_path("dr_spaam_ros"))
        for candidate in package_root.rglob("*.py"):
            try:
                text = candidate.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if "class SingleUserTracker" in text and "class DrSpaamROS" in text:
                return candidate.resolve()
    except Exception:
        pass
    raise FileNotFoundError(
        "Could not locate the modified detector script containing SingleUserTracker. "
        "Pass --detector-script explicitly."
    )


def discover_detector_weight(explicit: Optional[Path], detector_script: Path, model: str) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    env_path = os.environ.get("DR_SPAAM_WEIGHT", "").strip()
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if path.is_file():
            return path
    roots = [detector_script.parent, detector_script.parent.parent]
    candidates: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for pattern in ("*.pth", "*.pt", "*.ckpt"):
            candidates.extend(root.rglob(pattern))
    if not candidates:
        raise FileNotFoundError(
            "Detector weight was not found automatically. Pass --detector-weight "
            "with the same weight_file used by dr_spaam_ros.launch."
        )
    model_key = model.lower().replace("-", "").replace("_", "")
    candidates.sort(
        key=lambda p: (
            model_key not in p.name.lower().replace("-", "").replace("_", ""),
            -p.stat().st_mtime,
        )
    )
    return candidates[0].resolve()


def install_offline_rospy_shims(running_module: Any, clock: BagClock, verbose: bool) -> None:
    # ModelPlanner uses rospy.get_time and log helpers even when its ROS I/O
    # class is replaced. Keep time deterministic and avoid requiring roscore.
    running_module.rospy.get_time = clock.now

    def emit(prefix: str, message: Any, *args: Any) -> None:
        if not verbose:
            return
        try:
            text = str(message) % args if args else str(message)
        except Exception:
            text = " ".join([str(message), *map(str, args)])
        print(f"[{prefix}] {text}")

    running_module.rospy.loginfo = lambda message, *args: emit("ros-info", message, *args)
    running_module.rospy.logwarn = lambda message, *args: emit("ros-warn", message, *args)
    running_module.rospy.logerr = lambda message, *args: emit("ros-error", message, *args)
    running_module.rospy.loginfo_throttle = lambda period, message, *args: emit("ros-info", message, *args)
    running_module.rospy.logwarn_throttle = lambda period, message, *args: emit("ros-warn", message, *args)
    running_module.rospy.logerr_throttle = lambda period, message, *args: emit("ros-error", message, *args)


def make_planner(
    running_module: Any,
    *,
    clock: BagClock,
    seed: OfflineInterfaceSeed,
    planning_path: np.ndarray,
    args: argparse.Namespace,
):
    def interface_factory(**kwargs: Any) -> OfflineRosInterface:
        return OfflineRosInterface(clock=clock, seed=seed, **kwargs)

    running_module.LiveRobotRosInterface = interface_factory
    install_offline_rospy_shims(running_module, clock, args.verbose_ros)

    checkpoint = args.ckpt
    if checkpoint is None:
        checkpoint = running_module.resolve_default_checkpoint()
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Planning checkpoint not found: {checkpoint}")

    segmentation_path = args.segmentation_path
    if segmentation_path is None:
        segmentation_path = running_module.DEFAULT_SEGMENTATION_PATH

    recorded_path_data = path_data_from_reference(planning_path)

    # ModelPlanner.__init__ unconditionally calls self._generate_new_path().
    # Temporarily replace that method so construction loads the recorded robot
    # trajectory instead of invoking PathGenerator and printing
    # "New path generated". Restore the class immediately afterwards.
    original_generate_new_path = running_module.ModelPlanner._generate_new_path

    def _load_recorded_robot_path(self: Any) -> None:
        self.current_path_data = copy.deepcopy(recorded_path_data)
        self._precompute_frenet_cache()
        self._reset_position()
        print(
            "Recorded robot path loaded: "
            f"length={float(self.current_path_data['length']):.1f}m"
        )

    running_module.ModelPlanner._generate_new_path = _load_recorded_robot_path
    try:
        planner = running_module.ModelPlanner(
            checkpoint_path=checkpoint,
            device=args.device,
            use_ema=not args.no_ema,
            action_mode=args.action_mode,
            k_lookahead=args.k_lookahead,
            frame_stride=args.frame_stride,
            path_length=float(recorded_path_data["length"]),
            corridor_width=2.2,
            obstacle_radius=0.1,
            leash_length=args.leash_length,
            robot_speed=args.robot_speed,
            robot_radius=args.robot_radius,
            human_radius=args.human_radius,
            fps=args.fps,
            inference_steps=args.inference_steps,
            turn_gain=args.turn_gain,
            curvature_slowdown=not args.no_curvature_slowdown,
            curvature_scale=args.curvature_scale,
            min_speed_scale=args.min_speed_scale,
            log_path=None,
            human_detection_log_path=None,
            eval_path=None,
            collect_enabled=False,
            visualizer=None,
            create_visualizer=False,
            collision_behavior="pause",
            safety_mode="human_robot_qp",
            pointcloud_mode="live",
            odom_topic=args.odom_topic,
            pointcloud_topic=args.pointcloud_topic,
            laser_scan_topic=args.scan_topic,
            range_source=args.range_source,
            human_detections_topic="/offline_dr_spaam_detections",
            human_source="detector",
            cmd_vel_topic="/offline_cmd_vel",
            max_angular_speed=args.max_angular_speed,
            ros_input_timeout=0.0,
            odom_timeout=9999.0,
            pointcloud_timeout=9999.0,
            human_detection_timeout=max(1.0, args.reset_gap),
            human_detector_frame="laser",
            human_world_frame="odom",
            human_detector_y_axis="right",
            human_tf_timeout=0.0,
            human_track_max_jump=args.human_track_max_jump,
            human_rear_sector_range=args.rear_max_range,
            human_rear_sector_angle_deg=args.rear_sector_angle_deg,
            human_kf_process_accel_std=args.human_kf_process_accel_std,
            human_kf_measurement_std=args.human_kf_measurement_std,
            human_kf_gate=args.human_kf_gate,
            human_kf_hold_timeout=args.human_kf_hold_timeout,
            human_kf_max_misses=args.human_kf_max_misses,
            human_continuity_mode=True,
            rosbag_loop_mode=False,
            lidar_height=args.lidar_height,
            enable_motion=False,
            safety_margin=args.safety_margin,
            safety_alpha=args.safety_alpha,
            safety_max_constraints=args.safety_max_constraints,
            safety_influence_distance=args.safety_influence_distance,
            safety_path_corridor=args.safety_path_corridor,
            safety_point_spacing=args.safety_point_spacing,
            debug_preview=False,
            debug_policy=False,
            debug_qp_log=False,
            interaction_segmentation=not args.no_interaction_segmentation,
            segmentation_path=Path(segmentation_path),
            segmentation_window=args.segmentation_window,
            segmentation_min_samples=args.segmentation_min_samples,
        )
    finally:
        running_module.ModelPlanner._generate_new_path = original_generate_new_path

    if planner.policy is None:
        raise RuntimeError("Checkpoint loaded no policy")

    # Defensive assignment in case a future running.py constructor changes its
    # initialization order. reset=False avoids a second physical-state reset.
    planner.set_path_data(recorded_path_data, reset=False)
    planner.use_policy = True
    planner.paused = False
    planner.collision_pause = False
    planner.obs_history.clear()
    planner.prev_robot_pos = None
    planner.interaction_segmenter.reset()
    planner._reset_runtime_caches()
    return planner


def make_detector(detector_module: Any, detector_weight: Path, args: argparse.Namespace):
    detector = detector_module.Detector(
        str(detector_weight),
        model=args.detector_model,
        gpu=not args.detector_cpu,
        stride=args.detector_stride,
        panoramic_scan=args.panoramic_scan,
    )
    tracker = detector_module.SingleUserTracker(
        leash_length=args.leash_length,
        max_jump=args.target_max_jump,
        hold_frames=args.target_hold_frames,
        smoothing_alpha=args.target_smoothing_alpha,
        prior_scale=args.rear_prior_scale,
        min_detect_weight=args.min_detect_weight,
        max_detect_weight=args.max_detect_weight,
        fallback_prior_gain=args.fallback_prior_gain,
        prior_gate=args.candidate_prior_gate,
        track_gate=args.candidate_track_gate,
        max_output_step=args.max_output_step,
        velocity_gain=args.target_velocity_gain,
        velocity_decay=args.target_velocity_decay,
        min_range=args.rear_min_range,
        max_range=args.rear_max_range,
        aperture_deg=args.rear_sector_angle_deg,
        continuous_output=True,
    )
    return detector, tracker


def run_detector_frame(
    detector_module: Any,
    detector: Any,
    tracker: Any,
    scan_message: Any,
    *,
    conf_thresh: float,
    rescue_conf_thresh: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    scan_fov_deg = abs(np.rad2deg(float(scan_message.angle_max) - float(scan_message.angle_min)))
    if not detector.is_ready():
        detector.set_laser_fov(scan_fov_deg)

    scan = np.asarray(scan_message.ranges, dtype=np.float32).copy()
    scan[scan == 0.0] = 29.99
    scan[~np.isfinite(scan)] = 29.99

    start = time.perf_counter()
    dets_xy, dets_cls, _ = detector(scan)
    detector_ms = (time.perf_counter() - start) * 1000.0
    dets_xy = np.asarray(dets_xy, dtype=np.float32).reshape(-1, 2)
    dets_cls = np.asarray(dets_cls, dtype=np.float32).reshape(-1)

    high_mask = dets_cls >= float(conf_thresh)
    rescue_mask = dets_cls >= min(float(conf_thresh), float(rescue_conf_thresh))
    rescue_xy = dets_xy[rescue_mask]
    rescue_cls = dets_cls[rescue_mask]
    rear_xy, rear_cls, rear_indices = detector_module.filter_rear_sector(
        rescue_xy,
        rescue_cls,
        min_range=tracker.min_range,
        max_range=tracker.max_range,
        aperture_deg=tracker.aperture_deg,
    )
    target_xy, track_info = tracker.update(rear_xy, rear_cls)
    output = (
        np.asarray(target_xy, dtype=np.float32).reshape(1, 2)
        if target_xy is not None
        else np.zeros((0, 2), dtype=np.float32)
    )
    info = dict(track_info)
    info.update(
        {
            "raw_count": int(len(dets_xy)),
            "high_conf_count": int(np.count_nonzero(high_mask)),
            "rescue_count": int(np.count_nonzero(rescue_mask)),
            "rear_count": int(len(rear_xy)),
            "rear_indices": rear_indices,
            "detector_ms": float(detector_ms),
            "scan_fov_deg": float(scan_fov_deg),
            "raw_detections_xy": dets_xy.copy(),
            "raw_scores": dets_cls.copy(),
            "rescue_detections_xy": rescue_xy.copy(),
            "rescue_scores": rescue_cls.copy(),
            "rear_detections_xy": rear_xy.copy(),
            "rear_scores": rear_cls.copy(),
            "target_xy": output[0].copy() if len(output) else None,
        }
    )
    return output, info


def compliance_config(running_module: Any, planner: Any):
    return running_module.ComplianceControlConfig(
        data_dt=float(planner.data_dt),
        sim_dt=float(planner.sim_dt),
        frame_stride=int(planner.frame_stride),
        turn_gain=float(planner.turn_gain),
        safety_mode="off",
        heading_control=False,
        forward_only_slowdown=True,
        preserve_heading=False,
        curvature_slowdown=bool(planner.curvature_slowdown),
        curvature_scale=float(planner.curvature_scale),
        min_speed_scale=float(planner.min_speed_scale),
        backoff_scales=tuple(float(v) for v in planner.safety_backoff_scales),
        stop_clearance=float(planner.safety_stop_clearance),
    )


def action_seq_to_deltas(
    planner: Any,
    action_seq: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    actions = np.asarray(action_seq, dtype=np.float32).reshape(-1, 2)
    labels = np.asarray(labels, dtype=object).reshape(-1)
    sim = copy.deepcopy(planner.physics)
    deltas: list[np.ndarray] = []
    for idx, action in enumerate(actions):
        tether = str(labels[min(idx, len(labels) - 1)]).lower() in ("leash", "tether")
        if planner.action_mode == "forward_heading":
            delta, sim = planner._forward_heading_action_to_nominal_delta(
                sim,
                action,
                bre_override=tether,
            )
        else:
            delta = planner._action_to_world_delta(action, sim.robot.position, sim.robot.heading)
            planner._simulate_delta_on_engine(
                sim,
                delta,
                obstacles=None,
                segment_obstacles=None,
                protect_robot=False,
                protect_human=False,
                bre_override=tether,
            )
        deltas.append(np.asarray(delta, dtype=np.float32))
    return np.asarray(deltas, dtype=np.float32)


def apply_safety(
    planner: Any,
    actions: np.ndarray,
    labels: np.ndarray,
    *,
    safety_mode: str,
    preserve_heading_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    old_mode = planner.safety_mode
    try:
        planner.safety_mode = safety_mode
        if safety_mode == "off":
            return action_seq_to_deltas(planner, actions, labels), {
                "modified_steps": 0,
                "constraint_count": 0,
                "min_clearance": float("inf"),
                "mean_shift": 0.0,
            }
        if planner.action_mode == "forward_heading":
            _nominal, safe, _infos = planner._apply_forward_heading_safety_filter(
                actions,
                preserve_heading_mask=preserve_heading_mask,
            )
            return np.asarray(safe, dtype=np.float32), copy.deepcopy(planner.last_safety_stats)
        safe_actions = planner._apply_safety_filter(actions)
        safe_deltas = action_seq_to_deltas(planner, safe_actions, labels)
        return safe_deltas, copy.deepcopy(planner.last_safety_stats)
    finally:
        planner.safety_mode = old_mode


def simulate_delta_horizon(
    planner: Any,
    deltas: np.ndarray,
    labels: np.ndarray,
    obstacles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool, str, int]:
    sim = copy.deepcopy(planner.physics)
    robot_path = [sim.robot.position.copy()]
    human_path = [sim.human.position.copy()]
    collision = False
    collision_who = ""
    collision_step = -1
    labels = np.asarray(labels, dtype=object).reshape(-1)

    global_step = 0
    for idx, delta in enumerate(np.asarray(deltas, dtype=np.float32).reshape(-1, 2)):
        label = str(labels[min(idx, len(labels) - 1)]).lower()
        tether = label in ("leash", "tether")
        forward, turn, _ = planner._delta_to_safe_control(
            delta,
            sim.robot.heading,
            dt=planner.data_dt,
        )
        sim.set_control(float(forward), float(turn), bool(tether))
        for _ in range(int(planner.frame_stride)):
            robot_state, human_state = sim.step()
            robot_path.append(robot_state.position.copy())
            human_path.append(human_state.position.copy())
            global_step += 1
            collided, info = sim.check_collision(
                obstacles,
                segment_obstacles=np.zeros((0, 4), dtype=np.float32),
            )
            if collided:
                collision = True
                collision_who = str(info.get("who", "unknown")) if info else "unknown"
                collision_step = int(global_step)
                break
        if collision:
            break
    return (
        np.asarray(robot_path, dtype=np.float32),
        np.asarray(human_path, dtype=np.float32),
        collision,
        collision_who,
        collision_step,
    )


def min_circle_clearance(path: np.ndarray, radius: float, obstacles: np.ndarray) -> float:
    points = np.asarray(path, dtype=np.float32).reshape(-1, 2)
    circles = np.asarray(obstacles, dtype=np.float32).reshape(-1, 3)
    if len(points) == 0 or len(circles) == 0:
        return float("inf")
    best = float("inf")
    chunk = 2048
    for start in range(0, len(circles), chunk):
        obs = circles[start : start + chunk]
        dist = np.linalg.norm(points[:, None, :] - obs[None, :, :2], axis=2)
        clearance = dist - float(radius) - obs[None, :, 2]
        best = min(best, float(np.min(clearance)))
    return best


def mean_path_deviation(path: np.ndarray, reference: np.ndarray) -> float:
    points = np.asarray(path, dtype=np.float32).reshape(-1, 2)
    ref = np.asarray(reference, dtype=np.float32).reshape(-1, 2)
    if len(points) == 0 or len(ref) == 0:
        return float("inf")
    values = []
    for start in range(0, len(points), 512):
        chunk = points[start : start + 512]
        distances = np.linalg.norm(chunk[:, None, :] - ref[None, :, :], axis=2)
        values.extend(np.min(distances, axis=1).tolist())
    return float(np.mean(values))


def path_progress_m(path: np.ndarray, start_xy: np.ndarray, end_xy: np.ndarray) -> float:
    """Signed nearest-point progress along a polyline."""
    ref = np.asarray(path, dtype=np.float32).reshape(-1, 2)
    if len(ref) < 2:
        return 0.0
    ds = np.linalg.norm(np.diff(ref, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(ds)]).astype(np.float32)
    start_idx = int(np.argmin(np.linalg.norm(ref - np.asarray(start_xy, dtype=np.float32), axis=1)))
    end_idx = int(np.argmin(np.linalg.norm(ref - np.asarray(end_xy, dtype=np.float32), axis=1)))
    return float(s[end_idx] - s[start_idx])


def mean_sequence_distance(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float32).reshape(-1, 2)
    second = np.asarray(second, dtype=np.float32).reshape(-1, 2)
    count = min(len(first), len(second))
    if count <= 0:
        return 0.0
    return float(np.mean(np.linalg.norm(first[:count] - second[:count], axis=1)))


def evaluate_variants(
    running_module: Any,
    planner: Any,
    raw_actions: np.ndarray,
    interaction_label: str,
) -> tuple[dict[str, VariantResult], dict[str, Any]]:
    raw_actions = np.asarray(raw_actions, dtype=np.float32).reshape(-1, 2)
    current_label = "tether" if str(interaction_label).lower() in ("leash", "tether") else "guide"
    current_labels = np.full((len(raw_actions),), current_label, dtype=object)
    tether_mask = np.asarray(current_labels == "tether", dtype=bool)
    cfg = compliance_config(running_module, planner)
    obstacles, _segments = planner._safety_obstacle_inputs()
    obstacles = (
        np.asarray(obstacles, dtype=np.float32).reshape(-1, 3)
        if obstacles is not None
        else np.zeros((0, 3), dtype=np.float32)
    )

    actions_by_mode: dict[str, np.ndarray] = {
        "diffusion": raw_actions.copy(),
        "diffusion_qp": raw_actions.copy(),
    }
    compliance_stats: dict[str, dict[str, Any]] = {
        "diffusion": {"compliance_steps": 0},
        "diffusion_qp": {"compliance_steps": 0},
    }

    always_result = running_module.apply_bre_compliance_control(
        action_seq=raw_actions,
        engine=planner.physics,
        config=cfg,
        safety_filter=planner.safety_filter,
        obstacles=obstacles,
        segment_obstacles=None,
        bre=True,
    )
    actions_by_mode["safe_compliance"] = np.asarray(always_result.actions, dtype=np.float32)
    compliance_stats["safe_compliance"] = copy.deepcopy(always_result.stats)

    ours_result = running_module.apply_interaction_aware_compliance_control(
        action_seq=raw_actions,
        interaction_labels=current_labels,
        engine=planner.physics,
        config=cfg,
        safety_filter=planner.safety_filter,
        obstacles=obstacles,
        segment_obstacles=None,
        bre=bool(current_label == "tether"),
        bre_sequence=tether_mask,
    )
    actions_by_mode["ours"] = np.asarray(ours_result.actions, dtype=np.float32)
    compliance_stats["ours"] = copy.deepcopy(ours_result.stats)

    raw_deltas = action_seq_to_deltas(planner, raw_actions, current_labels)
    results: dict[str, VariantResult] = {}
    details: dict[str, Any] = {}

    for mode in MODE_ORDER:
        start = time.perf_counter()
        actions = actions_by_mode[mode]
        if mode == "diffusion":
            labels = current_labels
            deltas = action_seq_to_deltas(planner, actions, labels)
            safety_stats = {
                "modified_steps": 0,
                "constraint_count": 0,
                "min_clearance": float("inf"),
                "mean_shift": 0.0,
            }
        elif mode == "diffusion_qp":
            labels = current_labels
            deltas, safety_stats = apply_safety(
                planner,
                actions,
                labels,
                safety_mode="human_robot_qp",
                preserve_heading_mask=np.zeros((len(actions),), dtype=bool),
            )
        elif mode == "safe_compliance":
            labels = np.full((len(actions),), "tether", dtype=object)
            deltas, safety_stats = apply_safety(
                planner,
                actions,
                labels,
                safety_mode="human_robot_qp",
                preserve_heading_mask=np.ones((len(actions),), dtype=bool),
            )
        else:
            labels = current_labels
            deltas, safety_stats = apply_safety(
                planner,
                actions,
                labels,
                safety_mode="human_robot_qp",
                preserve_heading_mask=tether_mask,
            )

        robot_path, human_path, collision, collision_who, collision_step = simulate_delta_horizon(
            planner,
            deltas,
            labels,
            obstacles,
        )
        runtime_ms = (time.perf_counter() - start) * 1000.0
        action_shift = float(np.mean(np.linalg.norm(actions - raw_actions, axis=1))) if len(actions) else 0.0
        delta_count = min(len(deltas), len(raw_deltas))
        delta_shift = (
            float(np.mean(np.linalg.norm(deltas[:delta_count] - raw_deltas[:delta_count], axis=1)))
            if delta_count
            else 0.0
        )
        compliance_steps = int(
            compliance_stats.get(mode, {}).get(
                "compliance_steps",
                len(actions) if mode == "safe_compliance" else int(np.count_nonzero(tether_mask)),
            )
        )
        result = VariantResult(
            mode=mode,
            actions=actions,
            deltas=deltas,
            robot_path=robot_path,
            human_path=human_path,
            projected_collision=bool(collision),
            collision_who=collision_who,
            collision_step=int(collision_step),
            robot_min_clearance=min_circle_clearance(robot_path, planner.physics.robot_radius, obstacles),
            human_min_clearance=min_circle_clearance(human_path, planner.physics.human_radius, obstacles),
            mean_action_shift=action_shift,
            mean_delta_shift=delta_shift,
            safety_modified_steps=int(safety_stats.get("modified_steps", 0)),
            safety_constraint_count=int(safety_stats.get("constraint_count", 0)),
            safety_min_clearance=float(safety_stats.get("min_clearance", float("inf"))),
            robot_horizon_displacement=float(np.linalg.norm(robot_path[-1] - robot_path[0])),
            human_horizon_displacement=float(np.linalg.norm(human_path[-1] - human_path[0])),
            path_deviation=mean_path_deviation(robot_path, planner.current_path_data["path"]),
            runtime_ms=float(runtime_ms),
            compliance_steps=compliance_steps,
        )
        results[mode] = result
        details[mode] = {
            "safety_stats": safety_stats,
            "compliance_stats": compliance_stats.get(mode, {}),
        }
    return results, details


def current_human_local_from_world(robot_position: np.ndarray, robot_yaw: float, human_world: np.ndarray) -> np.ndarray:
    relative = np.asarray(human_world, dtype=np.float32).reshape(2) - np.asarray(robot_position, dtype=np.float32).reshape(2)
    c = float(np.cos(robot_yaw))
    s = float(np.sin(robot_yaw))
    x_forward = c * relative[0] + s * relative[1]
    y_left = -s * relative[0] + c * relative[1]
    return np.array([x_forward, -y_left], dtype=np.float32)  # detector y-right


def reset_temporal_state(planner: Any, tracker: Any) -> None:
    tracker.reset()
    planner.interaction_segmenter.reset()
    planner._human_kf.reset()
    planner._tracked_human_position = None
    planner._tracked_human_velocity = np.zeros((2,), dtype=np.float32)
    planner._human_detection_seq = -1
    planner._human_kf_last_measurement_stamp = 0.0
    planner._human_kf_consecutive_misses = 0
    planner.obs_history.clear()
    planner.prev_robot_pos = None
    planner._reset_runtime_caches()



def downsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    """Deterministically cap visualization-only point arrays saved to JSON."""
    array = np.asarray(points)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    max_points = int(max_points)
    if max_points <= 0 or len(array) <= max_points:
        return array.copy()
    indices = np.linspace(0, len(array) - 1, num=max_points, dtype=np.int64)
    return array[indices].copy()


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    run_wall_start = time.perf_counter()

    bag_path = args.bag.expanduser().resolve()
    running_script = args.running_script.expanduser().resolve()
    detector_script = discover_detector_script(args.detector_script, running_script)

    print(f"[1/5] Inspecting bag: {bag_path}")
    inspection = inspect_bag(
        bag_path,
        odom_topic=args.odom_topic,
        scan_topic=args.scan_topic,
        cloud_topic=args.pointcloud_topic,
    )
    print(
        f"      duration={inspection.duration:.2f}s, odom={len(inspection.odom_samples)}, "
        f"scan={len(inspection.scan_stamps)}, median_scan_dt={inspection.median_scan_dt:.4f}s"
    )

    running_module = load_source_module("guide_running_offline", running_script)
    detector_module = load_source_module("dr_spaam_offline_source", detector_script)
    planning_path = load_recorded_robot_path(
        inspection,
        spacing=args.reference_spacing,
        smoothing_window=args.reference_smoothing_window,
    )
    planning_path_source = "recorded_robot_odom"
    print(
        f"[2/5] Recorded robot path: {len(planning_path)} points, "
        f"source={planning_path_source}"
    )

    detector_weight = discover_detector_weight(
        args.detector_weight,
        detector_script,
        args.detector_model,
    )
    print(f"      running={running_script}")
    print(f"      detector={detector_script}")
    print(f"      detector_weight={detector_weight}")

    first_position, first_yaw = odom_to_xy_yaw(inspection.first_odom)
    first_stamp = message_stamp(inspection.first_odom, inspection.bag_start)
    first_scan_xyz = laser_scan_to_xyz(inspection.first_scan)
    first_cloud_xyz = (
        pointcloud_to_xyz(inspection.first_cloud)
        if inspection.first_cloud is not None
        else np.zeros((0, 3), dtype=np.float32)
    )
    seed = OfflineInterfaceSeed(
        odom_position=first_position,
        odom_yaw=first_yaw,
        odom_stamp=first_stamp,
        scan_xyz=first_scan_xyz,
        cloud_xyz=first_cloud_xyz,
        human_local_xy=np.array([[-args.leash_length, 0.0]], dtype=np.float32),
        detector_to_base_xy=np.array(
            [args.detector_frame_x, args.detector_frame_y], dtype=np.float32
        ),
        detector_to_base_yaw=np.deg2rad(args.detector_frame_yaw_deg),
    )
    clock = BagClock(first_stamp)

    print("[3/5] Loading detector and planning policy")
    detector, tracker = make_detector(detector_module, detector_weight, args)
    planner = make_planner(
        running_module,
        clock=clock,
        seed=seed,
        planning_path=planning_path,
        args=args,
    )
    offline_io: OfflineRosInterface = planner.ros_io
    print(
        f"      observation_mode={planner.observation_mode}, action_mode={planner.action_mode}, "
        f"obs_dim={planner.obs_dim}, action_horizon={planner.n_action_steps}, "
        f"device={planner.device}"
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / args.result_json_name

    detection_frames: list[dict[str, Any]] = []
    planning_frames: list[dict[str, Any]] = []

    import rosbag

    latest_odom_position = first_position.copy()
    latest_odom_yaw = float(first_yaw)
    latest_odom_stamp = float(first_stamp)
    last_scan_stamp: Optional[float] = None
    scan_idx = 0
    plan_idx = 0
    bag_start_stamp = (
        float(inspection.scan_stamps[0])
        if len(inspection.scan_stamps)
        else inspection.bag_start
    )
    start_limit = bag_start_stamp + max(0.0, args.start_offset)
    end_limit = (
        float("inf")
        if args.duration <= 0.0
        else start_limit + float(args.duration)
    )

    print("[4/5] Processing bag frames")
    topics = [args.odom_topic, args.scan_topic]
    if args.pointcloud_topic:
        topics.append(args.pointcloud_topic)

    with rosbag.Bag(str(bag_path), "r") as bag:
        for topic, message, bag_time in bag.read_messages(topics=topics):
            stamp = message_stamp(message, bag_time)

            if topic == args.odom_topic:
                latest_odom_position, latest_odom_yaw = odom_to_xy_yaw(message)
                latest_odom_stamp = stamp
                offline_io.update_odom(
                    latest_odom_position,
                    latest_odom_yaw,
                    stamp,
                )
                continue

            if topic == args.pointcloud_topic:
                cloud_xyz = pointcloud_to_xyz(message)
                finite = np.isfinite(cloud_xyz).all(axis=1)
                offline_io.update_cloud(cloud_xyz[finite], stamp)
                continue

            if topic != args.scan_topic:
                continue
            if stamp < start_limit:
                continue
            if stamp > end_limit:
                break
            if args.max_scans > 0 and scan_idx >= args.max_scans:
                break

            frame_wall_start = time.perf_counter()
            clock.set(stamp)
            offline_io.update_odom(
                latest_odom_position,
                latest_odom_yaw,
                latest_odom_stamp,
            )

            scan_preprocess_start = time.perf_counter()
            scan_xyz = laser_scan_to_xyz(message)
            offline_io.update_scan(scan_xyz, stamp)
            scan_preprocess_ms = (
                time.perf_counter() - scan_preprocess_start
            ) * 1000.0

            temporal_reset = False
            if last_scan_stamp is not None:
                dt = stamp - last_scan_stamp
                if dt < -0.25 or dt > args.reset_gap:
                    reset_temporal_state(planner, tracker)
                    temporal_reset = True
            last_scan_stamp = stamp

            perception_start = time.perf_counter()
            human_local_candidates, detector_info = run_detector_frame(
                detector_module,
                detector,
                tracker,
                message,
                conf_thresh=args.conf_thresh,
                rescue_conf_thresh=args.rescue_conf_thresh,
            )
            offline_io.update_human_local(human_local_candidates, stamp)

            planner._synchronize_robot_from_odometry()
            human_ok = planner._refresh_human_from_detector()
            downstream_prior_fallback = False
            if not human_ok:
                downstream_prior_fallback = True
                prior_world = planner._rear_leash_prior_position()
                fallback_local = current_human_local_from_world(
                    latest_odom_position,
                    latest_odom_yaw,
                    prior_world,
                )
                offline_io.update_human_local(
                    fallback_local.reshape(1, 2),
                    stamp,
                )
                planner._refresh_human_from_detector()

            planner._update_interaction_segmentation()
            interaction_label = planner._current_interaction_label()
            planner.bre = interaction_label == "tether"
            perception_pipeline_ms = (
                time.perf_counter() - perception_start
            ) * 1000.0

            human_world = planner.physics.human.position.copy()
            human_local = current_human_local_from_world(
                planner.physics.robot.position,
                planner.physics.robot.heading,
                human_world,
            )

            detection_record: dict[str, Any] = {
                "scan_idx": int(scan_idx),
                "stamp": float(stamp),
                "relative_time": float(stamp - bag_start_stamp),
                "robot_position": planner.physics.robot.position.copy(),
                "robot_yaw": float(planner.physics.robot.heading),
                "human_world_position": human_world.copy(),
                "human_local_position": human_local.copy(),
                "accepted": bool(detector_info.get("accepted", False)),
                "fallback": bool(detector_info.get("fallback", False)),
                "fallback_kind": str(detector_info.get("fallback_kind", "")),
                "confidence": float(detector_info.get("confidence", 0.0)),
                "raw_count": int(detector_info.get("raw_count", 0)),
                "high_conf_count": int(
                    detector_info.get("high_conf_count", 0)
                ),
                "rescue_count": int(detector_info.get("rescue_count", 0)),
                "rear_count": int(detector_info.get("rear_count", 0)),
                "tracker_reason": str(detector_info.get("reason", "")),
                "interaction_label": str(interaction_label),
                "segmentation_raw_label": str(
                    planner.interaction_segmenter.current_label
                ),
                "segmentation_samples": int(
                    len(planner.interaction_segmenter.samples)
                ),
                "segmentation_decode_ms": float(
                    planner.interaction_segmenter.last_decode_ms
                ),
                "downstream_tracking_mode": str(
                    planner._human_tracking_mode
                ),
                "downstream_kf_prediction": bool(
                    planner._human_kf_using_prediction
                ),
                "downstream_kf_misses": int(
                    planner._human_kf_consecutive_misses
                ),
                "downstream_prior_fallback": bool(
                    downstream_prior_fallback
                ),
                "temporal_reset": bool(temporal_reset),
                "timing": {
                    "scan_preprocess_ms": float(scan_preprocess_ms),
                    "detector_ms": float(
                        detector_info.get("detector_ms", 0.0)
                    ),
                    "segmentation_decode_ms": float(
                        planner.interaction_segmenter.last_decode_ms
                    ),
                    "perception_pipeline_ms": float(
                        perception_pipeline_ms
                    ),
                    "frame_total_ms": None,
                },
                "scan_points_local": downsample_points(
                    scan_xyz[:, :2],
                    args.json_max_scan_points,
                ),
                "detector": copy.deepcopy(detector_info),
                "planning_plan_idx": None,
            }

            planning_due = (
                scan_idx % max(1, args.plan_every)
            ) == 0
            planning_allowed = (
                planning_due
                and (
                    not args.skip_fallback_plans
                    or not detector_info.get("fallback", False)
                )
            )

            if planning_allowed:
                planning_wall_start = time.perf_counter()

                obs_start = time.perf_counter()
                obs = planner._build_obs(
                    planner.physics.robot.position,
                    planner.physics.human.position,
                    planner.physics.robot.heading,
                )
                if len(planner.obs_history) == 0:
                    for _ in range(planner.n_obs_steps):
                        planner.obs_history.append(obs.copy())
                else:
                    planner.obs_history.append(obs.copy())
                planner.prev_robot_pos = (
                    planner.physics.robot.position.copy()
                )
                observation_ms = (
                    time.perf_counter() - obs_start
                ) * 1000.0

                inference_start = time.perf_counter()
                raw_actions = planner._predict_action()
                inference_ms = (
                    time.perf_counter() - inference_start
                ) * 1000.0

                preprocess_start = time.perf_counter()
                raw_labels = np.full(
                    (len(raw_actions),),
                    (
                        "tether"
                        if interaction_label == "tether"
                        else "guide"
                    ),
                    dtype=object,
                )
                raw_deltas = action_seq_to_deltas(
                    planner,
                    raw_actions,
                    raw_labels,
                )
                raw_preview = planner._deltas_to_path(
                    raw_deltas,
                    protect_robot=False,
                    protect_human=False,
                    bre_override=interaction_label == "tether",
                )
                point_stats = planner._prepare_safety_point_obstacles(
                    raw_preview
                )
                preprocess_ms = (
                    time.perf_counter() - preprocess_start
                ) * 1000.0

                variant_eval_start = time.perf_counter()
                variants, variant_details = evaluate_variants(
                    running_module,
                    planner,
                    raw_actions,
                    interaction_label,
                )
                variant_evaluation_ms = (
                    time.perf_counter() - variant_eval_start
                ) * 1000.0

                compliant_reference = variants[
                    "safe_compliance"
                ].deltas
                diffusion_result = variants["diffusion"]
                diffusion_clearance = min(
                    float(diffusion_result.robot_min_clearance),
                    float(diffusion_result.human_min_clearance),
                )
                challenge_frame = bool(
                    diffusion_result.projected_collision
                    or diffusion_clearance
                    < float(args.challenge_clearance_threshold)
                )

                obstacle_points, _ = planner._safety_obstacle_inputs()
                obstacle_points = (
                    np.asarray(
                        obstacle_points,
                        dtype=np.float32,
                    ).reshape(-1, 3)
                    if obstacle_points is not None
                    else np.zeros((0, 3), dtype=np.float32)
                )

                mode_records: dict[str, Any] = {}
                for mode in MODE_ORDER:
                    result = variants[mode]
                    combined_clearance = min(
                        float(result.robot_min_clearance),
                        float(result.human_min_clearance),
                    )
                    horizon_duration = max(
                        1e-6,
                        float(len(result.deltas))
                        * float(planner.data_dt),
                    )
                    progress = path_progress_m(
                        planner.current_path_data["path"],
                        result.robot_path[0],
                        result.robot_path[-1],
                    )
                    progress_speed = progress / horizon_duration
                    safety_violation = bool(
                        result.projected_collision
                        or combined_clearance
                        < float(args.safety_eval_clearance)
                    )
                    strong_intervention = bool(
                        float(result.mean_delta_shift)
                        > float(
                            args.strong_intervention_threshold
                        )
                    )
                    tether_response_error = (
                        mean_sequence_distance(
                            result.deltas,
                            compliant_reference,
                        )
                        if interaction_label == "tether"
                        else float("nan")
                    )
                    guide_unnecessary_intervention = (
                        float(result.mean_delta_shift)
                        if interaction_label == "guide"
                        else float("nan")
                    )
                    safety_fraction = (
                        float(result.safety_modified_steps)
                        / max(1, len(result.deltas))
                    )
                    compliance_fraction = (
                        float(result.compliance_steps)
                        / max(1, len(result.actions))
                    )

                    payload = asdict(result)
                    payload.update(
                        {
                            "combined_min_clearance": float(
                                combined_clearance
                            ),
                            "horizon_duration_sec": float(
                                horizon_duration
                            ),
                            "path_progress_m": float(progress),
                            "progress_speed_mps": float(
                                progress_speed
                            ),
                            "safety_violation": bool(
                                safety_violation
                            ),
                            "strong_intervention": bool(
                                strong_intervention
                            ),
                            "tether_response_error_m": float(
                                tether_response_error
                            ),
                            "guide_unnecessary_intervention_m": float(
                                guide_unnecessary_intervention
                            ),
                            "safety_intervention_fraction": float(
                                safety_fraction
                            ),
                            "compliance_intervention_fraction": float(
                                compliance_fraction
                            ),
                            "safety_and_compliance": copy.deepcopy(
                                variant_details[mode]
                            ),
                        }
                    )
                    mode_records[mode] = payload

                planning_pipeline_ms = (
                    time.perf_counter() - planning_wall_start
                ) * 1000.0
                planning_record = {
                    "plan_idx": int(plan_idx),
                    "scan_idx": int(scan_idx),
                    "stamp": float(stamp),
                    "relative_time": float(
                        stamp - bag_start_stamp
                    ),
                    "robot_position": planner.physics.robot.position.copy(),
                    "robot_heading": float(
                        planner.physics.robot.heading
                    ),
                    "human_position": planner.physics.human.position.copy(),
                    "human_local_position": human_local.copy(),
                    "interaction_label": str(interaction_label),
                    "challenge_frame": bool(challenge_frame),
                    "detector_fallback": bool(
                        detector_info.get("fallback", False)
                    ),
                    "detector_confidence": float(
                        detector_info.get("confidence", 0.0)
                    ),
                    "point_stats": copy.deepcopy(point_stats),
                    "timing": {
                        "observation_ms": float(observation_ms),
                        "inference_ms": float(inference_ms),
                        "planning_preprocess_ms": float(
                            preprocess_ms
                        ),
                        "variant_evaluation_ms": float(
                            variant_evaluation_ms
                        ),
                        "planning_pipeline_ms": float(
                            planning_pipeline_ms
                        ),
                    },
                    "obstacles_world": downsample_points(
                        obstacle_points,
                        args.json_max_obstacle_points,
                    ),
                    "modes": mode_records,
                }
                planning_frames.append(planning_record)
                detection_record["planning_plan_idx"] = int(plan_idx)
                plan_idx += 1

            detection_record["timing"]["frame_total_ms"] = float(
                (time.perf_counter() - frame_wall_start) * 1000.0
            )
            detection_frames.append(detection_record)

            scan_idx += 1
            if scan_idx % max(1, args.progress_every) == 0:
                print(
                    f"      scans={scan_idx}/{len(inspection.scan_stamps)}, "
                    f"plans={plan_idx}, "
                    f"t={stamp - bag_start_stamp:.1f}s, "
                    f"human={interaction_label}, "
                    f"det={'fallback' if detector_info.get('fallback', False) else 'accepted'}"
                )

    print("[5/5] Writing raw benchmark JSON")
    result = {
        "schema_version": 2,
        "benchmark_type": "static_open_loop_real_world_rosbag",
        "description": (
            "Raw benchmark output. No aggregate statistics or plots are "
            "computed here; use analyze_benchmark.py."
        ),
        "bag": {
            "path": str(bag_path),
            "duration_sec": float(inspection.duration),
            "bag_start": float(inspection.bag_start),
            "bag_end": float(inspection.bag_end),
            "topic_types": inspection.topic_types,
            "median_scan_dt": float(inspection.median_scan_dt),
            "odom_message_count": int(
                len(inspection.odom_samples)
            ),
            "scan_message_count": int(
                len(inspection.scan_stamps)
            ),
        },
        "inputs": {
            "running_script": str(running_script),
            "detector_script": str(detector_script),
            "detector_weight": str(detector_weight),
            "checkpoint": str(
                args.ckpt
                or running_module.resolve_default_checkpoint()
            ),
            "planning_path_source": planning_path_source,
        },
        "configuration": {
            key: json_safe(value)
            for key, value in vars(args).items()
        },
        "recorded_robot_path": planning_path,
        "recorded_odom_samples": inspection.odom_samples,
        "detection_frames": detection_frames,
        "planning_frames": planning_frames,
        "run": {
            "processed_scan_frames": int(len(detection_frames)),
            "processed_planning_frames": int(len(planning_frames)),
            "wall_time_sec": float(
                time.perf_counter() - run_wall_start
            ),
        },
    }

    safe_result = json_safe(result)
    with result_path.open("w", encoding="utf-8") as fp:
        json.dump(
            safe_result,
            fp,
            ensure_ascii=False,
            indent=2 if args.pretty_json else None,
            allow_nan=False,
        )

    print(
        json.dumps(
            {
                "result_json": str(result_path),
                "scan_frames": len(detection_frames),
                "planning_frames": len(planning_frames),
                "wall_time_sec": result["run"]["wall_time_sec"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate raw static/offline rosbag benchmark results. "
            "Analysis is performed separately by analyze_benchmark.py."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bag",
        type=Path,
        default=Path(
            "/home/yyf/IROS2026/FollowDataset/logs/"
            "2026-08-02-18-18-16.bag"
        ),
        help="Input rosbag path",
    )
    parser.add_argument(
        "--running-script",
        type=Path,
        default=Path(__file__).resolve().parent / "running.py",
        help="Dynamic running.py used by the online system",
    )
    parser.add_argument(
        "--detector-script",
        type=Path,
        default=Path(
            "/home/yyf/IROS2026/dr_spaam_ws/src/dr_spaam_ros/"
            "src/dr_spaam_ros/dr_spaam_ros.py"
        ),
        help=(
            "Modified dr_spaam_ros Python file containing "
            "SingleUserTracker"
        ),
    )
    parser.add_argument(
        "--detector-weight",
        type=Path,
        default=Path(
            "/home/yyf/IROS2026/dr_spaam_ws/models/"
            "self_supervised_person_detection/"
            "ckpt_jrdb_ann_dr_spaam_e20.pth"
        ),
        help="Path to the detector weight file",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=Path(
            "/home/yyf/IROS2026/diffusion_policy/data/outputs/"
            "2026.01.21/"
            "14.14.46_train_diffusion_unet_lowdim_guide_guide_lowdim/"
            "checkpoints/epoch=0090-test_mean_score=0.630.ckpt"
        ),
        help="Diffusion policy checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "tests/artifacts/2026-08-02-18-18-16.bag"
        ),
    )
    parser.add_argument(
        "--result-json-name",
        default="benchmark_results.json",
    )
    parser.add_argument(
        "--pretty-json",
        action="store_true",
        help="Indent JSON for manual inspection (larger file)",
    )
    parser.add_argument(
        "--json-max-scan-points",
        type=int,
        default=2000,
        help=(
            "Maximum LaserScan XY points saved per frame for "
            "offline visualization; <=0 saves all"
        ),
    )
    parser.add_argument(
        "--json-max-obstacle-points",
        type=int,
        default=1500,
        help=(
            "Maximum safety obstacle points saved per planning "
            "frame for offline visualization; <=0 saves all"
        ),
    )

    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--scan-topic", default="/front/scan")
    parser.add_argument("--pointcloud-topic", default="/livox/lidar")
    parser.add_argument(
        "--range-source",
        choices=("laser_scan", "point_cloud"),
        default="laser_scan",
        help=(
            "Range source used by the planning observation; "
            "detection always uses LaserScan"
        ),
    )
    parser.add_argument("--start-offset", type=float, default=0.0)
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="0 means until bag end",
    )
    parser.add_argument(
        "--max-scans",
        type=int,
        default=0,
        help="0 means all scans",
    )
    parser.add_argument(
        "--plan-every",
        type=int,
        default=1,
        help="Plan every N LaserScan frames",
    )
    parser.add_argument("--skip-fallback-plans", action="store_true")
    parser.add_argument(
        "--reset-gap",
        type=float,
        default=1.0,
        help="Reset temporal filters after a larger scan gap",
    )
    parser.add_argument("--progress-every", type=int, default=100)

    # Kept for command-line compatibility with the previous script.
    parser.add_argument(
        "--representative-count",
        type=int,
        default=5,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--save-plan-arrays",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    parser.add_argument("--reference-spacing", type=float, default=0.10)
    parser.add_argument(
        "--reference-smoothing-window",
        type=int,
        default=9,
    )

    parser.add_argument("--detector-model", default="DR-SPAAM")
    parser.add_argument("--detector-stride", type=int, default=1)
    parser.add_argument("--detector-cpu", action="store_true")
    parser.add_argument(
        "--panoramic-scan",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--conf-thresh", type=float, default=0.30)
    parser.add_argument(
        "--rescue-conf-thresh",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--rear-sector-angle-deg",
        type=float,
        default=90.0,
    )
    parser.add_argument(
        "--detector-frame-x",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--detector-frame-y",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--detector-frame-yaw-deg",
        type=float,
        default=0.0,
    )
    parser.add_argument("--rear-min-range", type=float, default=0.30)
    parser.add_argument("--rear-max-range", type=float, default=3.0)
    parser.add_argument("--target-max-jump", type=float, default=0.8)
    parser.add_argument("--target-hold-frames", type=int, default=5)
    parser.add_argument(
        "--target-smoothing-alpha",
        type=float,
        default=0.65,
    )
    parser.add_argument("--rear-prior-scale", type=float, default=0.60)
    parser.add_argument("--min-detect-weight", type=float, default=0.10)
    parser.add_argument("--max-detect-weight", type=float, default=0.90)
    parser.add_argument(
        "--fallback-prior-gain",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--candidate-prior-gate",
        type=float,
        default=1.20,
    )
    parser.add_argument(
        "--candidate-track-gate",
        type=float,
        default=1.00,
    )
    parser.add_argument("--max-output-step", type=float, default=0.25)
    parser.add_argument(
        "--target-velocity-gain",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--target-velocity-decay",
        type=float,
        default=0.65,
    )

    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--action-mode", default=None)
    parser.add_argument("--k-lookahead", type=int, default=None)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--frame-stride", type=int, default=None)
    parser.add_argument("--inference-steps", type=int, default=8)
    parser.add_argument("--leash-length", type=float, default=1.2)
    parser.add_argument("--robot-speed", type=float, default=1.0)
    parser.add_argument(
        "--max-angular-speed",
        type=float,
        default=1.0,
    )
    parser.add_argument("--robot-radius", type=float, default=0.10)
    parser.add_argument("--human-radius", type=float, default=0.10)
    parser.add_argument("--lidar-height", type=float, default=0.40)
    parser.add_argument("--turn-gain", type=float, default=1.2)
    parser.add_argument(
        "--no-curvature-slowdown",
        action="store_true",
    )
    parser.add_argument("--curvature-scale", type=float, default=0.7)
    parser.add_argument("--min-speed-scale", type=float, default=0.25)

    parser.add_argument("--safety-margin", type=float, default=0.20)
    parser.add_argument("--safety-alpha", type=float, default=1.0)
    parser.add_argument(
        "--safety-max-constraints",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--safety-influence-distance",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--safety-path-corridor",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--safety-point-spacing",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--safety-eval-clearance",
        type=float,
        default=None,
        help=(
            "Clearance below which a predicted path violates "
            "the safety envelope; defaults to --safety-margin"
        ),
    )
    parser.add_argument(
        "--challenge-clearance-threshold",
        type=float,
        default=0.3,
        help=(
            "Select challenging frames using diffusion minimum "
            "clearance"
        ),
    )
    parser.add_argument(
        "--strong-intervention-threshold",
        type=float,
        default=0.05,
        help=(
            "Mean delta-sequence shift [m] required to count a "
            "frame as strongly intervened"
        ),
    )

    parser.add_argument(
        "--human-track-max-jump",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--human-kf-process-accel-std",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--human-kf-measurement-std",
        type=float,
        default=0.18,
    )
    parser.add_argument("--human-kf-gate", type=float, default=11.83)
    parser.add_argument(
        "--human-kf-hold-timeout",
        type=float,
        default=1.5,
    )
    parser.add_argument("--human-kf-max-misses", type=int, default=30)

    parser.add_argument(
        "--segmentation-path",
        type=Path,
        default=Path("/home/yyf/workspace/segmentation.py"),
    )
    parser.add_argument("--segmentation-window", type=int, default=120)
    parser.add_argument(
        "--segmentation-min-samples",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--no-interaction-segmentation",
        action="store_true",
    )
    parser.add_argument("--verbose-ros", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.bag.expanduser().is_file():
        raise FileNotFoundError(args.bag)
    if not args.running_script.expanduser().is_file():
        raise FileNotFoundError(args.running_script)
    if args.plan_every < 1:
        raise ValueError("--plan-every must be >= 1")
    if args.frame_stride is not None and args.frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")
    if not 0.0 < args.rear_sector_angle_deg <= 180.0:
        raise ValueError(
            "--rear-sector-angle-deg must be in (0, 180]"
        )
    if not 0.0 <= args.rear_min_range < args.rear_max_range:
        raise ValueError(
            "Require 0 <= rear-min-range < rear-max-range"
        )
    if args.safety_eval_clearance is None:
        args.safety_eval_clearance = float(args.safety_margin)
    if args.challenge_clearance_threshold is None:
        args.challenge_clearance_threshold = float(
            args.safety_eval_clearance
        )
    if args.safety_eval_clearance < 0.0:
        raise ValueError(
            "--safety-eval-clearance must be >= 0"
        )
    if args.challenge_clearance_threshold < 0.0:
        raise ValueError(
            "--challenge-clearance-threshold must be >= 0"
        )
    if args.strong_intervention_threshold < 0.0:
        raise ValueError(
            "--strong-intervention-threshold must be >= 0"
        )
    if args.json_max_scan_points < 0:
        raise ValueError(
            "--json-max-scan-points must be >= 0"
        )
    if args.json_max_obstacle_points < 0:
        raise ValueError(
            "--json-max-obstacle-points must be >= 0"
        )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_args(args)
    run_benchmark(args)


if __name__ == "__main__":
    main()