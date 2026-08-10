#!/usr/bin/env python3
from __future__ import annotations
"""
Guide Dog Robot Planning Tool (hybrid real-robot runtime with selectable human input)

Controls:
    P     Toggle policy/manual control
    C     Toggle LaserScan/PointCloud input and display
    H     Toggle simulated/detector human
    SPACE Pause/Resume
    R     Reset position
    N     Generate new path
    O     Add obstacle ahead
    ESC   Exit
    Arrows Manual control (when policy disabled)
"""
import argparse
import copy
import json
import importlib
import re
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pygame
import torch
import dill
import rospy
import rostopic
import tf2_ros
from geometry_msgs.msg import PoseArray, Twist
from nav_msgs.msg import Odometry
from sensor_msgs import point_cloud2
from sensor_msgs.msg import LaserScan, PointCloud2

# Allow importing project modules when running from the FollowDataset directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FOLLOWDATASET_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(FOLLOWDATASET_DIR) not in sys.path:
    sys.path.insert(0, str(FOLLOWDATASET_DIR))

# Add diffusion_policy submodule to Python path
DIFFUSION_POLICY_DIR = PROJECT_ROOT / "diffusion_policy"
if str(DIFFUSION_POLICY_DIR) not in sys.path:
    sys.path.insert(0, str(DIFFUSION_POLICY_DIR))

from src.path_generator import PathGenerator
from src.physics import PhysicsEngine
from src.visualizer import Visualizer
from src.data_storage import DataStorage
from src.scoring import TrajectoryScorer
from src.safety_filter import QPSafetyFilter
from src.compliance_control import (
    ComplianceControlConfig,
    apply_bre_compliance_control,
    apply_interaction_aware_compliance_control,
)
from diffusion_policy.common.guide_mid360 import (
    GuideMid360ObservationConfig,
    encode_mid360_scan_from_local_points,
)


DEFAULT_SEGMENTATION_PATH = Path(
    "/home/yyf/workspace/segmentation.py"
)


class OnlineInteractionSegmenter:
    """Causal guide/tether recognition using a sliding trajectory window."""

    def __init__(
        self,
        segmentation_path: Path,
        *,
        enabled: bool = True,
        window_size: int = 120,
        min_samples: int = 12,
    ) -> None:
        self.enabled = bool(enabled)
        self.segmentation_path = Path(segmentation_path)
        self.window_size = max(12, int(window_size))
        self.min_samples = max(3, min(int(min_samples), self.window_size))
        self.samples: deque[dict] = deque(maxlen=self.window_size)
        self.module = None
        self.feature_config = None
        self.model = None
        self.current_label = "unknown"
        self.last_decode_ms = 0.0
        self.last_error: Optional[str] = None

        if not self.enabled:
            print("[segmentation] disabled; using manual B/timed state")
            return
        try:
            self._initialize()
        except Exception as exc:
            self.enabled = False
            self.last_error = f"{type(exc).__name__}: {exc}"
            print(
                "[segmentation] initialization failed; "
                f"using manual B/timed state: {self.last_error}"
            )

    @property
    def has_label(self) -> bool:
        return self.enabled and self.current_label in ("guide", "tether")

    def reset(self) -> None:
        self.samples.clear()
        self.current_label = "unknown"
        self.last_decode_ms = 0.0
        self.last_error = None

    def _initialize(self) -> None:
        if not self.segmentation_path.is_file():
            raise FileNotFoundError(self.segmentation_path)
        spec = importlib.util.spec_from_file_location(
            "guidedog_interaction_segmentation_runtime",
            self.segmentation_path,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import {self.segmentation_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        feature_config = module.FeatureConfig(
            smoothing_window=5,
            lead_distance_mm=120.0,
            link_rate_mm_per_frame=1.0,
            speed_balance_mm_per_frame=1.0,
            causal_max_lag_frames=6,
            causal_window=9,
            evidence_threshold=0.0,
        )
        calibration_frame = self._build_calibration_frame()
        calibration_features = module.extract_interaction_features(
            calibration_frame,
            feature_config,
        )
        calibration_trajectory = module.Trajectory(
            csv_path=Path("runtime_segmentation_calibration.csv"),
            frame=calibration_frame,
            features=calibration_features,
        )
        model = module.train_drag_threshold_model(
            [calibration_trajectory],
            module.HMMConfig(min_state_duration=6),
            feature_config,
            threshold_percentile=40.0,
        )

        self.module = module
        self.feature_config = feature_config
        self.model = model
        print(
            "[segmentation] ready: "
            f"path={self.segmentation_path}, window={self.window_size}, "
            f"min_samples={self.min_samples}, classifier={model.classifier_kind}"
        )

    @staticmethod
    def _build_calibration_frame() -> pd.DataFrame:
        """Build the same calibration trajectory used by the reference demo."""
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
        truth = np.where(tether_alpha >= 0.5, "leash", "guide")

        progress = np.zeros((frame_count,), dtype=float)
        for idx in range(1, frame_count):
            progress[idx] = progress[idx - 1] + (18.0 - 6.0 * tether_alpha[idx])
        frame_idx = np.arange(frame_count, dtype=float)
        center_y = 18.0 * np.sin(frame_idx / 17.0)
        lead = (1.0 - tether_alpha) * 300.0 + tether_alpha * -300.0
        lateral = 18.0 * np.sin(frame_idx / 19.0 + tether_alpha * np.pi)
        robot = np.column_stack(
            (progress + 0.5 * lead, center_y + 0.5 * lateral)
        )
        human = np.column_stack(
            (progress - 0.5 * lead, center_y - 0.5 * lateral)
        )
        return pd.DataFrame(
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

    def update(self, robot_state, human_state) -> tuple[str, bool]:
        """Append one measured/simulated pair and decode the newest state."""
        if not self.enabled:
            return self.current_label, False

        robot_position = np.asarray(robot_state.position, dtype=float).reshape(2)
        human_position = np.asarray(human_state.position, dtype=float).reshape(2)
        self.samples.append(
            {
                # segmentation.py uses mocap positions in millimetres.
                "robot_x": float(robot_position[0] * 1000.0),
                "robot_y": float(robot_position[1] * 1000.0),
                "robot_orientation": float(getattr(robot_state, "heading", 0.0)),
                "robot_occlusion": False,
                "human_x": float(human_position[0] * 1000.0),
                "human_y": float(human_position[1] * 1000.0),
                "human_rotation": float(getattr(human_state, "heading", 0.0)),
                "human_occlusion": False,
                "state": "unknown",
            }
        )
        if len(self.samples) < self.min_samples:
            return self.current_label, False

        previous_label = self.current_label
        try:
            frame = pd.DataFrame(list(self.samples))
            start = time.perf_counter()
            features = self.module.extract_interaction_features(
                frame,
                self.feature_config,
            )
            labeled = self.module.decode_trajectory(frame, features, self.model)
            self.last_decode_ms = (time.perf_counter() - start) * 1000.0
            raw_label = str(labeled["semantic_label"].iloc[-1]).strip().lower()
            if raw_label == "leash":
                raw_label = "tether"
            self.current_label = (
                raw_label if raw_label in ("guide", "tether") else "unknown"
            )
            self.last_error = None
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            print(f"[segmentation] decode failed: {self.last_error}")
            return self.current_label, False

        changed = (
            self.current_label in ("guide", "tether")
            and self.current_label != previous_label
        )
        return self.current_label, changed


class LiveRobotRosInterface:
    """Thread-safe ROS I/O boundary for measured robot and human inputs."""

    def __init__(
        self,
        *,
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
        self.odom_topic = str(odom_topic)
        self.pointcloud_topic = str(pointcloud_topic)
        self.laser_scan_topic = str(laser_scan_topic)
        self.range_source = normalize_range_source(range_source)
        self.human_detections_topic = str(human_detections_topic)
        self.odom_timeout = float(odom_timeout)
        self.pointcloud_timeout = float(pointcloud_timeout)
        self.human_detection_timeout = max(0.0, float(human_detection_timeout))
        self.human_detector_frame = str(human_detector_frame or "").strip()
        self.human_world_frame = str(human_world_frame or "").strip()
        self.human_detector_y_axis = str(human_detector_y_axis).strip().lower()
        if self.human_detector_y_axis not in ("left", "right"):
            raise ValueError(
                "human_detector_y_axis must be 'left' or 'right', "
                f"got {human_detector_y_axis!r}"
            )
        self.human_tf_timeout = max(0.0, float(human_tf_timeout))
        self.max_linear_speed = float(max_linear_speed)
        self.max_angular_speed = float(max_angular_speed)
        self.enabled = bool(enabled)
        self._lock = threading.RLock()
        self._odom_pose: Optional[tuple[np.ndarray, float]] = None
        self._odom_frame = ""
        self._odom_stamp = 0.0
        # rosbag play -l preserves message header stamps.  When playback wraps
        # from the end back to the beginning, this stamp moves backwards.  A
        # monotonically increasing replay epoch lets the planner distinguish a
        # bag-loop reset from a real localization discontinuity.
        self._odom_message_stamp = 0.0
        self._odom_replay_epoch = 0
        self._odom_rewind_count = 0
        self._odom_rewind_threshold = 0.25
        self._cloud_xyz = np.zeros((0, 3), dtype=np.float32)
        self._cloud_fields = ["x", "y", "z"]
        self._cloud_stamp = 0.0
        self._cloud_seq = 0
        self._scan_xyz = np.zeros((0, 3), dtype=np.float32)
        self._scan_fields = ["x", "y", "z"]
        self._scan_stamp = 0.0
        self._scan_seq = 0
        self._human_local_xy = np.zeros((0, 2), dtype=np.float32)
        self._human_source_frame = ""
        self._human_message_stamp = rospy.Time(0)
        self._human_message_stamp_seconds = 0.0
        self._human_receive_stamp = 0.0
        self._human_seq = 0
        self._human_world_cache_seq = -1
        self._human_world_cache = np.zeros((0, 2), dtype=np.float32)
        self._human_world_cache_target = ""
        self._human_transform_error: Optional[str] = None
        self._human_transform_mode = "none"
        self._human_rewind_count = 0
        self._last_human_resubscribe_monotonic = 0.0
        # Continuity-first fallback: when exact stamped TF is temporarily
        # unavailable after rosbag -l rewinds, transform detector-local points
        # with the newest odometry pose instead of returning no human.
        self._allow_human_odom_transform_fallback = True

        self._tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)

        self._cmd_pub = rospy.Publisher(cmd_vel_topic, Twist, queue_size=1)
        self._odom_sub = rospy.Subscriber(
            self.odom_topic, Odometry, self._on_odom, queue_size=1
        )
        self._human_sub = rospy.Subscriber(
            self.human_detections_topic,
            PoseArray,
            self._on_human_detections,
            queue_size=1,
        )
        self._scan_sub = rospy.Subscriber(
            self.laser_scan_topic,
            LaserScan,
            self._on_scan,
            queue_size=1,
        )
        self._cloud_sub = None
        self._cloud_cls = None
        # Resolve the Livox topic without blocking startup.  LaserScan is the
        # default source, so the program must still start when /livox/lidar is
        # temporarily absent.  Pressing C retries point-cloud subscription.
        self._ensure_pointcloud_subscription(blocking=False)
        rospy.on_shutdown(self.stop)
        cloud_type = (
            self._cloud_cls._type if self._cloud_cls is not None else "unresolved"
        )
        print(
            f"ROS input: odom={self.odom_topic}, "
            f"scan={self.laser_scan_topic} (sensor_msgs/LaserScan), "
            f"cloud={self.pointcloud_topic} ({cloud_type}), "
            f"active={self.range_source}, cmd={cmd_vel_topic}, "
            f"human={self.human_detections_topic} (geometry_msgs/PoseArray), "
            f"motion={'ENABLED' if self.enabled else 'DISABLED'}"
        )

    def _ensure_pointcloud_subscription(self, *, blocking: bool = False) -> bool:
        if self._cloud_sub is not None:
            return True
        cloud_cls, resolved_topic, _ = rostopic.get_topic_class(
            self.pointcloud_topic,
            blocking=bool(blocking),
        )
        if cloud_cls is None:
            return False
        self.pointcloud_topic = str(resolved_topic or self.pointcloud_topic)
        self._cloud_cls = cloud_cls
        self._cloud_sub = rospy.Subscriber(
            self.pointcloud_topic,
            cloud_cls,
            self._on_cloud,
            queue_size=1,
            buff_size=64 * 1024 * 1024,
        )
        print(
            f"[range input] subscribed point cloud: "
            f"{self.pointcloud_topic} ({cloud_cls._type})"
        )
        return True

    def set_range_source(self, source: str) -> None:
        source = normalize_range_source(source)
        if source == "point_cloud" and not self._ensure_pointcloud_subscription(
            blocking=False
        ):
            raise RuntimeError(
                f"Point-cloud topic is unavailable: {self.pointcloud_topic}"
            )

        now = float(rospy.get_time())
        with self._lock:
            if source == "laser_scan":
                seq = int(self._scan_seq)
                age = now - self._scan_stamp
                topic = self.laser_scan_topic
            else:
                seq = int(self._cloud_seq)
                age = now - self._cloud_stamp
                topic = self.pointcloud_topic
        if seq <= 0 or age > self.pointcloud_timeout:
            raise RuntimeError(
                f"Range source not ready: source={source}, topic={topic}, "
                f"age={age:.3f}s, seq={seq}"
            )
        self.range_source = source

    def _clear_tf_buffer_for_replay(self, reason: str) -> None:
        """Clear future-stamped TF data when rosbag playback rewinds.

        With ``rosbag play -l``, the next loop reuses earlier message stamps.
        A tf2 buffer that still contains the previous loop endpoint may reject
        the new transforms as old data. Clearing the buffer lets the new loop's
        transforms populate it immediately.
        """
        try:
            clear_fn = getattr(self._tf_buffer, "clear", None)
            if callable(clear_fn):
                clear_fn()
            else:
                # Compatibility fallback for tf2 versions without clear().
                self._tf_buffer = tf2_ros.Buffer(
                    cache_time=rospy.Duration(10.0)
                )
                self._tf_listener = tf2_ros.TransformListener(
                    self._tf_buffer
                )
        except Exception as exc:
            rospy.logwarn_throttle(
                1.0,
                "[rosbag loop] failed to clear TF buffer: "
                f"{type(exc).__name__}: {exc}",
            )
        with self._lock:
            self._human_world_cache_seq = -1
            self._human_world_cache = np.zeros((0, 2), dtype=np.float32)
            self._human_world_cache_target = ""
            self._human_transform_error = None
            self._human_transform_mode = "tf_reacquire"
        rospy.loginfo(
            f"[rosbag loop] TF buffer cleared for replay rewind: {reason}"
        )

    def _ensure_human_subscription_alive(self, stale_after: float) -> None:
        """Re-register the detector subscriber after a dropped TCPROS link."""
        stale_after = max(0.5, float(stale_after))
        now_ros = float(rospy.get_time())
        now_mono = time.monotonic()
        with self._lock:
            seq = int(self._human_seq)
            age = (
                now_ros - self._human_receive_stamp
                if seq > 0
                else float("inf")
            )
        if seq > 0 and age <= stale_after:
            return
        if now_mono - self._last_human_resubscribe_monotonic < 1.0:
            return
        self._last_human_resubscribe_monotonic = now_mono
        try:
            old_sub = self._human_sub
            if old_sub is not None:
                old_sub.unregister()
            self._human_sub = rospy.Subscriber(
                self.human_detections_topic,
                PoseArray,
                self._on_human_detections,
                queue_size=1,
            )
            rospy.logwarn(
                "[human detector] re-registered subscriber after stale link: "
                f"topic={self.human_detections_topic}, age={age:.3f}s"
            )
        except Exception as exc:
            rospy.logwarn_throttle(
                1.0,
                "[human detector] subscriber re-registration failed: "
                f"{type(exc).__name__}: {exc}",
            )

    @staticmethod
    def _stamp_seconds(message) -> float:
        header = getattr(message, "header", None)
        stamp = getattr(header, "stamp", None)
        if stamp is not None and hasattr(stamp, "to_sec"):
            value = float(stamp.to_sec())
            if value > 0.0:
                return value
        return float(rospy.get_time())

    def _on_odom(self, message: Odometry) -> None:
        pose = message.pose.pose
        q = pose.orientation
        yaw = float(
            np.arctan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z),
            )
        )
        position = np.array([pose.position.x, pose.position.y], dtype=np.float32)
        frame_id = str(getattr(message.header, "frame_id", "") or "").strip()
        message_stamp = self._stamp_seconds(message)
        rewind_detected = False
        previous_stamp = 0.0
        replay_epoch = 0
        with self._lock:
            previous_stamp = float(self._odom_message_stamp)
            if (
                previous_stamp > 0.0
                and message_stamp
                < previous_stamp - float(self._odom_rewind_threshold)
            ):
                self._odom_replay_epoch += 1
                self._odom_rewind_count += 1
                rewind_detected = True
            self._odom_message_stamp = float(message_stamp)
            replay_epoch = int(self._odom_replay_epoch)
            self._odom_pose = (position, yaw)
            if frame_id:
                self._odom_frame = frame_id
            self._odom_stamp = float(message_stamp)
        if rewind_detected:
            self._clear_tf_buffer_for_replay(
                "odometry stamp "
                f"{previous_stamp:.3f}->{message_stamp:.3f}"
            )
            rospy.loginfo(
                "[rosbag loop] odometry timestamp rewound: "
                f"{previous_stamp:.3f} -> {message_stamp:.3f}; "
                f"replay_epoch={replay_epoch}"
            )

    def _on_human_detections(self, message: PoseArray) -> None:
        """Cache detector outputs without blocking the ROS callback on TF."""
        local_xy = np.asarray(
            [
                (float(pose.position.x), float(pose.position.y))
                for pose in message.poses
            ],
            dtype=np.float32,
        ).reshape(-1, 2)
        if local_xy.size > 0:
            local_xy = local_xy[np.isfinite(local_xy).all(axis=1)]

        header = getattr(message, "header", None)
        frame_id = str(getattr(header, "frame_id", "") or "").strip()
        stamp = getattr(header, "stamp", None)
        if stamp is None or not hasattr(stamp, "to_sec"):
            stamp = rospy.Time(0)
        stamp_seconds = float(stamp.to_sec())

        with self._lock:
            previous_stamp = float(self._human_message_stamp_seconds)
            human_rewind = bool(
                previous_stamp > 0.0
                and stamp_seconds > 0.0
                and stamp_seconds
                < previous_stamp - float(self._odom_rewind_threshold)
            )
            self._human_message_stamp_seconds = float(stamp_seconds)
            if human_rewind:
                self._human_rewind_count += 1
            self._human_local_xy = local_xy
            self._human_source_frame = frame_id
            self._human_message_stamp = stamp
            self._human_receive_stamp = float(rospy.get_time())
            self._human_seq += 1
            self._human_world_cache_seq = -1
            self._human_transform_error = None
            self._human_transform_mode = "pending"

        if human_rewind:
            self._clear_tf_buffer_for_replay(
                "human detection stamp "
                f"{previous_stamp:.3f}->{stamp_seconds:.3f}"
            )

    @staticmethod
    def _rotation_matrix_from_quaternion(quaternion) -> np.ndarray:
        """Return the 3-D rotation matrix for a ROS quaternion."""
        x = float(quaternion.x)
        y = float(quaternion.y)
        z = float(quaternion.z)
        w = float(quaternion.w)
        norm = float(np.sqrt(x * x + y * y + z * z + w * w))
        if norm <= 1e-12:
            return np.eye(3, dtype=np.float64)
        x /= norm
        y /= norm
        z /= norm
        w /= norm
        return np.array(
            [
                [
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y - z * w),
                    2.0 * (x * z + y * w),
                ],
                [
                    2.0 * (x * y + z * w),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z - x * w),
                ],
                [
                    2.0 * (x * z - y * w),
                    2.0 * (y * z + x * w),
                    1.0 - 2.0 * (x * x + y * y),
                ],
            ],
            dtype=np.float64,
        )

    def _human_world_from_current_odom(
        self,
        local_ros: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Approximate local detector points in odom using the current pose.

        This continuity fallback assumes the detector frame is close to the
        robot base frame. Exact stamped TF remains the preferred path.
        """
        with self._lock:
            odom_pose = self._odom_pose
        if odom_pose is None:
            return None
        robot_position, robot_yaw = odom_pose
        c = float(np.cos(robot_yaw))
        s = float(np.sin(robot_yaw))
        local_xy = np.asarray(local_ros, dtype=np.float64)[:, :2]
        world_xy = np.empty_like(local_xy)
        world_xy[:, 0] = (
            float(robot_position[0])
            + c * local_xy[:, 0]
            - s * local_xy[:, 1]
        )
        world_xy[:, 1] = (
            float(robot_position[1])
            + s * local_xy[:, 0]
            + c * local_xy[:, 1]
        )
        return world_xy.astype(np.float32)

    def _lookup_human_transform(self, target_frame: str, source_frame: str, stamp):
        timeout = rospy.Duration(self.human_tf_timeout)
        stamp_seconds = (
            float(stamp.to_sec())
            if stamp is not None and hasattr(stamp, "to_sec")
            else 0.0
        )
        if stamp_seconds > 0.0:
            # A stamped detection must use the TF from the same sensor frame.
            # Never fall back to Time(0): after ``rosbag play -l`` rewinds,
            # the latest TF can still be the previous loop's endpoint and
            # would place a new detection at that stale world position.
            return self._tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                stamp,
                timeout,
            )
        return self._tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            rospy.Time(0),
            timeout,
        )

    def human_detections_world(
        self,
    ) -> Optional[tuple[np.ndarray, int, float]]:
        """Return fresh detector candidates expressed in the odometry frame.

        The supplied DR-SPAAM publisher documents x-forward/y-right coordinates.
        ROS frames use y-left, so y is negated before TF when configured as
        ``human_detector_y_axis='right'``.
        """
        now = float(rospy.get_time())
        self._ensure_human_subscription_alive(
            max(1.0, 2.0 * self.human_detection_timeout)
        )
        with self._lock:
            seq = int(self._human_seq)
            receive_stamp = float(self._human_receive_stamp)
            age = now - receive_stamp
            local_xy = self._human_local_xy.copy()
            source_frame = self._human_source_frame or self.human_detector_frame
            target_frame = self.human_world_frame or self._odom_frame
            message_stamp = self._human_message_stamp
            if (
                seq > 0
                and seq == self._human_world_cache_seq
                and target_frame == self._human_world_cache_target
                and age <= self.human_detection_timeout
            ):
                return (
                    self._human_world_cache.copy(),
                    seq,
                    receive_stamp,
                )

        if seq <= 0 or age > self.human_detection_timeout:
            return None
        if local_xy.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.float32), seq, receive_stamp
        if not source_frame:
            error = (
                "human PoseArray has an empty frame_id; set "
                "--human-detector-frame"
            )
            with self._lock:
                self._human_transform_error = error
            rospy.logwarn_throttle(2.0, f"[human detector] {error}")
            return None
        if not target_frame:
            error = (
                "odometry frame is unknown; set --human-world-frame or "
                "publish Odometry.header.frame_id"
            )
            with self._lock:
                self._human_transform_error = error
            rospy.logwarn_throttle(2.0, f"[human detector] {error}")
            return None

        local_ros = np.zeros((len(local_xy), 3), dtype=np.float64)
        local_ros[:, 0] = local_xy[:, 0]
        local_ros[:, 1] = local_xy[:, 1]
        if self.human_detector_y_axis == "right":
            local_ros[:, 1] *= -1.0

        transform_error = None
        transform_mode = "tf_exact"
        try:
            if source_frame == target_frame:
                world_xy = local_ros[:, :2]
                transform_mode = "identity"
            else:
                transform = self._lookup_human_transform(
                    target_frame,
                    source_frame,
                    message_stamp,
                )
                rotation = self._rotation_matrix_from_quaternion(
                    transform.transform.rotation
                )
                translation = np.array(
                    [
                        transform.transform.translation.x,
                        transform.transform.translation.y,
                        transform.transform.translation.z,
                    ],
                    dtype=np.float64,
                )
                world_xyz = local_ros @ rotation.T + translation[None, :]
                world_xy = world_xyz[:, :2]
        except Exception as exc:
            transform_error = (
                f"TF {source_frame!r}->{target_frame!r} unavailable: "
                f"{type(exc).__name__}: {exc}"
            )
            world_xy = None
            if self._allow_human_odom_transform_fallback:
                world_xy = self._human_world_from_current_odom(local_ros)
            if world_xy is None:
                with self._lock:
                    self._human_transform_error = transform_error
                    self._human_transform_mode = "failed"
                rospy.logwarn_throttle(
                    2.0, f"[human detector] {transform_error}"
                )
                return None
            transform_mode = "odom_pose_fallback"
            rospy.logwarn_throttle(
                1.0,
                "[human detector] exact stamped TF unavailable after replay "
                f"rewind; using odometry-pose fallback: {transform_error}",
            )

        world_xy = np.asarray(world_xy, dtype=np.float32)
        with self._lock:
            if seq == self._human_seq:
                self._human_world_cache_seq = seq
                self._human_world_cache = world_xy.copy()
                self._human_world_cache_target = target_frame
                self._human_transform_error = transform_error
                self._human_transform_mode = transform_mode
        return world_xy, seq, receive_stamp

    def human_detection_status(self) -> dict:
        now = float(rospy.get_time())
        with self._lock:
            age = (
                now - self._human_receive_stamp
                if self._human_seq > 0
                else float("inf")
            )
            message_stamp = (
                float(self._human_message_stamp.to_sec())
                if self._human_message_stamp is not None
                and hasattr(self._human_message_stamp, "to_sec")
                else 0.0
            )
            return {
                "seq": int(self._human_seq),
                "count": int(len(self._human_local_xy)),
                "age": float(age),
                "message_stamp": float(message_stamp),
                "receive_stamp": float(self._human_receive_stamp),
                "source_frame": (
                    self._human_source_frame or self.human_detector_frame
                ),
                "target_frame": self.human_world_frame or self._odom_frame,
                "world_cache_seq": int(self._human_world_cache_seq),
                "transform_error": self._human_transform_error,
                "transform_mode": str(self._human_transform_mode),
                "human_rewind_count": int(self._human_rewind_count),
            }

    @staticmethod
    def _pointcloud2_to_xyz(message: PointCloud2) -> np.ndarray:
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
        if values.size == 0:
            return np.zeros((0, 3), dtype=np.float32)
        return values.reshape(-1, 3)

    @staticmethod
    def _livox_custom_to_xyz(message) -> np.ndarray:
        points = getattr(message, "points", ())
        if not points:
            return np.zeros((0, 3), dtype=np.float32)
        xyz = np.fromiter(
            (value for point in points for value in (point.x, point.y, point.z)),
            dtype=np.float32,
            count=3 * len(points),
        )
        return xyz.reshape(-1, 3)

    def _on_scan(self, message: LaserScan) -> None:
        ranges = np.asarray(message.ranges, dtype=np.float32)
        if ranges.size == 0:
            xyz = np.zeros((0, 3), dtype=np.float32)
        else:
            angles = (
                float(message.angle_min)
                + np.arange(ranges.size, dtype=np.float32)
                * float(message.angle_increment)
            )
            keep = np.isfinite(ranges)
            if np.isfinite(float(message.range_min)):
                keep &= ranges >= float(message.range_min)
            if np.isfinite(float(message.range_max)) and float(message.range_max) > 0.0:
                keep &= ranges <= float(message.range_max)
            valid_ranges = ranges[keep]
            valid_angles = angles[keep]
            xyz = np.column_stack(
                (
                    valid_ranges * np.cos(valid_angles),
                    valid_ranges * np.sin(valid_angles),
                    np.zeros_like(valid_ranges),
                )
            ).astype(np.float32, copy=False)
        with self._lock:
            self._scan_xyz = xyz
            self._scan_stamp = self._stamp_seconds(message)
            self._scan_seq += 1

    def _on_cloud(self, message) -> None:
        message_type = str(getattr(message, "_type", ""))
        if message_type == "sensor_msgs/PointCloud2":
            xyz = self._pointcloud2_to_xyz(message)
        elif message_type.endswith("/CustomMsg"):
            xyz = self._livox_custom_to_xyz(message)
        else:
            rospy.logerr_throttle(5.0, f"Unsupported Livox message type: {message_type}")
            return
        finite = np.isfinite(xyz).all(axis=1)
        with self._lock:
            self._cloud_xyz = xyz[finite]
            self._cloud_stamp = self._stamp_seconds(message)
            self._cloud_seq += 1

    def wait_until_ready(self, timeout: float) -> None:
        deadline = time.monotonic() + float(timeout)
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if self.range_source == "point_cloud":
                self._ensure_pointcloud_subscription(blocking=False)
            with self._lock:
                range_ready = (
                    self._scan_seq > 0
                    if self.range_source == "laser_scan"
                    else self._cloud_seq > 0
                )
                ready = self._odom_pose is not None and range_ready
            if ready:
                return
            if time.monotonic() >= deadline:
                active_topic = (
                    self.laser_scan_topic
                    if self.range_source == "laser_scan"
                    else self.pointcloud_topic
                )
                raise TimeoutError(
                    f"Timed out waiting for {self.odom_topic} and "
                    f"{active_topic} ({self.range_source})"
                )
            rate.sleep()

    def robot_pose(self) -> tuple[np.ndarray, float]:
        with self._lock:
            if self._odom_pose is None:
                raise RuntimeError("No odometry received")
            age = float(rospy.get_time()) - self._odom_stamp
            # if age > self.odom_timeout:
            #     raise RuntimeError(f"Stale odometry: age={age:.3f}s")
            return self._odom_pose[0].copy(), float(self._odom_pose[1])

    def odom_replay_status(self) -> dict:
        """Return odometry timestamp-rewind metadata for rosbag loop handling."""
        with self._lock:
            return {
                "message_stamp": float(self._odom_message_stamp),
                "replay_epoch": int(self._odom_replay_epoch),
                "rewind_count": int(self._odom_rewind_count),
            }

    def assert_fresh(self) -> None:
        """Fail closed when odometry or the selected range source stops updating."""
        now = float(rospy.get_time())
        with self._lock:
            odom_age = now - self._odom_stamp
            odom_ready = self._odom_pose is not None
            if self.range_source == "laser_scan":
                range_age = now - self._scan_stamp
                range_ready = self._scan_seq > 0
                range_seq = int(self._scan_seq)
                label = "Laser-scan"
            else:
                range_age = now - self._cloud_stamp
                range_ready = self._cloud_seq > 0
                range_seq = int(self._cloud_seq)
                label = "Point-cloud"
        # if not odom_ready or odom_age > self.odom_timeout:
        #     raise RuntimeError(f"Odometry watchdog expired: age={odom_age:.3f}s")
        # if not range_ready or range_age > self.pointcloud_timeout:
        #     raise RuntimeError(
        #         f"{label} watchdog expired: age={range_age:.3f}s, seq={range_seq}"
        #     )

    def pointcloud(self) -> tuple[np.ndarray, list[str], int]:
        """Return the selected range source as local XYZ points."""
        now = float(rospy.get_time())
        with self._lock:
            if self.range_source == "laser_scan":
                age = now - self._scan_stamp
                seq = int(self._scan_seq)
                # if seq == 0 or age > self.pointcloud_timeout:
                #     raise RuntimeError(
                #         f"Stale LaserScan: age={age:.3f}s, seq={seq}"
                #     )
                return self._scan_xyz.copy(), list(self._scan_fields), seq

            age = now - self._cloud_stamp
            seq = int(self._cloud_seq)
            if seq == 0 or age > self.pointcloud_timeout:
                raise RuntimeError(
                    f"Stale Livox point cloud: age={age:.3f}s, seq={seq}"
                )
            return self._cloud_xyz.copy(), list(self._cloud_fields), seq

    def range_source_status(self) -> dict:
        now = float(rospy.get_time())
        with self._lock:
            if self.range_source == "laser_scan":
                return {
                    "source": "laser_scan",
                    "topic": self.laser_scan_topic,
                    "seq": int(self._scan_seq),
                    "age": float(now - self._scan_stamp),
                    "count": int(len(self._scan_xyz)),
                }
            return {
                "source": "point_cloud",
                "topic": self.pointcloud_topic,
                "seq": int(self._cloud_seq),
                "age": float(now - self._cloud_stamp),
                "count": int(len(self._cloud_xyz)),
            }

    def publish_control(self, forward: float, turn: float) -> None:
        if not self.enabled:
            self.stop()
            return
        command = Twist()
        command.linear.x = float(np.clip(forward, -1.0, 1.0)) * self.max_linear_speed
        command.angular.z = float(np.clip(turn, -1.0, 1.0)) * self.max_angular_speed
        self._cmd_pub.publish(command)

    def stop(self) -> None:
        try:
            self._cmd_pub.publish(Twist())
        except Exception:
            pass


def _resolve_class(dotted_path: str):
    module_name, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def load_workspace_from_checkpoint(checkpoint_path: Path):
    """
    Load a diffusion_policy workspace from a checkpoint by reading `cfg._target_`.
    This supports both UNet and Transformer workspaces (and any future workspace types)
    without hard-coding the class.
    """
    payload = torch.load(checkpoint_path.open("rb"), pickle_module=dill)
    cfg = payload.get("cfg")

    target = None
    if cfg is not None:
        try:
            target = cfg.get("_target_")
        except Exception:
            target = None
        if not isinstance(target, str) or not target:
            target = getattr(cfg, "_target_", None)

    if not isinstance(target, str) or not target:
        raise ValueError(f"Checkpoint missing cfg._target_: {checkpoint_path}")

    workspace_cls = _resolve_class(target)
    workspace = workspace_cls(cfg)
    # For inference we don't need optimizer state; skip to reduce load time/memory.
    workspace.load_payload(payload, exclude_keys=("optimizer",))
    return workspace


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def filter_points_in_robot_rear_sector(
    points_world: np.ndarray,
    robot_position: np.ndarray,
    robot_heading: float,
    radius: float,
    aperture_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep world-frame points in the robot's rear-facing circular sector.

    ``aperture_deg`` is the total sector opening. For example, 45 degrees
    means +/-22.5 degrees around the direction opposite ``robot_heading``.
    The second return value contains the retained indices in the input array.
    """
    points = np.asarray(points_world, dtype=np.float32)
    if points.size == 0:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    points = points.reshape(-1, 2)
    robot_position = np.asarray(robot_position, dtype=np.float32).reshape(2)

    radius = float(radius)
    aperture_deg = float(aperture_deg)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError(f"rear-sector radius must be > 0, got {radius!r}")
    if (
        not np.isfinite(aperture_deg)
        or aperture_deg <= 0.0
        or aperture_deg > 360.0
    ):
        raise ValueError(
            "rear-sector aperture must be in (0, 360] degrees, "
            f"got {aperture_deg!r}"
        )

    relative = points - robot_position[None, :]
    distances = np.linalg.norm(relative, axis=1)
    point_angles = np.arctan2(relative[:, 1], relative[:, 0])
    rear_heading = wrap_angle(float(robot_heading) + np.pi)
    angular_error = np.abs(wrap_angle(point_angles - rear_heading))
    half_aperture = 0.5 * np.deg2rad(aperture_deg)

    mask = (
        np.isfinite(points).all(axis=1)
        & (distances > 1e-6)
        & (distances <= radius)
        & (angular_error <= half_aperture + 1e-9)
    )
    retained_indices = np.flatnonzero(mask).astype(np.int64)
    return points[retained_indices].copy(), retained_indices


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # return torch.device("cpu")
    return torch.device(device)


def resolve_default_checkpoint() -> Path:
    outputs_dir = PROJECT_ROOT / "diffusion_policy" / "data" / "outputs"
    fallback = Path(
        "/home/yyf/IROS2026/diffusion_policy/data/outputs/2026.01.21/14.14.46_train_diffusion_unet_lowdim_guide_guide_lowdim/checkpoints/epoch=0090-test_mean_score=0.630.ckpt"
    )
    if not outputs_dir.exists():
        return fallback

    score_pattern = re.compile(r"test_mean_score=([0-9.]+)\.ckpt$")
    mid360_scored: list[tuple[float, float, Path]] = []
    mid360_latest: list[tuple[float, Path]] = []

    for ckpt in outputs_dir.rglob("*.ckpt"):
        path_str = str(ckpt)
        if "guide_mid360" not in path_str:
            continue
        try:
            mtime = ckpt.stat().st_mtime
        except OSError:
            continue
        match = score_pattern.search(ckpt.name)
        if match is not None:
            try:
                score = float(match.group(1))
            except ValueError:
                score = float("-inf")
            mid360_scored.append((score, mtime, ckpt))
        elif ckpt.name == "latest.ckpt":
            mid360_latest.append((mtime, ckpt))

    if mid360_scored:
        mid360_scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return mid360_scored[0][2]
    if mid360_latest:
        mid360_latest.sort(key=lambda item: item[0], reverse=True)
        return mid360_latest[0][1]
    return fallback


def normalize_safety_mode(mode: Optional[str]) -> str:
    mode = str(mode or "off").lower()
    aliases = {
        "off": "off",
        "none": "off",
        "diffusion": "off",
        "qp": "robot_qp",
        "robot": "robot_qp",
        "robot_qp": "robot_qp",
        "human_robot": "human_robot_qp",
        "human_robot_qp": "human_robot_qp",
        "human+robot": "human_robot_qp",
    }
    if mode not in aliases:
        raise ValueError(
            f"Unsupported safety_mode={mode!r} (expected off, robot_qp, or human_robot_qp)"
        )
    return aliases[mode]


def normalize_pointcloud_mode(mode: Optional[str]) -> str:
    mode = str(mode or "live").lower()
    aliases = {
        "auto": "live",
        "live": "live",
        "livox": "live",
        "ros": "live",
        "off": "off",
        "none": "off",
    }
    if mode not in aliases:
        raise ValueError(
            f"Unsupported pointcloud_mode={mode!r} "
            "(expected live or off)"
        )
    return aliases[mode]


def normalize_range_source(source: Optional[str]) -> str:
    source = str(source or "laser_scan").strip().lower()
    aliases = {
        "scan": "laser_scan",
        "laser": "laser_scan",
        "laser_scan": "laser_scan",
        "laserscan": "laser_scan",
        "point": "point_cloud",
        "cloud": "point_cloud",
        "pointcloud": "point_cloud",
        "point_cloud": "point_cloud",
        "livox": "point_cloud",
    }
    if source not in aliases:
        raise ValueError(
            f"Unsupported range_source={source!r} "
            "(expected laser_scan or point_cloud)"
        )
    return aliases[source]


def normalize_human_source(source: Optional[str]) -> str:
    source = str(source or "sim").strip().lower()
    aliases = {
        "sim": "sim",
        "simulation": "sim",
        "physics": "sim",
        "detector": "detector",
        "real": "detector",
        "ros": "detector",
        "topic": "detector",
    }
    if source not in aliases:
        raise ValueError(
            f"Unsupported human_source={source!r} "
            "(expected sim or detector)"
        )
    return aliases[source]


class ConstantVelocityKalman2D:
    """Small dependency-free Kalman filter for a 2-D pedestrian track.

    State order is ``[x, y, vx, vy]``.  The process model assumes constant
    velocity with white acceleration noise.  Detector observations measure
    position only.  A weak, high-noise simulation pseudo-measurement can be
    fused without overriding the real detector.
    """

    def __init__(
        self,
        *,
        process_accel_std: float = 1.5,
        measurement_std: float = 0.18,
        max_speed: float = 3.0,
    ) -> None:
        self.process_accel_std = max(1e-4, float(process_accel_std))
        self.measurement_std = max(1e-4, float(measurement_std))
        self.max_speed = max(0.0, float(max_speed))
        self._x = np.zeros((4,), dtype=np.float64)
        self._P = np.eye(4, dtype=np.float64)
        self._stamp = 0.0
        self._initialized = False

    @property
    def initialized(self) -> bool:
        return bool(self._initialized)

    @property
    def position(self) -> np.ndarray:
        return self._x[:2].astype(np.float32, copy=True)

    @property
    def velocity(self) -> np.ndarray:
        return self._x[2:].astype(np.float32, copy=True)

    @property
    def stamp(self) -> float:
        return float(self._stamp)

    @property
    def state(self) -> np.ndarray:
        """Return [x, y, vx, vy] for diagnostics without exposing internals."""
        return self._x.astype(np.float64, copy=True)

    @property
    def covariance_diag(self) -> np.ndarray:
        """Return the covariance diagonal for compact detector logs."""
        return np.diag(self._P).astype(np.float64, copy=True)

    def reset(self) -> None:
        self._x.fill(0.0)
        self._P = np.eye(4, dtype=np.float64)
        self._stamp = 0.0
        self._initialized = False

    def initialize(
        self,
        position: np.ndarray,
        timestamp: float,
        velocity: Optional[np.ndarray] = None,
    ) -> None:
        position = np.asarray(position, dtype=np.float64).reshape(2)
        if velocity is None:
            velocity = np.zeros((2,), dtype=np.float64)
        velocity = np.asarray(velocity, dtype=np.float64).reshape(2)
        self._x = np.concatenate([position, velocity], axis=0)
        pos_var = max(self.measurement_std, 0.15) ** 2
        self._P = np.diag([pos_var, pos_var, 1.0, 1.0]).astype(np.float64)
        self._stamp = float(timestamp)
        self._initialized = True
        self._clip_velocity()

    def _clip_velocity(self) -> None:
        speed = float(np.linalg.norm(self._x[2:]))
        if self.max_speed > 0.0 and speed > self.max_speed:
            self._x[2:] *= self.max_speed / max(speed, 1e-9)

    def predict(self, timestamp: float) -> tuple[np.ndarray, np.ndarray]:
        if not self._initialized:
            raise RuntimeError("Kalman filter is not initialized")
        timestamp = float(timestamp)
        dt = timestamp - self._stamp
        if not np.isfinite(dt) or dt <= 1e-6:
            return self.position, self.velocity

        # Bound one prediction interval so a clock jump cannot launch the track
        # across the map. Repeated normal calls still advance it continuously.
        dt_model = min(dt, 1.0)
        dt2 = dt_model * dt_model
        dt3 = dt2 * dt_model
        dt4 = dt2 * dt2
        F = np.array(
            [
                [1.0, 0.0, dt_model, 0.0],
                [0.0, 1.0, 0.0, dt_model],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        q = self.process_accel_std * self.process_accel_std
        Q = q * np.array(
            [
                [0.25 * dt4, 0.0, 0.5 * dt3, 0.0],
                [0.0, 0.25 * dt4, 0.0, 0.5 * dt3],
                [0.5 * dt3, 0.0, dt2, 0.0],
                [0.0, 0.5 * dt3, 0.0, dt2],
            ],
            dtype=np.float64,
        )
        self._x = F @ self._x
        self._P = F @ self._P @ F.T + Q
        self._P = 0.5 * (self._P + self._P.T)
        self._stamp = timestamp
        self._clip_velocity()
        return self.position, self.velocity

    def innovation(
        self,
        measurement: np.ndarray,
        measurement_std: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        if not self._initialized:
            raise RuntimeError("Kalman filter is not initialized")
        z = np.asarray(measurement, dtype=np.float64).reshape(2)
        std = self.measurement_std if measurement_std is None else max(
            1e-4, float(measurement_std)
        )
        H = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        residual = z - H @ self._x
        S = H @ self._P @ H.T + np.eye(2, dtype=np.float64) * (std * std)
        try:
            mahalanobis_sq = float(residual.T @ np.linalg.solve(S, residual))
        except np.linalg.LinAlgError:
            mahalanobis_sq = float("inf")
        return residual, S, mahalanobis_sq

    def correct(
        self,
        measurement: np.ndarray,
        measurement_std: Optional[float] = None,
    ) -> float:
        residual, S, mahalanobis_sq = self.innovation(
            measurement,
            measurement_std=measurement_std,
        )
        std = self.measurement_std if measurement_std is None else max(
            1e-4, float(measurement_std)
        )
        H = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        try:
            K = np.linalg.solve(S, H @ self._P).T
        except np.linalg.LinAlgError:
            return float("inf")
        self._x = self._x + K @ residual
        # Joseph form is slightly more expensive but keeps P positive and
        # symmetric during long real-robot runs.
        I = np.eye(4, dtype=np.float64)
        R = np.eye(2, dtype=np.float64) * (std * std)
        IKH = I - K @ H
        self._P = IKH @ self._P @ IKH.T + K @ R @ K.T
        self._P = 0.5 * (self._P + self._P.T)
        self._clip_velocity()
        return mahalanobis_sq

    def blend_velocity(
        self,
        velocity: np.ndarray,
        gain: float,
    ) -> None:
        """Blend a model-predicted velocity into the KF state.

        The PhysicsEngine provides a useful short-horizon human motion model,
        especially while the detector is missing or produces a large outlier.
        This operation is deliberately bounded and does not overwrite the
        position estimate.
        """
        if not self._initialized:
            return
        velocity = np.asarray(velocity, dtype=np.float64).reshape(2)
        if not np.isfinite(velocity).all():
            return
        gain = float(np.clip(gain, 0.0, 1.0))
        self._x[2:] = (1.0 - gain) * self._x[2:] + gain * velocity
        self._clip_velocity()


class ModelPlanner:
    """Run simulation and control robot with a trained policy."""

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        device: str = "auto",
        use_ema: bool = True,
        action_mode: Optional[str] = None,
        k_lookahead: Optional[int] = None,
        frame_stride: Optional[int] = None,
        path_length: float = 50.0,
        corridor_width: float = 2.1,
        obstacle_radius: float = 0.3,
        leash_length: float = 1.5,
        robot_speed: float = 1.0,
        robot_radius: float = 0.1,
        human_radius: float = 0.1,
        fps: int = 20,
        inference_steps: int = 8,
        turn_gain: float = 1.2,
        curvature_slowdown: bool = True,
        curvature_scale: float = 0.7,
        min_speed_scale: float = 0.25,
        log_path: Optional[Path] = None,
        human_detection_log_path: Optional[Path] = None,
        human_detection_log_interval: int = 1,
        human_detection_log_max_candidates: int = 64,
        human_detection_console: bool = False,
        eval_path: Optional[Path] = None,
        log_interval: int = 1,
        collect_enabled: bool = False,
        visualizer: Optional[Visualizer] = None,
        create_visualizer: bool = True,
        collision_behavior: str = "reset",
        safety_mode: str = "off",
        pointcloud_mode: str = "live",
        odom_topic: str = "/odom",
        pointcloud_topic: str = "/livox/lidar",
        laser_scan_topic: str = "/front/scan",
        range_source: str = "laser_scan",
        human_detections_topic: str = "/dr_spaam_detections",
        human_source: str = "sim",
        cmd_vel_topic: str = "/cmd_vel",
        max_angular_speed: float = 1.0,
        ros_input_timeout: float = 5.0,
        odom_timeout: float = 5.0,
        pointcloud_timeout: float = 5.0,
        human_detection_timeout: float = 1.5,
        human_detector_frame: str = "",
        human_world_frame: str = "",
        human_detector_y_axis: str = "right",
        human_tf_timeout: float = 0.05,
        human_track_max_jump: float = 1.5,
        human_rear_sector_range: float = 3.0,
        human_rear_sector_angle_deg: float = 45.0,
        human_kf_process_accel_std: float = 1.5,
        human_kf_measurement_std: float = 0.18,
        human_kf_gate: float = 11.83,
        human_kf_sim_prior_std: float = 0.75,
        human_kf_sim_prior_max_error: float = 1.25,
        human_rear_prior_calibration: bool = True,
        human_rear_prior_scale: float = 0.60,
        human_rear_prior_min_detect_weight: float = 0.05,
        human_rear_prior_max_detect_weight: float = 0.90,
        human_kf_hold_timeout: float = 1.5,
        human_kf_max_misses: int = 30,
        human_kf_sector_margin_deg: float = 12.0,
        human_kf_range_margin: float = 0.4,
        human_kf_sim_fallback_std: float = 0.30,
        human_kf_sim_velocity_gain: float = 0.60,
        human_sim_max_distance: float = 6.0,
        human_sim_full_loss_timeout: float = 0.0,
        human_continuity_mode: bool = True,
        human_robot_jump_threshold: float = 1.0,
        human_robot_heading_jump_deg: float = 90.0,
        rosbag_loop_mode: bool = True,
        rosbag_loop_origin_radius: float = 0.75,
        lidar_height: float = 0.4,
        enable_motion: bool = False,
        safety_margin: float = 0.02,
        safety_alpha: float = 1.0,
        safety_max_constraints: int = 8,
        safety_influence_distance: float = 1.0,
        safety_path_corridor: float = 0.8,
        safety_point_spacing: float = 0.1,
        debug_preview: bool = False,
        debug_preview_limit: int = 5,
        debug_policy: bool = False,
        debug_qp_log: bool = True,
        interaction_segmentation: bool = True,
        segmentation_path: Path = DEFAULT_SEGMENTATION_PATH,
        segmentation_window: int = 120,
        segmentation_min_samples: int = 12,
    ):
        self.fps = fps
        self.sim_dt = 1.0 / fps
        self.leash_length = leash_length
        self.lidar_height = float(lidar_height)
        self.human_source = normalize_human_source(human_source)
        self.human_track_max_jump = max(0.0, float(human_track_max_jump))
        self.human_rear_sector_range = float(human_rear_sector_range)
        self.human_rear_sector_angle_deg = float(human_rear_sector_angle_deg)
        self.human_kf_process_accel_std = max(
            1e-4, float(human_kf_process_accel_std)
        )
        self.human_kf_measurement_std = max(
            1e-4, float(human_kf_measurement_std)
        )
        self.human_kf_gate = max(0.0, float(human_kf_gate))
        self.human_kf_sim_prior_std = max(
            self.human_kf_measurement_std,
            float(human_kf_sim_prior_std),
        )
        self.human_kf_sim_prior_max_error = max(
            0.0, float(human_kf_sim_prior_max_error)
        )
        self.human_rear_prior_calibration = bool(
            human_rear_prior_calibration
        )
        self.human_rear_prior_scale = max(
            1e-3, float(human_rear_prior_scale)
        )
        self.human_rear_prior_min_detect_weight = float(
            np.clip(human_rear_prior_min_detect_weight, 0.0, 1.0)
        )
        self.human_rear_prior_max_detect_weight = float(
            np.clip(human_rear_prior_max_detect_weight, 0.0, 1.0)
        )
        if (
            self.human_rear_prior_min_detect_weight
            > self.human_rear_prior_max_detect_weight
        ):
            raise ValueError(
                "human_rear_prior_min_detect_weight must be <= "
                "human_rear_prior_max_detect_weight"
            )
        self.human_kf_hold_timeout = max(0.0, float(human_kf_hold_timeout))
        self.human_kf_max_misses = max(0, int(human_kf_max_misses))
        self.human_kf_sector_margin_deg = max(
            0.0, float(human_kf_sector_margin_deg)
        )
        self.human_kf_range_margin = max(0.0, float(human_kf_range_margin))
        self.human_kf_sim_fallback_std = max(
            1e-4, float(human_kf_sim_fallback_std)
        )
        self.human_kf_sim_velocity_gain = float(
            np.clip(human_kf_sim_velocity_gain, 0.0, 1.0)
        )
        self.human_sim_max_distance = max(
            float(self.leash_length) * 1.5,
            float(human_sim_max_distance),
        )
        self.human_sim_full_loss_timeout = max(
            0.0, float(human_sim_full_loss_timeout)
        )
        # Continuity-first mode: detector dropouts, empty PoseArrays, TF
        # delays, rosbag rewinds and detector outliers fall back to the
        # PhysicsEngine/rear-leash prior instead of entering LOST.
        self.human_continuity_mode = bool(human_continuity_mode)
        self.human_robot_jump_threshold = max(
            0.0, float(human_robot_jump_threshold)
        )
        self.human_robot_heading_jump_rad = np.deg2rad(
            max(0.0, float(human_robot_heading_jump_deg))
        )
        self.rosbag_loop_mode = bool(rosbag_loop_mode)
        self.rosbag_loop_origin_radius = max(
            0.0, float(rosbag_loop_origin_radius)
        )
        if (
            not np.isfinite(self.human_rear_sector_range)
            or self.human_rear_sector_range <= 0.0
        ):
            raise ValueError(
                "--human-rear-sector-range must be > 0, "
                f"got {human_rear_sector_range!r}"
            )
        if (
            not np.isfinite(self.human_rear_sector_angle_deg)
            or self.human_rear_sector_angle_deg <= 0.0
            or self.human_rear_sector_angle_deg > 360.0
        ):
            raise ValueError(
                "--human-rear-sector-angle-deg must be in (0, 360], "
                f"got {human_rear_sector_angle_deg!r}"
            )

        # Initialize modules
        self.path_generator = PathGenerator(
            target_length=path_length,
            corridor_width=corridor_width,
            obstacle_radius=obstacle_radius,
        )
        self.visualizer: Optional[Visualizer] = visualizer
        if self.visualizer is None and create_visualizer:
            self.visualizer = Visualizer()

        # Load policy if checkpoint is provided
        self.device = resolve_device(device)
        self.requested_pointcloud_mode = normalize_pointcloud_mode(pointcloud_mode)
        self.range_source = normalize_range_source(range_source)
        self.ros_io = LiveRobotRosInterface(
            odom_topic=odom_topic,
            pointcloud_topic=pointcloud_topic,
            laser_scan_topic=laser_scan_topic,
            range_source=self.range_source,
            human_detections_topic=human_detections_topic,
            cmd_vel_topic=cmd_vel_topic,
            max_linear_speed=robot_speed,
            max_angular_speed=max_angular_speed,
            odom_timeout=odom_timeout,
            pointcloud_timeout=pointcloud_timeout,
            human_detection_timeout=human_detection_timeout,
            human_detector_frame=human_detector_frame,
            human_world_frame=human_world_frame,
            human_detector_y_axis=human_detector_y_axis,
            human_tf_timeout=human_tf_timeout,
            enabled=enable_motion,
        )
        self.ros_io.wait_until_ready(ros_input_timeout)
        self._last_mid360_cloud_seq: Optional[int] = None
        self._mid360_visual_max_points = 4096
        self.workspace = None
        self.policy = None
        if checkpoint_path is not None and checkpoint_path.exists():
            self.workspace = load_workspace_from_checkpoint(checkpoint_path)
            if use_ema and getattr(self.workspace, "ema_model", None) is not None:
                self.policy = self.workspace.ema_model
            else:
                self.policy = self.workspace.model
            self.policy.to(self.device)
            self.policy.eval()
        else:
            # No checkpoint - will use manual control only
            print("Warning: No checkpoint provided. Running in manual control mode only.")
            print("Press 'P' key is disabled. Use arrow keys for manual control.")

        # Load configuration from checkpoint if available, otherwise use defaults
        if self.workspace is not None:
            cfg_action_mode = None
            try:
                cfg_action_mode = self.workspace.cfg.task.dataset.get("action_mode")
            except Exception:
                cfg_action_mode = None
            self.action_mode = action_mode or cfg_action_mode or "forward_heading"

            cfg_robot_frame = None
            try:
                cfg_robot_frame = self.workspace.cfg.task.dataset.get("robot_frame")
            except Exception:
                cfg_robot_frame = None
            self.robot_frame = bool(cfg_robot_frame) if cfg_robot_frame is not None else False
            if self.action_mode == "forward_heading" and not self.robot_frame:
                raise ValueError("action_mode='forward_heading' requires robot_frame=True")

            cfg_robot_state = None
            try:
                cfg_robot_state = self.workspace.cfg.task.dataset.get("robot_state")
            except Exception:
                cfg_robot_state = None
            self.robot_state = str(cfg_robot_state or "zero").lower()

            cfg_k_lookahead = None
            try:
                cfg_k_lookahead = self.workspace.cfg.task.get("k_lookahead")
            except Exception:
                cfg_k_lookahead = None
            if cfg_k_lookahead is None:
                try:
                    cfg_k_lookahead = self.workspace.cfg.task.dataset.get("k_lookahead")
                except Exception:
                    cfg_k_lookahead = None
            if cfg_k_lookahead is None:
                try:
                    cfg_k_lookahead = self.workspace.cfg.task.get("lookahead_stride")
                except Exception:
                    cfg_k_lookahead = None
            if cfg_k_lookahead is None:
                try:
                    cfg_k_lookahead = self.workspace.cfg.task.dataset.get("lookahead_stride")
                except Exception:
                    cfg_k_lookahead = None

            cfg_n_lookahead = None
            try:
                cfg_n_lookahead = self.workspace.cfg.task.get("n_lookahead")
            except Exception:
                cfg_n_lookahead = None
            if cfg_n_lookahead is None:
                try:
                    cfg_n_lookahead = self.workspace.cfg.task.dataset.get("n_lookahead")
                except Exception:
                    cfg_n_lookahead = None

            cfg_n_obstacle_circles = None
            try:
                cfg_n_obstacle_circles = self.workspace.cfg.task.get("n_obstacle_circles")
            except Exception:
                cfg_n_obstacle_circles = None
            if cfg_n_obstacle_circles is None:
                try:
                    cfg_n_obstacle_circles = self.workspace.cfg.task.dataset.get("n_obstacle_circles")
                except Exception:
                    cfg_n_obstacle_circles = None

            cfg_n_obstacle_segments = None
            try:
                cfg_n_obstacle_segments = self.workspace.cfg.task.get("n_obstacle_segments")
            except Exception:
                cfg_n_obstacle_segments = None
            if cfg_n_obstacle_segments is None:
                try:
                    cfg_n_obstacle_segments = self.workspace.cfg.task.dataset.get("n_obstacle_segments")
                except Exception:
                    cfg_n_obstacle_segments = None

            cfg_obstacle_include_radius = None
            try:
                cfg_obstacle_include_radius = self.workspace.cfg.task.dataset.get("obstacle_include_radius")
            except Exception:
                cfg_obstacle_include_radius = None

            cfg_obstacle_include_human_clearance = None
            try:
                cfg_obstacle_include_human_clearance = self.workspace.cfg.task.dataset.get(
                    "obstacle_include_human_clearance"
                )
            except Exception:
                cfg_obstacle_include_human_clearance = None

            cfg_segment_repr = None
            try:
                cfg_segment_repr = self.workspace.cfg.task.dataset.get("segment_repr")
            except Exception:
                cfg_segment_repr = None

            cfg_frame_stride = None
            try:
                cfg_frame_stride = self.workspace.cfg.task.get("frame_stride")
            except Exception:
                cfg_frame_stride = None
            if cfg_frame_stride is None:
                try:
                    cfg_frame_stride = self.workspace.cfg.task.dataset.get("frame_stride")
                except Exception:
                    cfg_frame_stride = None

            cfg_task_name = None
            try:
                cfg_task_name = self.workspace.cfg.task.get("name")
            except Exception:
                cfg_task_name = None

            cfg_dataset_target = None
            try:
                cfg_dataset_target = self.workspace.cfg.task.dataset.get("_target_")
            except Exception:
                cfg_dataset_target = None

            cfg_lidar_num_bins = None
            try:
                cfg_lidar_num_bins = self.workspace.cfg.task.get("lidar_num_bins")
            except Exception:
                cfg_lidar_num_bins = None
            if cfg_lidar_num_bins is None:
                try:
                    cfg_lidar_num_bins = self.workspace.cfg.task.dataset.get("lidar_num_bins")
                except Exception:
                    cfg_lidar_num_bins = None

            def _cfg_dataset_value(key: str, default=None):
                try:
                    value = self.workspace.cfg.task.dataset.get(key)
                except Exception:
                    value = None
                if value is None:
                    return default
                return value

            detected_observation_mode = "mid360" if (
                cfg_lidar_num_bins is not None
                or str(cfg_task_name or "").lower().endswith("mid360")
                or "GuideMid360Dataset" in str(cfg_dataset_target or "")
            ) else "lowdim"
            self.observation_mode = detected_observation_mode
            self.pointcloud_mode = self.requested_pointcloud_mode
            if self.observation_mode == "mid360" and self.pointcloud_mode == "off":
                raise ValueError(
                    "pointcloud_mode='off' is incompatible with a guide_mid360 checkpoint. "
                    "Use pointcloud_mode='live'."
                )

            self.obs_dim = int(self.policy.obs_dim)
            self.action_dim = int(self.policy.action_dim)
            self.n_obs_steps = int(self.policy.n_obs_steps)
            self.n_action_steps = int(self.policy.n_action_steps)
            self.lidar_num_bins = int(cfg_lidar_num_bins or 128)
            self.mid360_obs_config = GuideMid360ObservationConfig(
                num_bins=self.lidar_num_bins,
                min_angle=float(_cfg_dataset_value("lidar_min_angle", -np.pi)),
                max_angle=float(_cfg_dataset_value("lidar_max_angle", np.pi)),
                min_range=float(_cfg_dataset_value("lidar_min_range", 0.2)),
                max_range=float(_cfg_dataset_value("lidar_max_range", 8.0)),
                ground_height=float(_cfg_dataset_value("lidar_ground_height", 0.1)),
                max_height=float(_cfg_dataset_value("lidar_max_height", 2.2)),
                use_world_height=bool(_cfg_dataset_value("lidar_use_world_height", True)),
            )
            self.mid360_simulator = None
            if self.observation_mode == "mid360":
                self.n_obstacle_circles = 0
                self.n_obstacle_segments = 0
                self.obstacle_include_radius = False
                self.obstacle_include_human_clearance = False
                self.segment_repr = "closest_dir"
                self.obstacle_obs_dim = 0
                if cfg_n_lookahead is None:
                    extra_obs = self.obs_dim - 4 - self.lidar_num_bins
                    if extra_obs < 0 or extra_obs % 2 != 0:
                        raise ValueError(
                            f"Unsupported obs_dim={self.obs_dim}, expected 4 + 2*N + {self.lidar_num_bins}"
                        )
                    self.n_lookahead = extra_obs // 2
                else:
                    self.n_lookahead = int(cfg_n_lookahead)
                    expected_obs_dim = 4 + 2 * self.n_lookahead + self.lidar_num_bins
                    if self.obs_dim != expected_obs_dim:
                        extra_obs = self.obs_dim - 4 - self.lidar_num_bins
                        if extra_obs >= 0 and extra_obs % 2 == 0:
                            self.n_lookahead = extra_obs // 2
                            print(
                                f"[warn] obs_dim mismatch (expected {expected_obs_dim}, got {self.obs_dim}); "
                                f"using derived n_lookahead={self.n_lookahead}"
                            )
                        else:
                            raise ValueError(
                                f"Unsupported obs_dim={self.obs_dim}, expected {expected_obs_dim}"
                            )
            else:
                self.n_obstacle_circles = max(0, int(cfg_n_obstacle_circles or 0))
                self.n_obstacle_segments = max(0, int(cfg_n_obstacle_segments or 0))
                self.obstacle_include_radius = (
                    True if cfg_obstacle_include_radius is None else bool(cfg_obstacle_include_radius)
                )
                self.obstacle_include_human_clearance = (
                    False
                    if cfg_obstacle_include_human_clearance is None
                    else bool(cfg_obstacle_include_human_clearance)
                )
                self.segment_repr = str(cfg_segment_repr or "endpoints").lower()
                if self.segment_repr not in ("endpoints", "closest_dir"):
                    raise ValueError(f"Unsupported segment_repr: {self.segment_repr}")
                circle_dim = 3 if self.obstacle_include_radius else 2
                clearance_dim = (
                    (self.n_obstacle_circles + self.n_obstacle_segments)
                    if self.obstacle_include_human_clearance
                    else 0
                )
                self.obstacle_obs_dim = (
                    self.n_obstacle_circles * circle_dim + self.n_obstacle_segments * 4 + clearance_dim
                )
                if cfg_n_lookahead is None:
                    extra_obs = self.obs_dim - 4 - self.obstacle_obs_dim
                    if extra_obs < 0 or extra_obs % 2 != 0:
                        raise ValueError(
                            f"Unsupported obs_dim={self.obs_dim}, expected 4 + 2*N + {self.obstacle_obs_dim}"
                        )
                    self.n_lookahead = extra_obs // 2
                else:
                    self.n_lookahead = int(cfg_n_lookahead)
                    expected_obs_dim = 4 + 2 * self.n_lookahead + self.obstacle_obs_dim
                    if self.obs_dim != expected_obs_dim:
                        extra_obs = self.obs_dim - 4 - self.obstacle_obs_dim
                        if extra_obs >= 0 and extra_obs % 2 == 0:
                            self.n_lookahead = extra_obs // 2
                            print(
                                f"[warn] obs_dim mismatch (expected {expected_obs_dim}, got {self.obs_dim}); "
                                f"using derived n_lookahead={self.n_lookahead}"
                            )
                        else:
                            raise ValueError(
                                f"Unsupported obs_dim={self.obs_dim}, expected {expected_obs_dim}"
                            )
            stride = k_lookahead if k_lookahead is not None else cfg_k_lookahead
            if stride is None:
                stride = 5
            self.lookahead_stride = max(1, int(stride))

            stride = frame_stride if frame_stride is not None else cfg_frame_stride
            if stride is None:
                stride = 1
            self.frame_stride = max(1, int(stride))
        else:
            # Default values when no checkpoint is provided
            self.action_mode = action_mode or "forward_heading"
            self.observation_mode = "lowdim"
            self.pointcloud_mode = self.requested_pointcloud_mode
            self.robot_frame = True
            self.robot_state = "vel"
            self.obs_dim = 44  # Default: 4 (robot+human) + 2*20 (lookahead) + 0 (no obstacles)
            self.action_dim = 2
            self.n_obs_steps = 1
            self.n_action_steps = 24
            self.n_obstacle_circles = 0
            self.n_obstacle_segments = 0
            self.obstacle_include_radius = False
            self.obstacle_include_human_clearance = False
            self.segment_repr = "endpoints"
            self.obstacle_obs_dim = 0
            self.lidar_num_bins = 128
            self.mid360_obs_config = GuideMid360ObservationConfig(
                num_bins=self.lidar_num_bins,
                min_angle=-np.pi,
                max_angle=np.pi,
                min_range=0.2,
                max_range=8.0,
                ground_height=0.1,
                max_height=2.2,
                use_world_height=True,
            )
            self.mid360_simulator = None
            self.n_lookahead = 20
            self.lookahead_stride = k_lookahead if k_lookahead is not None else 5
            self.frame_stride = frame_stride if frame_stride is not None else 1
        self.data_dt = self.sim_dt * self.frame_stride
        self.physics = PhysicsEngine(
            leash_length=leash_length,
            robot_speed=robot_speed,
            dt=self.sim_dt,
            robot_radius=robot_radius,
            human_radius=human_radius,
        )
        self._human_kf = ConstantVelocityKalman2D(
            process_accel_std=self.human_kf_process_accel_std,
            measurement_std=self.human_kf_measurement_std,
            max_speed=3.0,
        )
        self._tracked_human_position: Optional[np.ndarray] = None
        self._tracked_human_velocity = np.zeros((2,), dtype=np.float32)
        self._human_detection_seq = -1
        self._human_detection_receive_stamp = 0.0
        self._human_kf_last_measurement_stamp = 0.0
        self._human_kf_consecutive_misses = 0
        self._human_kf_using_prediction = False
        self._human_kf_last_mahalanobis_sq = float("inf")
        self._human_detector_waiting = False
        self._human_detection_rejected = False
        self._human_tracking_mode = "uninitialized"
        self._human_sim_fallback_start_stamp = 0.0
        self._human_last_failure_reason = ""
        self._last_robot_odom_position: Optional[np.ndarray] = None
        self._last_robot_odom_heading: Optional[float] = None
        self._last_robot_odom_time = 0.0
        self._robot_localization_jump_pending = False
        self._robot_localization_jump_reason = ""
        replay_status = self.ros_io.odom_replay_status()
        self._last_odom_replay_epoch = int(replay_status["replay_epoch"])
        self._rosbag_initial_odom_position: Optional[np.ndarray] = None
        self._rosbag_loop_reset_pending = False
        self._rosbag_loop_reacquire_seq = -1
        self._rosbag_loop_reset_count = 0
        self._rosbag_loop_reset_reason = ""

        self.scorer = None
        self.current_path_data = None
        self.running = True
        self.paused = False
        self.bre = False
        self.bre_toggle_times = (4, 5, 9, 10, 14, 15, 19, 20, 24, 25, 29, 30)  # Seconds at which to toggle BRE on/off for testing
        self.bre_toggle_times = (5, 10, 15, 20, 25, 30)  # Seconds at which to toggle BRE on/off for testing
        self.triggered_bre_toggle_times = set()
        self.bre_timer_start_frame = None
        self.interaction_segmenter = OnlineInteractionSegmenter(
            segmentation_path=segmentation_path,
            enabled=interaction_segmentation,
            window_size=segmentation_window,
            min_samples=segmentation_min_samples,
        )
        self.use_policy = False
        self.collision_pause = False
        self.collision_happened = False
        self.collision_info = None
        collision_behavior = str(collision_behavior or "reset").lower()
        if collision_behavior == "freeze":
            collision_behavior = "pause"
        if collision_behavior not in ("reset", "pause"):
            raise ValueError(
                f"Unsupported collision_behavior={collision_behavior!r} (expected 'reset' or 'pause')"
            )
        self.collision_behavior = collision_behavior

        self.robot_trajectory = []
        self.human_trajectory = []
        self.planned_path = None
        self.nominal_planned_path = None
        self.safe_planned_path = None
        self.lookahead_world = None
        self.current_mid360_points_world = None
        # All filtered Livox points represented as QP circle obstacles:
        # [x_world, y_world, radius].  A 5 cm diameter means a 2.5 cm radius.
        self.current_mid360_point_obstacles = None
        self.mid360_point_obstacle_radius = 0.025
        # A lightweight copy used by SafeFilter. It is rebuilt after each
        # diffusion inference from points close to the raw diffusion path.
        self.current_safety_point_obstacles = None
        self.safety_path_corridor = max(0.0, float(safety_path_corridor))
        self.safety_point_spacing = max(0.0, float(safety_point_spacing))
        self.last_safety_pointcloud_stats = {
            "raw_count": 0,
            "corridor_count": 0,
            "sparse_count": 0,
        }
        self.frame_count = 0
        self.prev_robot_pos = None
        self.data_step_idx = 0
        self.log_fp = None
        self.human_detection_log_fp = None
        self.human_detection_log_path = (
            Path(human_detection_log_path)
            if human_detection_log_path is not None
            else None
        )
        self.human_detection_log_interval = max(
            1, int(human_detection_log_interval)
        )
        self.human_detection_log_max_candidates = max(
            1, int(human_detection_log_max_candidates)
        )
        self.human_detection_console = bool(human_detection_console)
        self._human_detection_log_refresh_idx = 0
        self._human_detection_log_record_idx = 0
        self._human_detection_log_outcomes: dict[str, int] = {}
        self.eval_fp = None
        self.eval_planning_idx = 0
        self.log_interval = max(1, int(log_interval))
        self.collect_enabled = bool(collect_enabled)
        self.recording = False
        self._last_recorded_cloud_seq = None
        self.storage = None
        self.collection_data_dir = FOLLOWDATASET_DIR / "data"
        if self.collect_enabled:
            self.storage = DataStorage(base_dir=str(self.collection_data_dir))
            print(
                f"Planning collection enabled: {type(self.storage).__name__} "
                f"-> {self.collection_data_dir}"
            )

        self.obs_history = deque(maxlen=self.n_obs_steps)
        
        # Performance optimization: cache actions and reduce inference frequency
        self.cached_action_seq = None
        self.cached_nominal_delta_seq = None
        self.cached_safe_delta_seq = None
        self.cached_safety_info_seq = None
        self.cached_interaction_labels_seq = None
        self.cached_uses_stashed_compliance_plan = False
        self.cached_action_idx = 0
        self.stashed_guide_action_seq = None
        self.stashed_guide_cursor = 0
        self.using_stashed_compliance_plan = False
        self.last_stashed_compliance_info = {}
        self.inference_interval = max(1, self.n_action_steps // 2)  # Infer every N frames
        self.frames_since_inference = 0
        self.cached_control = (0.0, 0.0)
        self.current_action = None
        self.current_delta = None
        self.latest_nominal_heading_delta = None
        self.turn_gain = float(turn_gain)
        self.curvature_slowdown = bool(curvature_slowdown)
        self.curvature_scale = float(curvature_scale)
        self.min_speed_scale = float(min_speed_scale)
        self.current_speed_scale = 1.0
        self.safety_mode = normalize_safety_mode(safety_mode)
        self.safety_filter = QPSafetyFilter(
            margin=float(safety_margin),
            alpha=float(safety_alpha),
            max_constraints=int(safety_max_constraints),
            influence_distance=float(safety_influence_distance),
        )
        self.safety_stop_clearance = max(0.0, float(safety_margin) * 0.75)
        self.safety_backoff_scales = (1.0, 0.75, 0.5, 0.25, 0.0)
        self.last_safety_stats = {
            "applied": False,
            "modified_steps": 0,
            "total_steps": 0,
            "mean_shift": 0.0,
            "constraint_count": 0,
            "min_clearance": float("inf"),
            "input_point_obstacle_count": 0,
            "input_segment_obstacle_count": 0,
        }
        self.last_compliance_stats = {
            "applied": False,
            "safety_applied": False,
            "modified_steps": 0,
            "safety_modified_steps": 0,
            "total_steps": 0,
            "mean_shift": 0.0,
            "mean_action_shift": 0.0,
            "constraint_count": 0,
            "min_clearance": float("inf"),
        }
        self.episode_safety_stats = {
            "modified_steps": 0,
            "total_steps": 0,
            "total_shift": 0.0,
            "constraint_count": 0,
            "min_clearance": float("inf"),
        }
        if self.safety_mode != "off" and self.action_mode not in ("forward_heading", "delta", "velocity"):
            print(
                f"[warn] safety_mode={self.safety_mode} is only supported for "
                f"forward_heading/delta/velocity; disabling for action_mode={self.action_mode}"
            )
            self.safety_mode = "off"
        self.debug_preview = bool(debug_preview)
        self.debug_preview_limit = max(1, int(debug_preview_limit))
        self.debug_policy = bool(debug_policy)
        self.debug_qp_log = bool(debug_qp_log)
        self.debug_inference_count = 0

        effective_inference_steps = int(inference_steps)
        if self.observation_mode == "mid360":
            effective_inference_steps = max(effective_inference_steps, 100)
        self.inference_steps = effective_inference_steps
        
        # Reduce inference steps for faster performance
        if self.policy is not None and hasattr(self.policy, 'num_inference_steps'):
            original_steps = self.policy.num_inference_steps
            self.policy.num_inference_steps = self.inference_steps
            if original_steps != self.inference_steps:
                print(
                    f"Set inference steps to {self.inference_steps} "
                    f"(original: {original_steps})"
                )

        if log_path is not None:
            log_path = Path(log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_fp = log_path.open("w", encoding="utf-8")
            self._log_event("init", {"log_path": str(log_path)})
            self._log_event(
                "config",
                {
                    "fps": float(self.fps),
                    "sim_dt": float(self.sim_dt),
                    "data_dt": float(self.data_dt),
                    "frame_stride": int(self.frame_stride),
                    "sim_time_scale": float(self.sim_dt * self.fps),
                    "data_time_scale": float(self.data_dt * self.fps),
                    "robot_speed": float(self.physics.robot_speed),
                    "turn_speed": float(self.physics.turn_speed),
                    "leash_length": float(self.leash_length),
                    "corridor_width": float(self.path_generator.corridor_width),
                    "obstacle_radius": float(self.path_generator.obstacle_radius),
                    "robot_radius": float(self.physics.robot_radius),
                    "human_radius": float(self.physics.human_radius),
                    "action_mode": self.action_mode,
                    "robot_frame": bool(self.robot_frame),
                    "robot_state": self.robot_state,
                    "observation_mode": self.observation_mode,
                    "pointcloud_mode": self.pointcloud_mode,
                    "human_source": self.human_source,
                    "human_detections_topic": self.ros_io.human_detections_topic,
                    "human_detection_timeout": float(
                        self.ros_io.human_detection_timeout
                    ),
                    "human_detector_frame": self.ros_io.human_detector_frame,
                    "human_world_frame": self.ros_io.human_world_frame,
                    "human_detector_y_axis": self.ros_io.human_detector_y_axis,
                    "human_track_max_jump": float(self.human_track_max_jump),
                    "human_kf_process_accel_std": float(
                        self.human_kf_process_accel_std
                    ),
                    "human_kf_measurement_std": float(
                        self.human_kf_measurement_std
                    ),
                    "human_kf_gate": float(self.human_kf_gate),
                    "human_kf_sim_prior_std": float(
                        self.human_kf_sim_prior_std
                    ),
                    "human_kf_sim_prior_max_error": float(
                        self.human_kf_sim_prior_max_error
                    ),
                    "human_rear_prior_calibration": bool(
                        self.human_rear_prior_calibration
                    ),
                    "human_rear_prior_scale": float(
                        self.human_rear_prior_scale
                    ),
                    "human_rear_prior_min_detect_weight": float(
                        self.human_rear_prior_min_detect_weight
                    ),
                    "human_rear_prior_max_detect_weight": float(
                        self.human_rear_prior_max_detect_weight
                    ),
                    "human_kf_hold_timeout": float(
                        self.human_kf_hold_timeout
                    ),
                    "human_kf_max_misses": int(self.human_kf_max_misses),
                    "human_kf_sector_margin_deg": float(
                        self.human_kf_sector_margin_deg
                    ),
                    "human_kf_range_margin": float(
                        self.human_kf_range_margin
                    ),
                    "human_kf_sim_fallback_std": float(
                        self.human_kf_sim_fallback_std
                    ),
                    "human_kf_sim_velocity_gain": float(
                        self.human_kf_sim_velocity_gain
                    ),
                    "human_sim_max_distance": float(
                        self.human_sim_max_distance
                    ),
                    "human_sim_full_loss_timeout": float(
                        self.human_sim_full_loss_timeout
                    ),
                    "human_continuity_mode": bool(
                        self.human_continuity_mode
                    ),
                    "human_robot_jump_threshold": float(
                        self.human_robot_jump_threshold
                    ),
                    "human_robot_heading_jump_deg": float(
                        np.rad2deg(self.human_robot_heading_jump_rad)
                    ),
                    "rosbag_loop_mode": bool(self.rosbag_loop_mode),
                    "rosbag_loop_origin_radius": float(
                        self.rosbag_loop_origin_radius
                    ),
                    "human_rear_sector_range": float(
                        self.human_rear_sector_range
                    ),
                    "human_rear_sector_angle_deg": float(
                        self.human_rear_sector_angle_deg
                    ),
                    "n_lookahead": int(self.n_lookahead),
                    "lookahead_stride": int(self.lookahead_stride),
                    "lidar_num_bins": int(self.lidar_num_bins),
                    "n_obstacle_circles": int(self.n_obstacle_circles),
                    "n_obstacle_segments": int(self.n_obstacle_segments),
                    "obstacle_obs_dim": int(self.obstacle_obs_dim),
                    "obstacle_include_radius": bool(self.obstacle_include_radius),
                    "obstacle_include_human_clearance": bool(
                        getattr(self, "obstacle_include_human_clearance", False)
                    ),
                    "segment_repr": self.segment_repr,
                    "inference_interval": int(self.inference_interval),
                    "inference_steps": int(self.inference_steps),
                    "turn_gain": float(self.turn_gain),
                    "curvature_slowdown": bool(self.curvature_slowdown),
                    "curvature_scale": float(self.curvature_scale),
                    "min_speed_scale": float(self.min_speed_scale),
                    "safety_mode": self.safety_mode,
                    "safety_margin": float(self.safety_filter.margin),
                    "safety_alpha": float(self.safety_filter.alpha),
                    "safety_max_constraints": int(self.safety_filter.max_constraints),
                    "safety_influence_distance": float(self.safety_filter.influence_distance),
                    "debug_qp_log": bool(self.debug_qp_log),
                },
            )

        if self.human_detection_log_path is not None:
            self.human_detection_log_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            self.human_detection_log_fp = self.human_detection_log_path.open(
                "w",
                encoding="utf-8",
            )
            self._log_human_system_event(
                "human_detection_log_started",
                {
                    "path": str(self.human_detection_log_path),
                    "topic": self.ros_io.human_detections_topic,
                    "log_interval": int(self.human_detection_log_interval),
                    "max_candidates_per_record": int(
                        self.human_detection_log_max_candidates
                    ),
                    "rear_sector_range": float(self.human_rear_sector_range),
                    "rear_sector_angle_deg": float(
                        self.human_rear_sector_angle_deg
                    ),
                    "tracking_sector_range": float(
                        self.human_rear_sector_range
                        + self.human_kf_range_margin
                    ),
                    "tracking_sector_angle_deg": float(
                        min(
                            360.0,
                            self.human_rear_sector_angle_deg
                            + 2.0 * self.human_kf_sector_margin_deg,
                        )
                    ),
                    "kf_gate": float(self.human_kf_gate),
                    "max_jump": float(self.human_track_max_jump),
                    "rear_prior_calibration": bool(
                        self.human_rear_prior_calibration
                    ),
                    "rear_prior_scale": float(
                        self.human_rear_prior_scale
                    ),
                    "rear_prior_min_detect_weight": float(
                        self.human_rear_prior_min_detect_weight
                    ),
                    "rear_prior_max_detect_weight": float(
                        self.human_rear_prior_max_detect_weight
                    ),
                },
            )
            print(
                "Human detection diagnostic log: "
                f"{self.human_detection_log_path}"
            )

        if eval_path is not None:
            eval_path = Path(eval_path)
            eval_path.parent.mkdir(parents=True, exist_ok=True)
            self.eval_fp = eval_path.open("w", encoding="utf-8")
            print(f"Planning eval log: {eval_path}")

        self._generate_new_path()

    def set_path_data(self, path_data: dict, reset: bool = True):
        """Override the current episode path/obstacles with externally-provided data."""
        self.current_path_data = copy.deepcopy(path_data)
        self._precompute_frenet_cache()
        if reset:
            self._reset_position()

    def _generate_new_path(self):
        """Generate new reference path."""
        self.current_path_data = self.path_generator.generate()
        self._precompute_frenet_cache()
        self._reset_position()
        obstacles = self.current_path_data.get("obstacles")
        obstacle_count = int(len(obstacles)) if obstacles is not None else 0
        segments = self.current_path_data.get("segment_obstacles")
        segment_count = int(len(segments)) if segments is not None else 0
        self._log_event(
            "new_path",
            {
                "path_length": float(self.current_path_data["length"]),
                "obstacle_count": obstacle_count,
                "segment_obstacle_count": segment_count,
                "start": self._rounded_list(self.current_path_data.get("start")),
                "end": self._rounded_list(self.current_path_data.get("end")),
                "obstacles": self._serialize_obstacles(obstacles) if self.debug_qp_log else None,
                "segment_obstacles": self._serialize_segments(segments) if self.debug_qp_log else None,
            },
        )
        print(f"New path generated: length={self.current_path_data['length']:.1f}m")

    def _reset_position(self):
        """Reset to start position."""
        if self.current_path_data is not None:
            start = self.current_path_data["start"]
            self.scorer = TrajectoryScorer(
                self.current_path_data["path"],
                self.leash_length,
            )
        else:
            start = np.array([0.0, 0.0])

        self.paused = False
        self.collision_pause = False
        self.bre = False
        self.triggered_bre_toggle_times.clear()
        self.bre_timer_start_frame = None
        self.interaction_segmenter.reset()
        self.physics.reset(start)
        # A user-requested reset establishes a new local comparison baseline.
        self._last_robot_odom_position = None
        self._last_robot_odom_heading = None
        self._last_robot_odom_time = 0.0
        self._synchronize_robot_from_odometry()
        robot_position = self.physics.robot.position.copy()
        robot_heading = float(self.physics.robot.heading)
        trailing_direction = np.array(
            [np.cos(robot_heading), np.sin(robot_heading)], dtype=np.float32
        )
        self.physics.human.position = (
            robot_position - float(self.leash_length) * trailing_direction
        ).astype(np.float32)
        if hasattr(self.physics.human, "velocity"):
            self.physics.human.velocity = np.zeros((2,), dtype=np.float32)
        self._human_kf.reset()
        self._tracked_human_position = None
        self._tracked_human_velocity = np.zeros((2,), dtype=np.float32)
        self._human_detection_seq = -1
        self._human_detection_receive_stamp = 0.0
        self._human_kf_last_measurement_stamp = 0.0
        self._human_kf_consecutive_misses = 0
        self._human_kf_using_prediction = False
        self._human_kf_last_mahalanobis_sq = float("inf")
        self._human_detection_rejected = False
        self._human_detector_waiting = False
        self._human_tracking_mode = "uninitialized"
        self._human_sim_fallback_start_stamp = 0.0
        self._human_last_failure_reason = ""
        self._robot_localization_jump_pending = False
        self._robot_localization_jump_reason = ""
        self._rosbag_loop_reset_pending = False
        self._rosbag_loop_reacquire_seq = -1
        self._rosbag_loop_reset_reason = ""
        self._last_odom_replay_epoch = int(
            self.ros_io.odom_replay_status()["replay_epoch"]
        )
        if self.human_source == "detector":
            self._refresh_human_from_detector()
        if self.recording:
            self._stop_recording()
        self._last_mid360_cloud_seq = None
        self.robot_trajectory = []
        self.human_trajectory = []
        self.planned_path = None
        self.nominal_planned_path = None
        self.safe_planned_path = None
        self.current_mid360_points_world = None
        self.current_mid360_point_obstacles = None
        self.current_safety_point_obstacles = None
        self.frame_count = 0
        self.prev_robot_pos = None
        self._seed_obs_history(self.physics.robot.position, self.physics.human.position)
        if self.policy is not None:
            self.policy.reset()
        # Reset action cache
        self._reset_runtime_caches()
        self.stashed_guide_action_seq = None
        self.stashed_guide_cursor = 0
        self.last_stashed_compliance_info = {}
        self.data_step_idx = 0
        self.episode_safety_stats = {
            "modified_steps": 0,
            "total_steps": 0,
            "total_shift": 0.0,
            "constraint_count": 0,
            "min_clearance": float("inf"),
        }
        self._log_event(
            "reset_position",
            {
                "robot_pos": self.physics.robot.position.tolist(),
                "human_pos": self.physics.human.position.tolist(),
                "human_source": self.human_source,
            },
        )

    def _update_timed_bre_toggle(self):
        if self.bre_timer_start_frame is None:
            return
        sim_time = float((self.frame_count - self.bre_timer_start_frame) * self.sim_dt)
        for toggle_time in self.bre_toggle_times:
            if toggle_time not in self.triggered_bre_toggle_times and sim_time >= toggle_time:
                detector_is_fallback = not self.interaction_segmenter.has_label
                if detector_is_fallback and not self.bre:
                    self._stash_current_cached_guide_plan()
                self.bre = not self.bre
                if detector_is_fallback:
                    self._reset_runtime_caches()
                self.triggered_bre_toggle_times.add(toggle_time)
                print(
                    f"Timed hidden-state toggle at {toggle_time:.1f}s: "
                    f"bre={self.bre}, detected={self._current_interaction_label()}"
                )
                self._log_event(
                    "timed_bre_toggle",
                    {"toggle_time": float(toggle_time), "bre": bool(self.bre)},
                )

    def _seed_obs_history(self, robot_pos: np.ndarray, human_pos: np.ndarray):
        obs = self._build_obs(robot_pos, human_pos, self.physics.robot.heading)
        self.obs_history.clear()
        for _ in range(self.n_obs_steps):
            self.obs_history.append(obs.copy())
        self.prev_robot_pos = robot_pos.copy()

    def _synchronize_robot_from_odometry(self) -> None:
        """Make measured odometry authoritative and classify discontinuities.

        A timestamp rewind is the strongest rosbag-loop signal because
        ``rosbag play -l`` preserves message header stamps.  In rosbag loop
        mode, a large jump back near the first odometry pose is also accepted
        as a fallback signal for bags whose timestamps are unavailable.

        Replay resets clear all temporal state and wait for a new detector
        message, but they are not reported as localization/detection failures.
        Other discontinuities retain the fail-closed localization-jump path.
        """
        position, heading = self.ros_io.robot_pose()
        position = np.asarray(position, dtype=np.float32).reshape(2)
        heading = float(heading)
        now = float(rospy.get_time())
        replay_status = self.ros_io.odom_replay_status()
        replay_epoch = int(replay_status["replay_epoch"])

        if self._rosbag_initial_odom_position is None:
            self._rosbag_initial_odom_position = position.copy()

        position_bad = False
        heading_bad = False
        position_jump = 0.0
        heading_jump = 0.0
        dynamic_position_limit = float(self.human_robot_jump_threshold)
        dynamic_heading_limit = float(self.human_robot_heading_jump_rad)

        if (
            self._last_robot_odom_position is not None
            and self._last_robot_odom_heading is not None
            and self.human_source == "detector"
        ):
            position_jump = float(
                np.linalg.norm(position - self._last_robot_odom_position)
            )
            heading_jump = abs(
                wrap_angle(heading - float(self._last_robot_odom_heading))
            )
            elapsed = max(
                0.0,
                now - float(self._last_robot_odom_time),
            )
            # Long diffusion/QP calls can delay this loop. Allow physically
            # plausible motion over that elapsed time before declaring an
            # odometry-frame discontinuity.
            dynamic_position_limit = max(
                self.human_robot_jump_threshold,
                2.0 * float(self.physics.robot_speed) * elapsed + 0.25,
            )
            dynamic_heading_limit = max(
                self.human_robot_heading_jump_rad,
                2.0 * float(self.physics.turn_speed) * elapsed
                + np.deg2rad(10.0),
            )
            position_bad = (
                self.human_robot_jump_threshold > 0.0
                and position_jump > dynamic_position_limit
            )
            heading_bad = (
                self.human_robot_heading_jump_rad > 0.0
                and heading_jump > dynamic_heading_limit
            )

        timestamp_rewound = replay_epoch != self._last_odom_replay_epoch
        near_bag_origin = False
        returned_from_away = False
        if self._rosbag_initial_odom_position is not None:
            distance_to_origin = float(
                np.linalg.norm(position - self._rosbag_initial_odom_position)
            )
            near_bag_origin = (
                self.rosbag_loop_origin_radius > 0.0
                and distance_to_origin <= self.rosbag_loop_origin_radius
            )
            if self._last_robot_odom_position is not None:
                previous_distance_to_origin = float(
                    np.linalg.norm(
                        self._last_robot_odom_position
                        - self._rosbag_initial_odom_position
                    )
                )
                returned_from_away = previous_distance_to_origin > max(
                    2.0 * self.rosbag_loop_origin_radius,
                    dynamic_position_limit,
                )

        geometric_loop_reset = bool(
            self.rosbag_loop_mode
            and (position_bad or heading_bad)
            and near_bag_origin
            and (returned_from_away or timestamp_rewound)
        )
        replay_reset = bool(
            self.rosbag_loop_mode
            and (timestamp_rewound or geometric_loop_reset)
        )

        # Make the newest measurement authoritative before reset handlers log
        # or rebuild dependent state.
        self.physics.robot.position = position.copy()
        self.physics.robot.heading = heading

        if replay_reset:
            reason_parts = []
            if timestamp_rewound:
                reason_parts.append(
                    "odometry timestamp rewind "
                    f"(epoch {self._last_odom_replay_epoch} -> {replay_epoch})"
                )
            if geometric_loop_reset:
                reason_parts.append(
                    "pose returned to rosbag origin "
                    f"(jump={position_jump:.2f} m, "
                    f"heading={np.rad2deg(heading_jump):.1f} deg)"
                )
            self._handle_rosbag_loop_reset("; ".join(reason_parts))
        elif position_bad or heading_bad:
            reason = (
                "robot localization jump: "
                f"position={position_jump:.2f} m "
                f"(dynamic_limit={dynamic_position_limit:.2f}), "
                f"heading={np.rad2deg(heading_jump):.1f} deg "
                f"(dynamic_limit={np.rad2deg(dynamic_heading_limit):.1f})"
            )
            self._handle_robot_localization_jump(reason)

        self._last_odom_replay_epoch = replay_epoch
        self._last_robot_odom_position = position.copy()
        self._last_robot_odom_heading = heading
        self._last_robot_odom_time = now

    def _handle_rosbag_loop_reset(self, reason: str) -> None:
        """Reset temporal state at a benign rosbag wrap without losing human.

        The old implementation cleared the KF and stopped until a later
        non-empty detector frame arrived.  In continuity mode we instead seed
        the human at the known rear-leash prior immediately, continue the
        PhysicsEngine, and let the first post-loop detector frame correct it.
        """
        self._rosbag_loop_reset_count += 1
        self._rosbag_loop_reset_reason = str(reason)
        self._robot_localization_jump_pending = False
        self._robot_localization_jump_reason = ""

        prior = self._rear_leash_prior_position()
        self.physics.human.position = prior.copy()
        if hasattr(self.physics.human, "velocity"):
            self.physics.human.velocity = np.zeros((2,), dtype=np.float32)

        now = float(rospy.get_time())
        self._human_kf.reset()
        self._human_kf.initialize(
            prior,
            now,
            velocity=np.zeros((2,), dtype=np.float32),
        )
        self._sync_tracked_human_from_kf()
        self._apply_detector_human_state()

        current_human_seq = int(
            self.ros_io.human_detection_status().get("seq", -1)
        )
        self._human_detection_seq = current_human_seq
        self._rosbag_loop_reacquire_seq = current_human_seq
        self._human_kf_last_measurement_stamp = 0.0
        self._human_kf_consecutive_misses = 0
        self._human_kf_using_prediction = True
        self._human_kf_last_mahalanobis_sq = float("inf")
        self._human_detection_rejected = False
        self._human_sim_fallback_start_stamp = now
        self._human_last_failure_reason = ""

        self._reset_runtime_caches()
        self.interaction_segmenter.reset()
        self.obs_history.clear()
        self.prev_robot_pos = None
        self.robot_trajectory = []
        self.human_trajectory = []
        self.planned_path = None
        self.nominal_planned_path = None
        self.safe_planned_path = None
        self.current_mid360_points_world = None
        self.current_mid360_point_obstacles = None
        self.current_safety_point_obstacles = None
        self._last_mid360_cloud_seq = None
        if self.scorer is not None:
            self.scorer.reset()

        if self.human_source == "detector":
            self._rosbag_loop_reset_pending = True
            self._human_detector_waiting = False
            self._human_tracking_mode = "rosbag_prior_fallback"
            self._seed_obs_history(
                self.physics.robot.position,
                self.physics.human.position,
            )
        else:
            self._rosbag_loop_reset_pending = False
            self._human_detector_waiting = False
            self._human_tracking_mode = "sim"
            self._seed_obs_history(
                self.physics.robot.position,
                self.physics.human.position,
            )

        print(
            "[rosbag loop] playback restarted; continuing from rear-prior "
            f"fallback while detector reacquires: {reason}"
        )
        self._log_human_system_event(
            "rosbag_loop_reset",
            {
                "reason": str(reason),
                "reset_count": int(self._rosbag_loop_reset_count),
                "reacquire_after_seq": int(self._rosbag_loop_reacquire_seq),
                "robot_pos": self.physics.robot.position.tolist(),
                "robot_heading": float(self.physics.robot.heading),
                "fallback_human_pos": self.physics.human.position.tolist(),
                "continuing": True,
            },
        )
        self._log_event(
            "rosbag_loop_reset",
            {
                "reason": str(reason),
                "reset_count": int(self._rosbag_loop_reset_count),
                "human_reacquire_after_seq": int(
                    self._rosbag_loop_reacquire_seq
                ),
                "robot_pos": self.physics.robot.position.tolist(),
                "robot_heading": float(self.physics.robot.heading),
                "continuing": True,
            },
        )

    def _wait_for_rosbag_human_reacquisition(self, reason: str) -> bool:
        """Continue on the fallback track while awaiting post-loop detector data."""
        self._rosbag_loop_reset_pending = True
        continued = self._force_continuity_fallback(
            reason=f"rosbag reacquire: {reason}",
            now=float(rospy.get_time()),
            mode="rosbag_prior_fallback",
        )
        rospy.loginfo_throttle(
            1.0,
            "[rosbag loop] detector reacquiring; fallback remains active: "
            f"{reason}",
        )
        return bool(continued)

    def _handle_robot_localization_jump(self, reason: str) -> None:
        """Re-anchor the human behind the newest robot pose after an odom jump."""
        if self._robot_localization_jump_pending:
            return

        if self.human_continuity_mode:
            now = float(rospy.get_time())
            prior = self._rear_leash_prior_position()
            self.physics.human.position = prior.copy()
            if hasattr(self.physics.human, "velocity"):
                self.physics.human.velocity = np.zeros((2,), dtype=np.float32)
            self._human_kf.reset()
            self._human_kf.initialize(
                prior,
                now,
                velocity=np.zeros((2,), dtype=np.float32),
            )
            self._sync_tracked_human_from_kf()
            self._apply_detector_human_state()
            self._human_detection_seq = int(
                self.ros_io.human_detection_status().get("seq", -1)
            )
            self._human_kf_last_measurement_stamp = 0.0
            self._human_kf_consecutive_misses = 0
            self._human_kf_using_prediction = True
            self._human_detection_rejected = False
            self._human_tracking_mode = "odom_jump_prior_fallback"
            self._human_last_failure_reason = str(reason)
            self._human_detector_waiting = False
            self._robot_localization_jump_pending = False
            self._robot_localization_jump_reason = ""
            self._reset_runtime_caches()
            self.interaction_segmenter.reset()
            self._seed_obs_history(
                self.physics.robot.position,
                self.physics.human.position,
            )
            rospy.logwarn(
                "[human continuity] odometry jump re-anchored human to rear "
                f"prior; control loop continues: {reason}"
            )
            self._log_event(
                "robot_localization_jump_fallback",
                {
                    "reason": str(reason),
                    "robot_pos": self.physics.robot.position.tolist(),
                    "robot_heading": float(self.physics.robot.heading),
                    "human_pos": self.physics.human.position.tolist(),
                    "continuing": True,
                },
            )
            return

        self._robot_localization_jump_pending = True
        self._robot_localization_jump_reason = str(reason)
        self._human_tracking_mode = "odom_jump"
        self._human_detection_rejected = True
        self._human_last_failure_reason = str(reason)
        self._human_kf.reset()
        self._tracked_human_position = None
        self._tracked_human_velocity = np.zeros((2,), dtype=np.float32)
        self._human_detection_seq = int(
            self.ros_io.human_detection_status().get("seq", -1)
        )
        self._human_kf_last_measurement_stamp = 0.0
        self._human_kf_consecutive_misses = 0
        self._human_kf_using_prediction = False
        self.ros_io.stop()
        self._set_human_detector_waiting(True, reason)
        self._log_event(
            "robot_localization_jump",
            {
                "reason": str(reason),
                "robot_pos": self.physics.robot.position.tolist(),
                "robot_heading": float(self.physics.robot.heading),
            },
        )

    def _set_human_detector_waiting(
        self,
        waiting: bool,
        reason: str,
    ) -> None:
        waiting = bool(waiting)
        if waiting == self._human_detector_waiting:
            return

        self._human_detector_waiting = waiting
        self._reset_runtime_caches()
        self.interaction_segmenter.reset()
        if waiting:
            self.ros_io.stop()
            print(f"[human detector] waiting: {reason}; robot stopped")
            event_name = "human_detector_lost"
        else:
            self._seed_obs_history(
                self.physics.robot.position,
                self.physics.human.position,
            )
            print(
                "[human detector] target available: "
                f"mode={self._human_tracking_mode}, "
                f"position={np.round(self.physics.human.position, 3).tolist()}"
            )
            event_name = "human_detector_acquired"
        self._log_event(
            event_name,
            {
                "reason": str(reason),
                "human_source": self.human_source,
                "tracking_mode": self._human_tracking_mode,
                "human_pos": self.physics.human.position.tolist(),
            },
        )

    def _apply_detector_human_state(self) -> None:
        if self._tracked_human_position is None:
            return
        self.physics.human.position = self._tracked_human_position.copy()
        if hasattr(self.physics.human, "velocity"):
            self.physics.human.velocity = self._tracked_human_velocity.copy()
        speed = float(np.linalg.norm(self._tracked_human_velocity))
        if speed > 0.05 and hasattr(self.physics.human, "heading"):
            self.physics.human.heading = float(
                np.arctan2(
                    self._tracked_human_velocity[1],
                    self._tracked_human_velocity[0],
                )
            )

    def _human_candidates_in_rear_sector(
        self,
        candidates_world: np.ndarray,
        *,
        robot_position: Optional[np.ndarray] = None,
        robot_heading: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Strict detector ROI used to initialize a new human track."""
        if robot_position is None:
            robot_position = self.physics.robot.position
        if robot_heading is None:
            robot_heading = self.physics.robot.heading
        return filter_points_in_robot_rear_sector(
            candidates_world,
            robot_position,
            float(robot_heading),
            self.human_rear_sector_range,
            self.human_rear_sector_angle_deg,
        )

    def _human_candidates_in_tracking_sector(
        self,
        candidates_world: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Relaxed ROI for an already initialized detector/simulation track."""
        aperture = min(
            360.0,
            self.human_rear_sector_angle_deg
            + 2.0 * self.human_kf_sector_margin_deg,
        )
        radius = self.human_rear_sector_range + self.human_kf_range_margin
        return filter_points_in_robot_rear_sector(
            candidates_world,
            self.physics.robot.position,
            float(self.physics.robot.heading),
            radius,
            aperture,
        )

    def _human_position_in_tracking_sector(
        self,
        position_world: np.ndarray,
    ) -> bool:
        kept, _ = self._human_candidates_in_tracking_sector(
            np.asarray(position_world, dtype=np.float32).reshape(1, 2)
        )
        return len(kept) > 0

    def _sync_tracked_human_from_kf(self) -> None:
        if not self._human_kf.initialized:
            return
        self._tracked_human_position = self._human_kf.position
        self._tracked_human_velocity = self._human_kf.velocity

    def _simulation_human_state_valid(
        self,
        sim_position: np.ndarray,
        sim_velocity: np.ndarray,
    ) -> tuple[bool, str]:
        sim_position = np.asarray(sim_position, dtype=np.float32).reshape(2)
        sim_velocity = np.asarray(sim_velocity, dtype=np.float32).reshape(2)
        if not np.isfinite(sim_position).all() or not np.isfinite(sim_velocity).all():
            return False, "simulation human state is non-finite"
        robot_distance = float(
            np.linalg.norm(sim_position - self.physics.robot.position)
        )
        if robot_distance > self.human_sim_max_distance:
            return (
                False,
                f"simulation human is {robot_distance:.2f} m from robot "
                f"(limit={self.human_sim_max_distance:.2f} m)",
            )
        return True, ""

    def _fuse_simulation_state(
        self,
        *,
        timestamp: float,
        sim_position: np.ndarray,
        sim_velocity: np.ndarray,
        fallback: bool,
    ) -> bool:
        """Predict with KF and correct toward the PhysicsEngine state."""
        if not self._human_kf.initialized:
            return False
        valid, _reason = self._simulation_human_state_valid(
            sim_position,
            sim_velocity,
        )
        if not valid:
            return False

        self._human_kf.predict(float(timestamp))
        if fallback:
            sim_std = self.human_kf_sim_fallback_std
            velocity_gain = self.human_kf_sim_velocity_gain
        else:
            sim_std = self.human_kf_sim_prior_std
            velocity_gain = min(0.25, self.human_kf_sim_velocity_gain)
        self._human_kf.correct(
            np.asarray(sim_position, dtype=np.float32),
            measurement_std=sim_std,
        )
        self._human_kf.blend_velocity(sim_velocity, velocity_gain)
        return True

    def _force_continuity_fallback(
        self,
        *,
        reason: str,
        now: float,
        mode: str = "sim_fallback",
        preferred_position: Optional[np.ndarray] = None,
        preferred_velocity: Optional[np.ndarray] = None,
        measurement_std: Optional[float] = None,
        reseed: bool = False,
    ) -> bool:
        """Produce a bounded human state even when the detector is unusable.

        Preference order:
        1. a finite PhysicsEngine human state inside the relaxed rear sector;
        2. the deterministic rear-leash prior derived from robot pose.

        This method is intentionally independent of detector/KF initialization,
        which removes the old startup/reset path that immediately entered LOST.
        """
        now = float(now)
        prior = self._rear_leash_prior_position()

        if preferred_position is None:
            position = np.asarray(
                self.physics.human.position, dtype=np.float32
            ).reshape(2)
        else:
            position = np.asarray(preferred_position, dtype=np.float32).reshape(2)
        if preferred_velocity is None:
            velocity = np.asarray(
                getattr(
                    self.physics.human,
                    "velocity",
                    np.zeros((2,), dtype=np.float32),
                ),
                dtype=np.float32,
            ).reshape(2)
        else:
            velocity = np.asarray(preferred_velocity, dtype=np.float32).reshape(2)

        valid, invalid_reason = self._simulation_human_state_valid(
            position, velocity
        )
        inside = bool(valid and self._human_position_in_tracking_sector(position))
        fallback_source = "simulation"
        if not inside:
            position = prior
            velocity = np.zeros((2,), dtype=np.float32)
            fallback_source = "rear_prior"
            if valid:
                invalid_reason = "simulation human outside relaxed rear sector"

        if reseed or not self._human_kf.initialized:
            self._human_kf.initialize(position, now, velocity=velocity)
        else:
            self._human_kf.predict(now)
            std = (
                self.human_kf_sim_fallback_std
                if measurement_std is None
                else max(1e-4, float(measurement_std))
            )
            self._human_kf.correct(position, measurement_std=std)
            self._human_kf.blend_velocity(
                velocity, self.human_kf_sim_velocity_gain
            )

        self._sync_tracked_human_from_kf()
        if self._tracked_human_position is None:
            return False
        self._apply_detector_human_state()
        self._human_kf_using_prediction = True
        self._human_detection_rejected = False
        self._human_tracking_mode = str(mode)
        self._human_last_failure_reason = str(reason)
        self._human_detector_waiting = False
        if self._human_sim_fallback_start_stamp <= 0.0:
            self._human_sim_fallback_start_stamp = now

        rospy.logwarn_throttle(
            1.0,
            "[human continuity] "
            f"mode={mode}, source={fallback_source}: {reason}"
            + (f"; {invalid_reason}" if invalid_reason else ""),
        )
        return True

    def _continue_with_simulation(
        self,
        reason: str,
        *,
        now: float,
        sim_position: np.ndarray,
        sim_velocity: np.ndarray,
        count_miss: bool,
        force_strong_fallback: bool = False,
    ) -> bool:
        """Bridge every detector failure with KF + simulation/rear prior."""
        if count_miss:
            self._human_kf_consecutive_misses += 1

        detector_gap = (
            float(now) - self._human_kf_last_measurement_stamp
            if self._human_kf_last_measurement_stamp > 0.0
            else float("inf")
        )
        short_bridge = (
            self._human_kf.initialized
            and not force_strong_fallback
            and detector_gap <= self.human_kf_hold_timeout
            and self._human_kf_consecutive_misses
            <= self.human_kf_max_misses
        )
        strong_fallback = not short_bridge

        if strong_fallback:
            if self._human_sim_fallback_start_stamp <= 0.0:
                self._human_sim_fallback_start_stamp = float(now)
            fallback_age = float(now) - self._human_sim_fallback_start_stamp
            # Continuity mode deliberately ignores the optional full-loss
            # timeout.  Disabling continuity restores the fail-closed timeout.
            if (
                not self.human_continuity_mode
                and self.human_sim_full_loss_timeout > 0.0
                and fallback_age > self.human_sim_full_loss_timeout
            ):
                self._human_last_failure_reason = (
                    f"simulation-only fallback exceeded "
                    f"{self.human_sim_full_loss_timeout:.2f}s"
                )
                return False
        else:
            fallback_age = 0.0
            self._human_sim_fallback_start_stamp = 0.0

        mode = "sim_fallback" if strong_fallback else "prediction_blend"
        std = (
            self.human_kf_sim_fallback_std
            if strong_fallback
            else self.human_kf_sim_prior_std
        )
        continued = self._force_continuity_fallback(
            reason=reason,
            now=now,
            mode=mode,
            preferred_position=sim_position,
            preferred_velocity=sim_velocity,
            measurement_std=std,
            reseed=False,
        )
        if continued:
            if strong_fallback:
                rospy.logwarn_throttle(
                    1.0,
                    "[human detector] simulation/rear-prior fallback: "
                    f"{reason}; misses={self._human_kf_consecutive_misses}, "
                    f"detector_gap={detector_gap:.2f}s, "
                    f"fallback_age={fallback_age:.2f}s",
                )
            else:
                rospy.logwarn_throttle(
                    1.0,
                    "[human detector] KF/simulation prediction bridge: "
                    f"{reason}; detector_gap={detector_gap:.3f}s",
                )
        return bool(continued)

    def _stop_for_human_tracking_failure(
        self,
        reason: str,
        *,
        mode: str = "lost",
    ) -> bool:
        """Use the deterministic prior before declaring a true hard failure."""
        if self.human_continuity_mode:
            prior = self._rear_leash_prior_position()
            continued = self._force_continuity_fallback(
                reason=f"forced fallback after {mode}: {reason}",
                now=float(rospy.get_time()),
                mode="rear_prior_fallback",
                preferred_position=prior,
                preferred_velocity=np.zeros((2,), dtype=np.float32),
                measurement_std=self.human_kf_sim_fallback_std,
                reseed=not self._human_kf.initialized,
            )
            if continued:
                return True

        self._human_detection_rejected = True
        self._human_kf_using_prediction = False
        self._human_tracking_mode = str(mode)
        self._human_last_failure_reason = str(reason)
        rospy.logwarn_throttle(1.0, f"[human detector] {mode}: {reason}")
        self._set_human_detector_waiting(True, reason)
        return False

    def _rear_leash_prior_position(
        self,
        *,
        robot_position: Optional[np.ndarray] = None,
        robot_heading: Optional[float] = None,
    ) -> np.ndarray:
        """Return the simple human prior: one leash length behind robot."""
        if robot_position is None:
            robot_position = self.physics.robot.position
        if robot_heading is None:
            robot_heading = self.physics.robot.heading
        robot_position = np.asarray(
            robot_position, dtype=np.float32
        ).reshape(2)
        heading = float(robot_heading)
        forward = np.array(
            [np.cos(heading), np.sin(heading)],
            dtype=np.float32,
        )
        return (
            robot_position - float(self.leash_length) * forward
        ).astype(np.float32)

    def _calibrate_human_detection_candidates(
        self,
        candidates: np.ndarray,
    ) -> tuple[np.ndarray, list[dict], np.ndarray]:
        """Pull detector positions toward the rear-leash prior adaptively.

        The detector weight follows a robust Cauchy-like curve:

            w_detect = 1 / (1 + (error / scale)^2)
            calibrated = (1 - w_detect) * rear_prior
                         + w_detect * detect_position

        A close detection keeps a large detector weight. A far detection is
        pulled strongly toward the position exactly one leash length behind
        the robot. Weight clipping prevents complete trust in either source.
        """
        candidates = np.asarray(
            candidates, dtype=np.float32
        ).reshape(-1, 2)
        prior = self._rear_leash_prior_position()
        if len(candidates) == 0:
            return candidates.copy(), [], prior

        errors = np.linalg.norm(candidates - prior[None, :], axis=1)
        if self.human_rear_prior_calibration:
            scale = max(float(self.human_rear_prior_scale), 1e-6)
            detect_weights = 1.0 / (1.0 + (errors / scale) ** 2)
            detect_weights = np.clip(
                detect_weights,
                self.human_rear_prior_min_detect_weight,
                self.human_rear_prior_max_detect_weight,
            )
        else:
            detect_weights = np.ones((len(candidates),), dtype=np.float32)

        prior_weights = 1.0 - detect_weights
        calibrated = (
            prior_weights[:, None] * prior[None, :]
            + detect_weights[:, None] * candidates
        ).astype(np.float32)
        corrections = np.linalg.norm(calibrated - candidates, axis=1)

        rows: list[dict] = []
        for idx in range(len(candidates)):
            rows.append(
                {
                    "detected_world_xy": candidates[idx].copy(),
                    "rear_prior_world_xy": prior.copy(),
                    "calibrated_world_xy": calibrated[idx].copy(),
                    "prior_error_m": float(errors[idx]),
                    "detect_weight": float(detect_weights[idx]),
                    "prior_weight": float(prior_weights[idx]),
                    "calibration_shift_m": float(corrections[idx]),
                }
            )
        return calibrated, rows, prior

    def _initialize_human_track(
        self,
        candidates: np.ndarray,
        *,
        timestamp: float,
        sim_position: np.ndarray,
        sim_velocity: np.ndarray,
        calibrated_position: Optional[np.ndarray] = None,
    ) -> bool:
        candidates = np.asarray(candidates, dtype=np.float32).reshape(-1, 2)
        if calibrated_position is None and len(candidates) == 0:
            return False
        if calibrated_position is not None:
            initial_position = np.asarray(
                calibrated_position, dtype=np.float32
            ).reshape(2)
        else:
            sim_distances = np.linalg.norm(
                candidates
                - np.asarray(sim_position, dtype=np.float32)[None, :],
                axis=1,
            )
            selected = candidates[int(np.argmin(sim_distances))]
            detector_var = self.human_kf_measurement_std ** 2
            sim_var = self.human_kf_sim_prior_std ** 2
            detector_weight = sim_var / max(
                detector_var + sim_var, 1e-9
            )
            initial_position = (
                detector_weight * selected
                + (1.0 - detector_weight) * sim_position
            ).astype(np.float32)
        self._human_kf.initialize(
            initial_position,
            timestamp,
            velocity=sim_velocity,
        )
        self._human_kf_last_mahalanobis_sq = 0.0
        self._human_kf_last_measurement_stamp = float(timestamp)
        self._human_detection_receive_stamp = float(timestamp)
        self._human_kf_consecutive_misses = 0
        self._human_kf_using_prediction = False
        self._human_detection_rejected = False
        self._human_tracking_mode = "detector_fused"
        self._human_sim_fallback_start_stamp = 0.0
        self._human_last_failure_reason = ""
        self._robot_localization_jump_pending = False
        self._robot_localization_jump_reason = ""
        self._rosbag_loop_reset_pending = False
        self._rosbag_loop_reset_reason = ""
        return True

    def _refresh_human_from_detector(self) -> bool:
        """Fuse detector measurements and write a complete decision trace."""
        if self.human_source != "detector":
            return True

        now = float(rospy.get_time())
        sim_position = np.asarray(
            self.physics.human.position,
            dtype=np.float32,
        ).reshape(2)
        sim_velocity = np.asarray(
            getattr(
                self.physics.human,
                "velocity",
                np.zeros((2,), dtype=np.float32),
            ),
            dtype=np.float32,
        ).reshape(2)
        trace = self._new_human_detection_trace(
            now=now,
            sim_position=sim_position,
            sim_velocity=sim_velocity,
        )

        def finish(
            outcome: str,
            success: bool,
            reason: str = "",
            *,
            important: bool = False,
            **extra,
        ) -> bool:
            return self._finish_human_detection_trace(
                trace,
                outcome=outcome,
                success=success,
                reason=reason,
                important=important,
                extra=extra,
            )

        detection = self.ros_io.human_detections_world()
        if detection is None:
            status = trace["detector_status"]
            if self._rosbag_loop_reset_pending:
                status_seq = int(status.get("seq", -1))
                if status_seq <= self._rosbag_loop_reacquire_seq:
                    reason = "no post-loop detector message yet"
                    result = self._wait_for_rosbag_human_reacquisition(reason)
                    return finish(
                        "rosbag_reacquire_wait",
                        result,
                        reason,
                        important=True,
                    )
                # A new post-loop PoseArray reached the callback, but exact TF
                # may still be rebuilding. Do not stay permanently latched in
                # rosbag_prior_fallback; continue with sim/prior and allow the
                # next detector frame to fuse normally.
                self._rosbag_loop_reset_pending = False
                self._rosbag_loop_reset_reason = ""
                self._human_detector_waiting = False
                rospy.logwarn(
                    "[rosbag loop] post-loop detector message received but "
                    "world transform is not ready; continuing fallback "
                    "without blocking reacquisition"
                )

            age = float(status["age"])
            if status["transform_error"]:
                reason = str(status["transform_error"])
                no_data_kind = "tf_unavailable"
            elif np.isfinite(age):
                reason = (
                    f"no fresh detection (age={age:.3f}s, timeout="
                    f"{self.ros_io.human_detection_timeout:.3f}s)"
                )
                no_data_kind = "stale_or_empty_cache"
            else:
                reason = "no detector message received"
                no_data_kind = "never_received"

            continued = self._continue_with_simulation(
                reason,
                now=now,
                sim_position=sim_position,
                sim_velocity=sim_velocity,
                count_miss=False,
            )
            if continued:
                return finish(
                    "no_fresh_detection_simulation",
                    True,
                    reason,
                    important=no_data_kind != "stale_or_empty_cache",
                    no_data_kind=no_data_kind,
                )

            mode = "odom_jump" if self._robot_localization_jump_pending else "lost"
            final_reason = (
                self._robot_localization_jump_reason
                if self._robot_localization_jump_pending
                else f"{reason}; no initialized detector/simulation track"
            )
            stopped = self._stop_for_human_tracking_failure(
                final_reason,
                mode=mode,
            )
            return finish(
                "no_fresh_detection_stopped",
                stopped,
                final_reason,
                important=True,
                no_data_kind=no_data_kind,
            )

        raw_candidates, seq, receive_stamp = detection
        seq = int(seq)
        receive_stamp = float(receive_stamp)
        if not np.isfinite(receive_stamp) or receive_stamp <= 0.0:
            receive_stamp = now
        trace["detector_frame"] = {
            "seq": int(seq),
            "receive_stamp": float(receive_stamp),
            "receive_age": float(max(0.0, now - receive_stamp)),
            "is_new_seq": bool(seq != self._human_detection_seq),
        }

        if (
            self._rosbag_loop_reset_pending
            and seq <= self._rosbag_loop_reacquire_seq
        ):
            reason = (
                f"detector seq={seq} has not advanced beyond "
                f"{self._rosbag_loop_reacquire_seq}"
            )
            result = self._wait_for_rosbag_human_reacquisition(reason)
            return finish(
                "rosbag_reacquire_old_frame",
                result,
                reason,
                important=True,
            )

        if (
            self._rosbag_loop_reset_pending
            and seq > self._rosbag_loop_reacquire_seq
        ):
            self._rosbag_loop_reset_pending = False
            self._rosbag_loop_reset_reason = ""
            self._human_detector_waiting = False
            rospy.loginfo(
                "[rosbag loop] post-loop human detector frame received; "
                "switching from prior fallback to detector fusion"
            )

        if seq == self._human_detection_seq:
            if self._rosbag_loop_reset_pending:
                reason = "waiting for another post-loop detector frame"
                result = self._wait_for_rosbag_human_reacquisition(reason)
                return finish(
                    "rosbag_reacquire_same_frame",
                    result,
                    reason,
                    important=False,
                )
            reason = "waiting for next detector frame"
            continued = self._continue_with_simulation(
                reason,
                now=now,
                sim_position=sim_position,
                sim_velocity=sim_velocity,
                count_miss=False,
            )
            if continued:
                return finish(
                    "same_detector_frame_prediction",
                    True,
                    reason,
                )
            mode = "odom_jump" if self._robot_localization_jump_pending else "lost"
            final_reason = (
                self._robot_localization_jump_reason
                if self._robot_localization_jump_pending
                else "no initialized human track"
            )
            stopped = self._stop_for_human_tracking_failure(
                final_reason,
                mode=mode,
            )
            return finish(
                "same_detector_frame_stopped",
                stopped,
                final_reason,
                important=True,
            )

        self._human_detection_seq = seq
        raw_candidates = np.asarray(
            raw_candidates,
            dtype=np.float32,
        ).reshape(-1, 2)
        trace["raw_candidate_count"] = int(len(raw_candidates))
        raw_geometry = self._human_candidate_geometry(raw_candidates)
        trace["raw_candidates"] = raw_geometry[
            : self.human_detection_log_max_candidates
        ]
        trace["raw_candidates_truncated"] = bool(
            len(raw_geometry) > self.human_detection_log_max_candidates
        )

        kf_was_initialized = self._human_kf.initialized
        if kf_was_initialized:
            roi_kind = "tracking_sector"
            roi_range = (
                self.human_rear_sector_range + self.human_kf_range_margin
            )
            roi_angle = min(
                360.0,
                self.human_rear_sector_angle_deg
                + 2.0 * self.human_kf_sector_margin_deg,
            )
            candidates, retained_indices = (
                self._human_candidates_in_tracking_sector(raw_candidates)
            )
        else:
            roi_kind = "initialization_sector"
            roi_range = self.human_rear_sector_range
            roi_angle = self.human_rear_sector_angle_deg
            candidates, retained_indices = (
                self._human_candidates_in_rear_sector(raw_candidates)
            )
        candidates = np.asarray(candidates, dtype=np.float32).reshape(-1, 2)
        retained_indices = np.asarray(
            retained_indices,
            dtype=np.int64,
        ).reshape(-1)

        # Simple and deliberately aggressive rescue: if the detector produced
        # candidates but the strict rear ROI rejected all of them, retain the
        # raw candidate closest to the rear-leash prior. The adaptive
        # calibration below then pulls it back toward the expected location.
        hard_roi_candidate_count = int(len(candidates))
        roi_rescued = False
        rescued_raw_index = None
        if (
            len(candidates) == 0
            and len(raw_candidates) > 0
            and self.human_rear_prior_calibration
        ):
            rescue_prior = self._rear_leash_prior_position()
            raw_prior_errors = np.linalg.norm(
                raw_candidates - rescue_prior[None, :],
                axis=1,
            )
            rescued_raw_index = int(np.argmin(raw_prior_errors))
            candidates = raw_candidates[[rescued_raw_index]].copy()
            retained_indices = np.array(
                [rescued_raw_index], dtype=np.int64
            )
            roi_kind = f"{roi_kind}_rear_prior_rescue"
            roi_rescued = True

        trace["roi"] = {
            "kind": roi_kind,
            "range": float(roi_range),
            "total_angle_deg": float(roi_angle),
            "hard_retained_count": int(hard_roi_candidate_count),
            "rear_prior_rescued": bool(roi_rescued),
            "rescued_raw_index": rescued_raw_index,
            "retained_raw_indices": retained_indices,
        }
        trace["roi_candidate_count"] = int(len(candidates))
        roi_geometry = self._human_candidate_geometry(
            candidates,
            retained_indices=retained_indices,
        )
        trace["roi_candidates"] = roi_geometry[
            : self.human_detection_log_max_candidates
        ]
        trace["roi_candidates_truncated"] = bool(
            len(roi_geometry) > self.human_detection_log_max_candidates
        )

        calibrated_candidates, calibration_rows, rear_prior = (
            self._calibrate_human_detection_candidates(candidates)
        )
        trace["rear_prior_calibration"] = {
            "enabled": bool(self.human_rear_prior_calibration),
            "rear_prior_world_xy": rear_prior.copy(),
            "leash_length": float(self.leash_length),
            "scale": float(self.human_rear_prior_scale),
            "min_detect_weight": float(
                self.human_rear_prior_min_detect_weight
            ),
            "max_detect_weight": float(
                self.human_rear_prior_max_detect_weight
            ),
            "candidates": calibration_rows[
                : self.human_detection_log_max_candidates
            ],
            "candidates_truncated": bool(
                len(calibration_rows)
                > self.human_detection_log_max_candidates
            ),
        }

        if len(candidates) == 0:
            if self._rosbag_loop_reset_pending:
                # Keep _rosbag_loop_reacquire_seq fixed at the detector
                # sequence observed at the loop boundary. A new empty frame is
                # post-loop data, not another old frame.
                reason = "post-loop detector frame has no candidate in the ROI"
                result = self._wait_for_rosbag_human_reacquisition(reason)
                return finish(
                    "rosbag_reacquire_empty_roi",
                    result,
                    reason,
                    important=True,
                )

            reason = (
                "latest PoseArray contains no pedestrian"
                if len(raw_candidates) == 0
                else (
                    "all detector candidates rejected by ROI "
                    f"(raw={len(raw_candidates)}, roi={roi_kind})"
                )
            )
            continued = self._continue_with_simulation(
                reason,
                now=now,
                sim_position=sim_position,
                sim_velocity=sim_velocity,
                count_miss=True,
            )
            if continued:
                return finish(
                    "empty_roi_simulation",
                    True,
                    reason,
                    important=True,
                )
            final_reason = f"{reason}; detector and simulation track unavailable"
            stopped = self._stop_for_human_tracking_failure(final_reason)
            return finish(
                "empty_roi_stopped",
                stopped,
                final_reason,
                important=True,
            )

        accepted_outcome = "detector_fused"
        if not self._human_kf.initialized:
            prior_errors = np.asarray(
                [row["prior_error_m"] for row in calibration_rows],
                dtype=np.float64,
            )
            selected_idx = int(np.argmin(prior_errors))
            selected = candidates[selected_idx]
            calibrated_selected = calibrated_candidates[selected_idx]
            sim_distances = np.linalg.norm(
                calibrated_candidates - sim_position[None, :],
                axis=1,
            )
            candidate_rows = []
            for idx in range(
                min(
                    len(candidates),
                    self.human_detection_log_max_candidates,
                )
            ):
                candidate_rows.append(
                    {
                        **roi_geometry[idx],
                        **calibration_rows[idx],
                        "association_world_xy": (
                            calibrated_candidates[idx].copy()
                        ),
                        "sim_distance": float(sim_distances[idx]),
                        "selected": bool(idx == selected_idx),
                    }
                )
            trace["candidate_scores"] = candidate_rows
            trace["selected_candidate"] = {
                **roi_geometry[selected_idx],
                **calibration_rows[selected_idx],
                "roi_index": int(selected_idx),
                "association_world_xy": calibrated_selected.copy(),
                "sim_distance": float(sim_distances[selected_idx]),
                "initial_fused_position": calibrated_selected.copy(),
                "decision": "initialize",
            }
            initialized = self._initialize_human_track(
                candidates,
                timestamp=receive_stamp,
                sim_position=sim_position,
                sim_velocity=sim_velocity,
                calibrated_position=calibrated_selected,
            )
            if not initialized:
                reason = "failed to initialize human track"
                stopped = self._stop_for_human_tracking_failure(reason)
                return finish(
                    "track_initialization_failed",
                    stopped,
                    reason,
                    important=True,
                )
            accepted_outcome = "track_initialized"
        else:
            trace["kf_prediction_input"] = {
                "prediction_timestamp": float(receive_stamp),
                "dt": float(receive_stamp - self._human_kf.stamp),
            }
            self._human_kf.predict(receive_stamp)
            trace["kf_after_motion_prediction"] = (
                self._human_kf_log_snapshot()
            )
            sim_prediction_error = float(
                np.linalg.norm(self._human_kf.position - sim_position)
            )
            sim_prior_applied = (
                self.human_kf_sim_prior_max_error <= 0.0
                or sim_prediction_error
                <= self.human_kf_sim_prior_max_error
            )
            trace["simulation_prior"] = {
                "prediction_error": float(sim_prediction_error),
                "max_error": float(self.human_kf_sim_prior_max_error),
                "measurement_std": float(self.human_kf_sim_prior_std),
                "applied": bool(sim_prior_applied),
            }
            if sim_prior_applied:
                self._human_kf.correct(
                    sim_position,
                    measurement_std=self.human_kf_sim_prior_std,
                )
                self._human_kf.blend_velocity(
                    sim_velocity,
                    min(0.25, self.human_kf_sim_velocity_gain),
                )
            trace["kf_before_detector_scoring"] = (
                self._human_kf_log_snapshot()
            )

            predicted_position = self._human_kf.position
            scores = np.zeros((len(candidates),), dtype=np.float64)
            mahalanobis_sq = np.zeros((len(candidates),), dtype=np.float64)
            sim_scale = max(self.human_kf_sim_prior_std, 1e-3)
            candidate_scores = []
            for idx, candidate in enumerate(calibrated_candidates):
                residual, innovation_cov, d2 = self._human_kf.innovation(
                    candidate
                )
                mahalanobis_sq[idx] = d2
                sim_distance = float(np.linalg.norm(candidate - sim_position))
                prediction_distance = float(
                    np.linalg.norm(candidate - predicted_position)
                )
                score = d2 + 0.20 * (sim_distance / sim_scale) ** 2
                scores[idx] = score
                jump_pass = (
                    self.human_track_max_jump <= 0.0
                    or prediction_distance <= self.human_track_max_jump
                )
                gate_pass = d2 <= self.human_kf_gate
                candidate_scores.append(
                    {
                        **roi_geometry[idx],
                        **calibration_rows[idx],
                        "roi_index": int(idx),
                        "association_world_xy": candidate.copy(),
                        "residual": residual,
                        "innovation_covariance_diag": np.diag(
                            innovation_cov
                        ),
                        "mahalanobis_sq": float(d2),
                        "kf_gate": float(self.human_kf_gate),
                        "gate_pass": bool(gate_pass),
                        "jump_from_prediction": float(prediction_distance),
                        "max_jump": float(self.human_track_max_jump),
                        "jump_pass": bool(jump_pass),
                        "sim_distance": float(sim_distance),
                        "score": float(score),
                        "plausible": bool(gate_pass or jump_pass),
                    }
                )

            selected_idx = int(np.argmin(scores))
            selected_raw = candidates[selected_idx]
            selected = calibrated_candidates[selected_idx]
            selected_detect_weight = float(
                calibration_rows[selected_idx]["detect_weight"]
            )
            best_d2 = float(mahalanobis_sq[selected_idx])
            jump = float(np.linalg.norm(selected - predicted_position))
            self._human_kf_last_mahalanobis_sq = best_d2
            candidate_scores[selected_idx]["selected"] = True
            trace["candidate_scores"] = candidate_scores[
                : self.human_detection_log_max_candidates
            ]
            trace["candidate_scores_truncated"] = bool(
                len(candidate_scores) > self.human_detection_log_max_candidates
            )

            detector_plausible = best_d2 <= self.human_kf_gate
            gate_pass = bool(detector_plausible)
            jump_pass = bool(
                self.human_track_max_jump > 0.0
                and jump <= self.human_track_max_jump
            )
            if self.human_track_max_jump > 0.0:
                detector_plausible = detector_plausible or jump_pass

            trace["selected_candidate"] = {
                **candidate_scores[selected_idx],
                "raw_detected_world_xy": selected_raw.copy(),
                "decision": (
                    "accept" if detector_plausible else "reject_outlier"
                ),
                "accepted_by_gate": gate_pass,
                "accepted_by_jump_override": jump_pass and not gate_pass,
            }

            if not detector_plausible:
                reason = (
                    "detector outlier; using simulation: "
                    f"d2={best_d2:.2f}, jump={jump:.2f} m"
                )
                continued = self._continue_with_simulation(
                    reason,
                    now=now,
                    sim_position=sim_position,
                    sim_velocity=sim_velocity,
                    count_miss=True,
                    force_strong_fallback=True,
                )
                if continued:
                    return finish(
                        "detector_outlier_simulation",
                        True,
                        reason,
                        important=True,
                    )
                final_reason = f"{reason}; simulation state invalid"
                stopped = self._stop_for_human_tracking_failure(
                    final_reason
                )
                return finish(
                    "detector_outlier_stopped",
                    stopped,
                    final_reason,
                    important=True,
                )

            gate_ratio = min(
                1.0,
                best_d2 / max(self.human_kf_gate, 1e-6),
            )
            calibration_noise_scale = 1.0 / np.sqrt(
                max(selected_detect_weight, 1e-3)
            )
            adaptive_detector_std = (
                self.human_kf_measurement_std
                * (1.0 + 1.5 * gate_ratio)
                * calibration_noise_scale
            )
            trace["detector_correction"] = {
                "base_measurement_std": float(
                    self.human_kf_measurement_std
                ),
                "gate_ratio": float(gate_ratio),
                "detect_weight": float(selected_detect_weight),
                "calibration_noise_scale": float(
                    calibration_noise_scale
                ),
                "adaptive_measurement_std": float(
                    adaptive_detector_std
                ),
            }
            self._human_kf.correct(
                selected,
                measurement_std=adaptive_detector_std,
            )
            self._human_kf_last_measurement_stamp = receive_stamp
            self._human_detection_receive_stamp = receive_stamp
            self._human_kf_consecutive_misses = 0
            self._human_kf_using_prediction = False
            self._human_detection_rejected = False
            self._human_tracking_mode = "detector_fused"
            self._human_sim_fallback_start_stamp = 0.0
            self._human_last_failure_reason = ""
            trace["kf_after_detector_correction"] = (
                self._human_kf_log_snapshot()
            )

        final_sim_fused = self._fuse_simulation_state(
            timestamp=now,
            sim_position=sim_position,
            sim_velocity=sim_velocity,
            fallback=False,
        )
        trace["final_simulation_calibration"] = {
            "applied": bool(final_sim_fused),
            "timestamp": float(now),
        }
        self._sync_tracked_human_from_kf()
        if self._tracked_human_position is None:
            reason = "KF did not produce a human state"
            stopped = self._stop_for_human_tracking_failure(reason)
            return finish(
                "kf_output_missing",
                stopped,
                reason,
                important=True,
            )

        in_tracking_sector = self._human_position_in_tracking_sector(
            self._tracked_human_position
        )
        trace["final_tracking_sector_check"] = {
            "inside": bool(in_tracking_sector),
            "tracked_position": self._tracked_human_position.copy(),
        }
        if not in_tracking_sector:
            reason = "fused human left relaxed detector sector"
            continued = self._continue_with_simulation(
                reason,
                now=now,
                sim_position=sim_position,
                sim_velocity=sim_velocity,
                count_miss=True,
                force_strong_fallback=True,
            )
            if continued:
                return finish(
                    "fused_track_outside_roi_simulation",
                    True,
                    reason,
                    important=True,
                )
            final_reason = f"{reason}; simulation state invalid"
            stopped = self._stop_for_human_tracking_failure(final_reason)
            return finish(
                "fused_track_outside_roi_stopped",
                stopped,
                final_reason,
                important=True,
            )

        self._apply_detector_human_state()
        self._set_human_detector_waiting(
            False,
            "detector/simulation fused target",
        )
        return finish(
            accepted_outcome,
            True,
            "detector candidate accepted",
            important=accepted_outcome == "track_initialized",
        )

    def _set_human_source(self, source: str) -> None:
        source = normalize_human_source(source)
        if source == self.human_source:
            return

        previous_source = self.human_source
        self.human_source = source
        self._human_kf.reset()
        self._tracked_human_position = None
        self._tracked_human_velocity = np.zeros((2,), dtype=np.float32)
        self._human_detection_seq = -1
        self._human_detection_receive_stamp = 0.0
        self._human_kf_last_measurement_stamp = 0.0
        self._human_kf_consecutive_misses = 0
        self._human_kf_using_prediction = False
        self._human_kf_last_mahalanobis_sq = float("inf")
        self._human_detection_rejected = False
        self._human_detector_waiting = False
        self._human_tracking_mode = "uninitialized"
        self._human_sim_fallback_start_stamp = 0.0
        self._human_last_failure_reason = ""
        self._robot_localization_jump_pending = False
        self._robot_localization_jump_reason = ""
        self._rosbag_loop_reset_pending = False
        self._rosbag_loop_reacquire_seq = -1
        self._rosbag_loop_reset_reason = ""

        detector_ready = True
        if source == "detector":
            detector_ready = self._refresh_human_from_detector()
        elif hasattr(self.physics.human, "velocity"):
            # Continue simulation from the most recently displayed position.
            self.physics.human.velocity = np.zeros((2,), dtype=np.float32)

        self.interaction_segmenter.reset()
        self._reset_runtime_caches()
        self._seed_obs_history(
            self.physics.robot.position,
            self.physics.human.position,
        )
        suffix = ""
        if source == "detector":
            suffix = " (ready)" if detector_ready else " (waiting; robot stopped)"
        print(f"Human input: {source}{suffix}")
        self._log_event(
            "human_source_changed",
            {
                "previous_source": previous_source,
                "human_source": source,
                "detector_ready": bool(detector_ready),
                "human_pos": self.physics.human.position.tolist(),
            },
        )

    def _toggle_human_source(self) -> None:
        next_source = "detector" if self.human_source == "sim" else "sim"
        self._set_human_source(next_source)

    def _reset_runtime_caches(self):
        self.cached_action_seq = None
        self.cached_nominal_delta_seq = None
        self.cached_safe_delta_seq = None
        self.cached_safety_info_seq = None
        self.cached_interaction_labels_seq = None
        self.cached_uses_stashed_compliance_plan = False
        self.cached_action_idx = 0
        self.frames_since_inference = 0
        self.cached_control = (0.0, 0.0)
        self.current_action = None
        self.current_delta = None
        self.latest_nominal_heading_delta = None
        self.current_speed_scale = 1.0
        self.planned_path = None
        self.nominal_planned_path = None
        self.safe_planned_path = None
        self.last_compliance_stats = {
            "applied": False,
            "safety_applied": False,
            "modified_steps": 0,
            "safety_modified_steps": 0,
            "total_steps": 0,
            "mean_shift": 0.0,
            "mean_action_shift": 0.0,
            "constraint_count": 0,
            "min_clearance": float("inf"),
        }
        self.using_stashed_compliance_plan = False

    @staticmethod
    def _compliance_mask_from_labels(labels: np.ndarray) -> np.ndarray:
        return np.isin(
            np.asarray(labels, dtype=str),
            np.array(["leash", "tether"], dtype=str),
        )

    def _stash_guide_action_seq(self, action_seq: np.ndarray, source: str) -> None:
        action_seq = np.asarray(action_seq, dtype=np.float32)
        if action_seq.ndim != 2 or len(action_seq) == 0:
            return
        front_half_len = max(1, len(action_seq) // 2)
        self.stashed_guide_action_seq = action_seq.copy()
        self.stashed_guide_cursor = 0
        self.last_stashed_compliance_info = {
            "source": str(source),
            "stashed_len": int(len(action_seq)),
            "stashed_front_half_len": int(front_half_len),
        }

    def _stash_current_cached_guide_plan(self) -> None:
        if self.cached_action_seq is None or len(self.cached_action_seq) == 0:
            return
        if self._current_interaction_label() != "guide":
            return
        start = min(max(0, int(self.cached_action_idx)), len(self.cached_action_seq) - 1)
        remaining = np.asarray(self.cached_action_seq[start:], dtype=np.float32)
        if len(remaining) == 0:
            remaining = np.asarray(self.cached_action_seq, dtype=np.float32)
        self._stash_guide_action_seq(remaining, source="cached_remaining")

    def _stashed_compliance_action_seq(
        self,
        requested_len: int,
        fallback_action_seq: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        requested_len = max(1, int(requested_len))
        fallback_action_seq = np.asarray(fallback_action_seq, dtype=np.float32)
        if (
            self.stashed_guide_action_seq is not None
            and len(self.stashed_guide_action_seq) > 0
        ):
            stash = np.asarray(self.stashed_guide_action_seq, dtype=np.float32)
            half_len = max(1, len(stash) // 2)
            start = max(0, int(self.stashed_guide_cursor))
            if start >= half_len:
                segment = np.zeros((1, stash.shape[1]), dtype=np.float32)
                start = half_len
            else:
                end = min(half_len, start + half_len)
                segment = stash[start:end].copy()
            info = {
                "using_stashed_compliance_plan": bool(start < half_len),
                "stashed_source": str(self.last_stashed_compliance_info.get("source", "guide")),
                "stashed_len": int(len(stash)),
                "stashed_cursor": int(start),
                "stashed_used_len": int(len(segment)),
                "stashed_front_half_len": int(half_len),
            }
            return segment.astype(np.float32), info

        action_dim = (
            int(fallback_action_seq.shape[1])
            if fallback_action_seq.ndim == 2 and fallback_action_seq.shape[1] > 0
            else int(self.action_dim)
        )
        fallback = np.zeros((requested_len, action_dim), dtype=np.float32)
        info = {
            "using_stashed_compliance_plan": False,
            "stashed_source": "fallback_stop_no_stash",
            "stashed_len": 0,
            "stashed_cursor": 0,
            "stashed_used_len": int(len(fallback)),
            "stashed_front_half_len": 0,
        }
        return fallback.astype(np.float32), info

    def _start_recording(self):
        if self.storage is None:
            return
        self.recording = True
        self.storage.start_recording()
        if self.bre_timer_start_frame is None:
            self.bre_timer_start_frame = self.frame_count
            print("Timed bre toggle timer started.")
        self._last_recorded_cloud_seq = None
        if self.scorer:
            self.scorer.reset()
        print("Recording started...")

    def _stop_recording(self):
        if self.storage is None:
            return
        self.recording = False
        print(f"Recording paused. Points: {self.storage.get_num_points()}")

    def _record_frame(self, robot_state, human_state):
        if self.storage is None:
            return
        timestamp = float(self.frame_count * self.sim_dt)
        # Keep the simulator's hidden/manual state as the recorded truth. The
        # segmentation result is the online estimate used by the controller.
        state = 0 if self.bre else 2
        self.storage.record_frame(
            robot_state.position,
            human_state.position,
            timestamp=timestamp,
            state=state,
        )

    def _save_episode(self):
        if self.storage is None:
            return
        if self.storage.get_num_points() == 0:
            print("No data to save!")
            return
        scores = self.scorer.get_scores() if self.scorer else {}
        extra_metadata = {"scores": scores, "source": "real_robot_hybrid"}
        episode_dir = self.storage.save_episode(
            reference_path=self.current_path_data["path"],
            start_pos=self.current_path_data["start"],
            end_pos=self.current_path_data["end"],
            obstacles=self.current_path_data.get("obstacles"),
            segment_obstacles=self.current_path_data.get("segment_obstacles"),
            extra_metadata=extra_metadata,
        )
        print(f"Saved planning episode: {episode_dir}")
        self.recording = False
        self.storage.clear()

    def _supported_pointcloud_modes(self) -> list[str]:
        return ["live"] if self.observation_mode == "mid360" else ["off", "live"]

    @staticmethod
    def _json_safe(value):
        """Convert NumPy/non-finite values to strict JSON-compatible values."""
        if isinstance(value, np.ndarray):
            return [ModelPlanner._json_safe(v) for v in value.tolist()]
        if isinstance(value, np.generic):
            return ModelPlanner._json_safe(value.item())
        if isinstance(value, dict):
            return {
                str(key): ModelPlanner._json_safe(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [ModelPlanner._json_safe(item) for item in value]
        if isinstance(value, float):
            return float(value) if np.isfinite(value) else None
        if isinstance(value, (str, int, bool)) or value is None:
            return value
        return str(value)

    def _human_kf_log_snapshot(self) -> dict:
        if not self._human_kf.initialized:
            return {
                "initialized": False,
                "stamp": None,
                "state": None,
                "covariance_diag": None,
            }
        return {
            "initialized": True,
            "stamp": float(self._human_kf.stamp),
            "state": self._human_kf.state,
            "position": self._human_kf.position,
            "velocity": self._human_kf.velocity,
            "covariance_diag": self._human_kf.covariance_diag,
        }

    def _human_candidate_geometry(
        self,
        candidates_world: np.ndarray,
        *,
        retained_indices: Optional[np.ndarray] = None,
    ) -> list[dict]:
        candidates = np.asarray(
            candidates_world,
            dtype=np.float32,
        ).reshape(-1, 2)
        if len(candidates) == 0:
            return []
        robot_position = np.asarray(
            self.physics.robot.position,
            dtype=np.float32,
        ).reshape(2)
        relative = candidates - robot_position[None, :]
        distances = np.linalg.norm(relative, axis=1)
        bearings = np.arctan2(relative[:, 1], relative[:, 0])
        rear_heading = wrap_angle(float(self.physics.robot.heading) + np.pi)
        signed_rear_error = np.array(
            [wrap_angle(float(angle) - rear_heading) for angle in bearings],
            dtype=np.float64,
        )
        if retained_indices is None:
            raw_indices = np.arange(len(candidates), dtype=np.int64)
        else:
            raw_indices = np.asarray(retained_indices, dtype=np.int64).reshape(-1)
        rows = []
        for idx, candidate in enumerate(candidates):
            rows.append(
                {
                    "candidate_index": int(idx),
                    "raw_index": int(raw_indices[idx]),
                    "world_xy": candidate,
                    "robot_distance": float(distances[idx]),
                    "bearing_world_deg": float(np.rad2deg(bearings[idx])),
                    "rear_angle_error_deg": float(
                        np.rad2deg(signed_rear_error[idx])
                    ),
                }
            )
        return rows

    def _new_human_detection_trace(
        self,
        *,
        now: float,
        sim_position: np.ndarray,
        sim_velocity: np.ndarray,
    ) -> dict:
        status = self.ros_io.human_detection_status()
        odom_replay = self.ros_io.odom_replay_status()
        self._human_detection_log_refresh_idx += 1
        return {
            "event": "human_detection_diagnostic",
            "refresh_index": int(self._human_detection_log_refresh_idx),
            "frame": int(self.frame_count),
            "data_step": int(self.data_step_idx),
            "ros_time": float(now),
            "wall_time": datetime.now().isoformat(timespec="milliseconds"),
            "_perf_start": time.perf_counter(),
            "robot": {
                "position": self.physics.robot.position.copy(),
                "heading_rad": float(self.physics.robot.heading),
                "heading_deg": float(np.rad2deg(self.physics.robot.heading)),
            },
            "simulation_human_before": {
                "position": np.asarray(sim_position, dtype=np.float32),
                "velocity": np.asarray(sim_velocity, dtype=np.float32),
            },
            "rear_prior_calibration_config": {
                "enabled": bool(self.human_rear_prior_calibration),
                "leash_length": float(self.leash_length),
                "scale": float(self.human_rear_prior_scale),
                "min_detect_weight": float(
                    self.human_rear_prior_min_detect_weight
                ),
                "max_detect_weight": float(
                    self.human_rear_prior_max_detect_weight
                ),
            },
            "detector_status": status,
            "odom_replay": odom_replay,
            "rosbag": {
                "loop_mode": bool(self.rosbag_loop_mode),
                "reset_pending": bool(self._rosbag_loop_reset_pending),
                "reset_count": int(self._rosbag_loop_reset_count),
                "reacquire_seq": int(self._rosbag_loop_reacquire_seq),
                "reset_reason": str(self._rosbag_loop_reset_reason),
            },
            "tracking_before": {
                "mode": str(self._human_tracking_mode),
                "detector_seq": int(self._human_detection_seq),
                "consecutive_misses": int(
                    self._human_kf_consecutive_misses
                ),
                "using_prediction": bool(self._human_kf_using_prediction),
                "detector_rejected": bool(self._human_detection_rejected),
                "last_measurement_stamp": float(
                    self._human_kf_last_measurement_stamp
                ),
                "last_mahalanobis_sq": float(
                    self._human_kf_last_mahalanobis_sq
                ),
                "last_failure_reason": str(self._human_last_failure_reason),
                "tracked_position": (
                    None
                    if self._tracked_human_position is None
                    else self._tracked_human_position.copy()
                ),
                "tracked_velocity": self._tracked_human_velocity.copy(),
            },
            "kf_before": self._human_kf_log_snapshot(),
        }

    def _finish_human_detection_trace(
        self,
        trace: dict,
        *,
        outcome: str,
        success: bool,
        reason: str = "",
        important: bool = False,
        extra: Optional[dict] = None,
    ) -> bool:
        perf_start = float(trace.pop("_perf_start", time.perf_counter()))
        trace["duration_ms"] = (
            time.perf_counter() - perf_start
        ) * 1000.0
        trace["outcome"] = str(outcome)
        trace["success"] = bool(success)
        trace["reason"] = str(reason)
        if extra:
            trace.update(extra)
        trace["kf_after"] = self._human_kf_log_snapshot()
        trace["tracking_after"] = {
            "mode": str(self._human_tracking_mode),
            "detector_seq": int(self._human_detection_seq),
            "consecutive_misses": int(self._human_kf_consecutive_misses),
            "using_prediction": bool(self._human_kf_using_prediction),
            "detector_rejected": bool(self._human_detection_rejected),
            "waiting": bool(self._human_detector_waiting),
            "last_measurement_stamp": float(
                self._human_kf_last_measurement_stamp
            ),
            "last_mahalanobis_sq": float(
                self._human_kf_last_mahalanobis_sq
            ),
            "last_failure_reason": str(self._human_last_failure_reason),
            "tracked_position": (
                None
                if self._tracked_human_position is None
                else self._tracked_human_position.copy()
            ),
            "tracked_velocity": self._tracked_human_velocity.copy(),
            "physics_human_position": self.physics.human.position.copy(),
            "physics_human_velocity": np.asarray(
                getattr(
                    self.physics.human,
                    "velocity",
                    np.zeros((2,), dtype=np.float32),
                ),
                dtype=np.float32,
            ),
        }

        self._human_detection_log_outcomes[outcome] = (
            self._human_detection_log_outcomes.get(outcome, 0) + 1
        )
        should_write = (
            self.human_detection_log_fp is not None
            and (
                important
                or self.human_detection_log_interval <= 1
                or self._human_detection_log_refresh_idx
                % self.human_detection_log_interval
                == 0
            )
        )
        if should_write:
            self._human_detection_log_record_idx += 1
            trace["record_index"] = int(
                self._human_detection_log_record_idx
            )
            payload = self._json_safe(trace)
            self.human_detection_log_fp.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
            self.human_detection_log_fp.flush()

        if self.human_detection_console:
            status = trace.get("detector_status", {})
            selected = trace.get("selected_candidate") or {}
            d2 = selected.get("mahalanobis_sq")
            jump = selected.get("jump_from_prediction")
            detect_weight = selected.get("detect_weight")
            prior_error = selected.get("prior_error_m")
            d2_text = "-" if d2 is None else f"{float(d2):.2f}"
            jump_text = "-" if jump is None else f"{float(jump):.2f}"
            weight_text = (
                "-"
                if detect_weight is None
                else f"{float(detect_weight):.2f}"
            )
            prior_error_text = (
                "-"
                if prior_error is None
                else f"{float(prior_error):.2f}"
            )
            print(
                "[human diagnostic] "
                f"frame={self.frame_count} seq={status.get('seq', -1)} "
                f"raw={trace.get('raw_candidate_count', 0)} "
                f"roi={trace.get('roi_candidate_count', 0)} "
                f"outcome={outcome} d2={d2_text} jump={jump_text} "
                f"prior_err={prior_error_text} w_det={weight_text} "
                f"mode={self._human_tracking_mode}"
            )
        return bool(success)

    def _log_human_system_event(
        self,
        name: str,
        extra: Optional[dict] = None,
    ) -> None:
        if self.human_detection_log_fp is None:
            return
        self._human_detection_log_record_idx += 1
        payload = {
            "event": "human_detection_system",
            "name": str(name),
            "record_index": int(self._human_detection_log_record_idx),
            "frame": int(getattr(self, "frame_count", 0)),
            "data_step": int(getattr(self, "data_step_idx", 0)),
            "ros_time": float(rospy.get_time()),
            "wall_time": datetime.now().isoformat(timespec="milliseconds"),
        }
        if extra:
            payload.update(extra)
        self.human_detection_log_fp.write(
            json.dumps(
                self._json_safe(payload),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )
        self.human_detection_log_fp.flush()

    def _log_event(self, name: str, extra: Optional[dict] = None):
        if self.log_fp is None:
            return
        payload = {
            "event": name,
            "frame": int(self.frame_count),
            "time_sec": float(self.frame_count * self.sim_dt),
            "data_step": int(self.data_step_idx),
            "data_time_sec": float(self.data_step_idx * self.data_dt),
        }
        if extra:
            payload.update(extra)
        self.log_fp.write(json.dumps(payload, ensure_ascii=True) + "\n")
        self.log_fp.flush()

    def _synchronize_timing_device(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _run_eval_safety_variant(
        self,
        action_seq: np.ndarray,
        safety_mode: str,
    ) -> tuple[np.ndarray, list[dict], dict, float]:
        saved_mode = self.safety_mode
        saved_stats = copy.deepcopy(self.last_safety_stats)
        saved_rng_state = np.random.get_state()
        try:
            self.safety_mode = normalize_safety_mode(safety_mode)
            start = time.perf_counter()
            if self.action_mode == "forward_heading":
                _nominal_deltas, safe_deltas, safety_infos = (
                    self._apply_forward_heading_safety_filter(action_seq)
                )
            else:
                safe_actions = self._apply_safety_filter(action_seq)
                safe_deltas = self._action_seq_to_nominal_delta_seq(safe_actions)
                safety_infos = []
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            stats = copy.deepcopy(self.last_safety_stats)
        finally:
            self.safety_mode = saved_mode
            self.last_safety_stats = saved_stats
            np.random.set_state(saved_rng_state)
        return safe_deltas, safety_infos, stats, elapsed_ms

    def _eval_paths_from_deltas(
        self,
        delta_seq: np.ndarray,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        saved_rng_state = np.random.get_state()
        try:
            sim = copy.deepcopy(self.physics)
            robot_points: list[np.ndarray] = []
            human_points: list[np.ndarray] = []
            for delta in np.asarray(delta_seq, dtype=np.float32):
                forward, turn, _speed_scale = self._delta_to_safe_control(
                    delta,
                    sim.robot.heading,
                    dt=self.data_dt,
                )
                sim.set_control(forward, turn, self.bre)
                for _ in range(int(self.frame_stride)):
                    robot_state, human_state = sim.step()
                    robot_points.append(robot_state.position.copy())
                    human_points.append(human_state.position.copy())
            robot_path = np.stack(robot_points, axis=0) if robot_points else None
            human_path = np.stack(human_points, axis=0) if human_points else None
            return robot_path, human_path
        finally:
            np.random.set_state(saved_rng_state)

    def _serialize_eval_safety_steps(self, safety_infos: list[dict]) -> list[dict]:
        rows = []
        for info in safety_infos:
            min_clearance = float(info.get("min_clearance", float("inf")))
            rows.append(
                {
                    "modified": bool(info.get("modified", False)),
                    "shift": round(float(info.get("shift", 0.0)), 6),
                    "constraint_count": int(info.get("constraint_count", 0)),
                    "min_clearance": (
                        round(min_clearance, 6) if np.isfinite(min_clearance) else "inf"
                    ),
                    "backoff_applied": bool(info.get("backoff_applied", False)),
                    "resolution_stage": str(info.get("resolution_stage", "none")),
                }
            )
        return rows

    def _serialize_eval_safety_stats(self, stats: dict) -> dict:
        min_clearance = float(stats.get("min_clearance", float("inf")))
        return {
            "modified_steps": int(stats.get("modified_steps", 0)),
            "total_steps": int(stats.get("total_steps", 0)),
            "mean_shift": round(float(stats.get("mean_shift", 0.0)), 6),
            "constraint_count": int(stats.get("constraint_count", 0)),
            "min_clearance": (
                round(min_clearance, 6) if np.isfinite(min_clearance) else "inf"
            ),
        }

    def _write_planning_eval(self, action_seq: np.ndarray, diffusion_time_ms: float):
        if self.eval_fp is None:
            return

        action_seq = np.asarray(action_seq, dtype=np.float32)
        saved_rng_state = np.random.get_state()
        try:
            origin_deltas = self._action_seq_to_nominal_delta_seq(action_seq)
        finally:
            np.random.set_state(saved_rng_state)
        origin_robot_path, origin_human_path = self._eval_paths_from_deltas(origin_deltas)

        robot_deltas, robot_infos, robot_stats, robot_qp_time_ms = (
            self._run_eval_safety_variant(action_seq, "robot_qp")
        )
        robot_safe_robot_path, robot_safe_human_path = self._eval_paths_from_deltas(robot_deltas)

        (
            human_robot_deltas,
            human_robot_infos,
            human_robot_stats,
            human_robot_qp_time_ms,
        ) = self._run_eval_safety_variant(action_seq, "human_robot_qp")
        human_robot_safe_robot_path, human_robot_safe_human_path = (
            self._eval_paths_from_deltas(human_robot_deltas)
        )

        diffusion_time_ms = float(diffusion_time_ms)
        obstacles, segments = self._safety_obstacle_inputs()
        stats["input_point_obstacle_count"] = int(
            len(obstacles) if obstacles is not None else 0
        )
        stats["input_segment_obstacle_count"] = int(
            len(segments) if segments is not None else 0
        )
        payload = {
            "event": "planning_eval",
            "planning_index": int(self.eval_planning_idx),
            "frame": int(self.frame_count),
            "time_sec": float(self.frame_count * self.sim_dt),
            "data_step": int(self.data_step_idx),
            "data_time_sec": float(self.data_step_idx * self.data_dt),
            "action_mode": self.action_mode,
            "executed_safety_mode": self.safety_mode,
            "robot_pos": self._rounded_list(self.physics.robot.position, decimals=6),
            "human_pos": self._rounded_list(self.physics.human.position, decimals=6),
            "robot_heading": round(float(self.physics.robot.heading), 6),
            "robot_radius": round(float(self.physics.robot_radius), 6),
            "human_radius": round(float(self.physics.human_radius), 6),
            "obstacles": self._serialize_obstacles(obstacles, decimals=6),
            "segment_obstacles": self._serialize_segments(segments, decimals=6),
            "origin_planning": {
                "diffusion_time_ms": round(diffusion_time_ms, 3),
                "qp_time_ms": 0.0,
                "total_time_ms": round(diffusion_time_ms, 3),
                "action_seq": self._rounded_list(action_seq, decimals=6),
                "delta_seq": self._rounded_list(origin_deltas, decimals=6),
                "path": self._rounded_list(origin_robot_path, decimals=6),
                "robot_path": self._rounded_list(origin_robot_path, decimals=6),
                "human_path": self._rounded_list(origin_human_path, decimals=6),
            },
            "robot_safe_planning": {
                "diffusion_time_ms": round(diffusion_time_ms, 3),
                "qp_time_ms": round(robot_qp_time_ms, 3),
                "total_time_ms": round(diffusion_time_ms + robot_qp_time_ms, 3),
                "delta_seq": self._rounded_list(robot_deltas, decimals=6),
                "path": self._rounded_list(robot_safe_robot_path, decimals=6),
                "robot_path": self._rounded_list(robot_safe_robot_path, decimals=6),
                "human_path": self._rounded_list(robot_safe_human_path, decimals=6),
                "safety_summary": self._serialize_eval_safety_stats(robot_stats),
                "qp_steps": self._serialize_eval_safety_steps(robot_infos),
            },
            "human_robot_safe_planning": {
                "diffusion_time_ms": round(diffusion_time_ms, 3),
                "qp_time_ms": round(human_robot_qp_time_ms, 3),
                "total_time_ms": round(diffusion_time_ms + human_robot_qp_time_ms, 3),
                "delta_seq": self._rounded_list(human_robot_deltas, decimals=6),
                "path": self._rounded_list(human_robot_safe_robot_path, decimals=6),
                "robot_path": self._rounded_list(human_robot_safe_robot_path, decimals=6),
                "human_path": self._rounded_list(human_robot_safe_human_path, decimals=6),
                "safety_summary": self._serialize_eval_safety_stats(human_robot_stats),
                "qp_steps": self._serialize_eval_safety_steps(human_robot_infos),
            },
        }
        self.eval_fp.write(json.dumps(payload, ensure_ascii=True) + "\n")
        self.eval_fp.flush()
        self.eval_planning_idx += 1

    def _log_step(
        self,
        frame_idx: int,
        robot_state,
        human_state,
        action: Optional[np.ndarray],
        delta: Optional[np.ndarray],
        forward: float,
        turn: float,
    ):
        if self.log_fp is None:
            return
        if frame_idx % self.log_interval != 0:
            return
        scores = self.scorer.get_scores() if self.scorer else {}
        payload = {
            "event": "step",
            "frame": int(frame_idx),
            "time_sec": float(frame_idx * self.sim_dt),
            "real_time_sec": float(frame_idx / max(1.0, self.fps)),
            "data_step": int(self.data_step_idx),
            "data_time_sec": float(self.data_step_idx * self.data_dt),
            "is_data_step": bool(frame_idx % self.frame_stride == 0),
            "robot_pos": [float(robot_state.position[0]), float(robot_state.position[1])],
            "robot_vel": [float(robot_state.velocity[0]), float(robot_state.velocity[1])],
            "human_pos": [float(human_state.position[0]), float(human_state.position[1])],
            "heading": float(robot_state.heading),
            "forward": float(forward),
            "turn": float(turn),
            "speed_scale": float(self.current_speed_scale),
            "use_policy": bool(self.use_policy),
            "paused": bool(self.paused),
            "scores": scores,
        }
        if self.robot_frame:
            rel = human_state.position - robot_state.position
            cos_h = float(np.cos(robot_state.heading))
            sin_h = float(np.sin(robot_state.heading))
            hx = cos_h * float(rel[0]) + sin_h * float(rel[1])
            hy = -sin_h * float(rel[0]) + cos_h * float(rel[1])
            payload["human_rel"] = [hx, hy]
        if action is not None:
            payload["action"] = [float(action[0]), float(action[1])]
        if delta is not None:
            payload["delta_world"] = [float(delta[0]), float(delta[1])]
        self.log_fp.write(json.dumps(payload, ensure_ascii=True) + "\n")
        self.log_fp.flush()

    def _rounded_list(self, values: Optional[np.ndarray], decimals: int = 4) -> Optional[list]:
        if values is None:
            return None
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0:
            return []
        return np.round(arr, decimals).tolist()

    def _serialize_obstacles(self, obstacles: Optional[np.ndarray], decimals: int = 4) -> list[dict]:
        if obstacles is None:
            return []
        rows: list[dict] = []
        for idx, obs in enumerate(obstacles):
            if isinstance(obs, dict):
                x = float(obs.get("x", 0.0))
                y = float(obs.get("y", 0.0))
                r = float(obs.get("r", 0.0))
            else:
                x = float(obs[0])
                y = float(obs[1])
                r = float(obs[2])
            rows.append(
                {
                    "idx": int(idx),
                    "x": round(x, decimals),
                    "y": round(y, decimals),
                    "r": round(r, decimals),
                }
            )
        return rows

    def _serialize_segments(self, segments: Optional[np.ndarray], decimals: int = 4) -> list[dict]:
        if segments is None:
            return []
        rows: list[dict] = []
        for idx, seg in enumerate(segments):
            if isinstance(seg, dict):
                if "p1" in seg and "p2" in seg:
                    p1 = np.asarray(seg["p1"], dtype=np.float32)
                    p2 = np.asarray(seg["p2"], dtype=np.float32)
                    x1, y1 = float(p1[0]), float(p1[1])
                    x2, y2 = float(p2[0]), float(p2[1])
                else:
                    x1 = float(seg.get("x1", 0.0))
                    y1 = float(seg.get("y1", 0.0))
                    x2 = float(seg.get("x2", 0.0))
                    y2 = float(seg.get("y2", 0.0))
            else:
                x1 = float(seg[0])
                y1 = float(seg[1])
                x2 = float(seg[2])
                y2 = float(seg[3])
            rows.append(
                {
                    "idx": int(idx),
                    "x1": round(x1, decimals),
                    "y1": round(y1, decimals),
                    "x2": round(x2, decimals),
                    "y2": round(y2, decimals),
                }
            )
        return rows

    def _serialize_safety_info(self, safety_info: Optional[dict], decimals: int = 4) -> dict:
        if safety_info is None:
            return {}

        payload = {
            "modified": bool(safety_info.get("modified", False)),
            "shift": round(float(safety_info.get("shift", 0.0)), decimals),
            "constraint_count": int(safety_info.get("constraint_count", 0)),
            "min_clearance": round(float(safety_info.get("min_clearance", float("inf"))), decimals)
            if np.isfinite(float(safety_info.get("min_clearance", float("inf"))))
            else "inf",
        }

        scalar_keys = [
            "nominal_delta_norm",
            "qp_delta_norm",
            "final_delta_norm",
            "robot_heading",
            "qp_total_constraint_count",
            "qp_candidate_count",
            "backoff_scale",
            "stop_clearance_threshold",
        ]
        for key in scalar_keys:
            if key in safety_info and safety_info[key] is not None:
                payload[key] = round(float(safety_info[key]), decimals)

        int_keys = ["qp_constraint_count"]
        for key in int_keys:
            if key in safety_info and safety_info[key] is not None:
                payload[key] = int(safety_info[key])

        bool_keys = [
            "protect_human",
            "qp_modified",
            "qp_ref_feasible",
            "collision_after_qp",
            "backoff_applied",
            "stop_triggered",
        ]
        for key in bool_keys:
            if key in safety_info and safety_info[key] is not None:
                payload[key] = bool(safety_info[key])

        list_keys = [
            "robot_pos",
            "human_pos",
            "nominal_delta",
            "qp_delta",
            "final_delta",
            "nominal_preview_robot_pos",
            "nominal_preview_human_pos",
        ]
        for key in list_keys:
            if key in safety_info:
                payload[key] = self._rounded_list(safety_info.get(key), decimals=decimals)

        text_keys = [
            "resolution_stage",
            "stop_reason",
            "qp_best_candidate_kind",
        ]
        for key in text_keys:
            if key in safety_info and safety_info[key] is not None:
                payload[key] = str(safety_info[key])

        if "qp_best_candidate_constraints" in safety_info:
            payload["qp_best_candidate_constraints"] = [
                int(v) for v in safety_info.get("qp_best_candidate_constraints", [])
            ]

        if "backoff_attempts" in safety_info:
            payload["backoff_attempts"] = [
                {
                    "scale": round(float(item.get("scale", 0.0)), decimals),
                    "collided": bool(item.get("collided", False)),
                    "delta": self._rounded_list(item.get("delta"), decimals=decimals),
                }
                for item in safety_info.get("backoff_attempts", [])
            ]

        if "qp_selected_constraints" in safety_info:
            payload["qp_selected_constraints"] = [
                {
                    "index": int(item.get("index", 0)),
                    "source": str(item.get("source", "")),
                    "clearance": round(float(item.get("clearance", 0.0)), decimals),
                    "predicted_clearance": round(
                        float(item.get("predicted_clearance", 0.0)), decimals
                    ),
                    "h": round(float(item.get("h", 0.0)), decimals),
                    "g": self._rounded_list(item.get("g"), decimals=decimals),
                    "ref_violation": round(float(item.get("ref_violation", 0.0)), decimals),
                }
                for item in safety_info.get("qp_selected_constraints", [])
            ]
        return payload

    def _log_robot_qp_step(
        self,
        action_idx: int,
        action: Optional[np.ndarray],
        delta: Optional[np.ndarray],
        forward: float,
        turn: float,
        speed_scale: float,
        safety_info: Optional[dict],
    ):
        if (
            not self.debug_qp_log
            or self.log_fp is None
            or self.safety_mode != "robot_qp"
            or self.action_mode != "forward_heading"
        ):
            return
        payload = {
            "action_idx": int(action_idx),
            "cached_action_len": int(len(self.cached_action_seq)) if self.cached_action_seq is not None else 0,
            "cached_safe_len": int(len(self.cached_safe_delta_seq)) if self.cached_safe_delta_seq is not None else 0,
            "robot_pos": self._rounded_list(self.physics.robot.position),
            "human_pos": self._rounded_list(self.physics.human.position),
            "robot_heading": round(float(self.physics.robot.heading), 4),
            "action": self._rounded_list(action),
            "executed_delta": self._rounded_list(delta),
            "forward_cmd": round(float(forward), 4),
            "turn_cmd": round(float(turn), 4),
            "speed_scale": round(float(speed_scale), 4),
            "safety": self._serialize_safety_info(safety_info),
        }
        self._log_event("robot_qp_step", payload)

    def _handle_input(self):
        """Handle keyboard input."""
        if self.visualizer is None:
            return
        for event in self.visualizer.get_events():
            if event.type == pygame.QUIT:
                self.running = False

            self.visualizer.handle_event(event)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    if self.collect_enabled:
                        if self.recording:
                            self._stop_recording()
                        else:
                            self._start_recording()
                    else:
                        self.paused = not self.paused
                elif event.key == pygame.K_s:
                    if self.collect_enabled:
                        self._save_episode()
                elif event.key == pygame.K_h:
                    self._toggle_human_source()
                elif event.key == pygame.K_b:
                    detector_is_fallback = not self.interaction_segmenter.has_label
                    if detector_is_fallback and not self.bre:
                        self._stash_current_cached_guide_plan()
                    self.bre = not self.bre
                    if detector_is_fallback:
                        self._reset_runtime_caches()
                    print(
                        "Hidden interaction state toggled: "
                        f"bre={self.bre}, detected={self._current_interaction_label()}"
                    )
                    self._log_event(
                        "manual_bre_toggle",
                        {
                            "bre": bool(self.bre),
                            "interaction_label": self._current_interaction_label(),
                        },
                    )
                elif event.key == pygame.K_r:
                    self._reset_position()
                    print("Position reset")
                elif event.key == pygame.K_m:
                    self._cycle_safety_mode()
                elif event.key == pygame.K_c:
                    self._toggle_range_source()
                elif event.key == pygame.K_n:
                    self._generate_new_path()
                elif event.key == pygame.K_p:
                    if self.policy is not None:
                        self.use_policy = not self.use_policy
                        mode = "policy" if self.use_policy else "manual"
                    else:
                        mode = "manual (no checkpoint)"
                    print(f"Control mode: {mode}")

    def _get_manual_control(self) -> Tuple[float, float]:
        keys = pygame.key.get_pressed()
        forward = 0.0
        turn = 0.0

        if keys[pygame.K_UP]:
            forward = 1.0
        elif keys[pygame.K_DOWN]:
            forward = -1.0

        if keys[pygame.K_LEFT]:
            turn = 1.0
        elif keys[pygame.K_RIGHT]:
            turn = -1.0
        return forward, turn

    def _predict_action(self) -> np.ndarray:
        obs_seq = np.stack(self.obs_history, axis=0)
        obs_tensor = torch.from_numpy(obs_seq).to(
            device=self.device, dtype=self.policy.dtype
        )[None, ...]
        obs_dict = {"obs": obs_tensor}
        with torch.no_grad():
            action_dict = self.policy.predict_action(obs_dict)
        action_seq = action_dict["action"].detach().cpu().numpy()[0]
        return action_seq.astype(np.float32)

    def _path_heading_delta(self, path: Optional[np.ndarray]) -> Optional[float]:
        if path is None or len(path) < 2:
            return None
        path = np.asarray(path, dtype=np.float32)
        for idx in range(len(path) - 1, 0, -1):
            delta = path[idx] - path[idx - 1]
            if float(np.linalg.norm(delta)) > 1e-6:
                final_heading = float(np.arctan2(delta[1], delta[0]))
                return float(wrap_angle(final_heading - self.physics.robot.heading))
        return None

    def _apply_bre_compliance_control(
        self,
        action_seq: np.ndarray,
        obstacles: Optional[np.ndarray] = None,
        segment_obstacles: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if obstacles is None and self.current_path_data is not None:
            obstacles = self.current_path_data.get("obstacles")
        if segment_obstacles is None and self.current_path_data is not None:
            segment_obstacles = self.current_path_data.get("segment_obstacles")

        result = apply_bre_compliance_control(
            action_seq=action_seq,
            engine=self.physics,
            config=ComplianceControlConfig(
                data_dt=float(self.data_dt),
                sim_dt=float(self.sim_dt),
                frame_stride=int(self.frame_stride),
                turn_gain=float(self.turn_gain),
                safety_mode="off",
                heading_control=False,
                forward_only_slowdown=True,
                preserve_heading=False,
                curvature_slowdown=bool(self.curvature_slowdown),
                curvature_scale=float(self.curvature_scale),
                min_speed_scale=float(self.min_speed_scale),
                backoff_scales=tuple(float(v) for v in self.safety_backoff_scales),
                stop_clearance=float(self.safety_stop_clearance),
            ),
            safety_filter=self.safety_filter,
            obstacles=obstacles,
            segment_obstacles=segment_obstacles,
            bre=self._interaction_is_tether(),
        )
        self.last_compliance_stats = copy.deepcopy(result.stats)
        return result.actions

    def _current_interaction_label(self) -> str:
        if self.interaction_segmenter.has_label:
            return self.interaction_segmenter.current_label
        return "tether" if self.bre else "guide"

    def _interaction_is_tether(self) -> bool:
        return self._current_interaction_label() == "tether"

    def _update_interaction_segmentation(self) -> None:
        old_label = self._current_interaction_label()
        label, changed = self.interaction_segmenter.update(
            self.physics.robot,
            self.physics.human,
        )
        if not changed or label == old_label:
            return

        if old_label == "guide" and label == "tether":
            self._stash_current_cached_guide_plan()
        self._reset_runtime_caches()
        print(
            "[segmentation] interaction state: "
            f"{old_label} -> {label} "
            f"(samples={len(self.interaction_segmenter.samples)}, "
            f"decode={self.interaction_segmenter.last_decode_ms:.1f} ms)"
        )
        self._log_event(
            "interaction_segmentation_changed",
            {
                "previous_label": old_label,
                "interaction_label": label,
                "sample_count": len(self.interaction_segmenter.samples),
                "decode_time_ms": round(
                    float(self.interaction_segmenter.last_decode_ms),
                    3,
                ),
            },
        )

    def _reset_compliance_stats_for_labels(self, labels: np.ndarray) -> None:
        labels = np.asarray(labels, dtype=object).reshape(-1)
        state_counts: dict[str, int] = {}
        for label in labels:
            key = str(label).strip().lower()
            state_counts[key] = state_counts.get(key, 0) + 1
        compliance_steps = int(
            sum(count for label, count in state_counts.items() if label in ("leash", "tether"))
        )
        self.last_compliance_stats = {
            "applied": True,
            "mode": "interaction_aware",
            "current_label": self._current_interaction_label(),
            "safety_applied": False,
            "modified_steps": 0,
            "safety_modified_steps": 0,
            "total_steps": int(len(labels)),
            "compliance_steps": int(compliance_steps),
            "guide_steps": int(len(labels) - compliance_steps),
            "bre_steps": int(compliance_steps),
            "state_counts": state_counts,
            "mean_shift": 0.0,
            "mean_action_shift": 0.0,
            "constraint_count": 0,
            "min_clearance": float("inf"),
        }

    def _interaction_labels_for_action_seq(self, action_count: int) -> np.ndarray:
        """Predict guide/leash labels over the cached action horizon."""
        action_count = max(0, int(action_count))
        labels = np.full(
            (action_count,),
            self._current_interaction_label(),
            dtype=object,
        )
        # Online segmentation estimates the current state. It cannot predict
        # future state switches, so hold the newest label over this short
        # diffusion horizon.
        if self.interaction_segmenter.has_label:
            return labels
        if action_count == 0 or self.bre_timer_start_frame is None:
            return labels

        current_time = float((self.frame_count - self.bre_timer_start_frame) * self.sim_dt)
        upcoming_toggles = sorted(
            float(t)
            for t in self.bre_toggle_times
            if t not in self.triggered_bre_toggle_times and float(t) >= current_time
        )
        if not upcoming_toggles:
            return labels

        state = bool(self.bre)
        toggle_idx = 0
        for action_idx in range(action_count):
            action_time = current_time + float(action_idx) * float(self.data_dt)
            while (
                toggle_idx < len(upcoming_toggles)
                and upcoming_toggles[toggle_idx] <= action_time + 1e-9
            ):
                state = not state
                toggle_idx += 1
            labels[action_idx] = "leash" if state else "guide"
        return labels

    def _apply_interaction_aware_compliance_control(
        self,
        action_seq: np.ndarray,
        obstacles: Optional[np.ndarray] = None,
        segment_obstacles: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if obstacles is None and self.current_path_data is not None:
            obstacles = self.current_path_data.get("obstacles")
        if segment_obstacles is None and self.current_path_data is not None:
            segment_obstacles = self.current_path_data.get("segment_obstacles")

        labels = self._interaction_labels_for_action_seq(len(action_seq))
        self.cached_interaction_labels_seq = labels.copy()
        compliance_mask = np.isin(labels.astype(str), np.array(["leash", "tether"], dtype=object))
        if not np.any(compliance_mask):
            self._reset_compliance_stats_for_labels(labels)
            return np.asarray(action_seq, dtype=np.float32).copy()

        result = apply_interaction_aware_compliance_control(
            action_seq=action_seq,
            interaction_labels=labels,
            engine=self.physics,
            config=ComplianceControlConfig(
                data_dt=float(self.data_dt),
                sim_dt=float(self.sim_dt),
                frame_stride=int(self.frame_stride),
                turn_gain=float(self.turn_gain),
                safety_mode="off",
                heading_control=False,
                forward_only_slowdown=True,
                preserve_heading=False,
                curvature_slowdown=bool(self.curvature_slowdown),
                curvature_scale=float(self.curvature_scale),
                min_speed_scale=float(self.min_speed_scale),
                backoff_scales=tuple(float(v) for v in self.safety_backoff_scales),
                stop_clearance=float(self.safety_stop_clearance),
            ),
            safety_filter=self.safety_filter,
            obstacles=obstacles,
            segment_obstacles=segment_obstacles,
            bre=self._interaction_is_tether(),
            bre_sequence=compliance_mask,
        )
        self.last_compliance_stats = copy.deepcopy(result.stats)
        self.last_compliance_stats["mode"] = "interaction_aware"
        self.last_compliance_stats["current_label"] = self._current_interaction_label()
        return result.actions

    def _cut_action(self, action_seq: np.ndarray) -> np.ndarray:
        """Stop before the planned robot velocity points behind the human->robot link."""
        action_seq = np.asarray(action_seq, dtype=np.float32)
        if action_seq.ndim != 2 or len(action_seq) == 0:
            return action_seq

        sim = copy.deepcopy(self.physics)
        keep_count = len(action_seq)
        for idx, action in enumerate(action_seq):
            _delta, forward, turn, _speed_scale = self._action_to_execution(
                action,
                sim.robot.position,
                sim.robot.heading,
            )
            sim.set_control(forward, turn, self._interaction_is_tether())
            invalid = False
            for _ in range(int(self.frame_stride)):
                sim.step()
                robot_vel = sim.robot.velocity
                link = sim.robot.position - sim.human.position
                link_angle = float(np.arctan2(link[1], link[0]))
                cos_a = float(np.cos(link_angle))
                sin_a = float(np.sin(link_angle))
                vel_local_x = cos_a * float(robot_vel[0]) + sin_a * float(robot_vel[1])
                if float(np.linalg.norm(link)) > 1e-6 and vel_local_x < 0.0:
                    invalid = True
                    break
            if invalid:
                keep_count = idx
                break

        if keep_count == len(action_seq):
            return action_seq
        if keep_count == 0:
            return np.zeros_like(action_seq[:1])
        return action_seq[:keep_count].copy()

    def _build_obs(
        self, robot_pos: np.ndarray, human_pos: np.ndarray, heading: float
    ) -> np.ndarray:
        if self.robot_frame:
            # robot-centric base state: [robot_state(2), human_rel(2)]
            human_rel = human_pos - robot_pos
            cos_h = float(np.cos(heading))
            sin_h = float(np.sin(heading))
            hx = cos_h * human_rel[0] + sin_h * human_rel[1]
            hy = -sin_h * human_rel[0] + cos_h * human_rel[1]
            if self.robot_state in ("vel", "velocity"):
                if self.prev_robot_pos is None:
                    vel_world = np.zeros((2,), dtype=np.float32)
                else:
                    vel_world = (robot_pos - self.prev_robot_pos).astype(np.float32) / float(
                        max(1e-6, self.data_dt)
                    )
                vx = cos_h * vel_world[0] + sin_h * vel_world[1]
                vy = -sin_h * vel_world[0] + cos_h * vel_world[1]
                robot_state = np.array([vx, vy], dtype=np.float32)
            else:
                robot_state = np.zeros((2,), dtype=np.float32)
            base = np.concatenate([robot_state, np.array([hx, hy], dtype=np.float32)], axis=0)
        else:
            base = np.concatenate([robot_pos, human_pos], axis=0).astype(np.float32)

        ref_features = (
            self._build_reference_features(robot_pos, heading)
            if self.n_lookahead > 0 and self.current_path_data is not None
            else np.zeros((self.n_lookahead * 2,), dtype=np.float32)
        )
        if self.observation_mode == "mid360":
            obs_features = self._build_mid360_features(robot_pos, human_pos, heading)
        else:
            if self.pointcloud_mode == "live":
                self._update_mid360_pointcloud(robot_pos, heading)
            else:
                self.current_mid360_points_world = None
                self.current_mid360_point_obstacles = None
                self.current_safety_point_obstacles = None
            obs_features = self._build_obstacle_features(robot_pos, human_pos, heading)
        if self.n_lookahead <= 0 or self.current_path_data is None:
            self.lookahead_world = None
        return np.concatenate([base, ref_features, obs_features], axis=0).astype(np.float32)

    def _build_reference_features(self, robot_pos: np.ndarray, heading: float) -> np.ndarray:
        ref_path = self.current_path_data["path"]
        if ref_path is None or len(ref_path) == 0:
            self.lookahead_world = None
            return np.zeros((self.n_lookahead * 2,), dtype=np.float32)
        diffs = ref_path - robot_pos
        idx = int(np.argmin(np.sum(diffs * diffs, axis=1)))
        indices = idx + np.arange(self.n_lookahead) * self.lookahead_stride
        indices = np.clip(indices, 0, len(ref_path) - 1)
        points = ref_path[indices]
        self.lookahead_world = points
        rel = points - robot_pos
        cos_h = float(np.cos(heading))
        sin_h = float(np.sin(heading))
        local_x = cos_h * rel[:, 0] + sin_h * rel[:, 1]
        local_y = -sin_h * rel[:, 0] + cos_h * rel[:, 1]
        return np.stack([local_x, local_y], axis=-1).reshape(-1).astype(np.float32)

    def _filter_mid360_pointcloud(self, frame: np.ndarray) -> np.ndarray:
        """Apply the same geometric validity limits used by the Mid-360 observation."""
        frame = np.asarray(frame, dtype=np.float32)
        if frame.ndim != 2 or frame.shape[1] < 3 or len(frame) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        if self.mid360_obs_config is None:
            return frame[:, :3]

        xy = frame[:, :2]
        ranges = np.linalg.norm(xy, axis=1)
        azimuth = np.arctan2(xy[:, 1], xy[:, 0])
        if self.range_source == "laser_scan":
            # LaserScan has no per-return height.  Place all beams at a valid
            # representative height so the original Mid-360 encoder can be
            # reused without discarding the entire 2-D scan.
            min_height = float(self.mid360_obs_config.ground_height)
            max_height = float(self.mid360_obs_config.max_height)
            representative_height = float(
                np.clip(self.lidar_height, min_height, max_height)
            )
            height = np.full(
                (len(frame),), representative_height, dtype=np.float32
            )
        else:
            height = frame[:, 2]
            if self.mid360_obs_config.use_world_height:
                height = height + self.lidar_height

        keep = np.isfinite(frame[:, :3]).all(axis=1)
        keep &= ranges >= float(self.mid360_obs_config.min_range)
        keep &= ranges <= float(self.mid360_obs_config.max_range)
        keep &= height >= float(self.mid360_obs_config.ground_height)
        keep &= height <= float(self.mid360_obs_config.max_height)

        min_angle = float(self.mid360_obs_config.min_angle)
        max_angle = float(self.mid360_obs_config.max_angle)
        if min_angle <= max_angle:
            keep &= (azimuth >= min_angle) & (azimuth <= max_angle)
        else:
            # Support a field of view that crosses the -pi/pi boundary.
            keep &= (azimuth >= min_angle) | (azimuth <= max_angle)
        return frame[keep, :3]

    def _update_mid360_pointcloud(
        self,
        robot_pos: np.ndarray,
        heading: float,
        frame: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if frame is None:
            frame, _field_names, seq = self.ros_io.pointcloud()
            self._last_mid360_cloud_seq = seq
        if frame is not None and len(frame) > 0:
            frame = self._filter_mid360_pointcloud(frame)
        if frame is None or len(frame) == 0:
            self.current_mid360_points_world = None
            self.current_mid360_point_obstacles = np.zeros((0, 3), dtype=np.float32)
            self.current_safety_point_obstacles = np.zeros((0, 3), dtype=np.float32)
            return np.zeros((0, 2), dtype=np.float32)

        local_xy = np.asarray(frame[:, :2], dtype=np.float32)
        cos_h = float(np.cos(heading))
        sin_h = float(np.sin(heading))
        rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]], dtype=np.float32)
        # Keep every filtered point for safety.  Only the visualization is
        # downsampled below.
        world_xy = local_xy @ rot.T + robot_pos.astype(np.float32)
        radii = np.full(
            (len(world_xy), 1),
            self.mid360_point_obstacle_radius,
            dtype=np.float32,
        )
        self.current_mid360_point_obstacles = np.concatenate(
            [world_xy, radii],
            axis=1,
        ).astype(np.float32, copy=False)
        # The path-dependent safety copy must be rebuilt after inference.
        self.current_safety_point_obstacles = None

        visual_xy = world_xy
        if len(visual_xy) > self._mid360_visual_max_points:
            stride = int(np.ceil(len(visual_xy) / self._mid360_visual_max_points))
            visual_xy = visual_xy[::stride]
        self.current_mid360_points_world = visual_xy
        return local_xy

    @staticmethod
    def _point_to_polyline_distance_sq(
        points: np.ndarray,
        path: np.ndarray,
        chunk_size: int = 4096,
    ) -> np.ndarray:
        """Return each 2D point's squared distance to a polyline."""
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        path = np.asarray(path, dtype=np.float32).reshape(-1, 2)
        if len(points) == 0:
            return np.zeros((0,), dtype=np.float32)
        if len(path) == 0:
            return np.full((len(points),), np.inf, dtype=np.float32)
        if len(path) == 1:
            return np.sum((points - path[0]) ** 2, axis=1)

        starts = path[:-1]
        vectors = path[1:] - starts
        lengths_sq = np.sum(vectors * vectors, axis=1)
        valid = lengths_sq > 1e-10
        if not np.any(valid):
            return np.sum((points - path[0]) ** 2, axis=1)
        starts = starts[valid]
        vectors = vectors[valid]
        lengths_sq = lengths_sq[valid]

        result = np.empty((len(points),), dtype=np.float32)
        for begin in range(0, len(points), chunk_size):
            chunk = points[begin : begin + chunk_size]
            relative = chunk[:, None, :] - starts[None, :, :]
            projection = np.sum(relative * vectors[None, :, :], axis=2)
            projection = np.clip(
                projection / lengths_sq[None, :],
                0.0,
                1.0,
            )
            closest = starts[None, :, :] + projection[:, :, None] * vectors[None, :, :]
            distance_sq = np.sum((chunk[:, None, :] - closest) ** 2, axis=2)
            result[begin : begin + len(chunk)] = np.min(distance_sq, axis=1)
        return result

    def _prepare_safety_point_obstacles(
        self,
        raw_diffusion_path: Optional[np.ndarray],
    ) -> dict:
        """Keep only path-near points, then retain at most one point per 2D voxel."""
        obstacles = self.current_mid360_point_obstacles
        if obstacles is None:
            obstacles = np.zeros((0, 3), dtype=np.float32)
        obstacles = np.asarray(obstacles, dtype=np.float32).reshape(-1, 3)
        raw_count = int(len(obstacles))

        path = np.asarray(
            raw_diffusion_path
            if raw_diffusion_path is not None
            else np.zeros((0, 2), dtype=np.float32),
            dtype=np.float32,
        ).reshape(-1, 2)
        robot_xy = np.asarray(self.physics.robot.position, dtype=np.float32).reshape(1, 2)
        if len(path) == 0:
            path = robot_xy
        elif np.linalg.norm(path[0] - robot_xy[0]) > 1e-4:
            path = np.concatenate([robot_xy, path], axis=0)

        if raw_count > 0 and self.safety_path_corridor > 0.0:
            corridor = self.safety_path_corridor
            lower = np.min(path, axis=0) - corridor
            upper = np.max(path, axis=0) + corridor
            in_path_box = np.all(
                (obstacles[:, :2] >= lower) & (obstacles[:, :2] <= upper),
                axis=1,
            )
            obstacles = obstacles[in_path_box]
            distance_sq = self._point_to_polyline_distance_sq(
                obstacles[:, :2],
                path,
            )
            obstacles = obstacles[
                distance_sq <= self.safety_path_corridor * self.safety_path_corridor
            ]
        corridor_count = int(len(obstacles))

        if len(obstacles) > 0 and self.safety_point_spacing > 0.0:
            voxel_xy = np.floor(
                obstacles[:, :2] / self.safety_point_spacing
            ).astype(np.int64)
            _unique_voxels, keep_indices = np.unique(
                voxel_xy,
                axis=0,
                return_index=True,
            )
            obstacles = obstacles[np.sort(keep_indices)]

        self.current_safety_point_obstacles = obstacles.astype(
            np.float32,
            copy=False,
        )
        self.last_safety_pointcloud_stats = {
            "raw_count": raw_count,
            "corridor_count": corridor_count,
            "sparse_count": int(len(obstacles)),
        }
        return dict(self.last_safety_pointcloud_stats)

    def _safety_obstacle_inputs(
        self,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return the obstacles used by QPSafetyFilter.

        Live point-cloud mode uses every filtered Livox point as a 5 cm
        diameter circle obstacle.  Segment obstacles from the synthetic path
        are intentionally excluded because they do not represent the measured
        real environment.
        """
        if self.pointcloud_mode == "live":
            if self.current_safety_point_obstacles is not None:
                return self.current_safety_point_obstacles, None
            if self.current_mid360_point_obstacles is None:
                return np.zeros((0, 3), dtype=np.float32), None
            return self.current_mid360_point_obstacles, None

        obstacles = self.current_path_data.get("obstacles") if self.current_path_data else None
        segments = (
            self.current_path_data.get("segment_obstacles")
            if self.current_path_data
            else None
        )
        return obstacles, segments

    def _human_cloud_keep_mask(
        self,
        local_xy: np.ndarray,
        robot_pos: np.ndarray,
        human_pos: np.ndarray,
        heading: float,
    ) -> np.ndarray:
        human_rel = np.asarray(human_pos, dtype=np.float32) - np.asarray(robot_pos, dtype=np.float32)
        cos_h = float(np.cos(heading))
        sin_h = float(np.sin(heading))
        human_local = np.array(
            [
                cos_h * human_rel[0] + sin_h * human_rel[1],
                -sin_h * human_rel[0] + cos_h * human_rel[1],
            ],
            dtype=np.float32,
        )
        distance = np.linalg.norm(local_xy[:, :2] - human_local[None, :], axis=1)
        return distance > float(self.physics.human_radius)

    def _build_mid360_features(
        self,
        robot_pos: np.ndarray,
        human_pos: np.ndarray,
        heading: float,
    ) -> np.ndarray:
        if self.mid360_obs_config is None:
            self.current_mid360_points_world = None
            self.current_mid360_point_obstacles = None
            self.current_safety_point_obstacles = None
            return np.zeros((self.lidar_num_bins,), dtype=np.float32)

        frame, _field_names, seq = self.ros_io.pointcloud()
        self._last_mid360_cloud_seq = seq
        if frame is not None and len(frame) > 0:
            keep = self._human_cloud_keep_mask(
                np.asarray(frame[:, :2], dtype=np.float32),
                robot_pos,
                human_pos,
                heading,
            )
            frame = frame[keep]
        self._update_mid360_pointcloud(robot_pos, heading, frame=frame)
        if frame is None or len(frame) == 0:
            return np.full(
                (self.lidar_num_bins,),
                self.mid360_obs_config.fill_value,
                dtype=np.float32,
            )

        # Preserve the encoder's existing behavior; it applies its own
        # validity mask internally.  The QP obstacle copy above uses the same
        # configured limits but keeps every surviving point.
        local_xy = np.asarray(frame[:, :2], dtype=np.float32)
        scan = encode_mid360_scan_from_local_points(
            local_xy=local_xy,
            config=self.mid360_obs_config,
            ranges=np.linalg.norm(frame[:, :2], axis=1).astype(np.float32),
            azimuth=np.arctan2(frame[:, 1], frame[:, 0]).astype(np.float32),
            height=(
                np.full(
                    (len(frame),),
                    float(
                        np.clip(
                            self.lidar_height,
                            self.mid360_obs_config.ground_height,
                            self.mid360_obs_config.max_height,
                        )
                    ),
                    dtype=np.float32,
                )
                if self.range_source == "laser_scan"
                else (
                    np.asarray(frame[:, 2], dtype=np.float32) + self.lidar_height
                    if self.mid360_obs_config.use_world_height
                    else np.asarray(frame[:, 2], dtype=np.float32)
                )
            ),
        )
        return scan.astype(np.float32, copy=False)

    def _select_obstacles_for_observation(
        self,
        robot_pos: np.ndarray,
        human_pos: np.ndarray,
        heading: Optional[float] = None,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self.current_path_data is None:
            return None, None

        path = self.current_path_data.get("path")
        path_s = self.current_path_data.get("_path_s")
        if path is None or len(path) == 0 or path_s is None:
            return None, None
        path = np.asarray(path, dtype=np.float32)
        path_s = np.asarray(path_s, dtype=np.float32)
        if len(path_s) != len(path):
            self._precompute_frenet_cache()
            path_s = self.current_path_data.get("_path_s")
            if path_s is None:
                return None, None
            path_s = np.asarray(path_s, dtype=np.float32)

        # Select k nearest *forward* obstacles w.r.t. the human progress along the
        # reference path (instead of the robot), matching training-time features.
        idx_h = self._nearest_path_index(path, human_pos)
        s_human = float(path_s[idx_h])

        selected_circles = None
        selected_segments = None

        if self.n_obstacle_circles > 0:
            obstacles = self.current_path_data.get("obstacles")
            if obstacles is not None and len(obstacles) > 0:
                circle_obs = np.asarray(obstacles, dtype=np.float32)
                if circle_obs.ndim == 2 and circle_obs.shape[1] >= 3:
                    circle_obs = circle_obs[:, :3]
                    circle_s = self.current_path_data.get("_circle_s")
                    if circle_s is None or len(circle_s) != len(circle_obs):
                        self._precompute_frenet_cache()
                        circle_s = self.current_path_data.get("_circle_s")
                    if circle_s is not None and len(circle_s) == len(circle_obs):
                        circle_s = np.asarray(circle_s, dtype=np.float32)
                        delta_s = circle_s - s_human
                        candidates = np.nonzero(delta_s >= 0.0)[0]
                        if len(candidates) > 0:
                            order = candidates[np.argsort(delta_s[candidates])]
                        else:
                            order = np.zeros((0,), dtype=np.int64)
                    else:
                        centers = circle_obs[:, :2]
                        rel = centers - human_pos
                        dist_sq = np.sum(rel * rel, axis=1)
                        order = np.argsort(dist_sq)
                    count = min(self.n_obstacle_circles, int(len(order)))
                    if count > 0:
                        selected_circles = circle_obs[order[:count], :3]

        if self.n_obstacle_segments > 0:
            segments = self.current_path_data.get("segment_obstacles")
            if segments is not None and len(segments) > 0:
                seg_obs = np.asarray(segments, dtype=np.float32)
                if seg_obs.ndim == 2 and seg_obs.shape[1] >= 4:
                    seg_obs = seg_obs[:, :4]
                    seg_s_min = self.current_path_data.get("_segment_s_min")
                    seg_s_max = self.current_path_data.get("_segment_s_max")
                    if (
                        seg_s_min is None
                        or seg_s_max is None
                        or len(seg_s_min) != len(seg_obs)
                        or len(seg_s_max) != len(seg_obs)
                    ):
                        self._precompute_frenet_cache()
                        seg_s_min = self.current_path_data.get("_segment_s_min")
                        seg_s_max = self.current_path_data.get("_segment_s_max")
                    if (
                        seg_s_min is not None
                        and seg_s_max is not None
                        and len(seg_s_min) == len(seg_obs)
                        and len(seg_s_max) == len(seg_obs)
                    ):
                        seg_s_min = np.asarray(seg_s_min, dtype=np.float32)
                        seg_s_max = np.asarray(seg_s_max, dtype=np.float32)
                        valid = seg_s_max >= s_human
                        candidates = np.nonzero(valid)[0]
                        if len(candidates) > 0:
                            delta = np.maximum(0.0, seg_s_min - s_human)
                            order = candidates[np.argsort(delta[candidates])]
                        else:
                            order = np.zeros((0,), dtype=np.int64)
                    else:
                        dist_sq = np.zeros((len(seg_obs),), dtype=np.float32)
                        for i, seg in enumerate(seg_obs):
                            dist_sq[i] = self._point_segment_dist_sq(
                                human_pos, seg[:2], seg[2:4]
                            )
                        order = np.argsort(dist_sq)
                    count = min(self.n_obstacle_segments, int(len(order)))
                    if count > 0:
                        selected_segments = seg_obs[order[:count], :4]

        return selected_circles, selected_segments

    def _precompute_frenet_cache(self):
        if self.current_path_data is None:
            return
        path = self.current_path_data.get("path")
        if path is None or len(path) == 0:
            self.current_path_data["_path_s"] = None
            self.current_path_data["_circle_s"] = None
            self.current_path_data["_segment_s_min"] = None
            self.current_path_data["_segment_s_max"] = None
            return
        path = np.asarray(path, dtype=np.float32)
        path_s = self._compute_path_s(path)
        self.current_path_data["_path_s"] = path_s

        obstacles = self.current_path_data.get("obstacles")
        circle_s = None
        if obstacles is not None and len(obstacles) > 0:
            circle_obs = np.asarray(obstacles, dtype=np.float32)
            if circle_obs.ndim == 2 and circle_obs.shape[1] >= 3:
                circle_obs = circle_obs[:, :3]
                circle_s = np.zeros((len(circle_obs),), dtype=np.float32)
                for i, center in enumerate(circle_obs[:, :2]):
                    idx = self._nearest_path_index(path, center)
                    circle_s[i] = path_s[idx]
        self.current_path_data["_circle_s"] = circle_s

        segments = self.current_path_data.get("segment_obstacles")
        seg_s_min = None
        seg_s_max = None
        if segments is not None and len(segments) > 0:
            seg_obs = np.asarray(segments, dtype=np.float32)
            if seg_obs.ndim == 2 and seg_obs.shape[1] >= 4:
                seg_obs = seg_obs[:, :4]
                s1 = np.zeros((len(seg_obs),), dtype=np.float32)
                s2 = np.zeros((len(seg_obs),), dtype=np.float32)
                for i, seg in enumerate(seg_obs):
                    idx1 = self._nearest_path_index(path, seg[:2])
                    idx2 = self._nearest_path_index(path, seg[2:4])
                    s1[i] = path_s[idx1]
                    s2[i] = path_s[idx2]
                seg_s_min = np.minimum(s1, s2).astype(np.float32)
                seg_s_max = np.maximum(s1, s2).astype(np.float32)
        self.current_path_data["_segment_s_min"] = seg_s_min
        self.current_path_data["_segment_s_max"] = seg_s_max

    def _compute_path_s(self, path: np.ndarray) -> np.ndarray:
        if path is None or len(path) == 0:
            return np.zeros((0,), dtype=np.float32)
        if len(path) == 1:
            return np.zeros((1,), dtype=np.float32)
        diffs = np.diff(path.astype(np.float32), axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1).astype(np.float32)
        s = np.zeros((len(path),), dtype=np.float32)
        s[1:] = np.cumsum(seg_lengths, axis=0).astype(np.float32)
        return s

    def _nearest_path_index(self, path: np.ndarray, point: np.ndarray) -> int:
        diffs = path - point.astype(np.float32)
        dist_sq = np.sum(diffs * diffs, axis=1)
        return int(np.argmin(dist_sq))

    def _build_obstacle_features(
        self, robot_pos: np.ndarray, human_pos: np.ndarray, heading: float
    ) -> np.ndarray:
        circle_dim = 3 if self.obstacle_include_radius else 2
        clearance_dim = (
            (self.n_obstacle_circles + self.n_obstacle_segments)
            if getattr(self, "obstacle_include_human_clearance", False)
            else 0
        )
        total_dim = self.n_obstacle_circles * circle_dim + self.n_obstacle_segments * 4 + clearance_dim
        if total_dim == 0:
            return np.zeros((0,), dtype=np.float32)

        feats = np.zeros((total_dim,), dtype=np.float32)
        offset = 0
        if self.current_path_data is None:
            return feats

        selected_circles, selected_segments = self._select_obstacles_for_observation(
            robot_pos, human_pos, heading
        )
        clearance_offset = self.n_obstacle_circles * circle_dim + self.n_obstacle_segments * 4
        circle_clear_offset = clearance_offset
        seg_clear_offset = circle_clear_offset + self.n_obstacle_circles
        if getattr(self, "obstacle_include_human_clearance", False) and clearance_dim > 0:
            feats[clearance_offset:] = 5.0

        if self.n_obstacle_circles > 0:
            count = 0 if selected_circles is None else len(selected_circles)
            for i in range(self.n_obstacle_circles):
                if i < count:
                    rel_i = selected_circles[i, :2] - robot_pos
                    if self.robot_frame:
                        rel_i = self._rotate_rel(rel_i, heading)
                    feats[offset : offset + 2] = rel_i
                    if self.obstacle_include_radius:
                        feats[offset + 2] = float(selected_circles[i, 2])
                    if getattr(self, "obstacle_include_human_clearance", False):
                        d = human_pos - selected_circles[i, :2]
                        dist = float(np.linalg.norm(d))
                        clearance = dist - float(selected_circles[i, 2] + self.physics.human_radius)
                        feats[circle_clear_offset + i] = float(clearance)
                offset += circle_dim

        if self.n_obstacle_segments > 0:
            count = 0 if selected_segments is None else len(selected_segments)
            for i in range(self.n_obstacle_segments):
                if i < count:
                    seg = selected_segments[i]
                    if self.segment_repr == "endpoints":
                        p1 = seg[:2] - robot_pos
                        p2 = seg[2:4] - robot_pos
                        if self.robot_frame:
                            p1 = self._rotate_rel(p1, heading)
                            p2 = self._rotate_rel(p2, heading)
                        feats[offset : offset + 4] = [p1[0], p1[1], p2[0], p2[1]]
                    else:  # closest_dir
                        p1_world = seg[:2].astype(np.float32)
                        p2_world = seg[2:4].astype(np.float32)
                        ab = p2_world - p1_world
                        denom = float(np.dot(ab, ab))
                        if denom < 1e-12:
                            closest = p1_world
                            direction = np.zeros((2,), dtype=np.float32)
                        else:
                            t_proj = float(np.dot(robot_pos - p1_world, ab)) / denom
                            t_proj = float(np.clip(t_proj, 0.0, 1.0))
                            closest = p1_world + t_proj * ab
                            direction = (ab / np.sqrt(denom)).astype(np.float32)
                            if (direction[0] < 0) or (
                                abs(direction[0]) < 1e-6 and direction[1] < 0
                            ):
                                direction = -direction

                        rel = (closest - robot_pos).astype(np.float32)
                        if self.robot_frame:
                            rel = self._rotate_rel(rel, heading)
                            direction = self._rotate_rel(direction, heading)
                        feats[offset : offset + 4] = [
                            float(rel[0]),
                            float(rel[1]),
                            float(direction[0]),
                            float(direction[1]),
                        ]
                    if getattr(self, "obstacle_include_human_clearance", False):
                        dist_sq_h = self._point_segment_dist_sq(human_pos, seg[:2], seg[2:4])
                        clearance = float(np.sqrt(dist_sq_h)) - float(self.physics.human_radius)
                        feats[seg_clear_offset + i] = float(clearance)
                offset += 4

        return feats

    def _rotate_rel(self, rel: np.ndarray, heading: float) -> np.ndarray:
        cos_h = float(np.cos(heading))
        sin_h = float(np.sin(heading))
        return np.array(
            [cos_h * rel[0] + sin_h * rel[1], -sin_h * rel[0] + cos_h * rel[1]],
            dtype=np.float32,
        )

    def _point_segment_dist_sq(self, point: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
        ab = p2 - p1
        denom = float(np.dot(ab, ab))
        if denom < 1e-12:
            diff = point - p1
            return float(np.dot(diff, diff))
        t = float(np.dot(point - p1, ab)) / denom
        t = float(np.clip(t, 0.0, 1.0))
        closest = p1 + t * ab
        diff = point - closest
        return float(np.dot(diff, diff))

    def _segments_closest_points_and_dirs(
        self,
        robot_pos: np.ndarray,
        segments: Optional[np.ndarray],
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if segments is None or len(segments) == 0:
            return None, None
        segs = np.asarray(segments, dtype=np.float32)
        if segs.ndim != 2 or segs.shape[1] < 4:
            return None, None
        segs = segs[:, :4]

        closest_points = np.zeros((len(segs), 2), dtype=np.float32)
        directions = np.zeros((len(segs), 2), dtype=np.float32)
        for i, seg in enumerate(segs):
            p1 = seg[:2]
            p2 = seg[2:4]
            ab = p2 - p1
            denom = float(np.dot(ab, ab))
            if denom < 1e-12:
                closest = p1
                direction = np.zeros((2,), dtype=np.float32)
            else:
                t_proj = float(np.dot(robot_pos - p1, ab)) / denom
                t_proj = float(np.clip(t_proj, 0.0, 1.0))
                closest = p1 + t_proj * ab
                direction = (ab / np.sqrt(denom)).astype(np.float32)
                if (direction[0] < 0) or (abs(direction[0]) < 1e-6 and direction[1] < 0):
                    direction = -direction
            closest_points[i] = closest
            directions[i] = direction

        return closest_points, directions

    def _actions_to_path(self, robot_pos: np.ndarray, action_seq: np.ndarray) -> Optional[np.ndarray]:
        if action_seq is None or action_seq.size == 0:
            return None

        pos = np.asarray(robot_pos, dtype=np.float32).copy()
        heading = float(self.physics.robot.heading)
        points: list[np.ndarray] = []
        action_seq = np.asarray(action_seq, dtype=np.float32)
        obstacles, segments = self._safety_obstacle_inputs()

        for action_idx, act in enumerate(action_seq):
            step_points, heading, collided = self._rollout_preview_action(
                pos=pos,
                heading=heading,
                action=act,
                obstacles=obstacles,
                segment_obstacles=segments,
            )
            if step_points:
                pos = step_points[-1].copy()
                points.extend(step_points)
            if self.debug_preview and action_idx < self.debug_preview_limit:
                end_pos = pos.tolist()
                print(
                    "[debug preview] "
                    f"idx={action_idx} action={np.round(act, 4).tolist()} "
                    f"end_pos={[round(v, 4) for v in end_pos]} heading={heading:.4f} "
                    f"collided={collided}"
                )
            if collided:
                break

        if not points:
            return None
        return np.stack(points, axis=0)

    def _deltas_to_path(
        self,
        delta_seq: np.ndarray,
        protect_robot: bool = True,
        protect_human: Optional[bool] = None,
        bre_override: Optional[bool] = None,
    ) -> Optional[np.ndarray]:
        if delta_seq is None or delta_seq.size == 0:
            return None

        delta_seq = np.asarray(delta_seq, dtype=np.float32)
        obstacles, segments = self._safety_obstacle_inputs()
        sim = copy.deepcopy(self.physics)
        points: list[np.ndarray] = []
        if protect_human is None:
            protect_human = self._safety_protects_human()

        for delta_idx, delta in enumerate(delta_seq):
            collided, step_points = self._simulate_delta_on_engine(
                sim,
                delta,
                obstacles=obstacles,
                segment_obstacles=segments,
                protect_robot=protect_robot,
                protect_human=protect_human,
                bre_override=bre_override,
            )
            if step_points:
                points.extend(step_points)
            if self.debug_preview and delta_idx < self.debug_preview_limit:
                end_pos = sim.robot.position.tolist()
                print(
                    "[debug preview] "
                    f"idx={delta_idx} safe_delta={np.round(delta, 4).tolist()} "
                    f"end_pos={[round(v, 4) for v in end_pos]} "
                    f"heading={sim.robot.heading:.4f} collided={collided}"
                )
            if collided:
                break

        if not points:
            return None
        return np.stack(points, axis=0)

    def _rollout_preview_action(
        self,
        pos: np.ndarray,
        heading: float,
        action: np.ndarray,
        obstacles: Optional[np.ndarray],
        segment_obstacles: Optional[np.ndarray],
    ) -> tuple[list[np.ndarray], float, bool]:
        pos = np.asarray(pos, dtype=np.float32).copy()
        heading = float(heading)
        action = np.asarray(action, dtype=np.float32)

        if self.action_mode == "forward_heading":
            _delta, forward_input, turn_input, _speed_scale = self._action_to_execution(
                action,
                pos,
                heading,
            )
        else:
            delta = self._action_to_world_delta(action, pos, heading)
            forward_input, turn_input, _speed_scale = self._delta_to_safe_control(
                delta,
                heading,
                dt=self.data_dt,
            )

        step_points: list[np.ndarray] = []
        collided = False
        for _ in range(int(self.frame_stride)):
            heading = wrap_angle(heading + float(turn_input) * float(self.physics.turn_speed) * float(self.sim_dt))
            step_dist = float(forward_input) * float(self.physics.robot_speed) * float(self.sim_dt)
            pos = pos + np.array([
                np.cos(heading) * step_dist,
                np.sin(heading) * step_dist,
            ], dtype=np.float32)
            step_points.append(pos.copy())
            if self._preview_robot_collision(pos, obstacles, segment_obstacles):
                collided = True
                break
        return step_points, heading, collided

    def _preview_robot_collision(
        self,
        robot_pos: np.ndarray,
        obstacles: Optional[np.ndarray],
        segment_obstacles: Optional[np.ndarray],
    ) -> bool:
        robot_pos = np.asarray(robot_pos, dtype=np.float32)
        robot_radius = float(self.physics.robot_radius)

        if obstacles is not None and len(obstacles) > 0:
            obs = np.asarray(obstacles, dtype=np.float32)
            if obs.ndim == 2 and obs.shape[1] >= 3:
                centers = obs[:, :2]
                radii = obs[:, 2]
                dists = np.linalg.norm(centers - robot_pos[None, :], axis=1)
                if np.any(dists <= (radii + robot_radius)):
                    return True

        if segment_obstacles is not None and len(segment_obstacles) > 0:
            segs = np.asarray(segment_obstacles, dtype=np.float32)
            if segs.ndim == 2 and segs.shape[1] >= 4:
                for seg in segs[:, :4]:
                    dist_sq = self._point_segment_dist_sq(robot_pos, seg[:2], seg[2:4])
                    if dist_sq <= robot_radius * robot_radius:
                        return True
        return False

    def _action_to_world_delta(
        self,
        action: np.ndarray,
        robot_pos: np.ndarray,
        heading: float,
    ) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        robot_pos = np.asarray(robot_pos, dtype=np.float32)

        if self.action_mode == "forward_heading":
            forward = float(action[0])
            return np.array(
                [np.cos(heading) * forward, np.sin(heading) * forward],
                dtype=np.float32,
            )

        if self.action_mode == "delta":
            delta = action[:2].astype(np.float32)
        elif self.action_mode == "position":
            delta = action[:2].astype(np.float32) - robot_pos[:2].astype(np.float32)
        else:
            delta = action[:2].astype(np.float32) * float(self.data_dt)

        if self.robot_frame and self.action_mode in ("delta", "velocity"):
            cos_h = float(np.cos(heading))
            sin_h = float(np.sin(heading))
            return np.array(
                [
                    cos_h * float(delta[0]) - sin_h * float(delta[1]),
                    sin_h * float(delta[0]) + cos_h * float(delta[1]),
                ],
                dtype=np.float32,
            )
        return delta.astype(np.float32)

    def _action_to_delta(self, action: np.ndarray, robot_pos: np.ndarray) -> np.ndarray:
        return self._action_to_world_delta(action, robot_pos, self.physics.robot.heading)

    def _world_delta_to_action(
        self,
        delta: np.ndarray,
        robot_pos: np.ndarray,
        heading: float,
    ) -> np.ndarray:
        delta = np.asarray(delta, dtype=np.float32).reshape(2)
        robot_pos = np.asarray(robot_pos, dtype=np.float32).reshape(2)

        if self.action_mode == "forward_heading":
            dist = float(np.linalg.norm(delta))
            if dist < 1e-6:
                return np.zeros((2,), dtype=np.float32)
            desired_heading = float(np.arctan2(delta[1], delta[0]))
            max_heading_delta = float(self.physics.turn_speed * self.data_dt) / max(
                float(self.turn_gain), 1e-6
            )
            heading_delta = wrap_angle(desired_heading - heading) / max(
                float(self.turn_gain), 1e-6
            )
            heading_delta = float(np.clip(heading_delta, -max_heading_delta, max_heading_delta))
            max_forward_delta = float(self.physics.robot_speed * self.data_dt)
            forward_delta = float(np.clip(dist, 0.0, max_forward_delta))
            return np.array([forward_delta, heading_delta], dtype=np.float32)

        local = delta.astype(np.float32)
        if self.robot_frame and self.action_mode in ("delta", "velocity"):
            cos_h = float(np.cos(heading))
            sin_h = float(np.sin(heading))
            local = np.array(
                [
                    cos_h * float(delta[0]) + sin_h * float(delta[1]),
                    -sin_h * float(delta[0]) + cos_h * float(delta[1]),
                ],
                dtype=np.float32,
            )

        if self.action_mode == "delta":
            return local.astype(np.float32)
        if self.action_mode == "velocity":
            return (local / float(max(self.data_dt, 1e-6))).astype(np.float32)
        return (robot_pos + delta).astype(np.float32)

    def _action_to_execution(
        self,
        action: np.ndarray,
        robot_pos: np.ndarray,
        heading: float,
    ) -> tuple[np.ndarray, float, float, float]:
        if self.action_mode == "forward_heading":
            forward_delta = float(action[0])
            heading_delta = float(action[1])
            turn_speed = float(self.physics.turn_speed)
            robot_speed = float(self.physics.robot_speed)
            turn_delta = heading_delta * float(self.turn_gain)
            turn = turn_delta / (turn_speed * self.data_dt) if turn_speed > 0 else 0.0
            forward = forward_delta / (robot_speed * self.data_dt) if robot_speed > 0 else 0.0
            speed_scale = 1.0
            if self.curvature_slowdown and turn_speed > 0:
                max_turn = turn_speed * self.data_dt
                if max_turn > 1e-6:
                    ratio = min(1.0, abs(turn_delta) / max_turn)
                    speed_scale = max(
                        float(self.min_speed_scale),
                        1.0 - float(self.curvature_scale) * ratio,
                    )
                    forward *= speed_scale
            forward = float(np.clip(forward, -1.0, 1.0))
            turn = float(np.clip(turn, -1.0, 1.0))
            delta = self._action_to_world_delta(action, robot_pos, heading)
            return delta, forward, turn, speed_scale

        delta = self._action_to_world_delta(action, robot_pos, heading)
        forward, turn = self._delta_to_control(delta, heading, dt=self.data_dt)
        return delta, forward, turn, 1.0

    def _simulate_action_on_engine(
        self,
        engine: PhysicsEngine,
        action: np.ndarray,
        obstacles: Optional[np.ndarray],
        segment_obstacles: Optional[np.ndarray],
        protect_robot: bool = True,
        protect_human: bool = True,
        bre_override: Optional[bool] = None,
    ) -> tuple[bool, list[np.ndarray]]:
        _, forward, turn, _speed_scale = self._action_to_execution(
            action,
            engine.robot.position,
            engine.robot.heading,
        )
        bre = self._interaction_is_tether() if bre_override is None else bool(bre_override)
        engine.set_control(forward, turn, bre)
        points: list[np.ndarray] = []
        for _ in range(int(self.frame_stride)):
            robot_state, _human_state = engine.step()
            points.append(robot_state.position.copy())
            if obstacles is not None or segment_obstacles is not None:
                collided, info = engine.check_collision(obstacles, segment_obstacles=segment_obstacles)
                who = info.get("who") if info else None
                relevant = (
                    collided
                    and (
                        (protect_robot and who == "robot")
                        or (protect_human and who == "human")
                    )
                )
                if relevant:
                    return True, points
        return False, points

    def _forward_heading_action_to_nominal_delta(
        self,
        engine: PhysicsEngine,
        action: np.ndarray,
        bre_override: Optional[bool] = None,
    ) -> tuple[np.ndarray, PhysicsEngine]:
        nominal_engine = copy.deepcopy(engine)
        start_pos = nominal_engine.robot.position.copy()
        self._simulate_action_on_engine(
            nominal_engine,
            action,
            obstacles=None,
            segment_obstacles=None,
            protect_robot=False,
            protect_human=False,
            bre_override=bre_override,
        )
        delta = (nominal_engine.robot.position - start_pos).astype(np.float32)
        return delta, nominal_engine

    def _action_seq_to_nominal_delta_seq(
        self,
        action_seq: np.ndarray,
        engine: Optional[PhysicsEngine] = None,
    ) -> np.ndarray:
        action_seq = np.asarray(action_seq, dtype=np.float32)
        if action_seq.size == 0:
            return np.zeros((0, 2), dtype=np.float32)

        sim = copy.deepcopy(engine if engine is not None else self.physics)
        deltas: list[np.ndarray] = []
        for action in action_seq:
            if self.action_mode == "forward_heading":
                delta, sim = self._forward_heading_action_to_nominal_delta(sim, action)
            else:
                delta = self._action_to_world_delta(action, sim.robot.position, sim.robot.heading)
                self._simulate_delta_on_engine(
                    sim,
                    delta,
                    obstacles=None,
                    segment_obstacles=None,
                    protect_robot=False,
                    protect_human=False,
                )
            deltas.append(delta.astype(np.float32))
        return np.asarray(deltas, dtype=np.float32)

    def _safety_protects_human(self) -> bool:
        return self.safety_mode == "human_robot_qp"

    def _empty_safety_info(self) -> dict:
        return {
            "modified": False,
            "shift": 0.0,
            "constraint_count": 0,
            "min_clearance": float("inf"),
        }

    def _update_episode_safety_stats(self, safety_info: dict):
        self.episode_safety_stats["total_steps"] += 1
        self.episode_safety_stats["total_shift"] += float(safety_info["shift"])
        self.episode_safety_stats["constraint_count"] = max(
            int(self.episode_safety_stats["constraint_count"]),
            int(safety_info["constraint_count"]),
        )
        self.episode_safety_stats["min_clearance"] = min(
            float(self.episode_safety_stats["min_clearance"]),
            float(safety_info["min_clearance"]),
        )
        if safety_info["modified"]:
            self.episode_safety_stats["modified_steps"] += 1

    def _safety_filter_delta(
        self,
        engine: PhysicsEngine,
        nominal_delta: np.ndarray,
        obstacles: Optional[np.ndarray],
        segment_obstacles: Optional[np.ndarray],
        nominal_preview: Optional[PhysicsEngine] = None,
    ) -> tuple[np.ndarray, PhysicsEngine, dict]:
        nominal_delta = np.asarray(nominal_delta, dtype=np.float32).reshape(2)
        info = self._empty_safety_info()
        if self.safety_mode == "off":
            trial_engine = copy.deepcopy(engine)
            self._simulate_delta_on_engine(
                trial_engine,
                nominal_delta,
                obstacles=None,
                segment_obstacles=None,
                protect_robot=False,
                protect_human=False,
            )
            return nominal_delta.astype(np.float32), trial_engine, info

        protect_human = self._safety_protects_human()
        robot_pos = engine.robot.position.copy()
        human_pos = engine.human.position.copy()
        robot_heading = float(engine.robot.heading)
        if nominal_preview is None:
            nominal_preview = copy.deepcopy(engine)
            self._simulate_delta_on_engine(
                nominal_preview,
                nominal_delta,
                obstacles=obstacles,
                segment_obstacles=segment_obstacles,
                protect_robot=False,
                protect_human=False,
            )

        extra_entities = [
            ("robot_future", nominal_preview.robot.position.copy(), self.physics.robot_radius)
        ]
        if protect_human:
            extra_entities.append(
                ("human_future", nominal_preview.human.position.copy(), self.physics.human_radius)
            )

        qp = self.safety_filter.project_delta(
            ref_delta=nominal_delta,
            robot_pos=engine.robot.position,
            robot_radius=self.physics.robot_radius,
            human_pos=engine.human.position,
            human_radius=self.physics.human_radius,
            circle_obstacles=obstacles,
            segment_obstacles=segment_obstacles,
            include_human=protect_human,
            extra_entities=extra_entities,
        )
        qp_delta = qp.delta.astype(np.float32)
        chosen_delta = qp_delta.copy()
        trial_engine = copy.deepcopy(engine)
        collided, _ = self._simulate_delta_on_engine(
            trial_engine,
            chosen_delta,
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
            for scale in self.safety_backoff_scales[1:]:
                backoff_delta = (chosen_delta * float(scale)).astype(np.float32)
                backoff_engine = copy.deepcopy(engine)
                collided, _ = self._simulate_delta_on_engine(
                    backoff_engine,
                    backoff_delta,
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
            "robot_pos": robot_pos.astype(np.float32),
            "human_pos": human_pos.astype(np.float32),
            "robot_heading": float(robot_heading),
            "nominal_delta": nominal_delta.astype(np.float32),
            "nominal_delta_norm": float(np.linalg.norm(nominal_delta)),
            "nominal_preview_robot_pos": nominal_preview.robot.position.copy().astype(np.float32),
            "nominal_preview_human_pos": nominal_preview.human.position.copy().astype(np.float32),
            "protect_human": bool(protect_human),
            "qp_delta": qp_delta.astype(np.float32),
            "qp_delta_norm": float(np.linalg.norm(qp_delta)),
            "qp_modified": bool(qp.modified),
            "qp_constraint_count": int(qp.constraint_count),
            "qp_total_constraint_count": int(getattr(qp, "total_constraint_count", qp.constraint_count)),
            "qp_ref_feasible": bool(getattr(qp, "ref_feasible", True)),
            "qp_candidate_count": int(getattr(qp, "candidate_count", 0)),
            "qp_best_candidate_kind": str(getattr(qp, "best_candidate_kind", "unknown")),
            "qp_best_candidate_constraints": list(
                getattr(qp, "best_candidate_constraints", [])
            ),
            "qp_selected_constraints": list(getattr(qp, "selected_constraints", [])),
            "collision_after_qp": bool(collision_after_qp),
            "backoff_attempts": backoff_attempts,
            "backoff_applied": bool(backoff_applied),
            "backoff_scale": float(backoff_scale),
            "stop_triggered": False,
            "stop_clearance_threshold": float(self.safety_stop_clearance),
            "stop_reason": None,
            "resolution_stage": resolution_stage,
        }
        # if info["min_clearance"] < self.safety_stop_clearance:
        #     chosen_delta = np.zeros((2,), dtype=np.float32)
        #     trial_engine = copy.deepcopy(engine)
        #     _collided, _ = self._simulate_delta_on_engine(
        #         trial_engine,
        #         chosen_delta,
        #         obstacles=obstacles,
        #         segment_obstacles=segment_obstacles,
        #         protect_robot=True,
        #         protect_human=protect_human,
        #     )
        #     info["modified"] = True
        #     info["shift"] = float(np.linalg.norm(chosen_delta - nominal_delta))
        #     info["stop_triggered"] = True
        #     info["stop_reason"] = "min_clearance_below_stop_threshold"
        #     info["resolution_stage"] = "stop"
        info["final_delta"] = chosen_delta.astype(np.float32)
        info["final_delta_norm"] = float(np.linalg.norm(chosen_delta))
        return chosen_delta.astype(np.float32), trial_engine, info

    def _forward_only_safety_filter_delta(
        self,
        engine: PhysicsEngine,
        nominal_delta: np.ndarray,
        obstacles: Optional[np.ndarray],
        segment_obstacles: Optional[np.ndarray],
        bre_override: Optional[bool] = None,
    ) -> tuple[np.ndarray, PhysicsEngine, dict]:
        nominal_delta = np.asarray(nominal_delta, dtype=np.float32).reshape(2)
        nominal_norm = float(np.linalg.norm(nominal_delta))
        has_obstacles = (
            obstacles is not None
            and len(obstacles) > 0
        ) or (
            segment_obstacles is not None
            and len(segment_obstacles) > 0
        )
        if self.safety_mode == "off" or not has_obstacles or nominal_norm < 1e-6:
            trial_engine = copy.deepcopy(engine)
            self._simulate_delta_on_engine(
                trial_engine,
                nominal_delta,
                obstacles=None,
                segment_obstacles=None,
                protect_robot=False,
                protect_human=False,
                bre_override=bre_override,
            )
            info = self._empty_safety_info()
            info.update(
                {
                    "forward_only": True,
                    "robot_heading": float(engine.robot.heading),
                    "nominal_delta": nominal_delta.astype(np.float32),
                    "nominal_delta_norm": nominal_norm,
                    "qp_delta": nominal_delta.astype(np.float32),
                    "qp_delta_norm": nominal_norm,
                    "final_delta": nominal_delta.astype(np.float32),
                    "final_delta_norm": nominal_norm,
                    "resolution_stage": "forward_only",
                    "protect_human": bool(self._safety_protects_human()),
                    "qp_modified": False,
                    "qp_ref_feasible": True,
                    "collision_after_qp": False,
                    "backoff_applied": False,
                    "backoff_scale": 1.0,
                    "stop_triggered": False,
                    "stop_clearance_threshold": float(self.safety_stop_clearance),
                    "stop_reason": None,
                }
            )
            return nominal_delta.astype(np.float32), trial_engine, info

        protect_human = self._safety_protects_human()
        attempts: list[dict] = []
        chosen_delta = np.zeros((2,), dtype=np.float32)
        chosen_engine = copy.deepcopy(engine)
        chosen_qp = None
        chosen_scale = 0.0
        chosen_collision = False
        resolution_stage = "forward_only_stop"

        for scale in self.safety_backoff_scales:
            candidate_delta = (nominal_delta * float(scale)).astype(np.float32)
            candidate_preview = copy.deepcopy(engine)
            collided, _ = self._simulate_delta_on_engine(
                candidate_preview,
                candidate_delta,
                obstacles=obstacles,
                segment_obstacles=segment_obstacles,
                protect_robot=True,
                protect_human=protect_human,
                bre_override=bre_override,
            )
            extra_entities = [
                (
                    "robot_future",
                    candidate_preview.robot.position.copy(),
                    self.physics.robot_radius,
                )
            ]
            if protect_human:
                extra_entities.append(
                    (
                        "human_future",
                        candidate_preview.human.position.copy(),
                        self.physics.human_radius,
                    )
                )
            qp = self.safety_filter.project_delta(
                ref_delta=candidate_delta,
                robot_pos=engine.robot.position,
                robot_radius=self.physics.robot_radius,
                human_pos=engine.human.position,
                human_radius=self.physics.human_radius,
                circle_obstacles=obstacles,
                segment_obstacles=segment_obstacles,
                include_human=protect_human,
                extra_entities=extra_entities,
            )
            ref_feasible = bool(getattr(qp, "ref_feasible", True)) and not bool(qp.modified)
            attempts.append(
                {
                    "scale": float(scale),
                    "delta": candidate_delta.astype(np.float32),
                    "collided": bool(collided),
                    "qp_modified": bool(qp.modified),
                    "qp_ref_feasible": bool(getattr(qp, "ref_feasible", True)),
                    "min_clearance": float(qp.min_clearance),
                    "constraint_count": int(qp.constraint_count),
                }
            )
            if ref_feasible and not collided:
                chosen_delta = candidate_delta
                chosen_engine = candidate_preview
                chosen_qp = qp
                chosen_scale = float(scale)
                chosen_collision = bool(collided)
                resolution_stage = "forward_only" if scale == 1.0 else "forward_only_backoff"
                break

        if chosen_qp is None:
            chosen_delta = np.zeros((2,), dtype=np.float32)
            chosen_engine = copy.deepcopy(engine)
            _collided, _ = self._simulate_delta_on_engine(
                chosen_engine,
                chosen_delta,
                obstacles=obstacles,
                segment_obstacles=segment_obstacles,
                protect_robot=True,
                protect_human=protect_human,
                bre_override=bre_override,
            )
            extra_entities = [
                ("robot_future", chosen_engine.robot.position.copy(), self.physics.robot_radius)
            ]
            if protect_human:
                extra_entities.append(
                    ("human_future", chosen_engine.human.position.copy(), self.physics.human_radius)
                )
            chosen_qp = self.safety_filter.project_delta(
                ref_delta=chosen_delta,
                robot_pos=engine.robot.position,
                robot_radius=self.physics.robot_radius,
                human_pos=engine.human.position,
                human_radius=self.physics.human_radius,
                circle_obstacles=obstacles,
                segment_obstacles=segment_obstacles,
                include_human=protect_human,
                extra_entities=extra_entities,
            )
            chosen_collision = bool(_collided)

        info = {
            "modified": bool(
                abs(chosen_scale - 1.0) > 1e-6
                or np.linalg.norm(chosen_delta - nominal_delta) > 1e-5
            ),
            "shift": float(np.linalg.norm(chosen_delta - nominal_delta)),
            "constraint_count": int(chosen_qp.constraint_count),
            "min_clearance": float(chosen_qp.min_clearance),
            "robot_pos": engine.robot.position.copy().astype(np.float32),
            "human_pos": engine.human.position.copy().astype(np.float32),
            "robot_heading": float(engine.robot.heading),
            "nominal_delta": nominal_delta.astype(np.float32),
            "nominal_delta_norm": nominal_norm,
            "nominal_preview_robot_pos": chosen_engine.robot.position.copy().astype(np.float32),
            "nominal_preview_human_pos": chosen_engine.human.position.copy().astype(np.float32),
            "protect_human": bool(protect_human),
            "qp_delta": chosen_qp.delta.astype(np.float32),
            "qp_delta_norm": float(np.linalg.norm(chosen_qp.delta)),
            "qp_modified": bool(chosen_qp.modified),
            "qp_constraint_count": int(chosen_qp.constraint_count),
            "qp_total_constraint_count": int(
                getattr(chosen_qp, "total_constraint_count", chosen_qp.constraint_count)
            ),
            "qp_ref_feasible": bool(getattr(chosen_qp, "ref_feasible", True)),
            "qp_candidate_count": int(getattr(chosen_qp, "candidate_count", 0)),
            "qp_best_candidate_kind": str(getattr(chosen_qp, "best_candidate_kind", "unknown")),
            "qp_best_candidate_constraints": list(
                getattr(chosen_qp, "best_candidate_constraints", [])
            ),
            "qp_selected_constraints": list(getattr(chosen_qp, "selected_constraints", [])),
            "collision_after_qp": bool(chosen_collision),
            "backoff_attempts": attempts,
            "backoff_applied": bool(chosen_scale < 1.0),
            "backoff_scale": float(chosen_scale),
            "stop_triggered": bool(chosen_scale == 0.0),
            "stop_clearance_threshold": float(self.safety_stop_clearance),
            "stop_reason": "forward_only_no_feasible_scale" if chosen_scale == 0.0 else None,
            "resolution_stage": resolution_stage,
            "forward_only": True,
            "final_delta": chosen_delta.astype(np.float32),
            "final_delta_norm": float(np.linalg.norm(chosen_delta)),
        }
        return chosen_delta.astype(np.float32), chosen_engine, info

    def _safety_filter_action(
        self,
        engine: PhysicsEngine,
        nominal_action: np.ndarray,
        obstacles: Optional[np.ndarray],
        segment_obstacles: Optional[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, PhysicsEngine, dict]:
        nominal_action = np.asarray(nominal_action, dtype=np.float32)
        info = self._empty_safety_info()
        ref_delta = self._action_to_world_delta(
            nominal_action, engine.robot.position, engine.robot.heading
        )
        if self.safety_mode == "off":
            return nominal_action, ref_delta, copy.deepcopy(engine), info

        nominal_preview = copy.deepcopy(engine)
        self._simulate_action_on_engine(
            nominal_preview,
            nominal_action,
            obstacles=obstacles,
            segment_obstacles=segment_obstacles,
            protect_robot=False,
            protect_human=False,
        )
        chosen_delta, trial_engine, info = self._safety_filter_delta(
            engine,
            ref_delta,
            obstacles=obstacles,
            segment_obstacles=segment_obstacles,
            nominal_preview=nominal_preview,
        )
        chosen_action = self._world_delta_to_action(
            chosen_delta,
            engine.robot.position,
            engine.robot.heading,
        )
        return chosen_action.astype(np.float32), chosen_delta.astype(np.float32), trial_engine, info

    def _apply_forward_heading_safety_filter(
        self,
        action_seq: np.ndarray,
        preserve_heading_mask: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        action_seq = np.asarray(action_seq, dtype=np.float32)
        if preserve_heading_mask is None:
            preserve_heading_mask = np.zeros((len(action_seq),), dtype=bool)
        else:
            preserve_heading_mask = np.asarray(preserve_heading_mask, dtype=bool).reshape(-1)
            if preserve_heading_mask.shape[0] != len(action_seq):
                preserve_heading_mask = np.zeros((len(action_seq),), dtype=bool)
        stats = {
            "applied": self.safety_mode != "off",
            "modified_steps": 0,
            "total_steps": int(len(action_seq)),
            "mean_shift": 0.0,
            "constraint_count": 0,
            "min_clearance": float("inf"),
            "input_point_obstacle_count": 0,
            "input_segment_obstacle_count": 0,
        }
        if action_seq.size == 0:
            self.last_safety_stats = stats
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0, 2), dtype=np.float32),
                [],
            )

        obstacles, segments = self._safety_obstacle_inputs()
        stats["input_point_obstacle_count"] = int(
            len(obstacles) if obstacles is not None else 0
        )
        stats["input_segment_obstacle_count"] = int(
            len(segments) if segments is not None else 0
        )
        sim = copy.deepcopy(self.physics)
        nominal_deltas: list[np.ndarray] = []
        safe_deltas: list[np.ndarray] = []
        safety_infos: list[dict] = []
        shifts = []

        for idx, nominal_action in enumerate(action_seq):
            if bool(preserve_heading_mask[idx]):
                nominal_delta, _nominal_preview = self._forward_heading_action_to_nominal_delta(
                    sim,
                    nominal_action,
                    bre_override=False,
                )
                chosen_delta, trial_engine, info = self._forward_only_safety_filter_delta(
                    sim,
                    nominal_delta,
                    obstacles=obstacles,
                    segment_obstacles=segments,
                    bre_override=False,
                )
            else:
                nominal_delta, nominal_preview = self._forward_heading_action_to_nominal_delta(
                    sim, nominal_action
                )
                if self.safety_mode == "off":
                    chosen_delta = nominal_delta
                    trial_engine = nominal_preview
                    info = self._empty_safety_info()
                else:
                    chosen_delta, trial_engine, info = self._safety_filter_delta(
                        sim,
                        nominal_delta,
                        obstacles=obstacles,
                        segment_obstacles=segments,
                        nominal_preview=nominal_preview,
                    )

            nominal_deltas.append(nominal_delta.astype(np.float32))
            safe_deltas.append(chosen_delta.astype(np.float32))
            safety_infos.append(info)
            stats["constraint_count"] = max(stats["constraint_count"], int(info["constraint_count"]))
            stats["min_clearance"] = min(stats["min_clearance"], float(info["min_clearance"]))
            shifts.append(float(info["shift"]))
            if info["modified"]:
                stats["modified_steps"] += 1
            sim = trial_engine

        if shifts:
            stats["mean_shift"] = float(np.mean(shifts))
        self.last_safety_stats = stats
        return (
            np.asarray(nominal_deltas, dtype=np.float32),
            np.asarray(safe_deltas, dtype=np.float32),
            safety_infos,
        )

    def _apply_safety_filter(self, action_seq: np.ndarray) -> np.ndarray:
        action_seq = np.asarray(action_seq, dtype=np.float32)
        stats = {
            "applied": self.safety_mode != "off",
            "modified_steps": 0,
            "total_steps": int(len(action_seq)),
            "mean_shift": 0.0,
            "constraint_count": 0,
            "min_clearance": float("inf"),
        }
        if (
            self.safety_mode == "off"
            or action_seq.size == 0
        ):
            self.last_safety_stats = stats
            return action_seq

        obstacles, segments = self._safety_obstacle_inputs()
        if (obstacles is None or len(obstacles) == 0) and (segments is None or len(segments) == 0):
            self.last_safety_stats = stats
            return action_seq

        sim = copy.deepcopy(self.physics)
        safe_actions = []
        shifts = []

        for nominal_action in action_seq:
            chosen_action, chosen_delta, trial_engine, info = self._safety_filter_action(
                sim,
                nominal_action,
                obstacles=obstacles,
                segment_obstacles=segments,
            )
            stats["constraint_count"] = max(stats["constraint_count"], int(info["constraint_count"]))
            stats["min_clearance"] = min(stats["min_clearance"], float(info["min_clearance"]))
            safe_actions.append(chosen_action.astype(np.float32))
            sim = trial_engine
            shift = float(info["shift"])
            shifts.append(shift)
            if info["modified"]:
                stats["modified_steps"] += 1

        if shifts:
            stats["mean_shift"] = float(np.mean(shifts))
        self.last_safety_stats = stats
        return np.asarray(safe_actions, dtype=np.float32)

    def _delta_to_control(
        self,
        delta: np.ndarray,
        heading: float,
        dt: Optional[float] = None,
    ) -> Tuple[float, float]:
        if dt is None:
            dt = self.data_dt
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm < 1e-6:
            return 0.0, 0.0
        desired_heading = float(np.arctan2(delta[1], delta[0]))
        heading_error = wrap_angle(desired_heading - heading)
        turn_speed = self.physics.turn_speed
        turn_input = heading_error / (turn_speed * dt) if turn_speed > 0 else 0.0
        turn_input = float(np.clip(turn_input, -1.0, 1.0))

        forward_input = delta_norm / (self.physics.robot_speed * dt)
        forward_input = float(np.clip(forward_input, -1.0, 1.0))
        return forward_input, turn_input

    def _delta_to_safe_control(
        self,
        delta: np.ndarray,
        heading: float,
        dt: Optional[float] = None,
    ) -> tuple[float, float, float]:
        if dt is None:
            dt = self.data_dt
        forward_input, turn_input = self._delta_to_control(delta, heading, dt=dt)
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm < 1e-6:
            return 0.0, 0.0, 1.0

        desired_heading = float(np.arctan2(delta[1], delta[0]))
        heading_error = wrap_angle(desired_heading - heading)
        turn_speed = float(self.physics.turn_speed)
        speed_scale = 1.0
        if self.curvature_slowdown and turn_speed > 0:
            max_turn = turn_speed * dt
            if max_turn > 1e-6:
                ratio = min(1.0, abs(heading_error) / max_turn)
                speed_scale = max(
                    float(self.min_speed_scale),
                    1.0 - float(self.curvature_scale) * ratio,
                )
                forward_input *= speed_scale
        return float(forward_input), float(turn_input), float(speed_scale)

    def _simulate_delta_on_engine(
        self,
        engine: PhysicsEngine,
        delta: np.ndarray,
        obstacles: Optional[np.ndarray],
        segment_obstacles: Optional[np.ndarray],
        protect_robot: bool = True,
        protect_human: bool = True,
        bre_override: Optional[bool] = None,
    ) -> tuple[bool, list[np.ndarray]]:
        forward, turn, _speed_scale = self._delta_to_safe_control(
            delta,
            engine.robot.heading,
            dt=self.data_dt,
        )
        bre = self._interaction_is_tether() if bre_override is None else bool(bre_override)
        engine.set_control(forward, turn, bre)
        points: list[np.ndarray] = []
        for _ in range(int(self.frame_stride)):
            robot_state, _human_state = engine.step()
            points.append(robot_state.position.copy())
            if obstacles is not None or segment_obstacles is not None:
                collided, info = engine.check_collision(obstacles, segment_obstacles=segment_obstacles)
                who = info.get("who") if info else None
                relevant = (
                    collided
                    and (
                        (protect_robot and who == "robot")
                        or (protect_human and who == "human")
                    )
                )
                if relevant:
                    return True, points
        return False, points
    
    def _cycle_safety_mode(self):
        """Cycle safety mode: off -> robot_qp -> human_robot_qp -> off."""
        modes = ["off", "robot_qp", "human_robot_qp"]
        current = normalize_safety_mode(self.safety_mode)
        next_idx = (modes.index(current) + 1) % len(modes)
        new_mode = modes[next_idx]

        # Keep compatibility with action mode constraints
        if new_mode != "off" and self.action_mode not in ("forward_heading", "delta", "velocity"):
            print(
                f"[warn] safety_mode={new_mode} is only supported for "
                f"forward_heading/delta/velocity; fallback to off"
            )
            new_mode = "off"

        self.safety_mode = new_mode

        # Clear caches so the new mode takes effect immediately
        self.cached_action_seq = None
        self.cached_nominal_delta_seq = None
        self.cached_safe_delta_seq = None
        self.cached_safety_info_seq = None
        self.cached_interaction_labels_seq = None
        self.cached_uses_stashed_compliance_plan = False
        self.cached_action_idx = 0
        self.frames_since_inference = 0
        self.cached_control = (0.0, 0.0)
        self.current_action = None
        self.current_delta = None
        self.current_speed_scale = 1.0

        # Clear preview paths
        self.planned_path = None
        self.nominal_planned_path = None
        self.safe_planned_path = None

        if self.log_fp is not None:
            self._log_event("safety_mode_changed", {"safety_mode": self.safety_mode})

        print(f"Safety mode: {self.safety_mode}")

    def _set_range_source(self, new_source: str) -> None:
        new_source = normalize_range_source(new_source)
        if new_source == self.range_source:
            return
        previous_source = self.range_source
        self.ros_io.set_range_source(new_source)
        self.range_source = new_source
        self.current_mid360_points_world = None
        self.current_mid360_point_obstacles = None
        self.current_safety_point_obstacles = None
        self._last_mid360_cloud_seq = None
        self._reset_runtime_caches()
        self._seed_obs_history(
            self.physics.robot.position,
            self.physics.human.position,
        )
        print(
            f"Range input: {self.range_source} "
            f"({self.ros_io.range_source_status()['topic']})"
        )
        self._log_event(
            "range_source_changed",
            {
                "previous_source": previous_source,
                "range_source": self.range_source,
                "topic": self.ros_io.range_source_status()["topic"],
            },
        )

    def _toggle_range_source(self) -> None:
        next_source = (
            "point_cloud"
            if self.range_source == "laser_scan"
            else "laser_scan"
        )
        try:
            self._set_range_source(next_source)
        except Exception as exc:
            print(
                f"[warn] cannot switch range input to {next_source}: "
                f"{type(exc).__name__}: {exc}"
            )

    def _set_pointcloud_mode(self, new_mode: str, *, rebuild_obs_history: bool = True):
        new_mode = normalize_pointcloud_mode(new_mode)
        if self.observation_mode == "mid360" and new_mode == "off":
            raise ValueError("guide_mid360 checkpoint requires pointcloud_mode != 'off'")
        if new_mode == self.pointcloud_mode:
            return
        self.pointcloud_mode = new_mode
        self.current_mid360_points_world = None
        self.current_mid360_point_obstacles = None
        self.current_safety_point_obstacles = None
        self._last_mid360_cloud_seq = None
        self._reset_runtime_caches()
        if rebuild_obs_history:
            self._seed_obs_history(self.physics.robot.position, self.physics.human.position)
        if self.log_fp is not None:
            self._log_event("pointcloud_mode_changed", {"pointcloud_mode": self.pointcloud_mode})

    def _cycle_pointcloud_mode(self):
        """Cycle point-cloud source during runtime."""
        modes = self._supported_pointcloud_modes()
        current = self.pointcloud_mode
        if current not in modes:
            current = modes[0]
        start_idx = modes.index(current)
        for offset in range(1, len(modes) + 1):
            candidate = modes[(start_idx + offset) % len(modes)]
            try:
                self._set_pointcloud_mode(candidate)
                print(f"Point cloud mode: {self.pointcloud_mode}")
                return
            except Exception as exc:
                print(f"[warn] failed to switch point cloud mode to {candidate}: {exc}")
        print(f"Point cloud mode unchanged: {self.pointcloud_mode}")

    def _step(self):
        """Run one hybrid step with a measured robot and selectable human source."""
        self.ros_io.assert_fresh()
        self._synchronize_robot_from_odometry()
        self.collision_happened = False
        self.collision_info = None
        if (
            self.human_source == "detector"
            and not self._refresh_human_from_detector()
        ):
            # Fail closed: never execute a cached action without a fresh,
            # transformable human detection.
            self.physics.set_control(0.0, 0.0, False)
            self.ros_io.stop()
            return self.physics.robot.copy(), self.physics.human.copy()
        if self.paused or self.collision_pause:
            # Freeze simulation state while paused (manual or auto-paused).
            self.physics.set_control(0.0, 0.0, False)
            self.ros_io.stop()
            return self.physics.robot.copy(), self.physics.human.copy()

        self._update_timed_bre_toggle()

        forward = 0.0
        turn = 0.0
        action = None
        delta = None
        is_data_step = (self.frame_count % self.frame_stride == 0)
        # Sample at the robot loop rate (normally 20 Hz), matching the
        # frame-based feature thresholds used by segmentation.py.
        self._update_interaction_segmentation()

        if not self.paused:
            if self.use_policy and self.policy is not None:
                # Update policy/action at data rate; hold control between data steps.
                if is_data_step:
                    if (self.cached_action_seq is None or
                        self.cached_action_idx >= len(self.cached_action_seq) or
                        self.frames_since_inference >= self.inference_interval):
                        # Run inference
                        # Synchronize CUDA so the printed diffusion time includes
                        # the actual GPU work rather than only kernel submission.
                        self._synchronize_timing_device()
                        diffusion_start = time.perf_counter()
                        action_seq = self._predict_action()
                        policy_action_seq = action_seq.copy()
                        stashed_compliance_info = {
                            "using_stashed_compliance_plan": False,
                            "stashed_source": "policy",
                            "stashed_len": int(
                                len(self.stashed_guide_action_seq)
                                if self.stashed_guide_action_seq is not None
                                else 0
                            ),
                            "stashed_cursor": int(self.stashed_guide_cursor),
                            "stashed_used_len": 0,
                            "stashed_front_half_len": 0,
                        }
                        self.using_stashed_compliance_plan = False
                        self._synchronize_timing_device()
                        diffusion_time_ms = (
                            time.perf_counter() - diffusion_start
                        ) * 1000.0
                        raw_nominal_delta_seq = self._action_seq_to_nominal_delta_seq(policy_action_seq)
                        raw_nominal_path = self._deltas_to_path(
                            raw_nominal_delta_seq,
                            protect_robot=False,
                            protect_human=False,
                        )
                        raw_heading_delta = self._path_heading_delta(raw_nominal_path)

                        # Cheap SafeFilter input reduction:
                        # 1) crop to a corridor around the raw diffusion path;
                        # 2) keep at most one point in each 2D voxel.
                        safety_preprocess_start = time.perf_counter()
                        safety_point_stats = self._prepare_safety_point_obstacles(
                            raw_nominal_path
                        )
                        safety_preprocess_time_ms = (
                            time.perf_counter() - safety_preprocess_start
                        ) * 1000.0

                        self._write_planning_eval(policy_action_seq, diffusion_time_ms)
                        safety_filter_start = time.perf_counter()
                        if self.action_mode == "forward_heading":
                            obstacles, segments = self._safety_obstacle_inputs()
                            labels = self._interaction_labels_for_action_seq(len(policy_action_seq))
                            if self._current_interaction_label() == "guide":
                                self._stash_guide_action_seq(policy_action_seq, source="policy")
                            elif np.any(self._compliance_mask_from_labels(labels)):
                                action_seq, stashed_compliance_info = self._stashed_compliance_action_seq(
                                    max(1, len(policy_action_seq) // 2),
                                    policy_action_seq,
                                )
                                self.using_stashed_compliance_plan = bool(
                                    stashed_compliance_info.get(
                                        "using_stashed_compliance_plan",
                                        False,
                                    )
                                )
                            action_seq = self._apply_interaction_aware_compliance_control(
                                action_seq,
                                obstacles=obstacles,
                                segment_obstacles=segments,
                            )
                        self.cached_action_seq = action_seq
                        self.cached_uses_stashed_compliance_plan = bool(
                            stashed_compliance_info.get("using_stashed_compliance_plan", False)
                        )
                        self.cached_action_idx = 0
                        self.frames_since_inference = 0

                        preview_action_seq = None
                        if self.action_mode == "forward_heading" and self.safety_mode != "off":
                            preserve_heading_mask = None
                            if self.cached_interaction_labels_seq is not None:
                                preserve_heading_mask = np.isin(
                                    np.asarray(self.cached_interaction_labels_seq, dtype=str),
                                    np.array(["leash", "tether"], dtype=str),
                                )
                            (
                                nominal_delta_seq,
                                safe_delta_seq,
                                safety_info_seq,
                            ) = self._apply_forward_heading_safety_filter(
                                action_seq,
                                preserve_heading_mask=preserve_heading_mask,
                            )
                            self.cached_nominal_delta_seq = nominal_delta_seq
                            self.cached_safe_delta_seq = safe_delta_seq
                            self.cached_safety_info_seq = safety_info_seq
                            preview_bre_override = (
                                False
                                if preserve_heading_mask is not None
                                and bool(np.any(preserve_heading_mask))
                                else None
                            )
                            self.nominal_planned_path = self._deltas_to_path(
                                nominal_delta_seq,
                                protect_robot=False,
                                protect_human=False,
                                bre_override=preview_bre_override,
                            )
                            self.safe_planned_path = self._deltas_to_path(
                                safe_delta_seq,
                                bre_override=preview_bre_override,
                            )
                            self.planned_path = self.safe_planned_path
                        else:
                            self.cached_nominal_delta_seq = None
                            self.cached_safe_delta_seq = None
                            self.cached_safety_info_seq = None
                            preview_action_seq = self._apply_safety_filter(action_seq)
                            self.nominal_planned_path = self._actions_to_path(
                                self.physics.robot.position, preview_action_seq
                            )
                            self.safe_planned_path = None
                            self.planned_path = self.nominal_planned_path

                        safety_filter_time_ms = (
                            safety_preprocess_time_ms
                            + (time.perf_counter() - safety_filter_start) * 1000.0
                        )
                        print(
                            "[timing] "
                            f"diffusion={diffusion_time_ms:.1f} ms | "
                            f"safefilter={safety_filter_time_ms:.1f} ms "
                            f"(preprocess={safety_preprocess_time_ms:.1f} ms) | "
                            "points="
                            f"{safety_point_stats['raw_count']}"
                            f"->{safety_point_stats['corridor_count']}"
                            f"->{safety_point_stats['sparse_count']}"
                        )

                        self.latest_nominal_heading_delta = self._path_heading_delta(
                            self.nominal_planned_path
                        )

                        if self.cached_action_seq is not None and len(self.cached_action_seq) > 0:
                            norms = np.linalg.norm(self.cached_action_seq, axis=1)
                            raw_heading = action_seq[:, 1] if action_seq.shape[1] > 1 else np.zeros(
                                (len(action_seq),), dtype=np.float32
                            )
                            log_payload = {
                                "action_len": int(len(self.cached_action_seq)),
                                "action_mean_norm": float(np.mean(norms)),
                                "action_max_norm": float(np.max(norms)),

                                "raw_forward_mean": float(np.mean(action_seq[:, 0])),
                                "raw_forward_std": float(np.std(action_seq[:, 0])),
                                "raw_heading_mean": float(np.mean(raw_heading)),
                                "raw_heading_std": float(np.std(raw_heading)),
                                "raw_heading_maxabs": float(np.max(np.abs(raw_heading))),

                                "raw_first3": np.round(action_seq[:3], 4).tolist(),
                                "raw_action_first3": np.round(action_seq[:3], 4).tolist(),
                                "policy_raw_first3": np.round(policy_action_seq[:3], 4).tolist(),
                                "policy_raw_forward_mean": float(np.mean(policy_action_seq[:, 0])),
                                "policy_raw_heading_maxabs": float(
                                    np.max(np.abs(policy_action_seq[:, 1]))
                                    if policy_action_seq.shape[1] > 1
                                    else 0.0
                                ),
                                "using_stashed_compliance_plan": bool(
                                    stashed_compliance_info.get(
                                        "using_stashed_compliance_plan",
                                        False,
                                    )
                                ),
                                "stashed_compliance_source": str(
                                    stashed_compliance_info.get("stashed_source", "policy")
                                ),
                                "stashed_compliance_len": int(
                                    stashed_compliance_info.get("stashed_len", 0)
                                ),
                                "stashed_compliance_cursor": int(
                                    stashed_compliance_info.get("stashed_cursor", 0)
                                ),
                                "stashed_compliance_used_len": int(
                                    stashed_compliance_info.get("stashed_used_len", 0)
                                ),
                                "stashed_compliance_front_half_len": int(
                                    stashed_compliance_info.get("stashed_front_half_len", 0)
                                ),

                                "safety_mode": self.safety_mode,
                                "safety_modified_steps": int(
                                    self.last_safety_stats.get("modified_steps", 0)
                                ),
                                "safety_mean_shift": float(
                                    self.last_safety_stats.get("mean_shift", 0.0)
                                ),
                                "safety_input_point_obstacle_count": int(
                                    self.last_safety_stats.get(
                                        "input_point_obstacle_count", 0
                                    )
                                ),
                                "safety_input_segment_obstacle_count": int(
                                    self.last_safety_stats.get(
                                        "input_segment_obstacle_count", 0
                                    )
                                ),
                                "safety_point_obstacle_diameter": float(
                                    2.0 * self.mid360_point_obstacle_radius
                                ),
                                "safety_pointcloud_raw_count": int(
                                    safety_point_stats["raw_count"]
                                ),
                                "safety_pointcloud_corridor_count": int(
                                    safety_point_stats["corridor_count"]
                                ),
                                "safety_pointcloud_sparse_count": int(
                                    safety_point_stats["sparse_count"]
                                ),
                                "safety_path_corridor": float(
                                    self.safety_path_corridor
                                ),
                                "safety_point_spacing": float(
                                    self.safety_point_spacing
                                ),
                                "diffusion_time_ms": float(diffusion_time_ms),
                                "safefilter_time_ms": float(
                                    safety_filter_time_ms
                                ),
                                "interaction_label": self._current_interaction_label(),
                                "interaction_label_source": (
                                    "segmentation"
                                    if self.interaction_segmenter.has_label
                                    else "manual_fallback"
                                ),
                                "segmentation_sample_count": int(
                                    len(self.interaction_segmenter.samples)
                                ),
                                "segmentation_decode_time_ms": float(
                                    self.interaction_segmenter.last_decode_ms
                                ),
                                "compliance_mode": str(
                                    self.last_compliance_stats.get("mode", "interaction_aware")
                                ),
                                "compliance_steps": int(
                                    self.last_compliance_stats.get("compliance_steps", 0)
                                ),
                                "compliance_guide_steps": int(
                                    self.last_compliance_stats.get("guide_steps", 0)
                                ),
                                "compliance_modified_steps": int(
                                    self.last_compliance_stats.get("modified_steps", 0)
                                ),
                                "compliance_mean_action_shift": float(
                                    self.last_compliance_stats.get("mean_action_shift", 0.0)
                                ),
                                "compliance_state_counts": dict(
                                    self.last_compliance_stats.get("state_counts", {})
                                ),
                            }
                            if self.cached_safe_delta_seq is not None:
                                safe_delta_norms = np.linalg.norm(self.cached_safe_delta_seq, axis=1)
                                log_payload.update(
                                    {
                                        "safe_mean_norm": float(np.mean(safe_delta_norms)),
                                        "safe_max_norm": float(np.max(safe_delta_norms)),
                                        "safe_delta_norm_mean": float(np.mean(safe_delta_norms)),
                                        "safe_delta_norm_max": float(np.max(safe_delta_norms)),
                                        "safe_delta_x_mean": float(np.mean(self.cached_safe_delta_seq[:, 0])),
                                        "safe_delta_y_mean": float(np.mean(self.cached_safe_delta_seq[:, 1])),
                                        "raw_action_seq": np.round(action_seq, 4).tolist(),
                                        "safe_delta_seq": np.round(self.cached_safe_delta_seq, 4).tolist(),
                                        "safe_first3": np.round(self.cached_safe_delta_seq[:3], 4).tolist(),
                                        "safe_delta_first3": np.round(
                                            self.cached_safe_delta_seq[:3], 4
                                        ).tolist(),
                                    }
                                )
                                if self.debug_qp_log and self.cached_safety_info_seq is not None:
                                    qp_min_clearance_seq = [
                                        float(info.get("min_clearance", float("inf")))
                                        for info in self.cached_safety_info_seq
                                    ]
                                    qp_constraint_count_seq = [
                                        int(info.get("constraint_count", 0))
                                        for info in self.cached_safety_info_seq
                                    ]
                                    qp_stop_steps = [
                                        int(idx)
                                        for idx, info in enumerate(self.cached_safety_info_seq)
                                        if bool(info.get("stop_triggered", False))
                                    ]
                                    qp_backoff_steps = [
                                        int(idx)
                                        for idx, info in enumerate(self.cached_safety_info_seq)
                                        if bool(info.get("backoff_applied", False))
                                    ]
                                    log_payload.update(
                                        {
                                            "robot_qp_min_clearance_seq": [
                                                round(v, 4) if np.isfinite(v) else "inf"
                                                for v in qp_min_clearance_seq
                                            ],
                                            "robot_qp_constraint_count_seq": qp_constraint_count_seq,
                                            "robot_qp_stop_steps": qp_stop_steps,
                                            "robot_qp_backoff_steps": qp_backoff_steps,
                                            "robot_qp_debug_seq": [
                                                self._serialize_safety_info(info)
                                                for info in self.cached_safety_info_seq
                                            ],
                                        }
                                    )
                            else:
                                safe_norms = np.linalg.norm(preview_action_seq, axis=1)
                                log_payload.update(
                                    {
                                        "safe_mean_norm": float(np.mean(safe_norms)),
                                        "safe_max_norm": float(np.max(safe_norms)),
                                        "safe_forward_mean": float(np.mean(preview_action_seq[:, 0])),
                                        "safe_forward_std": float(np.std(preview_action_seq[:, 0])),
                                        "safe_heading_mean": float(np.mean(preview_action_seq[:, 1])),
                                        "safe_heading_std": float(np.std(preview_action_seq[:, 1])),
                                        "safe_heading_maxabs": float(
                                            np.max(np.abs(preview_action_seq[:, 1]))
                                        ),
                                        "safe_first3": np.round(preview_action_seq[:3], 4).tolist(),
                                    }
                                )

                            self._log_event("inference", log_payload)
                            if self.debug_policy:
                                self.debug_inference_count += 1
                                preview_count = min(self.debug_preview_limit, len(self.cached_action_seq))
                                preview_actions = np.round(self.cached_action_seq[:preview_count], 4).tolist()
                                if self.cached_safe_delta_seq is not None:
                                    preview_safe_deltas = np.round(
                                        self.cached_safe_delta_seq[:preview_count], 4
                                    ).tolist()
                                    safe_delta_norms = np.linalg.norm(self.cached_safe_delta_seq, axis=1)
                                    print(
                                        "[debug policy] "
                                        f"inference={self.debug_inference_count} "
                                        f"len={len(self.cached_action_seq)} "
                                        f"raw_actions={preview_actions} "
                                        f"safe_deltas={preview_safe_deltas} "
                                        f"raw_heading_mean={float(np.mean(raw_heading)):.4f} "
                                        f"raw_heading_maxabs={float(np.max(np.abs(raw_heading))):.4f} "
                                        f"safe_delta_norm_mean={float(np.mean(safe_delta_norms)):.4f} "
                                        f"safe_delta_norm_max={float(np.max(safe_delta_norms)):.4f}"
                                    )
                                else:
                                    print(
                                        "[debug policy] "
                                        f"inference={self.debug_inference_count} "
                                        f"len={len(self.cached_action_seq)} "
                                        f"mean_norm={float(np.mean(norms)):.4f} "
                                        f"max_norm={float(np.max(norms)):.4f} "
                                        f"actions={preview_actions}"
                                    )
                    else:
                        # Reuse cached actions
                        self.frames_since_inference += 1

                    # Get current action from cached sequence
                    action_idx = 0
                    if self.cached_action_seq is not None and len(self.cached_action_seq) > 0:
                        action_idx = min(self.cached_action_idx, len(self.cached_action_seq) - 1)
                        action = self.cached_action_seq[action_idx]
                        # Hold the last action once the cached sequence is exhausted.
                        self.cached_action_idx = min(action_idx + 1, len(self.cached_action_seq) - 1)
                        if (
                            self.cached_uses_stashed_compliance_plan
                            and self.stashed_guide_action_seq is not None
                            and len(self.stashed_guide_action_seq) > 0
                        ):
                            stashed_front_half_len = max(
                                1,
                                len(self.stashed_guide_action_seq) // 2,
                            )
                            self.stashed_guide_cursor = min(
                                int(self.stashed_guide_cursor) + 1,
                                stashed_front_half_len,
                            )
                    else:
                        action = np.zeros(self.action_dim)

                    safety_info = None
                    if self.safety_mode != "off" and self.action_mode == "forward_heading":
                        if self.cached_safe_delta_seq is not None and len(self.cached_safe_delta_seq) > 0:
                            delta_idx = min(action_idx, len(self.cached_safe_delta_seq) - 1)
                            delta = self.cached_safe_delta_seq[delta_idx].astype(np.float32)
                            if (
                                self.cached_safety_info_seq is not None
                                and delta_idx < len(self.cached_safety_info_seq)
                            ):
                                safety_info = self.cached_safety_info_seq[delta_idx]
                            else:
                                safety_info = self._empty_safety_info()
                        else:
                            obstacles, segments = self._safety_obstacle_inputs()
                            nominal_delta, nominal_preview = self._forward_heading_action_to_nominal_delta(
                                self.physics, action
                            )
                            if self._current_interaction_label() in ("leash", "tether"):
                                nominal_delta, nominal_preview = (
                                    self._forward_heading_action_to_nominal_delta(
                                        self.physics,
                                        action,
                                        bre_override=False,
                                    )
                                )
                                delta, _trial_engine, safety_info = self._forward_only_safety_filter_delta(
                                    self.physics,
                                    nominal_delta,
                                    obstacles=obstacles,
                                    segment_obstacles=segments,
                                    bre_override=False,
                                )
                            else:
                                delta, _trial_engine, safety_info = self._safety_filter_delta(
                                    self.physics,
                                    nominal_delta,
                                    obstacles=obstacles,
                                    segment_obstacles=segments,
                                    nominal_preview=nominal_preview,
                                )
                        self._update_episode_safety_stats(safety_info)
                        forward, turn, speed_scale = self._delta_to_safe_control(
                            delta,
                            self.physics.robot.heading,
                        )
                    elif self.safety_mode != "off":
                        obstacles, segments = self._safety_obstacle_inputs()
                        safe_action, delta, _trial_engine, safety_info = self._safety_filter_action(
                            self.physics,
                            action,
                            obstacles=obstacles,
                            segment_obstacles=segments,
                        )
                        action = safe_action
                        self._update_episode_safety_stats(safety_info)
                        forward, turn, speed_scale = self._delta_to_safe_control(
                            delta,
                            self.physics.robot.heading,
                        )
                    else:
                        delta, forward, turn, speed_scale = self._action_to_execution(
                            action,
                            self.physics.robot.position,
                            self.physics.robot.heading,
                        )
                    self.current_action = action
                    self.current_speed_scale = speed_scale
                    self.current_delta = delta
                    self.cached_control = (forward, turn)
                    if is_data_step and self.safety_mode == "robot_qp" and self.action_mode == "forward_heading":
                        self._log_robot_qp_step(
                            action_idx=action_idx,
                            action=action,
                            delta=delta,
                            forward=forward,
                            turn=turn,
                            speed_scale=speed_scale,
                            safety_info=safety_info,
                        )
                else:
                    forward, turn = self.cached_control
                    action = self.current_action
                    delta = self.current_delta
            else:
                forward, turn = self._get_manual_control()
                self.planned_path = None
                self.nominal_planned_path = None
                self.safe_planned_path = None
                # Reset cache when switching to manual
                self.cached_action_seq = None
                self.cached_nominal_delta_seq = None
                self.cached_safe_delta_seq = None
                self.cached_safety_info_seq = None
                self.cached_interaction_labels_seq = None
                self.cached_uses_stashed_compliance_plan = False
                self.cached_action_idx = 0
                self.frames_since_inference = 0
                self.cached_control = (forward, turn)
                self.current_action = None
                self.current_delta = None
                self.current_speed_scale = 1.0

        # self.bre remains the simulator's hidden interaction state. The
        # controller never reads it while segmentation has a valid label.
        self.physics.set_control(forward, turn, self.bre)
        self.ros_io.publish_control(forward, turn)
        _simulated_robot_state, human_state = self.physics.step()
        self._synchronize_robot_from_odometry()
        if self.human_source == "detector":
            if not self._refresh_human_from_detector():
                # Hold the last measured target for display/state consistency.
                self._apply_detector_human_state()
                self.ros_io.stop()
            human_state = self.physics.human.copy()
        robot_state = self.physics.robot.copy()

        if self._check_collision():
            return self.physics.robot.copy(), self.physics.human.copy()

        if self.current_path_data is not None:
            path = self.current_path_data.get("path")
            path_s = self.current_path_data.get("_path_s")
            if path is not None and len(path) > 0:
                if path_s is None or len(path_s) != len(path):
                    self._precompute_frenet_cache()
                    path_s = self.current_path_data.get("_path_s")
                if path_s is not None and len(path_s) == len(path):
                    path = np.asarray(path, dtype=np.float32)
                    path_s = np.asarray(path_s, dtype=np.float32)
                    idx_h = self._nearest_path_index(path, human_state.position)
                    s_human = float(path_s[idx_h])
                    s_end = float(path_s[-1])
                    if s_human >= s_end - 1e-3:
                        self.paused = True
                        self.physics.set_control(0.0, 0.0, False)
                        self._log_event(
                            "goal_reached",
                            {
                                "human_pos": human_state.position.tolist(),
                                "s_human": float(s_human),
                                "s_end": float(s_end),
                            },
                        )
                        print("Human reached end of path. Simulation paused (SPACE to resume, N for new path).")

        self.robot_trajectory.append(robot_state.position.copy())
        self.human_trajectory.append(human_state.position.copy())
        if self.recording:
            self._record_frame(robot_state, human_state)

        max_trail = 5000
        if len(self.robot_trajectory) > max_trail:
            self.robot_trajectory = self.robot_trajectory[-max_trail:]
            self.human_trajectory = self.human_trajectory[-max_trail:]

        if self.scorer:
            self.scorer.update(robot_state.position, human_state.position)

        if is_data_step:
            obs = self._build_obs(
                robot_state.position, human_state.position, robot_state.heading
            )
            self.obs_history.append(obs)
            self.prev_robot_pos = robot_state.position.copy()
        # self._log_step(self.frame_count, robot_state, human_state, action, delta, forward, turn)
        if is_data_step:
            self.data_step_idx += 1
        self.frame_count += 1
        return robot_state, human_state

    def _check_collision(self) -> bool:
        obstacles = self.current_path_data.get("obstacles") if self.current_path_data else None
        segments = self.current_path_data.get("segment_obstacles") if self.current_path_data else None
        collided, info = self.physics.check_collision(obstacles, segment_obstacles=segments)
        if collided:
            self.collision_happened = True
            self.collision_info = info
            who = info.get("who", "agent") if info else "agent"
            idx = info.get("idx") if info else None
            obstacle = info.get("obstacle") if info else None
            obs_type = info.get("type") if info else None
            self._log_event(
                "collision",
                {
                    "who": who,
                    "obstacle_type": obs_type,
                    "obstacle_idx": idx,
                    "obstacle": obstacle,
                    "robot_pos": self.physics.robot.position.tolist(),
                    "human_pos": self.physics.human.position.tolist(),
                },
            )
            label = obs_type or "obstacle"
            if self.collision_behavior == "pause":
                print(f"Collision detected ({who}, {label} {idx}), pausing at collision.")
                self.collision_pause = True
                self.physics.set_control(0.0, 0.0, False)
                return True
            print(f"Collision detected ({who}, {label} {idx}), resetting.")
            self._reset_position()
            return True
        return False

    def _render(self, robot_state, human_state, actual_fps: float):
        scores = self.scorer.get_scores() if self.scorer else {}
        mode = "policy" if self.use_policy else "manual"
        safety_label = "diffusion" if self.safety_mode == "off" else self.safety_mode
        mode += f" [{safety_label}]"
        mode += f" [{self._current_interaction_label()}]"
        mode += f" [human:{self.human_source}]"
        if self.human_source == "detector":
            mode += f" [{self._human_tracking_mode}]"
        if self.paused:
            mode += " (paused)"
        elif self.collision_pause:
            mode += " (collision)"
        elif self.human_source == "detector" and self._human_detector_waiting:
            mode += " (waiting detector)"

        raw_detected_humans = np.zeros((0, 2), dtype=np.float32)
        detected_humans = np.zeros((0, 2), dtype=np.float32)
        best_human_index: Optional[int] = None
        if (
            self.human_source == "detector"
            and not self._rosbag_loop_reset_pending
        ):
            detection = self.ros_io.human_detections_world()
            if detection is not None:
                raw_detected_humans = np.asarray(
                    detection[0], dtype=np.float32
                ).reshape(-1, 2)
                # detected_humans, _ = self._human_candidates_in_rear_sector(
                #     raw_detected_humans,
                #     robot_position=robot_state.position,
                #     robot_heading=robot_state.heading,
                # )
                detected_humans = raw_detected_humans
                # print("raw",len(raw_detected_humans), "filtered", len(detected_humans))
                # Avoid printing at the rendering frame rate. Kalman rejection
                # and prediction-hold transitions are already throttled above.
                if (
                    len(detected_humans) > 0
                    and self._tracked_human_position is not None
                    and not self._human_detection_rejected
                ):
                    best_human_index = int(
                        np.argmin(
                            np.linalg.norm(
                                detected_humans
                                - self._tracked_human_position[None, :],
                                axis=1,
                            )
                        )
                    )

        human_detection_status = self.ros_io.human_detection_status()
        range_status = self.ros_io.range_source_status()
        info = {
            "fps": actual_fps,
            "path_length": self.current_path_data["length"] if self.current_path_data else 0,
            "robot_x": robot_state.position[0],
            "robot_y": robot_state.position[1],
            "num_points": self.storage.get_num_points() if self.storage is not None else self.frame_count,
            "recording": self.recording,
            "scores": scores,
            "mode": mode,
            "safety_mode": self.safety_mode,
            "pointcloud_mode": self.pointcloud_mode,
            "range_source": self.range_source,
            "range_topic": range_status["topic"],
            "range_point_count": range_status["count"],
            "range_age": range_status["age"],
            "human_source": self.human_source,
            "human_detector_waiting": self._human_detector_waiting,
            "human_tracking_mode": self._human_tracking_mode,
            "human_using_sim_fallback": bool(
                self._human_tracking_mode == "sim_fallback"
            ),
            "human_detection_rejected": self._human_detection_rejected,
            "robot_localization_jump_pending": (
                self._robot_localization_jump_pending
            ),
            "rosbag_loop_mode": bool(self.rosbag_loop_mode),
            "rosbag_loop_reset_pending": bool(
                self._rosbag_loop_reset_pending
            ),
            "rosbag_loop_reset_count": int(self._rosbag_loop_reset_count),
            "human_detection_count": int(len(detected_humans)),
            "human_detection_raw_count": int(len(raw_detected_humans)),
            "human_detection_age": human_detection_status["age"],
            "human_detection_frame": human_detection_status["source_frame"],
            "human_best_candidate_index": best_human_index,
            "human_rear_sector_range": self.human_rear_sector_range,
            "human_rear_sector_angle_deg": self.human_rear_sector_angle_deg,
            "interaction_label": self._current_interaction_label(),
            "interaction_label_source": (
                "segmentation"
                if self.interaction_segmenter.has_label
                else "manual_fallback"
            ),
            "segmentation_samples": len(self.interaction_segmenter.samples),
            "compliance_steps": int(self.last_compliance_stats.get("compliance_steps", 0)),
            "compliance_total_steps": int(self.last_compliance_stats.get("total_steps", 0)),
            "robot_radius": self.physics.robot_radius,
            "human_radius": self.physics.human_radius,
            "nominal_heading_delta": self.latest_nominal_heading_delta,
            "controls": [
                "P: Policy/Manual",
                "C: LaserScan/PointCloud",
                "H: Sim/Detector Human",
                "SPACE: Record/Pause" if self.collect_enabled else "SPACE: Pause",
                "S: Save episode" if self.collect_enabled else "M: Safety mode",
                "B: Hidden Guide/Tether",
                "R: Reset",
                "N: New Path",
                "Arrows: Manual control",
                "Scroll: Zoom",
                "ESC: Exit",
            ],
        }

        obs_obstacles, obs_segment_obstacles = self._select_obstacles_for_observation(
            robot_state.position, human_state.position, robot_state.heading
        )
        seg_closest_pts, seg_dirs = self._segments_closest_points_and_dirs(
            robot_state.position, obs_segment_obstacles
        )
        self.visualizer.render(
            robot_pos=robot_state.position,
            robot_heading=robot_state.heading,
            human_pos=human_state.position,
            reference_path=self.current_path_data["path"] if self.current_path_data else None,
            robot_trajectory=self.robot_trajectory,
            human_trajectory=self.human_trajectory,
            planned_path=self.planned_path,
            nominal_planned_path=self.nominal_planned_path,
            safe_planned_path=self.safe_planned_path,
            lookahead_points=self.lookahead_world,
            point_cloud=self.current_mid360_points_world,
            obstacles=self.current_path_data.get("obstacles") if self.current_path_data else None,
            segment_obstacles=self.current_path_data.get("segment_obstacles") if self.current_path_data else None,
            obs_obstacles=obs_obstacles,
            obs_segment_obstacles=obs_segment_obstacles,
            obs_segment_closest_points=seg_closest_pts,
            obs_segment_dirs=seg_dirs,
            obstacle_inflation=(self.physics.robot_radius, self.physics.human_radius),
            robot_radius=self.physics.robot_radius,
            human_radius=self.physics.human_radius,
            start_pos=self.current_path_data["start"] if self.current_path_data else None,
            end_pos=self.current_path_data["end"] if self.current_path_data else None,
            leash_tension=self.physics.get_leash_tension(),
            info=info,
            detected_humans=detected_humans,
            best_human_index=best_human_index,
            human_detection_sector_range=(
                self.human_rear_sector_range
                if self.human_source == "detector"
                else None
            ),
            human_detection_sector_angle_deg=(
                self.human_rear_sector_angle_deg
                if self.human_source == "detector"
                else None
            ),
        )

    def run(self):
        if self.visualizer is None:
            raise RuntimeError(
                "ModelPlanner.run() requires a Visualizer. "
                "Pass create_visualizer=True or provide a visualizer instance."
            )
        print("=" * 60)
        print("Guide Dog Robot Planning Tool")
        print("=" * 60)
        print(
            "Controls: P=Policy/Manual | C=LaserScan/PointCloud | "
            "H=Sim/Detector Human | SPACE=Pause | "
            "R=Reset | N=NewPath | ESC=Exit"
        )
        print(
            f"Range input: {self.range_source}; "
            f"scan topic={self.ros_io.laser_scan_topic}; "
            f"point-cloud topic={self.ros_io.pointcloud_topic}"
        )
        print(
            f"Human input: {self.human_source}; "
            f"detector topic={self.ros_io.human_detections_topic}"
        )
        print(
            "Human detector ROI: robot rear, "
            f"{self.human_rear_sector_angle_deg:g} deg total "
            f"(+/-{0.5 * self.human_rear_sector_angle_deg:g} deg), "
            f"range <= {self.human_rear_sector_range:g} m"
        )
        print("=" * 60)

        try:
            while self.running and not rospy.is_shutdown():
                self._handle_input()
                robot_state, human_state = self._step()
                actual_fps = self.visualizer.tick(self.fps)
                self._render(robot_state, human_state, actual_fps)
        finally:
            self.ros_io.stop()
            self.visualizer.quit()
        if self.log_fp is not None:
            total_steps = max(1, int(self.episode_safety_stats["total_steps"]))
            self._log_event(
                "shutdown",
                {
                    "episode_safety_stats": {
                        "modified_steps": int(self.episode_safety_stats["modified_steps"]),
                        "total_steps": int(self.episode_safety_stats["total_steps"]),
                        "mean_shift": round(
                            float(self.episode_safety_stats["total_shift"]) / float(total_steps),
                            4,
                        ),
                        "constraint_count": int(self.episode_safety_stats["constraint_count"]),
                        "min_clearance": round(
                            float(self.episode_safety_stats["min_clearance"]), 4
                        )
                        if np.isfinite(float(self.episode_safety_stats["min_clearance"]))
                        else "inf",
                    }
                },
            )
            self.log_fp.close()
        if self.human_detection_log_fp is not None:
            self._log_human_system_event(
                "human_detection_summary",
                {
                    "refresh_count": int(
                        self._human_detection_log_refresh_idx
                    ),
                    "record_count": int(
                        self._human_detection_log_record_idx
                    ),
                    "outcomes": dict(
                        sorted(self._human_detection_log_outcomes.items())
                    ),
                    "rosbag_loop_reset_count": int(
                        self._rosbag_loop_reset_count
                    ),
                },
            )
            self.human_detection_log_fp.close()
            print(
                "Human detection log summary: "
                f"{dict(sorted(self._human_detection_log_outcomes.items()))}"
            )
        if self.eval_fp is not None:
            self.eval_fp.close()
        print("Program exit")


def main():
    parser = argparse.ArgumentParser(description="Guide-follow planning with a trained policy")
    
    # Default checkpoint path - prefer the best available guide_mid360 checkpoint.
    default_ckpt = resolve_default_checkpoint()
    
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=default_ckpt,
        help="Path to the trained checkpoint (.ckpt). If not found, will run in manual control mode.",
    )
    parser.add_argument("--device", default="auto", help="cpu, cuda:0, or auto")
    parser.add_argument("--no-ema", action="store_true", help="Use non-EMA model")
    parser.add_argument(
        "--action-mode",
        default=None,
        help="delta | forward_heading | position | velocity",
    )
    parser.add_argument(
        "--k-lookahead",
        "--lookahead-stride",
        dest="k_lookahead",
        type=int,
        default=None,
        help="Sample one lookahead point every k path points (e.g., 5)",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=None,
        help="Downsample factor for simulated steps (matches dataset frame_stride)",
    )
    parser.add_argument(
        "--pointcloud-mode",
        default="live",
        choices=("live", "off"),
        help="Use the live Livox topic, or disable point-cloud observations.",
    )
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--pointcloud-topic", default="/livox/lidar")
    parser.add_argument("--laser-scan-topic", default="/front/scan")
    parser.add_argument(
        "--range-source",
        default="laser_scan",
        choices=("laser_scan", "point_cloud"),
        help=(
            "Initial range input. Press C to switch between /front/scan "
            "and /livox/lidar at runtime."
        ),
    )
    parser.add_argument(
        "--human-detections-topic",
        default="/dr_spaam_detections",
        help="PoseArray topic published by the independently running detector.",
    )
    parser.add_argument(
        "--human-source",
        default="sim",
        choices=("sim", "detector"),
        help="Initial human source. Press H to switch at runtime.",
    )
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument("--max-angular-speed", type=float, default=1.0)
    parser.add_argument("--ros-input-timeout", type=float, default=10.0)
    parser.add_argument("--odom-timeout", type=float, default=5.0)
    parser.add_argument("--pointcloud-timeout", type=float, default=5.0)
    parser.add_argument(
        "--human-detection-timeout",
        type=float,
        default=1.5,
        help=(
            "A PoseArray older than this is treated as unavailable and the "
            "tracker switches to simulation fallback when a track exists."
        ),
    )
    parser.add_argument(
        "--human-detector-frame",
        default="",
        help=(
            "Fallback detector frame when PoseArray.header.frame_id is empty. "
            "Normally leave empty and use the message frame."
        ),
    )
    parser.add_argument(
        "--human-world-frame",
        default="",
        help=(
            "Target frame for detected humans. Empty uses "
            "Odometry.header.frame_id; an override must match the frame of "
            "the robot position used by this planner."
        ),
    )
    parser.add_argument(
        "--human-detector-y-axis",
        default="right",
        choices=("right", "left"),
        help=(
            "Detector coordinate convention. The supplied DR-SPAAM code uses "
            "x-forward/y-right, so the default is right."
        ),
    )
    parser.add_argument(
        "--human-tf-timeout",
        type=float,
        default=0.05,
        help="Maximum TF lookup wait per new detector message in seconds.",
    )
    parser.add_argument(
        "--human-track-max-jump",
        type=float,
        default=1.5,
        help=(
            "Fallback Euclidean association limit in meters. Kalman "
            "Mahalanobis gating is primary; <=0 disables this fallback."
        ),
    )
    parser.add_argument(
        "--human-kf-process-accel-std",
        type=float,
        default=1.5,
        help="Kalman constant-velocity process acceleration standard deviation.",
    )
    parser.add_argument(
        "--human-kf-measurement-std",
        type=float,
        default=0.18,
        help="Expected detector position noise in meters.",
    )
    parser.add_argument(
        "--human-kf-gate",
        type=float,
        default=11.83,
        help=(
            "Squared Mahalanobis association gate (11.83 is approximately "
            "a 3-sigma gate for a 2-D observation)."
        ),
    )
    parser.add_argument(
        "--human-kf-sim-prior-std",
        type=float,
        default=0.75,
        help=(
            "Noise assigned to the simulated-human pseudo-measurement. "
            "Larger values make detector measurements more dominant."
        ),
    )
    parser.add_argument(
        "--human-kf-sim-prior-max-error",
        type=float,
        default=1.25,
        help=(
            "Fuse the simulation prior only when it is within this many "
            "meters of the KF prediction; <=0 always permits fusion."
        ),
    )
    parser.add_argument(
        "--no-human-rear-prior-calibration",
        action="store_true",
        help=(
            "Disable adaptive rear-leash calibration. By default each "
            "detected human position is blended with the position exactly "
            "one leash length behind the robot."
        ),
    )
    parser.add_argument(
        "--human-rear-prior-scale",
        type=float,
        default=0.60,
        help=(
            "Error scale in meters for adaptive detector weighting. At this "
            "error the unclipped detector weight is 0.5."
        ),
    )
    parser.add_argument(
        "--human-rear-prior-min-detect-weight",
        type=float,
        default=0.05,
        help=(
            "Minimum detector weight for a detection far from the rear-leash "
            "prior."
        ),
    )
    parser.add_argument(
        "--human-rear-prior-max-detect-weight",
        type=float,
        default=0.90,
        help=(
            "Maximum detector weight even when the detection matches the "
            "rear-leash prior exactly."
        ),
    )
    parser.add_argument(
        "--human-kf-hold-timeout",
        type=float,
        default=1.5,
        help=(
            "Duration of the weak KF/simulation bridge. After this gap, "
            "simulation becomes the dominant fallback instead of stopping."
        ),
    )
    parser.add_argument(
        "--human-kf-max-misses",
        type=int,
        default=30,
        help=(
            "Number of invalid new detector messages allowed in the weak "
            "prediction bridge before simulation becomes dominant."
        ),
    )
    parser.add_argument(
        "--human-kf-sector-margin-deg",
        type=float,
        default=12.0,
        help=(
            "Per-side angular margin used only to retain an established KF "
            "track; new detections still use the strict rear sector."
        ),
    )
    parser.add_argument(
        "--human-kf-range-margin",
        type=float,
        default=0.4,
        help=(
            "Range margin in meters used only to retain an established KF "
            "track."
        ),
    )
    parser.add_argument(
        "--human-kf-sim-fallback-std",
        type=float,
        default=0.30,
        help=(
            "Simulation pseudo-measurement noise during detector dropout or "
            "large detector outliers. Smaller values follow simulation more."
        ),
    )
    parser.add_argument(
        "--human-kf-sim-velocity-gain",
        type=float,
        default=0.60,
        help="Velocity blend gain from PhysicsEngine during simulation fallback.",
    )
    parser.add_argument(
        "--human-sim-max-distance",
        type=float,
        default=6.0,
        help=(
            "Declare the simulation human state invalid when its distance "
            "from the robot exceeds this value."
        ),
    )
    parser.add_argument(
        "--human-sim-full-loss-timeout",
        type=float,
        default=0.0,
        help=(
            "Optional maximum simulation-only duration before stopping; "
            "0 keeps valid simulation fallback indefinitely."
        ),
    )
    parser.add_argument(
        "--no-human-continuity-mode",
        action="store_true",
        help=(
            "Disable continuity-first human fallback and restore fail-closed "
            "stopping when both detector and simulation tracking fail."
        ),
    )
    parser.add_argument(
        "--human-robot-jump-threshold",
        type=float,
        default=1.0,
        help=(
            "Stop and reacquire after an odometry position discontinuity "
            "larger than this many meters; 0 disables position checking."
        ),
    )
    parser.add_argument(
        "--human-robot-heading-jump-deg",
        type=float,
        default=90.0,
        help=(
            "Stop and reacquire after an odometry heading discontinuity "
            "larger than this many degrees; 0 disables heading checking."
        ),
    )
    parser.add_argument(
        "--no-rosbag-loop-mode",
        action="store_true",
        help=(
            "Disable benign rosbag play -l wraparound handling. By default, "
            "odometry timestamp rewinds and large returns to the first bag "
            "pose reset temporal state without raising an anomaly."
        ),
    )
    parser.add_argument(
        "--rosbag-loop-origin-radius",
        type=float,
        default=0.75,
        help=(
            "Fallback radius around the first odometry pose used to recognize "
            "a rosbag loop when header timestamps cannot be used."
        ),
    )
    parser.add_argument(
        "--human-rear-sector-range",
        type=float,
        default=3.0,
        help=(
            "Maximum detector range behind the robot in meters "
            "(default: 3.0)."
        ),
    )
    parser.add_argument(
        "--human-rear-sector-angle-deg",
        type=float,
        default=90.0,
        help=(
            "Total opening of the rear-facing detector sector in degrees; "
            "45 means +/-22.5 deg around the exact rear direction."
        ),
    )
    parser.add_argument(
        "--lidar-height",
        type=float,
        default=0.4,
        help="Livox origin height above the ground, matching base_link->livox_frame.",
    )
    parser.add_argument(
        "--enable-motion",
        action="store_true",
        help="Actually publish non-zero /cmd_vel. Default is visualization-only.",
    )
    parser.add_argument("--path-length", type=float, default=50.0)
    parser.add_argument("--leash-length", type=float, default=1.5)
    parser.add_argument("--robot-speed", type=float, default=1.5)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--inference-steps", type=int, default=64,
                        help="Number of diffusion inference steps (lower=faster, default=64)")
    parser.add_argument(
        "--turn-gain",
        type=float,
        default=1.2,
        help="Scale heading_delta before converting to turn input.",
    )
    parser.add_argument(
        "--no-curvature-slowdown",
        action="store_true",
        help="Disable curvature-based forward slowdown.",
    )
    parser.add_argument(
        "--curvature-scale",
        type=float,
        default=0.7,
        help="Slowdown strength based on heading_delta ratio.",
    )
    parser.add_argument(
        "--min-speed-scale",
        type=float,
        default=0.25,
        help="Lower bound for curvature slowdown.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "logs",
        help="Directory for planning logs (jsonl).",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="Log every N frames (default: 1).",
    )
    parser.add_argument(
        "--human-detection-log-interval",
        type=int,
        default=1,
        help=(
            "Write one detailed human-detection diagnostic record every N "
            "control refreshes. Important failures are always logged."
        ),
    )
    parser.add_argument(
        "--human-detection-log-max-candidates",
        type=int,
        default=64,
        help=(
            "Maximum raw/ROI candidates serialized in each detector record "
            "(default: 64)."
        ),
    )
    parser.add_argument(
        "--human-detection-console",
        action="store_true",
        help=(
            "Print a compact detector decision line for every refresh in "
            "addition to the detailed JSONL file."
        ),
    )
    parser.add_argument(
        "--no-human-detection-log",
        action="store_true",
        help="Disable the dedicated human_detection_*.jsonl diagnostic log.",
    )
    parser.add_argument(
        "-e",
        "--eval",
        action="store_true",
        help="Write each origin/robot-safe/human-robot-safe planning result and timing to JSONL.",
    )
    parser.add_argument(
        "-c",
        "--collect",
        action="store_true",
        help="Enable collect-style episode recording. SPACE toggles recording; S saves.",
    )
    parser.add_argument(
        "--safety-mode",
        default="human_robot_qp",
        help="off | robot_qp | human_robot_qp. Apply online safety filtering before execution.",
    )
    parser.add_argument(
        "--safety-margin",
        type=float,
        default=0.2,
        help="Extra safety margin in meters added around obstacles for QP filtering.",
    )
    parser.add_argument(
        "--safety-alpha",
        type=float,
        default=1.0,
        help="Barrier gain for QP halfspace constraints.",
    )
    parser.add_argument(
        "--safety-max-constraints",
        type=int,
        default=8,
        help="Maximum nearby obstacle constraints kept in each 2D QP projection.",
    )
    parser.add_argument(
        "--safety-influence-distance",
        type=float,
        default=2.0,
        help="Only obstacles within this clearance band are included in the QP.",
    )
    parser.add_argument(
        "--safety-path-corridor",
        type=float,
        default=0.8,
        help=(
            "Keep filtered Livox points within this many meters of the raw "
            "diffusion path before SafeFilter (default: 0.8)."
        ),
    )
    parser.add_argument(
        "--safety-point-spacing",
        type=float,
        default=0.1,
        help=(
            "2D voxel size in meters; SafeFilter keeps at most one point "
            "obstacle per voxel (default: 0.1)."
        ),
    )
    parser.add_argument(
        "--segmentation-path",
        type=Path,
        default=DEFAULT_SEGMENTATION_PATH,
        help="Path to the guide/tether segmentation.py module.",
    )
    parser.add_argument(
        "--segmentation-window",
        type=int,
        default=120,
        help="Number of recent robot/human states used for online decoding.",
    )
    parser.add_argument(
        "--segmentation-min-samples",
        type=int,
        default=12,
        help="Minimum trajectory samples required before the first decode.",
    )
    parser.add_argument(
        "--no-interaction-segmentation",
        action="store_true",
        help="Disable online guide/tether recognition and use manual B/timed state.",
    )
    parser.add_argument(
        "--debug-preview",
        action="store_true",
        help="Print preview-rollout diagnostics from _actions_to_path.",
    )
    parser.add_argument(
        "--debug-policy",
        action="store_true",
        help="Print predicted action summaries after each inference.",
    )
    parser.add_argument(
        "--no-debug-qp-log",
        action="store_true",
        help="Disable detailed robot_qp diagnostics in the jsonl log.",
    )
    parser.add_argument(
        "--debug-preview-limit",
        type=int,
        default=5,
        help="Maximum number of actions printed for each debug preview/policy dump.",
    )
    parser.add_argument("--no-log", action="store_true", help="Disable planning logs")
    args = parser.parse_args()

    rospy.init_node("guide_real_robot_model_planner", anonymous=False)

    # Check if checkpoint file exists, but don't exit if it doesn't (will use manual control)
    checkpoint_path = args.ckpt if args.ckpt.exists() else None
    if checkpoint_path is None:
        print(f"Warning: Checkpoint file not found: {args.ckpt}")
        print("Running in manual control mode only (no policy available).")
        print("\nAvailable checkpoints:")
        outputs_dir = Path("/home/yyf/IROS2026/diffusion_policy/data/outputs")
        if outputs_dir.exists():
            for ckpt in sorted(outputs_dir.rglob("*.ckpt")):
                print(f"  {ckpt}")
        print("\nContinuing with manual control...")

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    planner = ModelPlanner(
        checkpoint_path=checkpoint_path,
        device=args.device,
        use_ema=not args.no_ema,
        action_mode=args.action_mode,
        k_lookahead=args.k_lookahead,
        frame_stride=args.frame_stride,
        pointcloud_mode=args.pointcloud_mode,
        odom_topic=args.odom_topic,
        pointcloud_topic=args.pointcloud_topic,
        laser_scan_topic=args.laser_scan_topic,
        range_source=args.range_source,
        human_detections_topic=args.human_detections_topic,
        human_source=args.human_source,
        cmd_vel_topic=args.cmd_vel_topic,
        max_angular_speed=args.max_angular_speed,
        ros_input_timeout=args.ros_input_timeout,
        odom_timeout=args.odom_timeout,
        pointcloud_timeout=args.pointcloud_timeout,
        human_detection_timeout=args.human_detection_timeout,
        human_detector_frame=args.human_detector_frame,
        human_world_frame=args.human_world_frame,
        human_detector_y_axis=args.human_detector_y_axis,
        human_tf_timeout=args.human_tf_timeout,
        human_track_max_jump=args.human_track_max_jump,
        human_rear_sector_range=args.human_rear_sector_range,
        human_rear_sector_angle_deg=args.human_rear_sector_angle_deg,
        human_kf_process_accel_std=args.human_kf_process_accel_std,
        human_kf_measurement_std=args.human_kf_measurement_std,
        human_kf_gate=args.human_kf_gate,
        human_kf_sim_prior_std=args.human_kf_sim_prior_std,
        human_kf_sim_prior_max_error=args.human_kf_sim_prior_max_error,
        human_rear_prior_calibration=(
            not args.no_human_rear_prior_calibration
        ),
        human_rear_prior_scale=args.human_rear_prior_scale,
        human_rear_prior_min_detect_weight=(
            args.human_rear_prior_min_detect_weight
        ),
        human_rear_prior_max_detect_weight=(
            args.human_rear_prior_max_detect_weight
        ),
        human_kf_hold_timeout=args.human_kf_hold_timeout,
        human_kf_max_misses=args.human_kf_max_misses,
        human_kf_sector_margin_deg=args.human_kf_sector_margin_deg,
        human_kf_range_margin=args.human_kf_range_margin,
        human_kf_sim_fallback_std=args.human_kf_sim_fallback_std,
        human_kf_sim_velocity_gain=args.human_kf_sim_velocity_gain,
        human_sim_max_distance=args.human_sim_max_distance,
        human_sim_full_loss_timeout=args.human_sim_full_loss_timeout,
        human_continuity_mode=not args.no_human_continuity_mode,
        human_robot_jump_threshold=args.human_robot_jump_threshold,
        human_robot_heading_jump_deg=args.human_robot_heading_jump_deg,
        rosbag_loop_mode=not args.no_rosbag_loop_mode,
        rosbag_loop_origin_radius=args.rosbag_loop_origin_radius,
        lidar_height=args.lidar_height,
        enable_motion=args.enable_motion,
        path_length=args.path_length,
        leash_length=args.leash_length,
        robot_speed=args.robot_speed,
        fps=args.fps,
        inference_steps=args.inference_steps,
        turn_gain=args.turn_gain,
        curvature_slowdown=not args.no_curvature_slowdown,
        curvature_scale=args.curvature_scale,
        min_speed_scale=args.min_speed_scale,
        log_path=None if args.no_log else args.log_dir / f"planning_{run_timestamp}.jsonl",
        human_detection_log_path=(
            None
            if args.no_log or args.no_human_detection_log
            else args.log_dir / f"human_detection_{run_timestamp}.jsonl"
        ),
        human_detection_log_interval=args.human_detection_log_interval,
        human_detection_log_max_candidates=(
            args.human_detection_log_max_candidates
        ),
        human_detection_console=args.human_detection_console,
        eval_path=(
            args.log_dir / f"planning_eval_{run_timestamp}.jsonl"
            if args.eval
            else None
        ),
        log_interval=args.log_interval,
        collect_enabled=args.collect,
        safety_mode=args.safety_mode,
        safety_margin=args.safety_margin,
        safety_alpha=args.safety_alpha,
        safety_max_constraints=args.safety_max_constraints,
        safety_influence_distance=args.safety_influence_distance,
        safety_path_corridor=args.safety_path_corridor,
        safety_point_spacing=args.safety_point_spacing,
        debug_preview=args.debug_preview,
        debug_preview_limit=args.debug_preview_limit,
        debug_policy=args.debug_policy,
        debug_qp_log=not args.no_debug_qp_log,
        interaction_segmentation=not args.no_interaction_segmentation,
        segmentation_path=args.segmentation_path,
        segmentation_window=args.segmentation_window,
        segmentation_min_samples=args.segmentation_min_samples,
    )
    planner.run()


if __name__ == "__main__":
    main()