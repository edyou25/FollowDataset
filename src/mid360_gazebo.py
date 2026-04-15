"""
Gazebo/ROS Mid-360 simulation bridge.
"""
from __future__ import annotations

import importlib
import math
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy.spatial.transform import Rotation


T_BASE_MID360 = np.array(
    [
        [-0.707, 0.000, 0.707, 0.170],
        [0.000, 1.000, 0.000, 0.000],
        [-0.707, 0.000, -0.707, 0.090],
        [0.000, 0.000, 0.000, 1.000],
    ],
    dtype=np.float64,
)


def transform_matrix_to_xyz_rpy(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotation = Rotation.from_matrix(transform[:3, :3])
    return transform[:3, 3].astype(np.float64).copy(), rotation.as_euler("xyz", degrees=False)


MID360_MOUNT_XYZ, MID360_MOUNT_RPY = transform_matrix_to_xyz_rpy(T_BASE_MID360)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _workspace_root() -> Path:
    return _repo_root().parent


def resolve_mid360_plugin_dir(explicit_path: Optional[str | os.PathLike[str]] = None) -> Path:
    candidates = []
    env_path = os.environ.get("MID360_PLUGIN_DIR")
    if explicit_path is not None:
        candidates.append(Path(explicit_path))
    if env_path:
        candidates.append(Path(env_path))

    candidates.extend(
        [
            _repo_root() / "Mid360_simulation_plugin",
            _workspace_root() / "Mid360_simulation_plugin",
            _workspace_root() / "_external" / "Mid360_simulation_plugin",
        ]
    )

    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if (candidate / "livox_laser_simulation").exists():
            return candidate

    searched = "\n".join(str(p.expanduser()) for p in candidates)
    raise RuntimeError(
        "Cannot find Mid360_simulation_plugin. Set MID360_PLUGIN_DIR or pass --mid360-plugin-dir.\n"
        f"Searched:\n{searched}"
    )


def resolve_mid360_plugin_library(
    plugin_dir: Path,
    explicit_path: Optional[str | os.PathLike[str]] = None,
) -> Optional[Path]:
    candidates = []
    env_path = os.environ.get("MID360_PLUGIN_LIBRARY")
    if explicit_path is not None:
        candidates.append(Path(explicit_path))
    if env_path:
        candidates.append(Path(env_path))

    candidates.extend(
        [
            plugin_dir / "devel" / "lib" / "liblivox_laser_simulation.so",
            plugin_dir / "build" / "devel" / "lib" / "liblivox_laser_simulation.so",
            plugin_dir / "build" / "lib" / "liblivox_laser_simulation.so",
            plugin_dir.parent / "devel" / "lib" / "liblivox_laser_simulation.so",
            plugin_dir.parent.parent / "devel" / "lib" / "liblivox_laser_simulation.so",
        ]
    )

    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate.exists():
            return candidate
    return None


@dataclass
class Mid360GazeboConfig:
    plugin_dir: Path
    plugin_library_path: Optional[Path] = None
    ros_topic: str = "/livox/lidar"
    frame_name: str = "mid360_link"
    samples: int = 20000
    downsample: int = 1
    update_rate: float = 10.0
    range_min: float = 0.1
    range_max: float = 40.0
    visualize_laser: bool = True
    gui: bool = True
    verbose: bool = True
    wall_height: float = 2.2
    wall_thickness: float = 0.12
    obstacle_min_height: float = 0.8
    obstacle_height_step: float = 0.35
    guide_strip_width: float = 0.16
    guide_strip_height: float = 0.02
    robot_base_size: tuple[float, float, float] = (0.55, 0.42, 0.25)
    human_radius: float = 0.22
    human_height: float = 1.72
    keep_runtime_artifacts: bool = True
    wait_for_first_cloud_sec: float = 10.0
    runtime_root: Optional[Path] = None

    def resolved_plugin_library(self) -> Path:
        if self.plugin_library_path is None:
            lib_path = resolve_mid360_plugin_library(self.plugin_dir)
            if lib_path is None:
                raise RuntimeError(
                    "Cannot find liblivox_laser_simulation.so.\n"
                    "Build Mid360_simulation_plugin in a catkin workspace first, "
                    "or set MID360_PLUGIN_LIBRARY."
                )
            self.plugin_library_path = lib_path
        return self.plugin_library_path


class Mid360GazeboSession:
    """Manage a Gazebo world with a Mid-360-equipped robot model."""

    def __init__(self, path_data: dict, config: Mid360GazeboConfig):
        self.path_data = path_data
        self.config = config
        runtime_root = config.runtime_root or (_repo_root() / "data" / "_mid360_runtime")
        runtime_root.mkdir(parents=True, exist_ok=True)
        self.runtime_dir = Path(
            tempfile.mkdtemp(prefix="mid360_", dir=str(runtime_root))
        ).resolve()

        self.world_path = self.runtime_dir / "world.sdf"
        self.robot_model_path = self.runtime_dir / "robot.sdf"
        self.human_model_path = self.runtime_dir / "human.sdf"
        self.roscore_log_path = self.runtime_dir / "roscore.log"
        self.gazebo_log_path = self.runtime_dir / "gazebo.log"

        self.roscore_process: Optional[subprocess.Popen] = None
        self.gazebo_process: Optional[subprocess.Popen] = None
        self.started_roscore = False
        self.started = False

        self.rospy = None
        self.point_cloud2 = None
        self.pointcloud_msg_type = None
        self.geometry_msgs = None
        self.gazebo_msgs = None
        self.gazebo_srvs = None

        self.spawn_model_srv = None
        self.delete_model_srv = None
        self.set_model_state_srv = None

        self._cloud_lock = threading.Lock()
        self._latest_cloud: Optional[dict[str, Any]] = None
        self._cloud_seq = 0

        self.robot_model_name = "followdataset_mid360_robot"
        self.human_model_name = "followdataset_human"

    def start(self):
        if self.started:
            return
        try:
            self._write_runtime_files()
            self._import_ros_modules()
            self._ensure_roscore()
            self._ensure_ros_node()
            self._launch_gazebo()
            self._bind_gazebo_services()
            self._spawn_robot()
            self._spawn_human()
            self.started = True
            self.wait_for_first_pointcloud(timeout_sec=self.config.wait_for_first_cloud_sec)
        except Exception:
            self.close()
            raise

    def close(self):
        if self.rospy is not None and self.delete_model_srv is not None:
            for name in (self.robot_model_name, self.human_model_name):
                try:
                    self.delete_model_srv(name)
                except Exception:
                    pass

        for proc in (self.gazebo_process, self.roscore_process if self.started_roscore else None):
            if proc is None:
                continue
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    proc.kill()

        self.gazebo_process = None
        self.roscore_process = None
        self.started = False

        if not self.config.keep_runtime_artifacts:
            shutil.rmtree(self.runtime_dir, ignore_errors=True)

    def metadata(self) -> dict:
        return {
            "backend": "mid360_gazebo",
            "mid360_topic": self.config.ros_topic,
            "mid360_frame": self.config.frame_name,
            "mid360_mount_matrix": T_BASE_MID360.tolist(),
            "mid360_mount_pose_xyz": MID360_MOUNT_XYZ.tolist(),
            "mid360_mount_pose_rpy": MID360_MOUNT_RPY.tolist(),
            "plugin_dir": str(self.config.plugin_dir),
            "plugin_library": str(self.config.plugin_library_path) if self.config.plugin_library_path else None,
            "runtime_dir": str(self.runtime_dir),
            "world_sdf": str(self.world_path),
            "robot_sdf": str(self.robot_model_path),
            "human_sdf": str(self.human_model_path),
            "wall_height": float(self.config.wall_height),
        }

    def update_entities(self, robot_state: Any, human_state: Any):
        if not self.started:
            return
        self._set_model_state(
            self.robot_model_name,
            x=float(robot_state.position[0]),
            y=float(robot_state.position[1]),
            z=0.5 * float(self.config.robot_base_size[2]),
            yaw=float(robot_state.heading),
        )
        self._set_model_state(
            self.human_model_name,
            x=float(human_state.position[0]),
            y=float(human_state.position[1]),
            z=0.5 * float(self.config.human_height),
            yaw=0.0,
        )

    def get_robot_base_pose(self, robot_state: Any) -> np.ndarray:
        return self._pose_vector_from_xyz_rpy(
            x=float(robot_state.position[0]),
            y=float(robot_state.position[1]),
            z=0.5 * float(self.config.robot_base_size[2]),
            roll=0.0,
            pitch=0.0,
            yaw=float(robot_state.heading),
        )

    def get_human_base_pose(self, human_state: Any) -> np.ndarray:
        return self._pose_vector_from_xyz_rpy(
            x=float(human_state.position[0]),
            y=float(human_state.position[1]),
            z=0.5 * float(self.config.human_height),
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
        )

    def get_mid360_pose(self, robot_state: Any) -> np.ndarray:
        base_rot = Rotation.from_euler("z", float(robot_state.heading), degrees=False).as_matrix()
        world_from_base = np.eye(4, dtype=np.float64)
        world_from_base[:3, :3] = base_rot
        world_from_base[:3, 3] = np.array(
            [
                float(robot_state.position[0]),
                float(robot_state.position[1]),
                0.5 * float(self.config.robot_base_size[2]),
            ],
            dtype=np.float64,
        )
        world_from_mid360 = world_from_base @ T_BASE_MID360
        rotation = Rotation.from_matrix(world_from_mid360[:3, :3]).as_quat()
        return np.array(
            [
                world_from_mid360[0, 3],
                world_from_mid360[1, 3],
                world_from_mid360[2, 3],
                rotation[0],
                rotation[1],
                rotation[2],
                rotation[3],
            ],
            dtype=np.float64,
        )

    def wait_for_first_pointcloud(self, timeout_sec: float):
        cloud = self.get_pointcloud(wait_timeout=timeout_sec)
        if cloud is None:
            raise RuntimeError(
                "No Mid-360 point cloud received from Gazebo.\n"
                "Check that ROS Noetic + Gazebo11 are installed and "
                "liblivox_laser_simulation.so is built and discoverable."
            )

    def get_pointcloud(
        self,
        *,
        after_seq: Optional[int] = None,
        wait_timeout: float = 0.0,
    ) -> Optional[dict[str, Any]]:
        deadline = time.time() + max(wait_timeout, 0.0)
        while True:
            with self._cloud_lock:
                cloud = None if self._latest_cloud is None else dict(self._latest_cloud)
            if cloud is not None and (after_seq is None or int(cloud["seq"]) > int(after_seq)):
                return cloud
            if time.time() >= deadline:
                return cloud if (cloud is not None and after_seq is None) else None
            time.sleep(0.01)

    def _write_runtime_files(self):
        self.world_path.write_text(self._generate_world_sdf(), encoding="utf-8")
        self.robot_model_path.write_text(self._generate_robot_sdf(), encoding="utf-8")
        self.human_model_path.write_text(self._generate_human_sdf(), encoding="utf-8")

    def _import_ros_modules(self):
        required_commands = ["roscore", "roslaunch"]
        for command in required_commands:
            if shutil.which(command) is None:
                raise RuntimeError(
                    f"Required command '{command}' not found. Install ROS Noetic + Gazebo11 first."
                )

        missing_modules = []
        try:
            self.rospy = importlib.import_module("rospy")
            self.point_cloud2 = importlib.import_module("sensor_msgs.point_cloud2")
            sensor_msgs_msg = importlib.import_module("sensor_msgs.msg")
            self.pointcloud_msg_type = sensor_msgs_msg.PointCloud2
            self.geometry_msgs = importlib.import_module("geometry_msgs.msg")
            self.gazebo_msgs = importlib.import_module("gazebo_msgs.msg")
            self.gazebo_srvs = importlib.import_module("gazebo_msgs.srv")
        except ModuleNotFoundError as exc:
            missing_modules.append(str(exc))

        if missing_modules:
            raise RuntimeError(
                "Missing ROS Python packages. Source your ROS environment before running "
                "`python collect.py --backend mid360`.\n"
                + "\n".join(missing_modules)
            )

    def _ensure_roscore(self):
        if self._is_port_open("127.0.0.1", 11311):
            return

        with open(self.roscore_log_path, "w", encoding="utf-8") as log_file:
            self.roscore_process = subprocess.Popen(
                ["roscore"],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
            )
        self.started_roscore = True

        deadline = time.time() + 20.0
        while time.time() < deadline:
            if self._is_port_open("127.0.0.1", 11311):
                return
            if self.roscore_process is not None and self.roscore_process.poll() is not None:
                raise RuntimeError(f"roscore exited unexpectedly. See {self.roscore_log_path}")
            time.sleep(0.2)
        raise RuntimeError("Timed out waiting for roscore on port 11311.")

    def _ensure_ros_node(self):
        if not self.rospy.core.is_initialized():
            self.rospy.init_node(
                "followdataset_mid360_bridge",
                anonymous=True,
                disable_signals=True,
            )
        self.rospy.Subscriber(self.config.ros_topic, self.pointcloud_msg_type, self._pointcloud_callback)

    def _launch_gazebo(self):
        plugin_library = self.config.resolved_plugin_library()
        env = os.environ.copy()
        plugin_dir = str(plugin_library.parent.resolve())
        existing_plugin_path = env.get("GAZEBO_PLUGIN_PATH", "")
        env["GAZEBO_PLUGIN_PATH"] = (
            plugin_dir if not existing_plugin_path else f"{plugin_dir}:{existing_plugin_path}"
        )

        cmd = [
            "roslaunch",
            "gazebo_ros",
            "empty_world.launch",
            f"world_name:={self.world_path}",
            f"gui:={'true' if self.config.gui else 'false'}",
            "paused:=false",
            "use_sim_time:=true",
            f"headless:={'false' if self.config.gui else 'true'}",
            f"verbose:={'true' if self.config.verbose else 'false'}",
            "debug:=false",
        ]
        with open(self.gazebo_log_path, "w", encoding="utf-8") as log_file:
            self.gazebo_process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
            )

    def _bind_gazebo_services(self):
        self.rospy.wait_for_service("/gazebo/spawn_sdf_model", timeout=30.0)
        self.rospy.wait_for_service("/gazebo/delete_model", timeout=30.0)
        self.rospy.wait_for_service("/gazebo/set_model_state", timeout=30.0)

        self.spawn_model_srv = self.rospy.ServiceProxy(
            "/gazebo/spawn_sdf_model", self.gazebo_srvs.SpawnModel
        )
        self.delete_model_srv = self.rospy.ServiceProxy(
            "/gazebo/delete_model", self.gazebo_srvs.DeleteModel
        )
        self.set_model_state_srv = self.rospy.ServiceProxy(
            "/gazebo/set_model_state", self.gazebo_srvs.SetModelState
        )

    def _spawn_robot(self):
        xml = self.robot_model_path.read_text(encoding="utf-8")
        self.spawn_model_srv(
            self.robot_model_name,
            xml,
            "",
            self._pose_msg(0.0, 0.0, 0.5 * float(self.config.robot_base_size[2]), 0.0, 0.0, 0.0),
            "world",
        )

    def _spawn_human(self):
        xml = self.human_model_path.read_text(encoding="utf-8")
        self.spawn_model_srv(
            self.human_model_name,
            xml,
            "",
            self._pose_msg(0.0, 0.0, 0.5 * float(self.config.human_height), 0.0, 0.0, 0.0),
            "world",
        )

    def _set_model_state(self, model_name: str, *, x: float, y: float, z: float, yaw: float):
        state = self.gazebo_msgs.ModelState()
        state.model_name = model_name
        state.pose = self._pose_msg(x, y, z, 0.0, 0.0, yaw)
        state.reference_frame = "world"
        self.set_model_state_srv(state)

    def _pose_msg(self, x: float, y: float, z: float, roll: float, pitch: float, yaw: float):
        pose = self.geometry_msgs.Pose()
        quat = Rotation.from_euler("xyz", [roll, pitch, yaw], degrees=False).as_quat()
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = float(z)
        pose.orientation.x = float(quat[0])
        pose.orientation.y = float(quat[1])
        pose.orientation.z = float(quat[2])
        pose.orientation.w = float(quat[3])
        return pose

    def _pose_vector_from_xyz_rpy(
        self,
        *,
        x: float,
        y: float,
        z: float,
        roll: float,
        pitch: float,
        yaw: float,
    ) -> np.ndarray:
        quat = Rotation.from_euler("xyz", [roll, pitch, yaw], degrees=False).as_quat()
        return np.asarray([x, y, z, quat[0], quat[1], quat[2], quat[3]], dtype=np.float64)

    def _pointcloud_callback(self, msg):
        field_names = [field.name for field in msg.fields]
        rows = list(self.point_cloud2.read_points(msg, field_names=field_names, skip_nans=False))
        if rows:
            points = np.asarray(rows, dtype=np.float32)
            if points.ndim == 1:
                points = points.reshape(1, -1)
        else:
            points = np.zeros((0, len(field_names)), dtype=np.float32)

        stamp = 0.0
        if hasattr(msg.header, "stamp") and hasattr(msg.header.stamp, "to_sec"):
            stamp = float(msg.header.stamp.to_sec())

        with self._cloud_lock:
            self._cloud_seq += 1
            self._latest_cloud = {
                "seq": self._cloud_seq,
                "stamp": stamp,
                "fields": field_names,
                "points": points,
            }

    def _generate_world_sdf(self) -> str:
        reference_path = np.asarray(self.path_data.get("waypoints", self.path_data.get("path", [])), dtype=np.float64)
        circle_obstacles = np.asarray(self.path_data.get("obstacles", []), dtype=np.float64)
        segment_obstacles = np.asarray(self.path_data.get("segment_obstacles", []), dtype=np.float64)

        models = [
            self._make_start_marker(np.asarray(self.path_data.get("start", [0.0, 0.0]), dtype=np.float64)),
            self._make_end_marker(np.asarray(self.path_data.get("end", [0.0, 0.0]), dtype=np.float64)),
        ]

        if len(reference_path) >= 2:
            for idx in range(len(reference_path) - 1):
                models.append(
                    self._box_model_xml(
                        name=f"guide_strip_{idx}",
                        center=(reference_path[idx] + reference_path[idx + 1]) * 0.5,
                        yaw=math.atan2(
                            reference_path[idx + 1][1] - reference_path[idx][1],
                            reference_path[idx + 1][0] - reference_path[idx][0],
                        ),
                        size=(
                            float(np.linalg.norm(reference_path[idx + 1] - reference_path[idx])),
                            float(self.config.guide_strip_width),
                            float(self.config.guide_strip_height),
                        ),
                        z=0.5 * float(self.config.guide_strip_height),
                        rgba=(0.86, 0.82, 0.38, 1.0),
                    )
                )

        for idx, seg in enumerate(segment_obstacles):
            p1 = seg[:2]
            p2 = seg[2:4]
            center = 0.5 * (p1 + p2)
            yaw = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
            length = float(np.linalg.norm(p2 - p1))
            if length < 1e-6:
                continue
            models.append(
                self._box_model_xml(
                    name=f"wall_{idx}",
                    center=center,
                    yaw=yaw,
                    size=(length, float(self.config.wall_thickness), float(self.config.wall_height)),
                    z=0.5 * float(self.config.wall_height),
                    rgba=(0.38, 0.44, 0.52, 1.0),
                )
            )

        for idx, obs in enumerate(circle_obstacles):
            height = float(self.config.obstacle_min_height + (idx % 4) * self.config.obstacle_height_step)
            models.append(
                self._cylinder_model_xml(
                    name=f"obstacle_{idx}",
                    center=obs[:2],
                    radius=float(obs[2]),
                    height=height,
                    rgba=(0.73, 0.36, 0.30, 1.0),
                )
            )

        models_xml = "\n".join(models)
        return f"""<?xml version='1.0'?>
<sdf version='1.6'>
  <world name='followdataset_mid360_world'>
    <include>
      <uri>model://sun</uri>
    </include>
    <include>
      <uri>model://ground_plane</uri>
    </include>
    <gravity>0 0 -9.81</gravity>
    <scene>
      <ambient>0.45 0.45 0.45 1</ambient>
      <background>0.72 0.78 0.85 1</background>
      <shadows>true</shadows>
    </scene>
    {models_xml}
  </world>
</sdf>
"""

    def _generate_robot_sdf(self) -> str:
        mesh_path = (
            self.config.plugin_dir
            / "livox_laser_simulation"
            / "meshes"
            / "livox_mid-360-90x.dae"
        )
        mesh_xml = ""
        if mesh_path.exists():
            mesh_xml = f"""
      <visual name='mid360_visual'>
        <geometry>
          <mesh>
            <uri>file://{mesh_path}</uri>
            <scale>1 1 1</scale>
          </mesh>
        </geometry>
      </visual>
"""

        return f"""<?xml version='1.0'?>
<sdf version='1.6'>
  <model name='followdataset_mid360_robot'>
    <static>false</static>
    <allow_auto_disable>false</allow_auto_disable>
    <link name='base_link'>
      <gravity>false</gravity>
      <self_collide>false</self_collide>
      <inertial>
        <mass>10.0</mass>
        <inertia>
          <ixx>0.4</ixx>
          <iyy>0.4</iyy>
          <izz>0.4</izz>
        </inertia>
      </inertial>
      <collision name='base_collision'>
        <geometry>
          <box>
            <size>{self.config.robot_base_size[0]} {self.config.robot_base_size[1]} {self.config.robot_base_size[2]}</size>
          </box>
        </geometry>
      </collision>
      <visual name='base_visual'>
        <geometry>
          <box>
            <size>{self.config.robot_base_size[0]} {self.config.robot_base_size[1]} {self.config.robot_base_size[2]}</size>
          </box>
        </geometry>
        <material>
          <ambient>0.28 0.46 0.78 1</ambient>
          <diffuse>0.28 0.46 0.78 1</diffuse>
        </material>
      </visual>
    </link>
    <link name='{self.config.frame_name}'>
      <pose>{MID360_MOUNT_XYZ[0]} {MID360_MOUNT_XYZ[1]} {MID360_MOUNT_XYZ[2]} {MID360_MOUNT_RPY[0]} {MID360_MOUNT_RPY[1]} {MID360_MOUNT_RPY[2]}</pose>
      <gravity>false</gravity>
      {mesh_xml}
      <sensor type='ray' name='laser_livox'>
        <pose>0 0 0 0 0 0</pose>
        <visualize>{'true' if self.config.visualize_laser else 'false'}</visualize>
        <always_on>true</always_on>
        <update_rate>{self.config.update_rate}</update_rate>
        <plugin name='gazebo_ros_laser_controller' filename='liblivox_laser_simulation.so'>
          <ray>
            <scan>
              <horizontal>
                <samples>100</samples>
                <resolution>1</resolution>
                <min_angle>-3.1415926535897931</min_angle>
                <max_angle>3.1415926535897931</max_angle>
              </horizontal>
              <vertical>
                <samples>50</samples>
                <resolution>1</resolution>
                <min_angle>-3.1415926535897931</min_angle>
                <max_angle>3.1415926535897931</max_angle>
              </vertical>
            </scan>
            <range>
              <min>{self.config.range_min}</min>
              <max>{self.config.range_max}</max>
              <resolution>1</resolution>
            </range>
            <noise>
              <type>gaussian</type>
              <mean>0.0</mean>
              <stddev>0.0</stddev>
            </noise>
          </ray>
          <visualize>{'True' if self.config.visualize_laser else 'False'}</visualize>
          <samples>{self.config.samples}</samples>
          <downsample>{self.config.downsample}</downsample>
          <csv_file_name>mid360-real-centr.csv</csv_file_name>
          <publish_pointcloud_type>2</publish_pointcloud_type>
          <ros_topic>{self.config.ros_topic}</ros_topic>
          <frameName>{self.config.frame_name}</frameName>
        </plugin>
      </sensor>
    </link>
    <joint name='mid360_mount_joint' type='fixed'>
      <parent>base_link</parent>
      <child>{self.config.frame_name}</child>
    </joint>
  </model>
</sdf>
"""

    def _generate_human_sdf(self) -> str:
        return f"""<?xml version='1.0'?>
<sdf version='1.6'>
  <model name='followdataset_human'>
    <static>false</static>
    <link name='human_link'>
      <gravity>false</gravity>
      <inertial>
        <mass>70.0</mass>
        <inertia>
          <ixx>1.0</ixx>
          <iyy>1.0</iyy>
          <izz>1.0</izz>
        </inertia>
      </inertial>
      <collision name='human_collision'>
        <geometry>
          <cylinder>
            <radius>{self.config.human_radius}</radius>
            <length>{self.config.human_height}</length>
          </cylinder>
        </geometry>
      </collision>
      <visual name='human_visual'>
        <geometry>
          <cylinder>
            <radius>{self.config.human_radius}</radius>
            <length>{self.config.human_height}</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.66 0.79 0.95 1</ambient>
          <diffuse>0.66 0.79 0.95 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""

    def _box_model_xml(
        self,
        *,
        name: str,
        center: np.ndarray,
        yaw: float,
        size: tuple[float, float, float],
        z: float,
        rgba: tuple[float, float, float, float],
    ) -> str:
        return f"""
    <model name='{name}'>
      <static>true</static>
      <pose>{center[0]} {center[1]} {z} 0 0 {yaw}</pose>
      <link name='{name}_link'>
        <collision name='{name}_collision'>
          <geometry>
            <box>
              <size>{size[0]} {size[1]} {size[2]}</size>
            </box>
          </geometry>
        </collision>
        <visual name='{name}_visual'>
          <geometry>
            <box>
              <size>{size[0]} {size[1]} {size[2]}</size>
            </box>
          </geometry>
          <material>
            <ambient>{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}</ambient>
            <diffuse>{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}</diffuse>
          </material>
        </visual>
      </link>
    </model>
"""

    def _cylinder_model_xml(
        self,
        *,
        name: str,
        center: np.ndarray,
        radius: float,
        height: float,
        rgba: tuple[float, float, float, float],
    ) -> str:
        return f"""
    <model name='{name}'>
      <static>true</static>
      <pose>{center[0]} {center[1]} {0.5 * height} 0 0 0</pose>
      <link name='{name}_link'>
        <collision name='{name}_collision'>
          <geometry>
            <cylinder>
              <radius>{radius}</radius>
              <length>{height}</length>
            </cylinder>
          </geometry>
        </collision>
        <visual name='{name}_visual'>
          <geometry>
            <cylinder>
              <radius>{radius}</radius>
              <length>{height}</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}</ambient>
            <diffuse>{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}</diffuse>
          </material>
        </visual>
      </link>
    </model>
"""

    def _make_start_marker(self, start_pos: np.ndarray) -> str:
        return self._cylinder_model_xml(
            name="start_marker",
            center=start_pos[:2],
            radius=0.22,
            height=0.05,
            rgba=(0.25, 0.85, 0.34, 1.0),
        )

    def _make_end_marker(self, end_pos: np.ndarray) -> str:
        return self._cylinder_model_xml(
            name="end_marker",
            center=end_pos[:2],
            radius=0.22,
            height=0.05,
            rgba=(0.88, 0.23, 0.23, 1.0),
        )

    @staticmethod
    def _is_port_open(host: str, port: int) -> bool:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            return False
