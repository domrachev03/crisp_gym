"""Structured leader-follower recording: per-arm state + high-rate F/T + cameras.

Unlike the default flat ``observation.state`` vector, this records four named
proprio/force keys plus camera frames, written through the *upstream* LeRobot
``LeRobotDataset`` (v3):

    observation.follower_state   (S,)   cartesian + joints + gripper [+ target]
    observation.leader_state     (S,)   same, leader arm
    observation.follower_ft      (W,6)  high-rate wrench window (~kHz)
    observation.leader_ft        (W,6)
    observation.{f,l}_ft_time    (W,)   per-sample times, relative to frame
    observation.images.<name>    (H,W,3)
    action                       (7,)   cartesian delta + gripper (relative)

High-rate F/T comes from background NetFT pollers feeding
:class:`crisp_gym.record.hrate_buffer.HrateRingBuffer`; cameras use upstream
LeRobot ``RealSenseCamera`` (pyrealsense2 direct), independent of any ROS camera
node.  Code lives in crisp_gym; the panda rig configuration lives in crisp_env
(``CRISP_CONFIG_PATH``).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import rclpy
from geometry_msgs.msg import WrenchStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data

from crisp_gym.record.hrate_buffer import HrateRingBuffer
from crisp_gym.record.record_functions import _leader_gripper_to_action
from crisp_gym.util.control_type import ControlType

logger = logging.getLogger(__name__)

# Env observation keys (flat per-component state produced by ManipulatorEnv._get_obs).
_CARTESIAN = "observation.state.cartesian"
_JOINTS = "observation.state.joints"
_GRIPPER = "observation.state.gripper"
_TARGET = "observation.state.target"

_FT_DOF = 6  # wrench: fx fy fz tx ty tz
_FT_NAMES = ["fx", "fy", "fz", "tx", "ty", "tz"]


# --------------------------------------------------------------------------- #
# High-rate F/T
# --------------------------------------------------------------------------- #
class _HrateFTPoller:
    """Subscribes to a WrenchStamped topic and appends every sample to a buffer."""

    def __init__(self, node, topic: str, buffer: HrateRingBuffer) -> None:
        self.buffer = buffer
        self._sub = node.create_subscription(
            WrenchStamped, topic, self._cb, qos_profile_sensor_data
        )
        logger.debug(f"HrateFTPoller subscribed to {topic}")

    def _cb(self, msg: WrenchStamped) -> None:
        # Use reception wall-clock so buffer times share one clock with the frame t_ref.
        t = time.time()
        w = msg.wrench
        self.buffer.append(
            t, (w.force.x, w.force.y, w.force.z, w.torque.x, w.torque.y, w.torque.z)
        )


class HrateFTManager:
    """Owns a dedicated spinning node + ring buffers for named F/T streams."""

    def __init__(self, specs: list[tuple[str, str, int]]) -> None:
        """Args: specs = list of (name, topic, window)."""
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node("hrate_ft_pollers")
        self.buffers: dict[str, HrateRingBuffer] = {}
        self._pollers: list[_HrateFTPoller] = []
        for name, topic, window in specs:
            buf = HrateRingBuffer(window=window, dof=_FT_DOF)
            self.buffers[name] = buf
            self._pollers.append(_HrateFTPoller(self.node, topic, buf))
        self._exec = MultiThreadedExecutor()
        self._exec.add_node(self.node)
        self._thread = threading.Thread(target=self._exec.spin, daemon=True)
        self._thread.start()

    def snapshot(self, name: str, t_ref: float) -> tuple[np.ndarray, np.ndarray]:
        values, times = self.buffers[name].snapshot(t_ref)
        return values.astype(np.float32), times.astype(np.float32)

    def close(self) -> None:
        try:
            self._exec.shutdown()
            self.node.destroy_node()
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------- #
# Cameras (upstream LeRobot RealSense)
# --------------------------------------------------------------------------- #
@dataclass
class CameraSpec:
    """One RealSense camera to record."""

    name: str
    serial: str
    width: int = 640
    height: int = 480
    fps: int = 30


class CameraReader:
    """Connects LeRobot RealSenseCamera objects and reads frames per step."""

    def __init__(self, specs: list[CameraSpec]) -> None:
        from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig

        self.specs = specs
        self.cameras: dict[str, object] = {}
        for s in specs:
            cfg = RealSenseCameraConfig(
                serial_number_or_name=s.serial, fps=s.fps, width=s.width, height=s.height
            )
            cam = RealSenseCamera(cfg)
            cam.connect()
            self.cameras[s.name] = cam
            logger.info(f"Connected RealSense '{s.name}' (sn={s.serial}) {s.width}x{s.height}@{s.fps}")

    def read(self) -> dict[str, np.ndarray]:
        return {
            f"observation.images.{name}": np.asarray(cam.async_read())
            for name, cam in self.cameras.items()
        }

    def close(self) -> None:
        for cam in self.cameras.values():
            try:
                cam.disconnect()
            except Exception:  # noqa: BLE001
                pass


# --------------------------------------------------------------------------- #
# Feature schema
# --------------------------------------------------------------------------- #
def _cartesian_names(env) -> list[str]:
    """Cartesian component names for the env orientation representation."""
    rep = str(getattr(env.config.orientation_representation, "value", env.config.orientation_representation))
    if "quat" in rep.lower():
        return ["x", "y", "z", "qw", "qx", "qy", "qz"]
    return ["x", "y", "z", "roll", "pitch", "yaw"]


def build_structured_features(
    env,
    hrate_window: int,
    cameras: list[CameraSpec],
    include_target: bool = True,
) -> dict:
    """Construct the LeRobotDataset feature dict for the structured schema."""
    cart = _cartesian_names(env)
    njoints = env.config.robot_config.num_joints()
    joint_names = [f"joint_{i}" for i in range(njoints)]
    state_names = list(cart) + joint_names + ["gripper"]
    if include_target:
        state_names += [f"target_{c}" for c in cart]
    state_len = len(state_names)

    feats: dict = {}
    for arm in ("follower", "leader"):
        feats[f"observation.{arm}_state"] = {
            "dtype": "float32",
            "shape": (state_len,),
            "names": state_names,
        }
        feats[f"observation.{arm}_ft"] = {
            "dtype": "float32",
            "shape": (hrate_window, _FT_DOF),
            "names": _FT_NAMES,
        }
        feats[f"observation.{arm}_ft_time"] = {
            "dtype": "float32",
            "shape": (hrate_window,),
            "names": None,
        }

    video_info = {
        "video.fps": float(env.config.control_frequency),
        "video.codec": "av1",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "has_audio": False,
    }
    for cam in cameras:
        feats[f"observation.images.{cam.name}"] = {
            "dtype": "video",
            "shape": (cam.height, cam.width, 3),
            "names": ["height", "width", "channels"],
            "video_info": video_info,
        }

    feats["action"] = {
        "dtype": "float32",
        "shape": (len(cart) + 1,),
        "names": list(cart) + ["gripper"],
    }
    return feats, state_names


# --------------------------------------------------------------------------- #
# State assembly
# --------------------------------------------------------------------------- #
def _f32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).flatten()


def follower_state_from_obs(obs: dict, env, include_target: bool) -> np.ndarray:
    """Assemble follower state vector from an env observation dict."""
    rep = env.config.orientation_representation
    parts = [_f32(obs[_CARTESIAN]), _f32(obs[_JOINTS]), _f32(obs[_GRIPPER])]
    if include_target:
        parts.append(_f32(obs.get(_TARGET, np.zeros(len(_cartesian_names(env))))))
    return np.concatenate(parts).astype(np.float32)


def leader_state(leader, env, include_target: bool) -> np.ndarray:
    """Assemble leader state vector directly from the leader TeleopRobot.

    Gripper/target are read defensively: a leader arm may expose no gripper
    (e.g. a master haptic device) or have no published target while idle.
    """
    rep = env.config.orientation_representation
    cart = _f32(leader.robot.end_effector_pose.to_array(rep))
    joints = _f32(leader.robot.joint_values)
    try:
        g = leader.gripper.value if leader.gripper is not None else 0.0
    except Exception:  # noqa: BLE001  (gripper not initialized)
        g = 0.0
    parts = [cart, joints, _f32(g)]
    if include_target:
        try:
            tgt = _f32(leader.robot.target_pose.to_array(rep))
        except Exception:  # noqa: BLE001  (no target published while idle)
            tgt = cart
        parts.append(tgt)
    return np.concatenate(parts).astype(np.float32)


# --------------------------------------------------------------------------- #
# Data functions (frame producers for RecordingManager.record_episode)
# --------------------------------------------------------------------------- #
def make_structured_teleop_fn(
    env,
    leader,
    ft: HrateFTManager,
    cams: CameraReader | None,
    include_target: bool = True,
) -> Callable:
    """Teleop-driven frame producer: drives env.step from the leader, records both arms."""
    state = {"prev_pose": leader.robot.end_effector_pose,
             "prev_joint": leader.robot.joint_values,
             "first": True}

    def _fn() -> tuple:
        t_ref = time.time()
        pose = leader.robot.end_effector_pose
        joint = leader.robot.joint_values
        if state["first"]:
            state["first"] = False
            state["prev_pose"], state["prev_joint"] = pose, joint
            return None, None

        if env.config.use_relative_actions:
            action_pose = pose - state["prev_pose"]
            action_joint = joint - state["prev_joint"]
        else:
            action_pose = pose
            action_joint = joint
        state["prev_pose"], state["prev_joint"] = pose, joint

        gripper_action = _leader_gripper_to_action(
            leader_value=leader.gripper.value if leader.gripper is not None else 0.0,
            follower_value=env.gripper.value if env.gripper is not None else 0.0,
            control_mode=env.config.gripper_mode,
        )
        if env.ctrl_type is ControlType.CARTESIAN:
            action = np.concatenate(
                [action_pose.to_array(env.config.orientation_representation), [gripper_action]]
            )
        else:
            action = np.concatenate([action_joint, [gripper_action]])

        follower_obs, *_ = env.step(action, block=False)
        obs = _assemble_obs(follower_obs, env, leader, ft, cams, t_ref, include_target)
        return obs, action.astype(np.float32)

    return _fn


def make_structured_sample_fn(
    env,
    leader,
    ft: HrateFTManager,
    cams: CameraReader | None,
    include_target: bool = True,
) -> Callable:
    """Read-only frame producer (NO motion / NO env.step): for verification without teleop."""
    action_dim = len(_cartesian_names(env)) + 1

    def _fn() -> tuple:
        t_ref = time.time()
        follower_obs = env._get_obs()
        obs = _assemble_obs(follower_obs, env, leader, ft, cams, t_ref, include_target)
        return obs, np.zeros(action_dim, dtype=np.float32)

    return _fn


def _assemble_obs(follower_obs, env, leader, ft, cams, t_ref, include_target) -> dict:
    obs = {
        "observation.follower_state": follower_state_from_obs(follower_obs, env, include_target),
        "observation.leader_state": leader_state(leader, env, include_target),
    }
    fv, ftt = ft.snapshot("follower", t_ref)
    lv, ltt = ft.snapshot("leader", t_ref)
    obs["observation.follower_ft"] = fv
    obs["observation.follower_ft_time"] = ftt
    obs["observation.leader_ft"] = lv
    obs["observation.leader_ft_time"] = ltt
    if cams is not None:
        obs.update(cams.read())
    return obs
