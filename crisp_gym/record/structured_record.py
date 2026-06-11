"""Structured leader-follower recording: per-arm state + multi-signal high-rate logs + cameras.

Records, through the *upstream* LeRobot ``LeRobotDataset`` (v3):

    observation.follower_state        (S,)    cartesian + joints + gripper [+ target]
    observation.leader_state          (S,)    same, leader arm
    observation.<arm>_<sig>_hrate     (W,D)   high-rate window of a signal (~its native rate)
    observation.<arm>_<sig>_hrate_time(W,)    per-sample times, relative to the frame
    observation.images.<name>         (H,W,3)
    action                            (7,)    cartesian delta + gripper (relative)

High-rate signals (``sig``) are polled off their ROS topics at native rate by a
background executor and snapshotted per recorded frame:

    ft      WrenchStamped  /<ns>/netft_data_unbiased_tcp   ~2.1 kHz   6 dof
    joints  JointState     /<ns>/joint_states              ~1 kHz     7 dof
    pose    PoseStamped    /<ns>/current_pose              ~250 Hz    7 dof (xyz + quat)
    twist   TwistStamped   /<ns>/current_twist             ~250 Hz    6 dof

Per-signal window sizes are FIXED and fps-independent (lerobot-panda style): the ft
signal gets ``ft_window`` samples and every other signal scales by its native rate,
so all cover the same time span no matter the record fps.
Cameras use upstream LeRobot ``RealSenseCamera`` (pyrealsense2 direct).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped, WrenchStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

from crisp_gym.record.hrate_buffer import HrateRingBuffer
from crisp_gym.record.record_functions import _leader_gripper_to_action
from crisp_gym.util.control_type import ControlType

logger = logging.getLogger(__name__)

# Env observation keys (flat per-component state produced by ManipulatorEnv._get_obs).
_CARTESIAN = "observation.state.cartesian"
_JOINTS = "observation.state.joints"
_GRIPPER = "observation.state.gripper"
_TARGET = "observation.state.target"


# --------------------------------------------------------------------------- #
# High-rate signal registry
# --------------------------------------------------------------------------- #
def _ft_extract(m):  # noqa: ANN001
    w = m.wrench
    return (w.force.x, w.force.y, w.force.z, w.torque.x, w.torque.y, w.torque.z)


def _pose_extract(m):  # noqa: ANN001
    p, o = m.pose.position, m.pose.orientation
    return (p.x, p.y, p.z, o.x, o.y, o.z, o.w)


def _twist_extract(m):  # noqa: ANN001
    t = m.twist
    return (t.linear.x, t.linear.y, t.linear.z, t.angular.x, t.angular.y, t.angular.z)


def _joints_extract(m):  # noqa: ANN001
    return tuple(m.position[:7])


# sig -> (msg type, topic suffix, dof, nominal rate Hz, extractor, component names)
HRATE_SIGNALS: dict[str, dict] = {
    "ft": dict(msg=WrenchStamped, topic="netft_data_unbiased_tcp", dof=6, rate=2100.0,
               extract=_ft_extract, names=["fx", "fy", "fz", "tx", "ty", "tz"]),
    "joints": dict(msg=JointState, topic="joint_states", dof=7, rate=1000.0,
                   extract=_joints_extract, names=[f"joint_{i}" for i in range(7)]),
    "pose": dict(msg=PoseStamped, topic="current_pose", dof=7, rate=250.0,
                 extract=_pose_extract, names=["x", "y", "z", "qx", "qy", "qz", "qw"]),
    "twist": dict(msg=TwistStamped, topic="current_twist", dof=6, rate=250.0,
                  extract=_twist_extract, names=["vx", "vy", "vz", "wx", "wy", "wz"]),
}
DEFAULT_SIGNALS = ["ft", "joints", "pose", "twist"]


# Reference rate (ft) that anchors the fixed hrate windows.
FT_RATE: float = HRATE_SIGNALS["ft"]["rate"]


def hrate_window(rate: float, ft_window: int) -> int:
    """Fixed, fps-INDEPENDENT window size (lerobot-panda style).

    The ft signal gets ``ft_window`` samples; every other signal's window is
    proportional to its native rate, so all signals cover the same time span
    (``ft_window / FT_RATE`` seconds) regardless of the record fps.
    """
    return max(1, int(round(ft_window * rate / FT_RATE)))


# --------------------------------------------------------------------------- #
# High-rate poller manager (multiple signals, both arms)
# --------------------------------------------------------------------------- #
class _HratePoller:
    def __init__(self, node, topic: str, msg_type, extract: Callable, buffer: HrateRingBuffer) -> None:
        self._extract = extract
        self._buf = buffer
        self._sub = node.create_subscription(msg_type, topic, self._cb, qos_profile_sensor_data)
        logger.debug(f"HratePoller subscribed to {topic}")

    def _cb(self, msg) -> None:  # noqa: ANN001
        # Reception wall-clock so buffer times share one clock with the frame t_ref.
        self._buf.append(time.time(), self._extract(msg))


class HrateManager:
    """Owns a dedicated spinning node + ring buffers for named high-rate streams."""

    def __init__(self, specs: list[tuple[str, str, str, int]]) -> None:
        """Args: specs = list of (key, topic, sig, window); sig in HRATE_SIGNALS."""
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node("hrate_pollers")
        self.buffers: dict[str, HrateRingBuffer] = {}
        self.dofs: dict[str, int] = {}
        self._pollers: list[_HratePoller] = []
        for key, topic, sig, window in specs:
            s = HRATE_SIGNALS[sig]
            buf = HrateRingBuffer(window=window, dof=s["dof"])
            self.buffers[key] = buf
            self.dofs[key] = s["dof"]
            self._pollers.append(_HratePoller(self.node, topic, s["msg"], s["extract"], buf))
        self._exec = MultiThreadedExecutor()
        self._exec.add_node(self.node)
        self._thread = threading.Thread(target=self._exec.spin, daemon=True)
        self._thread.start()

    def snapshot(self, key: str, t_ref: float) -> tuple[np.ndarray, np.ndarray]:
        values, times = self.buffers[key].snapshot(t_ref)
        return values.astype(np.float32), times.astype(np.float32)

    def close(self) -> None:
        try:
            self._exec.shutdown()
            self.node.destroy_node()
        except Exception:  # noqa: BLE001
            pass


def build_hrate_specs(
    arms: dict[str, str],
    signals: list[str],
    ft_window: int = 150,
    ft_topic_template: str = "/{ns}/{topic}",
) -> list[tuple[str, str, str, int]]:
    """Build (key, topic, sig, window) specs for the given arms x signals.

    Args:
        arms: mapping of arm label ("follower"/"leader") -> namespace ("right"/"left").
        signals: which HRATE_SIGNALS keys to record.
        ft_window: fixed ft window in samples (fps-independent); others scale by rate.
        ft_topic_template: topic format with {ns} and {topic}.
    """
    specs = []
    for arm, ns in arms.items():
        for sig in signals:
            s = HRATE_SIGNALS[sig]
            topic = ft_topic_template.format(ns=ns, topic=s["topic"])
            window = hrate_window(s["rate"], ft_window)
            specs.append((f"{arm}_{sig}", topic, sig, window))
    return specs


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


def load_camera_specs(config_name: str) -> list[CameraSpec]:
    """Load a list of CameraSpec from a YAML under CRISP config paths.

    YAML shape::

        cameras:
          - {name: wrist, serial: "130322271369", width: 640, height: 480, fps: 30}
    """
    import yaml

    from crisp_gym.config.path import find_config

    rel = config_name if config_name.endswith((".yaml", ".yml")) else f"cameras/{config_name}.yaml"
    path = find_config(rel)
    if path is None:
        raise FileNotFoundError(f"Camera config '{rel}' not found in CRISP config paths.")
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return [CameraSpec(**c) for c in data.get("cameras", [])]


class CameraReader:
    """Connects LeRobot RealSenseCamera objects and reads frames per step.

    A read that times out raises (RealSense ``async_read`` TimeoutError). The recording
    manager catches it and DISCARDS the current episode rather than crashing the run --
    a dropped camera frame nullifies that episode, the session continues. The per-read
    timeout is a little more tolerant than LeRobot's 200 ms default to avoid nullifying
    an episode over a single slightly-late frame.
    """

    def __init__(self, specs: list[CameraSpec], read_timeout_ms: int = 300) -> None:
        from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig

        self.specs = specs
        self.read_timeout_ms = read_timeout_ms
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
            f"observation.images.{name}": np.asarray(cam.async_read(timeout_ms=self.read_timeout_ms))
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
    rep = str(getattr(env.config.orientation_representation, "value", env.config.orientation_representation))
    if "quat" in rep.lower():
        return ["x", "y", "z", "qw", "qx", "qy", "qz"]
    return ["x", "y", "z", "roll", "pitch", "yaw"]


def build_structured_features(
    env,
    cameras: list[CameraSpec],
    fps: float,
    signals: list[str] | None = None,
    ft_window: int = 150,
    include_target: bool = True,
    arms: tuple[str, str] = ("follower", "leader"),
    bilateral_dof: int | None = None,
) -> tuple[dict, list[str]]:
    """Construct the LeRobotDataset feature dict for the structured schema.

    When ``bilateral_dof`` is given (the controller's generalized-coordinate
    dimension: 6 cartesian, 7 joint), the bilateral telemetry fields are added and
    the recorded ``action`` is the commanded follower target (+ gripper) rather than
    the cartesian-delta action, so the dataset reflects what the teleop scheme drove.
    """
    signals = signals if signals is not None else DEFAULT_SIGNALS
    cart = _cartesian_names(env)
    njoints = env.config.robot_config.num_joints()
    joint_names = [f"joint_{i}" for i in range(njoints)]
    state_names = list(cart) + joint_names + ["gripper"]
    if include_target:
        state_names += [f"target_{c}" for c in cart]
    state_len = len(state_names)

    feats: dict = {}
    for arm in arms:
        feats[f"observation.{arm}_state"] = {
            "dtype": "float32", "shape": (state_len,), "names": state_names,
        }
        for sig in signals:
            s = HRATE_SIGNALS[sig]
            w = hrate_window(s["rate"], ft_window)
            feats[f"observation.{arm}_{sig}_hrate"] = {
                "dtype": "float32", "shape": (w, s["dof"]), "names": s["names"],
            }
            feats[f"observation.{arm}_{sig}_hrate_time"] = {
                "dtype": "float32", "shape": (w,), "names": None,
            }

    video_info = {
        "video.fps": float(fps), "video.codec": "av1", "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False, "has_audio": False,
    }
    for cam in cameras:
        feats[f"observation.images.{cam.name}"] = {
            "dtype": "video", "shape": (cam.height, cam.width, 3),
            "names": ["height", "width", "channels"], "video_info": video_info,
        }

    if bilateral_dof is not None:
        from crisp_gym.record.bilateral_frame import _component_names, bilateral_feature_specs

        feats.update(bilateral_feature_specs(bilateral_dof))
        act_names = _component_names(bilateral_dof) + ["gripper"]
        feats["action"] = {"dtype": "float32", "shape": (bilateral_dof + 1,), "names": act_names}
    else:
        feats["action"] = {"dtype": "float32", "shape": (len(cart) + 1,), "names": list(cart) + ["gripper"]}
    return feats, state_names


# --------------------------------------------------------------------------- #
# State assembly
# --------------------------------------------------------------------------- #
def _f32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).flatten()


def follower_state_from_obs(obs: dict, env, include_target: bool) -> np.ndarray:
    parts = [_f32(obs[_CARTESIAN]), _f32(obs[_JOINTS]), _f32(obs[_GRIPPER])]
    if include_target:
        parts.append(_f32(obs.get(_TARGET, np.zeros(len(_cartesian_names(env))))))
    return np.concatenate(parts).astype(np.float32)


def leader_state(leader, env, include_target: bool) -> np.ndarray:
    """Leader state vector; gripper/target read defensively (may be absent/idle)."""
    rep = env.config.orientation_representation
    cart = _f32(leader.robot.end_effector_pose.to_array(rep))
    joints = _f32(leader.robot.joint_values)
    try:
        g = leader.gripper.value if leader.gripper is not None else 0.0
    except Exception:  # noqa: BLE001
        g = 0.0
    parts = [cart, joints, _f32(g)]
    if include_target:
        try:
            tgt = _f32(leader.robot.target_pose.to_array(rep))
        except Exception:  # noqa: BLE001
            tgt = cart
        parts.append(tgt)
    return np.concatenate(parts).astype(np.float32)


# --------------------------------------------------------------------------- #
# Data functions (frame producers for RecordingManager.record_episode)
# --------------------------------------------------------------------------- #
def make_structured_teleop_fn(env, leader, hrate: HrateManager, cams: CameraReader | None,
                              signals: list[str] | None = None, include_target: bool = True) -> Callable:
    """Teleop-driven frame producer: drives env.step from the leader, records both arms."""
    state = {"prev_pose": leader.robot.end_effector_pose,
             "prev_joint": leader.robot.joint_values, "first": True}
    signals = signals if signals is not None else DEFAULT_SIGNALS

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
            action_pose, action_joint = pose, joint
        state["prev_pose"], state["prev_joint"] = pose, joint

        gripper_action = _leader_gripper_to_action(
            leader_value=leader.gripper.value if leader.gripper is not None else 0.0,
            follower_value=env.gripper.value if env.gripper is not None else 0.0,
            control_mode=env.config.gripper_mode,
        )
        if env.ctrl_type is ControlType.CARTESIAN:
            action = np.concatenate(
                [action_pose.to_array(env.config.orientation_representation), [gripper_action]])
        else:
            action = np.concatenate([action_joint, [gripper_action]])

        follower_obs, *_ = env.step(action, block=False)
        obs = _assemble_obs(follower_obs, env, leader, hrate, cams, t_ref, signals, include_target)
        return obs, action.astype(np.float32)

    return _fn


def make_structured_sample_fn(env, leader, hrate: HrateManager, cams: CameraReader | None,
                              signals: list[str] | None = None, include_target: bool = True) -> Callable:
    """Read-only frame producer (NO motion / NO env.step): for verification without teleop."""
    action_dim = len(_cartesian_names(env)) + 1
    signals = signals if signals is not None else DEFAULT_SIGNALS

    def _fn() -> tuple:
        t_ref = time.time()
        follower_obs = env._get_obs()
        obs = _assemble_obs(follower_obs, env, leader, hrate, cams, t_ref, signals, include_target)
        return obs, np.zeros(action_dim, dtype=np.float32)

    return _fn


def make_structured_bilateral_fn(env, leader, controller, hrate: HrateManager,
                                 cams: CameraReader | None, signals: list[str] | None = None,
                                 include_target: bool = True) -> Callable:
    """Controller-driven frame producer for any bilateral scheme.

    The ``BilateralController`` commands BOTH arms each tick (forward position,
    reflected force, 4-channel force/spring, optional TDPA); we then read the
    follower observation read-only and record both arms' structured state, the
    high-rate windows, the cameras, and the control telemetry. The recorded action
    is the commanded (DELAYED) follower target plus the gripper command.
    """
    from crisp_gym.record.bilateral_frame import bilateral_action, telemetry_obs_fields

    signals = signals if signals is not None else DEFAULT_SIGNALS

    def _fn() -> tuple:
        t_ref = time.time()
        tel = controller.step()             # commands leader + follower
        follower_obs = env._get_obs()       # read-only: controller already commanded
        obs = _assemble_obs(follower_obs, env, leader, hrate, cams, t_ref, signals, include_target)
        obs.update(telemetry_obs_fields(tel))
        gripper_action = _leader_gripper_to_action(
            leader_value=leader.gripper.value if leader.gripper is not None else 0.0,
            follower_value=env.gripper.value if env.gripper is not None else 0.0,
            control_mode=env.config.gripper_mode,
        )
        action = bilateral_action(tel, gripper_action)
        return obs, action

    return _fn


def _assemble_obs(follower_obs, env, leader, hrate, cams, t_ref, signals, include_target) -> dict:
    obs = {
        "observation.follower_state": follower_state_from_obs(follower_obs, env, include_target),
        "observation.leader_state": leader_state(leader, env, include_target),
    }
    for arm in ("follower", "leader"):
        for sig in signals:
            values, times = hrate.snapshot(f"{arm}_{sig}", t_ref)
            obs[f"observation.{arm}_{sig}_hrate"] = values
            obs[f"observation.{arm}_{sig}_hrate_time"] = times
    if cams is not None:
        obs.update(cams.read())
    return obs
