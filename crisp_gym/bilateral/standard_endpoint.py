"""ROS 2 standard-message data plane for mixed-distro FR3 teleoperation.

This module deliberately has no controller-manager, action, Franka, or crisp_py
dependency.  Humble and Jazzy endpoint supervisors own lifecycle locally; the
inference PC only exchanges stable geometry messages with both machines.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class EndpointPose:
    """Minimal crisp-compatible Cartesian pose."""

    position: np.ndarray
    orientation: Rotation


@dataclass(frozen=True)
class EndpointTwist:
    """Minimal crisp-compatible Cartesian twist."""

    linear: np.ndarray
    angular: np.ndarray


class StandardMessageEndpoint:
    """One FR3 endpoint using only Pose/Twist/Wrench stamped messages."""

    def __init__(
        self,
        node,
        *,
        namespace: str,
        base_frame: str,
        tcp_frame: str,
        wrench_topic: str,
    ) -> None:
        # Keep ROS imports local so pure control-law tests need no ROS installation.
        from geometry_msgs.msg import PoseStamped, TwistStamped, WrenchStamped
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

        ns = namespace.strip("/")
        if not ns:
            raise ValueError("endpoint namespace cannot be empty")
        self.node = node
        self.namespace = ns
        self.base_frame = base_frame
        self.tcp_frame = tcp_frame
        self.pose_topic = f"/{ns}/current_pose"
        self.twist_topic = f"/{ns}/current_twist"
        self.target_pose_topic = f"/{ns}/target_pose"
        self.target_wrench_topic = f"/{ns}/target_wrench"
        self.wrench_topic = wrench_topic if wrench_topic.startswith("/") else f"/{ns}/{wrench_topic}"

        self._lock = threading.Lock()
        self._pose: EndpointPose | None = None
        self._twist: EndpointTwist | None = None
        self._wrench = np.zeros(6, dtype=float)
        self._bias = np.zeros(6, dtype=float)
        self._received_at = {"pose": 0.0, "twist": 0.0, "wrench": 0.0}
        self._sequence = {"pose": 0, "twist": 0, "wrench": 0}
        self._frame_errors: list[str] = []

        state_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        command_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._pose_pub = node.create_publisher(PoseStamped, self.target_pose_topic, command_qos)
        self._wrench_pub = node.create_publisher(
            WrenchStamped, self.target_wrench_topic, command_qos
        )
        self._subscriptions = [
            node.create_subscription(PoseStamped, self.pose_topic, self._on_pose, state_qos),
            node.create_subscription(TwistStamped, self.twist_topic, self._on_twist, state_qos),
            node.create_subscription(WrenchStamped, self.wrench_topic, self._on_wrench, state_qos),
        ]

    def _accept_frame(self, stream: str, frame_id: str) -> bool:
        # crisp_controllers publishes Cartesian pose in the robot base, while
        # its LOCAL Jacobian twist and the NetFT TCP wrench are TCP-expressed.
        # Checking these independently catches a silent frame mismatch without
        # rejecting the controller's intentional split-frame state contract.
        expected = self.base_frame if stream == "pose" else self.tcp_frame
        if frame_id == expected:
            return True
        error = f"{self.namespace} {stream}: expected frame {expected!r}, got {frame_id!r}"
        with self._lock:
            if not self._frame_errors or self._frame_errors[-1] != error:
                self._frame_errors.append(error)
        return False

    def _mark(self, stream: str) -> None:
        self._received_at[stream] = time.monotonic()
        self._sequence[stream] += 1

    def _on_pose(self, msg) -> None:
        if not self._accept_frame("pose", msg.header.frame_id):
            return
        q = np.array(
            [msg.pose.orientation.x, msg.pose.orientation.y,
             msg.pose.orientation.z, msg.pose.orientation.w], dtype=float
        )
        p = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        if not np.all(np.isfinite(p)) or not np.all(np.isfinite(q)) or np.linalg.norm(q) < 0.5:
            return
        pose = EndpointPose(p, Rotation.from_quat(q / np.linalg.norm(q)))
        with self._lock:
            self._pose = pose
            self._mark("pose")

    def _on_twist(self, msg) -> None:
        if not self._accept_frame("twist", msg.header.frame_id):
            return
        linear = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])
        angular = np.array([msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z])
        if not np.all(np.isfinite(linear)) or not np.all(np.isfinite(angular)):
            return
        with self._lock:
            self._twist = EndpointTwist(linear, angular)
            self._mark("twist")

    def _on_wrench(self, msg) -> None:
        if not self._accept_frame("wrench", msg.header.frame_id):
            return
        value = np.array(
            [msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
             msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z]
        )
        if not np.all(np.isfinite(value)):
            return
        with self._lock:
            self._wrench = value
            self._mark("wrench")

    @property
    def end_effector_pose(self) -> EndpointPose:
        with self._lock:
            if self._pose is None:
                raise RuntimeError(f"no pose received from {self.pose_topic}")
            return EndpointPose(self._pose.position.copy(), self._pose.orientation)

    @property
    def end_effector_twist(self) -> EndpointTwist:
        with self._lock:
            if self._twist is None:
                raise RuntimeError(f"no twist received from {self.twist_topic}")
            return EndpointTwist(self._twist.linear.copy(), self._twist.angular.copy())

    @property
    def wrench(self) -> np.ndarray:
        with self._lock:
            return self._wrench.copy() - self._bias

    @property
    def frame_errors(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._frame_errors)

    def stream_age(self, stream: str) -> float:
        with self._lock:
            stamp = self._received_at[stream]
        return float("inf") if stamp == 0.0 else time.monotonic() - stamp

    def ready(self, *, require_wrench: bool = True, max_age_s: float = 0.25) -> bool:
        streams = ("pose", "twist", "wrench") if require_wrench else ("pose", "twist")
        return not self.frame_errors and all(self.stream_age(name) <= max_age_s for name in streams)

    def wait_ready(self, timeout_s: float, *, require_wrench: bool = True) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.ready(require_wrench=require_wrench):
                return
            if self.frame_errors:
                raise RuntimeError(self.frame_errors[-1])
            time.sleep(0.02)
        ages = {name: self.stream_age(name) for name in self._received_at}
        raise TimeoutError(f"endpoint {self.namespace} not ready; stream ages={ages}")

    def capture_wrench_bias(self, duration_s: float = 1.0) -> np.ndarray:
        """Average distinct, fresh samples while the endpoint is stationary."""
        deadline = time.monotonic() + duration_s
        samples: list[np.ndarray] = []
        last_seq = -1
        while time.monotonic() < deadline:
            with self._lock:
                seq = self._sequence["wrench"]
                value = self._wrench.copy()
            if seq != last_seq:
                samples.append(value)
                last_seq = seq
            time.sleep(0.001)
        if len(samples) < max(20, int(duration_s * 100.0)):
            raise RuntimeError(f"only {len(samples)} distinct wrench samples captured")
        bias = np.mean(samples, axis=0)
        with self._lock:
            self._bias = bias
        return bias

    def assert_command_ownership(self) -> None:
        """Require this process to be the sole publisher on both command topics."""
        for topic in (self.target_pose_topic, self.target_wrench_topic):
            count = self.node.count_publishers(topic)
            if count != 1:
                raise RuntimeError(f"expected one publisher on {topic}, found {count}")

    def set_target(self, *, pose) -> None:
        from geometry_msgs.msg import PoseStamped

        msg = PoseStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = self.base_frame
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, pose.position)
        q = pose.orientation.as_quat()
        msg.pose.orientation.x = float(q[0])
        msg.pose.orientation.y = float(q[1])
        msg.pose.orientation.z = float(q[2])
        msg.pose.orientation.w = float(q[3])
        self._pose_pub.publish(msg)

    @staticmethod
    def pose_from_vec(vec: np.ndarray) -> EndpointPose:
        """Convert the adapter's w-first seven-vector to an endpoint pose."""
        value = np.asarray(vec, dtype=float)
        return EndpointPose(
            value[:3].copy(),
            Rotation.from_quat([value[4], value[5], value[6], value[3]]),
        )

    def set_target_wrench(self, *, force=None, torque=None) -> None:
        from geometry_msgs.msg import WrenchStamped

        force = np.zeros(3) if force is None else np.asarray(force, dtype=float)
        torque = np.zeros(3) if torque is None else np.asarray(torque, dtype=float)
        if force.shape != (3,) or torque.shape != (3,) or not np.all(np.isfinite([*force, *torque])):
            raise ValueError("force and torque must be finite 3-vectors")
        msg = WrenchStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = self.tcp_frame
        msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z = map(float, force)
        msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z = map(float, torque)
        self._wrench_pub.publish(msg)

    def publish_hold(self) -> None:
        """Refresh the current pose and zero wrench for the endpoint watchdog."""
        self.set_target(pose=self.end_effector_pose)
        self.set_target_wrench(force=np.zeros(3), torque=np.zeros(3))
