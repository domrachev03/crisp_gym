"""ROS monitor backing the recording dashboard (rclpy).

Subscribes the recorder's ``/record_status`` (JSON) plus each arm's NetFT + EE pose,
keeps the latest of each, and exposes ``snapshot()`` (merged status for the web app)
and ``send_control(cmd)`` (publish to ``/record_transition``). Runs its own spinning
node so the FastAPI app can read a fresh snapshot any time.
"""

from __future__ import annotations

import json
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, WrenchStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String


def _resolve_gripper_config(name: str):  # noqa: ANN201
    """Load a crisp_gym gripper yaml (by name or path) into a GripperConfig, or None.

    Mirrors how the env resolves it: crisp_gym ``find_config`` locates the yaml under the
    CRISP config paths, then ``GripperConfig.from_yaml``. Returns None if not found there
    so the caller can fall back to crisp_py's built-in config names.
    """
    try:
        from crisp_gym.config.path import find_config
        from crisp_py.gripper.gripper_config import GripperConfig

        rel = name if name.endswith((".yaml", ".yml")) else f"grippers/{name}.yaml"
        path = find_config(rel)
        if path is None:
            return None
        return GripperConfig.from_yaml(path=path.resolve())
    except Exception:  # noqa: BLE001 -- fall back to crisp_py name lookup
        return None


class DashboardMonitor:
    def __init__(self, arms: tuple[str, ...] = ("right", "left"),
                 status_topic: str = "/record_status",
                 control_topic: str = "/record_transition",
                 wrench_template: str = "/{ns}/netft_data_unbiased_tcp",
                 pose_template: str = "/{ns}/current_pose",
                 gripper_config: str | None = None,
                 gripper_namespace: str = "right",
                 stale_s: float = 2.0) -> None:
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node("record_dashboard")
        self.arms = arms
        self.stale_s = stale_s
        self._lock = threading.Lock()
        self._recorder: dict = {}
        self._recorder_t = 0.0
        self._arms: dict[str, dict] = {a: {} for a in arms}

        self.node.create_subscription(String, status_topic, self._on_status, 10)
        self._control_pub = self.node.create_publisher(String, control_topic, 10)
        for a in arms:
            self.node.create_subscription(
                WrenchStamped, wrench_template.format(ns=a),
                lambda m, a=a: self._on_wrench(a, m), qos_profile_sensor_data)
            self.node.create_subscription(
                PoseStamped, pose_template.format(ns=a),
                lambda m, a=a: self._on_pose(a, m), qos_profile_sensor_data)

        self._exec = MultiThreadedExecutor()
        self._exec.add_node(self.node)

        # Optional gripper: a crisp_py Gripper driven by the Open/Close buttons
        # (set_target 1.0 / 0.0, like franka_server_standalone). Non-fatal: if it
        # cannot be created, the buttons just report "unavailable".
        self._gripper = None
        if gripper_config:
            try:
                from crisp_py.gripper.gripper import make_gripper

                # Resolve crisp_gym gripper yamls (e.g. gripper_right_v2) via crisp_gym's
                # own config paths first -- crisp_py's make_gripper only searches its own
                # config dir. Fall back to crisp_py's built-in configs by name.
                cfg = _resolve_gripper_config(gripper_config)
                if cfg is not None:
                    self._gripper = make_gripper(
                        None, gripper_config=cfg, namespace=gripper_namespace, spin_node=False)
                else:
                    self._gripper = make_gripper(
                        gripper_config, namespace=gripper_namespace, spin_node=False)
                self._exec.add_node(self._gripper.node)
                self.node.get_logger().info(
                    f"gripper '{gripper_config}' (ns={gripper_namespace}) ready.")
            except Exception as e:  # noqa: BLE001
                self._gripper = None
                self.node.get_logger().warning(
                    f"gripper '{gripper_config}' (ns={gripper_namespace}) unavailable: {e}")

        self._thread = threading.Thread(target=self._exec.spin, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _on_status(self, msg: String) -> None:
        with self._lock:
            try:
                self._recorder = json.loads(msg.data)
            except Exception:  # noqa: BLE001
                pass
            self._recorder_t = time.time()

    def _on_wrench(self, arm: str, msg: WrenchStamped) -> None:
        w = msg.wrench
        with self._lock:
            self._arms[arm]["force"] = float(np.linalg.norm([w.force.x, w.force.y, w.force.z]))
            self._arms[arm]["torque"] = float(np.linalg.norm([w.torque.x, w.torque.y, w.torque.z]))

    def _on_pose(self, arm: str, msg: PoseStamped) -> None:
        p = msg.pose.position
        with self._lock:
            self._arms[arm]["pos"] = [round(p.x, 3), round(p.y, 3), round(p.z, 3)]

    def snapshot(self) -> dict:
        with self._lock:
            connected = bool(self._recorder) and (time.time() - self._recorder_t) < self.stale_s
            rec = dict(self._recorder)
            rec["connected"] = connected
            arms = {a: dict(v) for a, v in self._arms.items()}
        return {"recorder": rec, "arms": arms}

    def send_control(self, cmd: str) -> bool:
        msg = String()
        msg.data = cmd
        self._control_pub.publish(msg)
        return True

    def set_gripper(self, width: float) -> bool:
        """Command the gripper to a normalized target width (1.0 open / 0.0 close)."""
        if self._gripper is None:
            return False
        try:
            self._gripper.set_target(target=float(width))
            return True
        except Exception as e:  # noqa: BLE001
            self.node.get_logger().warning(f"gripper set_target failed: {e}")
            return False

    def close(self) -> None:
        try:
            self._exec.shutdown()
            self.node.destroy_node()
        except Exception:  # noqa: BLE001
            pass
