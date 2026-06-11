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


class DashboardMonitor:
    def __init__(self, arms: tuple[str, ...] = ("right", "left"),
                 status_topic: str = "/record_status",
                 control_topic: str = "/record_transition",
                 wrench_template: str = "/{ns}/netft_data_unbiased_tcp",
                 pose_template: str = "/{ns}/current_pose",
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

    def close(self) -> None:
        try:
            self._exec.shutdown()
            self.node.destroy_node()
        except Exception:  # noqa: BLE001
            pass
