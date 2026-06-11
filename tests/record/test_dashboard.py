"""Recording dashboard FastAPI app (routes + control validation), with a fake monitor.

No ROS: the app talks to a monitor object; the real rclpy monitor is exercised on HW.
"""

import pytest

from crisp_gym.record.dashboard import VALID_COMMANDS, create_app

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


class _FakeMonitor:
    def __init__(self):
        self.sent = []

    def snapshot(self):
        return {
            "recorder": {"connected": True, "state": "recording", "episode": 3,
                         "num_episodes": 10, "fps": 59.1, "frames": 120,
                         "repo_id": "x/y", "last_event": "ok"},
            "arms": {"right": {"force": 4.0, "pos": [0.1, 0.2, 0.3]}, "left": {}},
        }

    def send_control(self, cmd):
        self.sent.append(cmd)
        return True


def test_valid_commands():
    assert set(VALID_COMMANDS) == {"record", "save", "delete", "exit"}


def test_status_route_returns_snapshot():
    c = TestClient(create_app(_FakeMonitor()))
    r = c.get("/status")
    assert r.status_code == 200
    assert r.json()["recorder"]["state"] == "recording"
    assert r.json()["arms"]["right"]["force"] == 4.0


def test_index_serves_html():
    c = TestClient(create_app(_FakeMonitor()))
    r = c.get("/")
    assert r.status_code == 200
    assert "recording" in r.text.lower() and "<html" in r.text.lower()


def test_valid_control_publishes():
    m = _FakeMonitor()
    c = TestClient(create_app(m))
    r = c.post("/control/record")
    assert r.status_code == 200 and r.json()["ok"] is True
    assert m.sent == ["record"]


def test_invalid_control_rejected_without_publishing():
    m = _FakeMonitor()
    c = TestClient(create_app(m))
    r = c.post("/control/launch_rockets")
    assert r.status_code == 400
    assert m.sent == []
