"""Recording dashboard FastAPI app (routes + control validation), with a fake monitor.

No ROS: the app talks to a monitor object; the real rclpy monitor is exercised on HW.
"""

import pytest

from crisp_gym.record.dashboard import GRIPPER_WIDTHS, VALID_COMMANDS, create_app

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


class _FakeMonitor:
    def __init__(self, gripper_ok=True):
        self.sent = []
        self.gripper = []
        self._gripper_ok = gripper_ok

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

    def set_gripper(self, width):
        self.gripper.append(width)
        return self._gripper_ok


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


def test_gripper_open_close_map_to_widths():
    assert GRIPPER_WIDTHS == {"open": 1.0, "close": 0.0}


def test_gripper_open_commands_width_1():
    m = _FakeMonitor()
    c = TestClient(create_app(m))
    r = c.post("/gripper/open")
    assert r.status_code == 200 and r.json()["ok"] is True
    assert m.gripper == [1.0]


def test_gripper_close_commands_width_0():
    m = _FakeMonitor()
    c = TestClient(create_app(m))
    r = c.post("/gripper/close")
    assert r.status_code == 200
    assert m.gripper == [0.0]


def test_gripper_invalid_action_rejected():
    m = _FakeMonitor()
    c = TestClient(create_app(m))
    r = c.post("/gripper/halfway")
    assert r.status_code == 400
    assert m.gripper == []


def test_gripper_unavailable_reports_not_ok():
    m = _FakeMonitor(gripper_ok=False)
    c = TestClient(create_app(m))
    r = c.post("/gripper/open")
    assert r.status_code == 200 and r.json()["ok"] is False


def test_layout_has_gripper_buttons_and_rerun_left():
    c = TestClient(create_app(_FakeMonitor()))
    html = c.get("/").text
    low = html.lower()
    assert "grip('open')" in low and "grip('close')" in low  # gripper buttons
    assert ">gripper<" in low  # gripper card label
    # rerun on the left, the stacked cards on the right
    assert 'class="left"' in low and 'class="right"' in low
    assert low.index('class="left"') < low.index('id="rerun"')


def test_index_has_inline_rerun_iframe():
    """The page ships the rerun iframe embedded inline (always visible, no toggle)."""
    c = TestClient(create_app(_FakeMonitor()))
    html = c.get("/").text.lower()
    assert "<iframe" in html
    assert 'id="rerun"' in html  # the iframe element
    assert "rerun viewer" in html  # inline panel label
    assert "togglererun" not in html  # the show/hide toggle is gone
    assert "window.location.hostname" in html  # host derived for ssh-tunnel/LAN use


def test_rerun_config_default_ports():
    """Without an override the config exposes the default 9090/9877 rerun ports."""
    c = TestClient(create_app(_FakeMonitor()))
    cfg = c.get("/rerun_config").json()
    assert cfg["web_port"] == 9090
    assert cfg["ws_port"] == 9877
    assert cfg["url"] is None  # let the page build it from window.location.hostname


def test_rerun_config_explicit_url():
    """An explicit --rerun-url is served verbatim for the iframe src."""
    url = "http://my-host:9090?url=ws://my-host:9877"
    c = TestClient(create_app(_FakeMonitor(), rerun_url=url))
    cfg = c.get("/rerun_config").json()
    assert cfg["url"] == url


def test_rerun_config_custom_ports():
    c = TestClient(create_app(_FakeMonitor(), rerun_web_port=8000, rerun_ws_port=8001))
    cfg = c.get("/rerun_config").json()
    assert cfg["web_port"] == 8000
    assert cfg["ws_port"] == 8001
