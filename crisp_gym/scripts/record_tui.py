"""Small terminal controller for ``crisp-record-structured``.

The recorder publishes JSON status on ``/record_status`` and accepts commands on
``/record_transition``.  This client keeps that ROS interface unchanged and adds
an SSH-friendly curses UI with one-key episode controls.
"""

from __future__ import annotations

import argparse
import curses
import json
import time

import rclpy
from std_msgs.msg import String

_KEY_ACTIONS = {
    ord("r"): "record",
    ord("s"): "save",
    ord("d"): "delete",
    ord("q"): "exit",
}


def _action_allowed(action: str, state: str | None, mode: str = "record") -> bool:
    """Return whether the recording manager accepts ``action`` in ``state``."""
    if mode == "replay":
        if action == "record":
            return state in {"is_waiting", "recording", "paused"}
        if action == "exit":
            return state in {"is_waiting", "paused", "finished"}
        return False
    if action == "record":
        return state in {"is_waiting", "recording"}
    if action in {"save", "delete"}:
        return state == "paused"
    if action == "exit":
        return state in {"is_waiting", "paused"}
    return False


def _state_label(state: str | None, mode: str = "record") -> str:
    if mode == "replay":
        return {
            "is_waiting": "READY TO REPLAY",
            "recording": "REPLAYING",
            "paused": "REPLAY PAUSED",
            "finished": "REPLAY COMPLETE",
            "exit": "EXITING",
        }.get(state, "REPLAY NOT DETECTED")
    return {
        "is_waiting": "WAITING",
        "recording": "RECORDING",
        "paused": "PAUSED - SAVE OR DELETE",
        "to_be_saved": "SAVING",
        "to_be_deleted": "DELETING",
        "exit": "EXITING",
    }.get(state, "NOT DETECTED")


class RecordingTUI:
    """ROS status subscriber and transition publisher rendered with curses."""

    def __init__(self, status_topic: str, transition_topic: str, stale_timeout: float) -> None:
        """Create the ROS publisher/subscriber pair used by the UI."""
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node("recording_tui")
        self.publisher = self.node.create_publisher(String, transition_topic, 10)
        self.node.create_subscription(String, status_topic, self._on_status, 10)
        self.status: dict | None = None
        self.status_time = 0.0
        self.stale_timeout = stale_timeout
        self.notice = "Waiting for recorder status..."

    def _on_status(self, msg: String) -> None:
        try:
            self.status = json.loads(msg.data)
            self.status_time = time.monotonic()
        except (json.JSONDecodeError, TypeError):
            self.notice = "Ignored malformed /record_status message"

    def _connected(self) -> bool:
        return self.status is not None and time.monotonic() - self.status_time <= self.stale_timeout

    def _publish(self, action: str) -> bool:
        state = self.status.get("state") if self._connected() else None
        mode = self.status.get("mode", "record") if self._connected() else "record"
        if not _action_allowed(action, state, mode):
            if action == "exit" and state == "recording":
                self.notice = f"Stop {mode} with [r] before exiting"
            elif state is None:
                self.notice = "Recorder not detected; [q] again exits this TUI only"
            else:
                self.notice = f"'{action}' is not valid while {_state_label(state, mode)}"
            return False
        msg = String()
        msg.data = action
        self.publisher.publish(msg)
        self.notice = f"Sent: {action}"
        return True

    @staticmethod
    def _write(screen, row: int, text: str, attr: int = 0) -> None:  # noqa: ANN001
        height, width = screen.getmaxyx()
        if row >= height or width < 2:
            return
        try:
            screen.addnstr(row, 0, text, width - 1, attr)
        except curses.error:
            pass

    def _draw(self, screen) -> None:  # noqa: ANN001
        screen.erase()
        connected = self._connected()
        status = self.status if connected else {}
        state = status.get("state")
        mode = status.get("mode", "record")
        state_attr = curses.A_BOLD
        if state == "recording" and curses.has_colors():
            state_attr |= curses.color_pair(1)

        title = "CRISP trajectory replay" if mode == "replay" else "CRISP structured recording"
        self._write(screen, 0, title, curses.A_BOLD)
        self._write(screen, 2, f"Recorder : {'connected' if connected else 'not detected / stale'}")
        self._write(screen, 3, f"Dataset  : {status.get('repo_id', '-')}")
        self._write(screen, 4, f"State    : {_state_label(state, mode)}", state_attr)
        if mode == "replay":
            episode = f"{status.get('episode', 0)}"
            frames = f"{status.get('frames', 0)} / {status.get('total_frames', '-')}"
            controls = "[r] start/pause/resume   [q] exit"
        else:
            episode = f"{status.get('episode', 0)} saved / {status.get('num_episodes', '-')}"
            frames = f"{status.get('frames', 0)}"
            controls = "[r] start/stop   [s] save   [d] delete   [q] exit"
        self._write(screen, 5, f"Episode  : {episode}")
        self._write(screen, 6, f"Frames   : {frames}")
        self._write(screen, 7, f"Rate     : {status.get('fps', 0.0)} FPS")
        self._write(screen, 8, f"Last     : {status.get('last_event', '-') or '-'}")
        self._write(screen, 10, controls)
        self._write(screen, 12, self.notice)
        screen.refresh()

    def run(self, screen) -> None:  # noqa: ANN001
        """Render status and process keys until the operator exits."""
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        screen.timeout(50)
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_RED, -1)

        quit_without_recorder = False
        while rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.0)
            self._draw(screen)
            key = screen.getch()
            if key == curses.KEY_RESIZE or key < 0:
                continue
            action = _KEY_ACTIONS.get(key)
            if action is None:
                self.notice = "Unknown key"
                continue
            if action == "exit" and not self._connected():
                if quit_without_recorder:
                    return
                quit_without_recorder = True
                self.notice = "Recorder not detected; press [q] again to close this TUI"
                continue
            quit_without_recorder = False
            if self._publish(action) and action == "exit":
                # Give the reliable publisher a moment to hand the command to DDS.
                for _ in range(4):
                    rclpy.spin_once(self.node, timeout_sec=0.05)
                return

    def close(self) -> None:
        """Release the TUI's ROS node and context."""
        try:
            self.node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()


def main() -> None:
    """Run the terminal recording controller."""
    p = argparse.ArgumentParser(description="Terminal UI for crisp_gym recording")
    p.add_argument("--status-topic", default="/record_status")
    p.add_argument("--transition-topic", default="/record_transition")
    p.add_argument("--stale-timeout", type=float, default=2.0)
    args = p.parse_args()

    tui = RecordingTUI(args.status_topic, args.transition_topic, args.stale_timeout)
    try:
        curses.wrapper(tui.run)
    except KeyboardInterrupt:
        pass
    finally:
        tui.close()


if __name__ == "__main__":
    main()
