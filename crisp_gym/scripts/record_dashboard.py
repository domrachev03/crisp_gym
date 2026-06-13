"""Serve the dataset-recording web dashboard.

Run alongside ``crisp-record-structured --recording-manager-type ros``: the recorder
publishes ``/record_status``; this serves a page showing state / episode / loop-rate /
per-arm force+pose, with start-stop / save / delete / exit buttons (which publish to
``/record_transition`` -- same control path as the ``rec-*`` pixi tasks).

    pixi run -e jazzy-lerobot crisp-record-dashboard     # -> http://<host>:8000
"""

import argparse
import ctypes
import logging
import signal

import uvicorn

from crisp_gym.record.dashboard import create_app
from crisp_gym.record.dashboard_monitor import DashboardMonitor


def _die_with_parent() -> None:
    """Linux: receive SIGTERM when the parent dies (e.g. the ``pixi run`` wrapper),

    so the dashboard never orphans and keeps holding its port. uvicorn handles the
    SIGTERM and shuts down gracefully, releasing the socket + rclpy.
    """
    try:
        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM, 0, 0, 0)  # PR_SET_PDEATHSIG
    except Exception:  # noqa: BLE001 -- best-effort; non-Linux just relies on signals
        pass


def main() -> None:
    _die_with_parent()
    p = argparse.ArgumentParser(description="crisp_gym recording dashboard")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--arms", type=str, nargs="+", default=["right", "left"])
    p.add_argument("--rerun-url", type=str, default=None,
                   help="Explicit rerun web-viewer URL for the embedded iframe. "
                        "Unset -> built from the page host as "
                        "http://<host>:<web-port>?url=ws://<host>:<ws-port> (works over ssh tunnel).")
    p.add_argument("--rerun-web-port", type=int, default=9090,
                   help="Rerun web-viewer HTML port (used when --rerun-url is unset).")
    p.add_argument("--rerun-ws-port", type=int, default=9877,
                   help="Rerun websocket data port (used when --rerun-url is unset).")
    p.add_argument("--gripper-config", type=str, default=None,
                   help="crisp_py gripper config for the Open/Close buttons (e.g. gripper_right_v2 "
                        "or gripper_franka). Unset -> gripper buttons report 'unavailable'.")
    p.add_argument("--gripper-namespace", type=str, default="right",
                   help="ROS namespace of the gripper to command (default: right).")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO)
    monitor = DashboardMonitor(arms=tuple(args.arms),
                               gripper_config=args.gripper_config,
                               gripper_namespace=args.gripper_namespace)
    monitor.start()
    app = create_app(monitor, rerun_url=args.rerun_url,
                     rerun_web_port=args.rerun_web_port, rerun_ws_port=args.rerun_ws_port)
    logging.getLogger(__name__).info(f"Dashboard on http://{args.host}:{args.port}")
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        monitor.close()


if __name__ == "__main__":
    main()
