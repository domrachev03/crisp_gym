"""Optional live rerun streaming for the recorder (lazy, non-fatal).

When the recorder is launched with ``--rerun``, a :class:`RerunStreamer` is created
with ``enabled=True``. It lazy-imports ``rerun`` (an optional lerobot-feature dep) and
opens one of three sinks depending on ``mode``:

    spawn  rr.init(...); rr.spawn()        native viewer window on the PC's display
                                           (needs DISPLAY set; default now there is one)
    web    rr.init(...); rr.serve_web(...)  headless web viewer for the dashboard iframe:
                                            http://<host>:<web_port>?url=ws://<host>:<ws_port>
    save   rr.init(...); rr.save(path)      writes an .rrd file for offline viewing

Each recorded frame is logged: camera image(s) always, and -- unless ``images_only`` --
the per-arm force magnitude and EE position too.

Design rules (so it never disturbs recording):
    * disabled  -> a total no-op; ``rerun`` is never imported.
    * import/serve failure -> ``active`` stays ``False``, logging is a no-op (warn once).
    * a logging exception is swallowed (warn once); a rerun hiccup must never crash the
      record loop.

The per-frame ``obs`` is exactly what ``structured_record._assemble_obs`` produces, so
the streamer reads ``observation.images.<name>`` (HxWx3), ``observation.<arm>_ft_hrate``
(W,6) and ``observation.<arm>_pose_hrate`` (W,7) when present.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_ARMS = ("follower", "leader")
RERUN_MODES = ("spawn", "web", "save")


class RerunStreamer:
    """Stream recorded frames to rerun (native window / web viewer / .rrd), optional & non-fatal."""

    def __init__(
        self,
        enabled: bool = False,
        mode: str = "spawn",
        web_port: int = 9090,
        ws_port: int = 9877,
        images_only: bool = True,
        save_path: str | None = None,
        app_id: str = "crisp_record",
    ) -> None:
        """Open the requested rerun sink (lazy, non-fatal).

        Args:
            enabled: Master switch. When ``False`` nothing is imported and every method
                is a no-op.
            mode: Sink to open -- ``"spawn"`` (native window, needs DISPLAY), ``"web"``
                (headless web viewer for the dashboard iframe) or ``"save"`` (write .rrd).
            web_port: Port for the rerun web-viewer HTML server (``mode="web"``).
            ws_port: Port for the rerun websocket data server (``mode="web"``).
            images_only: When ``True`` (default) only camera images are logged; F/T and
                pose streams are skipped (lean native image preview).
            save_path: Output .rrd path (``mode="save"``); defaults to ``crisp_record.rrd``.
            app_id: rerun application id (recording stream name).
        """
        self.enabled = bool(enabled)
        self.mode = mode
        self.web_port = web_port
        self.ws_port = ws_port
        self.images_only = bool(images_only)
        self.save_path = save_path or "crisp_record.rrd"
        self.app_id = app_id
        self._rr = None
        self._warned = False
        self._frame = 0

        if not self.enabled:
            return
        if mode not in RERUN_MODES:
            logger.warning("unknown rerun mode %r (expected one of %s); disabling.",
                           mode, RERUN_MODES)
            return
        try:
            import rerun as rr

            if rr is None:  # monkeypatched-absent in tests / shadowed import
                raise ImportError("rerun is unavailable")
            rr.init(app_id, spawn=False)
            if mode == "spawn":
                rr.spawn()
                logger.info("rerun native viewer spawned (needs DISPLAY on this PC).")
            elif mode == "web":
                rr.serve_web(open_browser=False, web_port=web_port, ws_port=ws_port)
                logger.info("rerun web viewer live: http://<host>:%d?url=ws://<host>:%d",
                            web_port, ws_port)
            else:  # save
                rr.save(self.save_path)
                logger.info("rerun logging to %s (offline .rrd).", self.save_path)
            self._rr = rr
        except Exception as exc:  # noqa: BLE001 -- optional dep / serve failure is non-fatal
            logger.warning(
                "rerun streaming requested (mode=%s) but unavailable (%s: %s); continuing without it.",
                mode, type(exc).__name__, exc,
            )
            self._rr = None

    @property
    def active(self) -> bool:
        """True iff rerun was imported and a sink is open."""
        return self._rr is not None

    def log_frame(self, obs: dict | None) -> None:
        """Log one recorded frame to rerun (images always; F/T+pose unless images_only).

        Never raises: a logging error is swallowed (warned once) so a rerun hiccup
        cannot crash the record loop.
        """
        if self._rr is None or obs is None:
            return
        rr = self._rr
        try:
            rr.set_time_sequence("frame", self._frame)
            self._frame += 1

            for key, value in obs.items():
                if key.startswith("observation.images."):
                    name = key[len("observation.images.") :]
                    rr.log(f"cameras/{name}", rr.Image(np.asarray(value)))

            if self.images_only:
                return

            for arm in _ARMS:
                ft = obs.get(f"observation.{arm}_ft_hrate")
                if ft is not None:
                    ft = np.asarray(ft, dtype=np.float32)
                    latest = ft[-1] if ft.ndim == 2 and len(ft) else ft.reshape(-1)
                    if latest.shape[0] >= 3:
                        mag = float(np.linalg.norm(latest[:3]))
                        rr.log(f"{arm}/force_mag", rr.Scalar(mag))
                pose = obs.get(f"observation.{arm}_pose_hrate")
                if pose is not None:
                    pose = np.asarray(pose, dtype=np.float32)
                    latest = pose[-1] if pose.ndim == 2 and len(pose) else pose.reshape(-1)
                    if latest.shape[0] >= 3:
                        rr.log(f"{arm}/ee", rr.Points3D(latest[:3].reshape(1, 3)))
        except Exception as exc:  # noqa: BLE001 -- never crash recording on a log error
            if not self._warned:
                logger.warning("rerun logging failed (%s: %s); muting further warnings.",
                               type(exc).__name__, exc)
                self._warned = True

    def close(self) -> None:
        """Release the rerun handle (the serve/spawn thread is a daemon; nothing to join)."""
        self._rr = None
