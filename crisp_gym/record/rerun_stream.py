"""Optional live rerun streaming for the recorder (lazy, async, non-fatal).

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

**Off the control loop, low latency.** ``log_frame`` only hands the obs to a background
worker thread and returns immediately, so image serialisation never blocks the 60 fps
record loop (logging inline caused the loop to lag). The queue is depth-1 **latest-wins**:
while the worker logs a frame, only the newest frame produced meanwhile survives (older
ones are dropped), so the viewer shows ~real-time instead of falling a queue-length
behind. Following the lerobot-panda recipe, images are logged ``static`` (always-latest,
no timeline accumulation) and the viewer is given a small ``memory_limit`` so it drops
old data; frames are JPEG-compressed and ``max_fps``-throttled so the raw stream never
floods the sink faster than it drains. The recorded *dataset* is produced by the writer
process and is unaffected by any drop.

Design rules (so it never disturbs recording):
    * disabled  -> a total no-op; ``rerun`` is never imported, no worker thread.
    * import/serve failure -> ``active`` stays ``False``, logging is a no-op (warn once).
    * a logging exception is swallowed (warn once); a rerun hiccup must never crash the
      record loop.

The per-frame ``obs`` is exactly what ``structured_record._assemble_obs`` produces, so
the streamer reads ``observation.images.<name>`` (HxWx3), ``observation.<arm>_ft_hrate``
(W,6) and ``observation.<arm>_pose_hrate`` (W,7) when present.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)

_ARMS = ("follower", "leader")
RERUN_MODES = ("spawn", "web", "save")
_SENTINEL = object()  # queue poison pill to stop the worker


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
        buffer: int = 1,
        max_fps: float = 30.0,
        jpeg_quality: int = 50,
        memory_limit: str = "10%",
        max_dim: int = 320,
        sync: bool = False,
    ) -> None:
        """Open the requested rerun sink (lazy, non-fatal) and start the logging worker.

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
            buffer: Worker queue depth; default 1 = latest-wins (lowest latency). Larger
                values buffer more frames before dropping (rarely wanted for a preview).
            max_fps: Cap on frames accepted per second (0 = unlimited); throttles the sink.
            jpeg_quality: JPEG quality for image frames (lower = less bandwidth/latency).
            memory_limit: rerun viewer store cap (low = drops old data, bounds latency).
            max_dim: downscale preview frames so max(H,W) <= this (0 = full res). Smaller
                = far less encode/transport/render work per frame -> lower latency.
            sync: log inline on the record loop instead of the worker thread (A/B knob).
                Now that ``_log`` is cheap (downscaled + JPEG), inline logging adds little
                to the loop but removes the worker's scheduling delay.
        """
        self.enabled = bool(enabled)
        self.sync = bool(sync)
        self.mode = mode
        self.web_port = web_port
        self.ws_port = ws_port
        self.images_only = bool(images_only)
        self.save_path = save_path or "crisp_record.rrd"
        self.app_id = app_id
        self.jpeg_quality = int(jpeg_quality)
        self.memory_limit = memory_limit
        self._max_dim = int(max_dim)
        # Flush the rerun micro-batcher per frame (no waiting to accumulate bytes), so a
        # frame hits the sink immediately instead of sitting in the batch buffer.
        if self.enabled:
            os.environ.setdefault("RERUN_FLUSH_NUM_BYTES", "1024")
        self._min_dt = (1.0 / max_fps) if max_fps and max_fps > 0 else 0.0
        self._last_accept = 0.0
        self._rr = None
        self._warned = False
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, int(buffer)))
        self._worker: threading.Thread | None = None

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
                rr.spawn(memory_limit=self.memory_limit)
                logger.info("rerun native viewer spawned (needs DISPLAY on this PC).")
            elif mode == "web":
                rr.serve_web(open_browser=False, web_port=web_port, ws_port=ws_port,
                             server_memory_limit=self.memory_limit)
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
            return

        # Async (default): logging runs on a daemon worker so the record loop never waits
        # on rr.log. sync=True logs inline instead (no worker) for A/B comparison.
        if not self.sync:
            self._worker = threading.Thread(target=self._run, name="rerun_log", daemon=True)
            self._worker.start()

    @property
    def active(self) -> bool:
        """True iff rerun was imported and a sink is open."""
        return self._rr is not None

    def log_frame(self, obs: dict | None) -> None:
        """Hand one recorded frame to the worker (non-blocking; drops oldest if backed up).

        Returns immediately -- the actual rr.log happens on the worker thread, so a slow
        image serialise cannot stall the record loop. Never raises.
        """
        if self._rr is None or obs is None:
            return
        if self.sync:  # inline (no worker): log right here on the record loop
            try:
                self._log(obs)
            except Exception as exc:  # noqa: BLE001 -- never crash recording on a log error
                if not self._warned:
                    logger.warning("rerun logging failed (%s: %s); muting further warnings.",
                                   type(exc).__name__, exc)
                    self._warned = True
            return
        if self._min_dt:  # throttle: drop frames arriving faster than max_fps
            now = time.time()
            if now - self._last_accept < self._min_dt:
                return
            self._last_accept = now
        try:
            self._queue.put_nowait(obs)
        except queue.Full:
            # Drop the oldest queued frame to make room for the newest (live preview).
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(obs)
            except queue.Full:
                pass

    def _run(self) -> None:
        """Worker loop: drain the queue and log each frame to rerun."""
        while True:
            obs = self._queue.get()
            if obs is _SENTINEL:
                self._queue.task_done()
                break
            try:
                self._log(obs)
            except Exception as exc:  # noqa: BLE001 -- never crash recording on a log error
                if not self._warned:
                    logger.warning("rerun logging failed (%s: %s); muting further warnings.",
                                   type(exc).__name__, exc)
                    self._warned = True
            finally:
                self._queue.task_done()

    def _log(self, obs: dict) -> None:
        """Map one obs to rerun entities (images always; F/T+pose unless images_only).

        Images are logged ``static`` (overwrite, no per-frame timeline point) so the
        viewer always shows the latest frame with no accumulation -- the lerobot-panda
        recipe -- and JPEG-compressed so raw frames don't flood the sink faster than the
        viewer drains (which was the residual lag in spawn mode).
        """
        rr = self._rr

        for key, value in obs.items():
            if key.startswith("observation.images."):
                name = key[len("observation.images.") :]
                arr = self._downscale(np.asarray(value))
                img = rr.Image(arr)
                try:
                    img = img.compress(jpeg_quality=self.jpeg_quality)
                except Exception:  # noqa: BLE001 -- fall back to raw if compress unsupported
                    img = rr.Image(arr)
                rr.log(f"cameras/{name}", img, static=True)

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

    def _downscale(self, arr: np.ndarray) -> np.ndarray:
        """Return a smaller copy for the preview (never mutates the recorded frame)."""
        if not self._max_dim or arr.ndim < 2:
            return arr
        h, w = arr.shape[:2]
        m = max(h, w)
        if m <= self._max_dim:
            return arr
        scale = self._max_dim / m
        try:
            import cv2

            return cv2.resize(arr, (max(1, int(w * scale)), max(1, int(h * scale))),
                              interpolation=cv2.INTER_AREA)
        except Exception:  # noqa: BLE001 -- no cv2: cheap stride subsample (a fresh copy)
            step = int(np.ceil(m / self._max_dim))
            return np.ascontiguousarray(arr[::step, ::step])

    def flush(self, timeout: float = 2.0) -> None:
        """Block until queued frames are logged (best-effort, bounded by ``timeout``)."""
        if self._worker is None:
            return
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._queue.unfinished_tasks == 0:
                return
            time.sleep(0.005)

    def close(self) -> None:
        """Flush, stop the worker, and release the rerun handle."""
        if self._worker is not None:
            self.flush()
            try:
                self._queue.put_nowait(_SENTINEL)
            except queue.Full:  # make room, then poison
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(_SENTINEL)
                except queue.Full:
                    pass
            self._worker.join(timeout=2.0)
            self._worker = None
        self._rr = None
