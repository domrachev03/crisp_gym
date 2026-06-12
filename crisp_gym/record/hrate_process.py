"""Out-of-process high-rate capture: a dedicated poller process per recorder.

``HrateManager`` runs its ROS spin thread in the recorder process, so it shares
one GIL with the 60 fps control + lerobot encode loop and the high-rate subs
(F/T ~2.1 kHz, joints ~1 kHz) starve to ~470 Hz. ``HrateProcessManager`` is a
drop-in replacement (same ``snapshot(key, t_ref)`` / ``close()``) that moves the
pollers into their own process with its own GIL, writing into shared-memory ring
buffers the recorder reads at frame time. Optional CPU-core affinity pins that
process off the recorder's cores so the kHz callbacks never contend.

The default child (`_hrate_child`) runs rclpy pollers; it is spawned with the
``spawn`` start method (fork + rclpy is unsafe). Tests inject a fake child target
over a ``fork`` context to exercise the IPC without ROS.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os

import numpy as np

from crisp_gym.record.hrate_shared import SharedHrateRing

logger = logging.getLogger(__name__)


def _set_affinity(core_affinity) -> None:  # noqa: ANN001
    if not core_affinity:
        return
    try:
        os.sched_setaffinity(0, set(core_affinity))
        logger.info(f"hrate process pinned to cores {sorted(set(core_affinity))}")
    except (AttributeError, OSError) as e:
        logger.warning(f"hrate core affinity {core_affinity} failed: {e}")


def _hrate_child(ring_metas: dict, specs: list, stop_evt, core_affinity) -> None:  # noqa: ANN001
    """Child entry: own rclpy context + pollers writing the shared rings."""
    _set_affinity(core_affinity)
    import rclpy
    from rclpy.executors import MultiThreadedExecutor

    from crisp_gym.record.structured_record import HRATE_SIGNALS, _HratePoller

    if not rclpy.ok():
        rclpy.init()
    node = rclpy.create_node("hrate_pollers")
    rings = {key: SharedHrateRing.attach(meta) for key, meta in ring_metas.items()}
    pollers = []
    for key, topic, sig, _window in specs:
        s = HRATE_SIGNALS[sig]
        pollers.append(_HratePoller(node, topic, s["msg"], s["extract"], rings[key]))

    # One thread per poller so the high-rate subs never queue behind each other.
    ex = MultiThreadedExecutor(num_threads=max(2, len(pollers)))
    ex.add_node(node)
    logger.info(f"hrate child: {len(pollers)} pollers, {ex.get_nodes() and ''}spinning")
    try:
        while rclpy.ok() and not stop_evt.is_set():
            ex.spin_once(timeout_sec=0.1)
    finally:
        for r in rings.values():
            r.close()
        try:
            node.destroy_node()
            rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass


class HrateProcessManager:
    """Drop-in for ``HrateManager`` that captures hrate in a separate process."""

    def __init__(self, specs: list, *, signals: dict | None = None,
                 core_affinity=None, oversize: int = 8,  # noqa: ANN001
                 mp_context: str = "spawn", child_target=None) -> None:  # noqa: ANN001
        """Allocate shared rings and spawn the poller process.

        Args:
            specs: list of (key, topic, sig, window) (from ``build_hrate_specs``).
            signals: sig -> {"dof": int} registry; defaults to ``HRATE_SIGNALS``.
            core_affinity: CPU cores to pin the poller process to (None = inherit).
            oversize: ring capacity multiple of the window.
            mp_context: multiprocessing start method ("spawn" on hardware).
            child_target: child entry fn; defaults to the rclpy poller loop.
        """
        if signals is None:
            from crisp_gym.record.structured_record import HRATE_SIGNALS
            signals = HRATE_SIGNALS
        child_target = child_target or _hrate_child

        self._rings: dict[str, SharedHrateRing] = {}
        ring_metas: dict[str, dict] = {}
        for key, _topic, sig, window in specs:
            dof = int(signals[sig]["dof"])
            ring = SharedHrateRing.create(window, dof, oversize=oversize)
            self._rings[key] = ring
            ring_metas[key] = ring.meta

        ctx = mp.get_context(mp_context)
        self._stop = ctx.Event()
        self._proc = ctx.Process(
            target=child_target, name="hrate_proc", daemon=True,
            args=(ring_metas, specs, self._stop, core_affinity))
        self._proc.start()

    def snapshot(self, key: str, t_ref: float) -> tuple[np.ndarray, np.ndarray]:
        """Latest window for ``key`` as float32 (values, times-t_ref)."""
        values, times = self._rings[key].snapshot(t_ref)
        return values.astype(np.float32), times.astype(np.float32)

    def close(self) -> None:
        """Stop the poller process and release shared memory."""
        try:
            self._stop.set()
        except Exception:  # noqa: BLE001
            pass
        if self._proc.is_alive():
            self._proc.join(timeout=3.0)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=1.0)
        for r in self._rings.values():
            r.close()
