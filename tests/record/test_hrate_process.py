"""Cross-process hrate manager: a separate poller process fills shared rings.

Exercises the orchestration (allocate shared rings, spawn the poller, read
windows, tear down) with a fake non-ROS feeder injected as the child target, so
the IPC path is verified under ``uv`` without rclpy. The real child runs rclpy
pollers on hardware.
"""

import time

import numpy as np

from crisp_gym.record.hrate_shared import SharedHrateRing


def _fake_feeder(ring_metas, specs, stop_evt, core_affinity):
    """Child target: attach rings, append a ramp of synthetic samples, then idle."""
    rings = {k: SharedHrateRing.attach(m) for k, m in ring_metas.items()}
    for i in range(500):
        if stop_evt.is_set():
            break
        for ring in rings.values():
            ring.append(t=float(i), value=np.full(ring.dof, float(i)))
    stop_evt.wait()  # hold until the parent closes us
    for r in rings.values():
        r.close()


def test_manager_reads_cross_process_writes():
    from crisp_gym.record.hrate_process import HrateProcessManager

    specs = [("follower_ft", "/right/netft", "ft", 3)]
    mgr = HrateProcessManager(
        specs, signals={"ft": {"dof": 6}},
        mp_context="fork", child_target=_fake_feeder, oversize=8)
    try:
        deadline = time.time() + 5.0
        vals = times = None
        while time.time() < deadline:
            vals, times = mgr.snapshot("follower_ft", t_ref=499.0)
            if vals[-1, 0] == 499.0:
                break
            time.sleep(0.02)
        assert vals is not None
        assert vals.dtype == np.float32 and times.dtype == np.float32
        assert vals.shape == (3, 6)
        assert np.allclose(vals[:, 0], [497.0, 498.0, 499.0])
        assert np.allclose(times, [-2.0, -1.0, 0.0])
    finally:
        mgr.close()


def test_snapshot_before_data_is_zero_padded():
    from crisp_gym.record.hrate_process import HrateProcessManager

    # No child started writing yet: a fresh ring returns an all-zero window.
    specs = [("follower_ft", "/right/netft", "ft", 4)]

    def _idle(ring_metas, specs, stop_evt, core_affinity):
        stop_evt.wait()

    mgr = HrateProcessManager(
        specs, signals={"ft": {"dof": 6}},
        mp_context="fork", child_target=_idle, oversize=8)
    try:
        vals, times = mgr.snapshot("follower_ft", t_ref=0.0)
        assert vals.shape == (4, 6)
        assert np.allclose(vals, 0.0)
        assert np.allclose(times, 0.0)
    finally:
        mgr.close()
