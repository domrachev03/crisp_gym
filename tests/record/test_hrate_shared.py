"""Shared-memory high-rate ring (cross-process hrate capture).

Same window semantics as ``HrateRingBuffer`` (right-aligned latest-N, front
zero-padded, times relative to the frame ``t_ref``) but backed by
``multiprocessing.shared_memory`` so a separate poller process (own GIL) can
write while the recorder process reads. Mirrors lerobot-panda's hrate schema so
the dataset path is unchanged.
"""

import numpy as np

from crisp_gym.record.hrate_shared import SharedHrateRing


def test_snapshot_returns_latest_n_with_relative_times():
    ring = SharedHrateRing.create(window=3, dof=2)
    try:
        for i in range(5):
            ring.append(t=1000.0 + i * 0.002, value=np.array([float(i), -float(i)]))
        values, times = ring.snapshot(t_ref=1000.0 + 4 * 0.002)
        assert values.shape == (3, 2)
        assert np.allclose(values[:, 0], [2.0, 3.0, 4.0])
        assert np.allclose(values[:, 1], [-2.0, -3.0, -4.0])
        assert np.allclose(times, [-0.004, -0.002, 0.0], atol=1e-9)
    finally:
        ring.close()


def test_zero_pads_front_before_filled():
    ring = SharedHrateRing.create(window=4, dof=1)
    try:
        ring.append(t=10.0, value=np.array([7.0]))
        values, times = ring.snapshot(t_ref=10.0)
        assert values.shape == (4, 1)
        assert np.allclose(values[:, 0], [0.0, 0.0, 0.0, 7.0])
        assert np.allclose(times[:3], 0.0)
        assert np.allclose(times[3], 0.0)
    finally:
        ring.close()


def test_window_zero_returns_empty():
    ring = SharedHrateRing.create(window=0, dof=6)
    try:
        ring.append(t=1.0, value=np.zeros(6))
        values, times = ring.snapshot(t_ref=1.0)
        assert values.shape == (0, 6)
        assert times.shape == (0,)
    finally:
        ring.close()


def test_wraparound_keeps_latest_window():
    # capacity = window * oversize; append well past capacity, latest-N still correct.
    ring = SharedHrateRing.create(window=3, dof=1, oversize=4)
    try:
        n = ring.capacity * 3 + 2  # lap the buffer several times
        for i in range(n):
            ring.append(t=float(i), value=np.array([float(i)]))
        values, times = ring.snapshot(t_ref=float(n - 1))
        last = n - 1
        assert np.allclose(values[:, 0], [last - 2, last - 1, last])
        assert np.allclose(times, [-2.0, -1.0, 0.0])
    finally:
        ring.close()


def test_count_tracks_total_appends():
    ring = SharedHrateRing.create(window=2, dof=1, oversize=4)
    try:
        assert ring.count == 0
        for i in range(10):  # past capacity (2*4=8): count keeps growing
            ring.append(t=float(i), value=np.array([float(i)]))
        assert ring.count == 10
        assert SharedHrateRing.attach(ring.meta).count == 10
    finally:
        ring.close()


def test_meta_roundtrip_attach_sees_writes():
    # A second handle attached by meta observes the creator's writes (same block).
    ring = SharedHrateRing.create(window=2, dof=1)
    try:
        ring.append(t=5.0, value=np.array([9.0]))
        other = SharedHrateRing.attach(ring.meta)
        try:
            values, times = other.snapshot(t_ref=5.0)
            assert np.allclose(values[:, 0], [0.0, 9.0])
            assert np.allclose(times, [0.0, 0.0])
        finally:
            other.close()
    finally:
        ring.close()
