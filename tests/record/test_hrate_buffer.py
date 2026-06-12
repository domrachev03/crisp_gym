"""Client-side high-rate ring buffer for sub-step F/T logging (LeRobot hrate windows).

A background poller appends native-rate samples; each recorded step snapshots the
latest N as (values (N,K), times (N,) relative to the frame time), zero-padded at
the front during startup. Mirrors lerobot-panda's hrate window schema so the
existing dataset path + reconstruction utility work unchanged.
"""

import numpy as np

from crisp_gym.record.hrate_buffer import HrateRingBuffer


def test_snapshot_returns_latest_n_with_relative_times():
    buf = HrateRingBuffer(window=3, dof=2)
    for i in range(5):
        buf.append(t=1000.0 + i * 0.002, value=np.array([float(i), -float(i)]))
    values, times = buf.snapshot(t_ref=1000.0 + 4 * 0.002)
    assert values.shape == (3, 2)
    # latest 3 samples: i=2,3,4
    assert np.allclose(values[:, 0], [2.0, 3.0, 4.0])
    # times relative to the last sample (t_ref): -0.004, -0.002, 0.0
    assert np.allclose(times, [-0.004, -0.002, 0.0], atol=1e-9)


def test_zero_pads_front_before_filled():
    buf = HrateRingBuffer(window=4, dof=1)
    buf.append(t=10.0, value=np.array([7.0]))
    values, times = buf.snapshot(t_ref=10.0)
    assert values.shape == (4, 1)
    # 3 zero-pads then the one real sample
    assert np.allclose(values[:, 0], [0.0, 0.0, 0.0, 7.0])
    assert np.allclose(times[:3], 0.0)


def test_count_tracks_total_appends():
    buf = HrateRingBuffer(window=2, dof=1)
    assert buf.count == 0
    for i in range(10):  # past capacity: count keeps growing
        buf.append(t=float(i), value=np.array([float(i)]))
    assert buf.count == 10


def test_window_zero_returns_empty():
    buf = HrateRingBuffer(window=0, dof=6)
    buf.append(t=1.0, value=np.zeros(6))
    values, times = buf.snapshot(t_ref=1.0)
    assert values.shape == (0, 6)
    assert times.shape == (0,)
