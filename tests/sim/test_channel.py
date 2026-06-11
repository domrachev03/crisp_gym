"""Tests for the delayed communication channel used to inject artificial latency."""

import numpy as np

from crisp_gym.sim import DelayedChannel


def test_zero_delay_returns_last_sent():
    ch = DelayedChannel(delay_steps=0, dof=2)
    ch.send(np.array([1.0, 2.0]))
    assert np.allclose(ch.receive(), [1.0, 2.0])


def test_n_step_delay_returns_value_from_n_steps_ago():
    ch = DelayedChannel(delay_steps=3, dof=1)
    seen = []
    for i in range(6):
        ch.send(np.array([float(i)]))
        seen.append(ch.receive()[0])
    # step i receives value sent 3 steps earlier; first 3 are the fill (0.0)
    assert seen == [0.0, 0.0, 0.0, 0.0, 1.0, 2.0]


def test_fill_value_before_buffer_filled():
    ch = DelayedChannel(delay_steps=2, dof=1, fill=np.array([9.0]))
    ch.send(np.array([5.0]))
    assert np.allclose(ch.receive(), [9.0])
