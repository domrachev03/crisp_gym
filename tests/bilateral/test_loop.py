"""Dry-run bilateral loop: position coupling + force reflection through delayed channels.

These tests pin the phenomenon TDPA must fix (step 2): naive force reflection is
passive (stable) with no channel delay, but injects energy and DIVERGES once the
round-trip delay is non-trivial. The MSD follower presses into a virtual wall.
"""

import numpy as np

from crisp_gym.sim import DelayedChannel, MSDRobot
from crisp_gym.bilateral.loop import run_bilateral


def _wall(x_wall: float, k_wall: float, dof: int = 1):
    def env(pos):
        pen = pos - x_wall
        f = np.zeros(dof)
        mask = pen > 0
        f[mask] = -k_wall * pen[mask]
        return f

    return env


def _make_pair(delay_steps: int):
    leader = MSDRobot(dof=1, mass=1.0, stiffness=0.0, damping=2.0)
    follower = MSDRobot(
        dof=1, mass=1.0, stiffness=300.0, damping=8.0, environment=_wall(0.05, 800.0)
    )
    ch_pos = DelayedChannel(delay_steps=delay_steps, dof=1)
    ch_force = DelayedChannel(delay_steps=delay_steps, dof=1)
    return leader, follower, ch_pos, ch_force


def test_no_delay_is_stable():
    leader, follower, ch_pos, ch_force = _make_pair(delay_steps=0)
    log = run_bilateral(
        leader, follower, ch_pos, ch_force,
        human_force=np.array([5.0]), n_steps=4000, dt=1e-3,
    )
    # leader settles against the reflected wall force near the wall, stays bounded
    assert np.max(np.abs(log["leader_pos"])) < 0.25
    assert np.isfinite(log["leader_pos"]).all()


def test_naive_reflection_diverges_under_delay():
    leader, follower, ch_pos, ch_force = _make_pair(delay_steps=30)  # 30 ms each way
    log = run_bilateral(
        leader, follower, ch_pos, ch_force,
        human_force=np.array([5.0]), n_steps=4000, dt=1e-3,
    )
    # delayed naive reflection pumps energy -> runaway oscillation far past the wall
    assert np.max(np.abs(log["leader_pos"])) > 1.0
