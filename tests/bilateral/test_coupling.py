"""Absolute vs relative position coupling option.

absolute: follower target = leader absolute position (offset erased; no drift).
relative: follower target accumulates leader position increments (preserves an
initial offset; the clutchable, drift-prone mode the current HW script uses).
"""

import numpy as np

from crisp_gym.sim import DelayedChannel, MSDRobot
from crisp_gym.bilateral.loop import run_bilateral


def _setup():
    # "human" holds the leader at 0.3 via its own spring; follower anchored at 1.0.
    leader = MSDRobot(dof=1, mass=1.0, stiffness=50.0, damping=20.0)
    leader.set_target_position(np.array([0.3]))
    follower = MSDRobot(dof=1, mass=1.0, stiffness=300.0, damping=20.0, x0=np.array([1.0]))
    return leader, follower, DelayedChannel(0, 1), DelayedChannel(0, 1)


def test_absolute_coupling_erases_offset():
    leader, follower, cp, cf = _setup()
    log = run_bilateral(leader, follower, cp, cf, human_force=np.array([0.0]),
                        n_steps=4000, dt=1e-3, coupling="absolute")
    assert abs(log["follower_pos"][-1, 0] - log["leader_pos"][-1, 0]) < 0.05


def test_relative_coupling_preserves_offset_and_tracks_increments():
    leader, follower, cp, cf = _setup()
    log = run_bilateral(leader, follower, cp, cf, human_force=np.array([0.0]),
                        n_steps=4000, dt=1e-3, coupling="relative")
    # follower kept its ~1.0 initial offset from the leader
    assert log["follower_pos"][-1, 0] - log["leader_pos"][-1, 0] > 0.85
    # but moved by the same displacement the leader did (0 -> ~0.3)
    leader_disp = log["leader_pos"][-1, 0]
    follower_disp = log["follower_pos"][-1, 0] - 1.0
    assert abs(follower_disp - leader_disp) < 0.05
