"""TDPA (MasterOnlyPOPC) passivates the delayed position-force loop.

The decisive test: the exact delay-30 scenario that DIVERGES with naive reflection
(test_loop.py) must stay BOUNDED once the time-domain passivity controller is in
the loop. Ported from franka_server_standalone (docs/design/tdpa_reference.md).
"""

import numpy as np

from crisp_gym.sim import DelayedChannel, MSDRobot
from crisp_gym.bilateral.loop import run_bilateral
from crisp_gym.bilateral.tdpa import MasterOnlyPOPC


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
    return leader, follower, DelayedChannel(delay_steps, 1), DelayedChannel(delay_steps, 1)


def test_tdpa_keeps_delayed_loop_bounded():
    leader, follower, ch_pos, ch_force = _make_pair(delay_steps=30)
    tdpa = MasterOnlyPOPC(dof=1, contact_threshold_n=1.0)
    log = run_bilateral(
        leader, follower, ch_pos, ch_force,
        human_force=np.array([5.0]), n_steps=4000, dt=1e-3, passivity=tdpa,
    )
    assert np.isfinite(log["leader_pos"]).all()
    # naive diverges to >1.0; TDPA must keep it bounded near the workspace
    assert np.max(np.abs(log["leader_pos"])) < 0.5


def test_modify_is_passthrough_when_port_dissipative():
    # Leader moving +, reflected force also + => power into port > 0 (dissipative/input):
    # no energy generated, so no damping injected -> reflected unchanged.
    tdpa = MasterOnlyPOPC(dof=1, contact_threshold_n=0.0)
    reflected = np.array([2.0])
    vel = np.array([1.0])
    out = tdpa.modify(reflected, vel, human_force=np.array([2.0]), dt=1e-3)
    assert np.allclose(out, reflected)


def test_modify_shape_and_finite():
    tdpa = MasterOnlyPOPC(dof=3)
    out = tdpa.modify(np.array([1.0, -2.0, 0.5]), np.array([0.1, -0.3, 0.0]),
                      human_force=np.array([5.0, 0.0, 0.0]), dt=1e-3)
    assert out.shape == (3,)
    assert np.isfinite(out).all()
