"""Joint-space P-F teleop (1:1 joint mapping) + TDPA, in the dof-generic dry-run.

Joint space is per-joint scalars (no orientation), so the same run_bilateral loop
applies directly with dof = n_joints: leader joint angles -> follower joint targets
(absolute = identical-config 1:1 map), follower joint torque reflected to the leader.
Confirms naive joint-space reflection diverges under delay and TDPA passivates it.
"""

import numpy as np

from crisp_gym.sim import DelayedChannel, MSDRobot
from crisp_gym.bilateral.loop import run_bilateral
from crisp_gym.bilateral.tdpa import MasterOnlyPOPC

N = 7  # Panda joints


def _joint_stops(limit: float, k_stop: float, dof: int = N):
    def env(q):
        over = q - limit
        tau = np.zeros(dof)
        mask = over > 0
        tau[mask] = -k_stop * over[mask]
        return tau

    return env


def _make_joint_pair(delay_steps: int):
    leader = MSDRobot(dof=N, mass=1.0, stiffness=0.0, damping=2.0)
    follower = MSDRobot(
        dof=N, mass=1.0, stiffness=300.0, damping=8.0, environment=_joint_stops(0.05, 800.0)
    )
    return leader, follower, DelayedChannel(delay_steps, N), DelayedChannel(delay_steps, N)


def test_joint_space_naive_diverges_under_delay():
    leader, follower, cp, cf = _make_joint_pair(delay_steps=30)
    log = run_bilateral(leader, follower, cp, cf,
                        human_force=np.full(N, 5.0), n_steps=4000, dt=1e-3, coupling="absolute")
    assert np.max(np.abs(log["leader_pos"])) > 1.0


def test_joint_space_tdpa_keeps_bounded():
    leader, follower, cp, cf = _make_joint_pair(delay_steps=30)
    tdpa = MasterOnlyPOPC(dof=N, contact_threshold_n=1.0)
    log = run_bilateral(leader, follower, cp, cf,
                        human_force=np.full(N, 5.0), n_steps=4000, dt=1e-3,
                        coupling="absolute", passivity=tdpa)
    assert np.isfinite(log["leader_pos"]).all()
    assert np.max(np.abs(log["leader_pos"])) < 0.5
