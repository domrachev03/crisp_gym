"""run_bilateral integrates the telemetry logger (comprehensive per-step capture)."""

import numpy as np

from crisp_gym.sim import DelayedChannel, MSDRobot
from crisp_gym.bilateral.loop import run_bilateral
from crisp_gym.bilateral.telemetry import TeleopLogger, analyze


def test_run_bilateral_populates_logger():
    leader = MSDRobot(dof=1, mass=1.0, stiffness=0.0, damping=2.0)
    follower = MSDRobot(dof=1, mass=1.0, stiffness=300.0, damping=8.0)
    ch_pos = DelayedChannel(delay_steps=5, dof=1)
    ch_force = DelayedChannel(delay_steps=5, dof=1)
    logger = TeleopLogger()
    run_bilateral(
        leader, follower, ch_pos, ch_force,
        human_force=np.array([5.0]), n_steps=200, dt=1e-3, logger=logger,
    )
    assert len(logger.records) == 200
    r0 = logger.records[0]
    for key in ("step", "t", "leader_pos", "follower_pos", "leader_vel", "follower_wrench", "reflected"):
        assert key in r0, key
    rep = analyze(logger)
    assert np.isclose(rep["loop_dt_mean"], 1e-3)
    assert "max_drift" in rep
