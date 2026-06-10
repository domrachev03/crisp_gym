"""Tests for the mass-spring-damper dry-run robot mock (pure python, no ROS)."""

import numpy as np
import pytest

from crisp_gym.sim import MSDRobot


def test_free_mass_under_constant_force_follows_kinematics():
    """k=0, b=0: a constant force gives x(t) = 0.5*(F/m)*t^2 (Newton)."""
    robot = MSDRobot(dof=1, mass=2.0, stiffness=0.0, damping=0.0)
    force = 4.0
    dt = 1e-3
    n = 1000  # 1 s
    for _ in range(n):
        robot.set_feedforward_force(np.array([force]))
        robot.step(dt)
    t = n * dt
    expected = 0.5 * (force / robot.mass) * t**2
    assert robot.position[0] == pytest.approx(expected, rel=2e-3)
    assert robot.velocity[0] == pytest.approx((force / robot.mass) * t, rel=2e-3)


def test_free_mass_no_force_stays_at_rest():
    robot = MSDRobot(dof=3, mass=1.0, stiffness=0.0, damping=0.0)
    for _ in range(100):
        robot.step(1e-3)
    assert np.allclose(robot.position, 0.0)
    assert np.allclose(robot.velocity, 0.0)
