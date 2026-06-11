"""RobotInterface protocol: the abstraction the bilateral control law operates on."""

import numpy as np

from crisp_gym.bilateral import RobotInterface
from crisp_gym.sim import MSDRobot


def test_msd_robot_satisfies_robot_interface():
    robot = MSDRobot(dof=3)
    assert isinstance(robot, RobotInterface)


def test_interface_round_trip_read_write():
    robot: RobotInterface = MSDRobot(dof=2, mass=1.0, stiffness=50.0)
    robot.set_target_position(np.array([1.0, 0.0]))
    robot.set_feedforward_force(np.array([0.0, 2.0]))
    for _ in range(200):
        robot.step(1e-3)
    # spring pulls axis 0 toward target 1.0; ff pushes axis 1
    assert robot.position[0] > 0.0
    assert robot.position[1] > 0.0
    assert robot.wrench.shape == (2,)
