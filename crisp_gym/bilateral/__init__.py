"""Bilateral teleoperation control law (no ROS dependency).

Pure-python position-force coupling + TDPA passivity, operating on the
``RobotInterface`` abstraction so it is testable against dry-run MSD plants and
deployable on real crisp_py robots via a runtime adapter.
"""

from crisp_gym.bilateral.interface import RobotInterface

__all__ = ["RobotInterface"]
