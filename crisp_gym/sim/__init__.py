"""Pure-python dry-run simulation substrate for testing teleop control laws.

No ROS/rclpy dependency: importable anywhere for TDD of the position-force and
TDPA control laws against mass-spring-damper plants with injectable channel delay.
"""

from crisp_gym.sim.channel import DelayedChannel
from crisp_gym.sim.msd_robot import MSDRobot

__all__ = ["MSDRobot", "DelayedChannel"]
