"""Mass-spring-damper dry-run robot mock.

Pure-python (numpy only, no ROS/rclpy) point-mass-per-axis plant used to test the
teleoperation control laws (position coupling, force reflection, TDPA) without
hardware. Each axis obeys

    m * a = F_ff + k * (x_target - x) - b * v + F_env

integrated with semi-implicit (symplectic) Euler. ``F_env`` is the contact force
from an optional environment (e.g. a virtual wall), and is what a force/torque
sensor on the robot would read back (the ``wrench`` property).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray


class MSDRobot:
    """A mass-spring-damper plant standing in for a real impedance-controlled arm."""

    def __init__(
        self,
        dof: int = 3,
        mass: float = 1.0,
        stiffness: float = 0.0,
        damping: float = 0.0,
        x0: NDArray | None = None,
        environment: Callable[[NDArray], NDArray] | None = None,
    ):
        """Create the plant.

        Args:
            dof: Number of independent axes.
            mass: Point mass (kg), shared across axes.
            stiffness: Impedance spring constant pulling toward the target position.
            damping: Velocity damping coefficient.
            x0: Initial position (defaults to zeros).
            environment: Optional callable mapping position -> external contact
                force (e.g. a virtual wall). Returns the force the environment
                applies to the robot; also reported via ``wrench``.
        """
        self.dof = dof
        self.mass = float(mass)
        self.stiffness = float(stiffness)
        self.damping = float(damping)
        self.environment = environment

        self._pos = np.zeros(dof) if x0 is None else np.array(x0, dtype=float)
        self._vel = np.zeros(dof)
        self._target = self._pos.copy()
        self._ff = np.zeros(dof)
        self._contact = np.zeros(dof)

    @property
    def position(self) -> NDArray:
        return self._pos.copy()

    @property
    def velocity(self) -> NDArray:
        return self._vel.copy()

    @property
    def wrench(self) -> NDArray:
        """Last environment contact force (what an F/T sensor would measure)."""
        return self._contact.copy()

    def set_target_position(self, x: NDArray) -> None:
        self._target = np.array(x, dtype=float)

    def set_feedforward_force(self, f: NDArray) -> None:
        self._ff = np.array(f, dtype=float)

    def step(self, dt: float) -> None:
        """Advance the plant by ``dt`` using semi-implicit Euler."""
        self._contact = (
            np.array(self.environment(self._pos), dtype=float)
            if self.environment is not None
            else np.zeros(self.dof)
        )
        force = (
            self._ff
            + self.stiffness * (self._target - self._pos)
            - self.damping * self._vel
            + self._contact
        )
        accel = force / self.mass
        self._vel = self._vel + accel * dt
        self._pos = self._pos + self._vel * dt
