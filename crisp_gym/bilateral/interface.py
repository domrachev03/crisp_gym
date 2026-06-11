"""The robot abstraction the bilateral control law operates on.

Keeping the control law (position coupling, force reflection, TDPA) behind this
minimal protocol lets it run identically against the pure-python ``MSDRobot``
dry-run plant (for TDD) and a real ``crisp_py`` robot (via a thin runtime
adapter), with no ROS dependency in the law itself.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from numpy.typing import NDArray


@runtime_checkable
class RobotInterface(Protocol):
    """Read robot state, write target position and feed-forward wrench."""

    @property
    def position(self) -> NDArray:
        """Current generalized position (e.g. end-effector position), shape (dof,)."""
        ...

    @property
    def velocity(self) -> NDArray:
        """Current generalized velocity, shape (dof,)."""
        ...

    @property
    def wrench(self) -> NDArray:
        """Measured external/contact wrench (force), shape (dof,)."""
        ...

    def set_target_position(self, x: NDArray) -> None:
        """Command an impedance target position."""
        ...

    def set_feedforward_force(self, f: NDArray) -> None:
        """Command a feed-forward force (e.g. reflected force on the leader)."""
        ...
