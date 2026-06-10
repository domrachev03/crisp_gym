"""Delayed communication channel for injecting artificial latency in dry-run.

Models a one-way link with a fixed integer transport delay measured in control
steps. ``send`` enqueues a value each tick; ``receive`` returns the value that was
sent ``delay_steps`` ticks earlier (a fill value until the pipeline is primed).
This is the dry-run stand-in for network latency between master and follower,
used to test that TDPA keeps the loop passive under delay.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from numpy.typing import NDArray


class DelayedChannel:
    """One-way channel delaying values by a fixed number of control steps."""

    def __init__(self, delay_steps: int = 0, dof: int = 6, fill: NDArray | None = None):
        """Create the channel.

        Args:
            delay_steps: Transport delay in control steps (0 = instantaneous).
            dof: Dimensionality of the transported vectors.
            fill: Value returned before the pipeline has been primed (default zeros).
        """
        if delay_steps < 0:
            raise ValueError("delay_steps must be >= 0")
        self.delay_steps = int(delay_steps)
        self.dof = int(dof)
        self._fill = np.zeros(dof) if fill is None else np.array(fill, dtype=float)
        self._buf: deque[NDArray] = deque(maxlen=self.delay_steps + 1)

    def send(self, value: NDArray) -> None:
        self._buf.append(np.array(value, dtype=float))

    def receive(self) -> NDArray:
        """Return the value sent ``delay_steps`` steps ago, else the fill value."""
        if len(self._buf) <= self.delay_steps:
            return self._fill.copy()
        return self._buf[0].copy()
