"""Client-side high-rate ring buffer for sub-step sensor logging.

A background poller appends native-rate samples (e.g. NetFT F/T at ~kHz); each
recorded step calls :meth:`snapshot` to attach the latest ``window`` samples to
the frame, with timestamps made relative to the frame time (to survive float32
storage). Front-padded with zeros during startup. The (values, times) layout
matches lerobot-panda's hrate windows, so its dataset path and
``hrate_reconstruction`` utility consume it unchanged.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from numpy.typing import NDArray


class HrateRingBuffer:
    """Fixed-window ring buffer of timestamped sensor samples."""

    def __init__(self, window: int, dof: int, oversize: int = 4):
        """Create the buffer.

        Args:
            window: Number of samples attached per recorded step (0 disables).
            dof: Sample dimensionality (e.g. 6 for a wrench).
            oversize: Internal capacity multiple of ``window`` to absorb poller jitter.
        """
        self.window = int(window)
        self.dof = int(dof)
        self._buf: deque[tuple[float, NDArray]] = deque(maxlen=max(1, self.window * oversize))
        self._count = 0

    @property
    def count(self) -> int:
        """Total samples appended since creation (monotonic; for rate diagnostics)."""
        return self._count

    def append(self, t: float, value) -> None:  # noqa: ANN001
        # Hot path (runs at the sensor's native kHz rate): store the raw value
        # cheaply (no per-sample numpy alloc) and defer array building to snapshot.
        self._buf.append((t, value))
        self._count += 1

    def snapshot(self, t_ref: float) -> tuple[NDArray, NDArray]:
        """Latest ``window`` samples as (values (N, dof), times (N,) - t_ref).

        Right-aligned: when fewer than ``window`` samples exist, the front is
        zero-padded (values 0, times 0).
        """
        n = self.window
        if n == 0:
            return np.zeros((0, self.dof)), np.zeros((0,))
        items = list(self._buf)[-n:]
        m = len(items)
        values = np.zeros((n, self.dof))
        times = np.zeros(n)
        if m:
            values[n - m:] = np.array([it[1] for it in items], dtype=float)
            times[n - m:] = np.fromiter((it[0] for it in items), dtype=float, count=m) - t_ref
        return values, times
