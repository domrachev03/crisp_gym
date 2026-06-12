"""Shared-memory high-rate ring buffer for cross-process sub-step logging.

Same window contract as :class:`crisp_gym.record.hrate_buffer.HrateRingBuffer`
(right-aligned latest-``window`` samples, front zero-padded, times made relative
to the frame ``t_ref``), but the storage lives in ``multiprocessing.shared_memory``
so a dedicated poller process can append at the sensor's native kHz rate on its
own GIL while the recorder process reads windows at frame rate. The in-process
``HrateManager`` starves the high-rate subs (F/T, joints) down to ~470 Hz because
its spin thread competes with the 60 fps record/encode loop for one GIL; moving
the pollers to their own process restores the full publish rate.

Memory layout (one block per signal)::

    [ write_count : int64 ][ times : capacity float64 ][ values : capacity*dof float64 ]

``write_count`` is bumped *after* the sample is written, so a reader that sees a
new count is guaranteed the data is already in place. A seqlock recheck guards
the rare case where the writer laps the window mid-read.
"""

from __future__ import annotations

from multiprocessing import shared_memory

import numpy as np
from numpy.typing import NDArray

_HEADER = 8  # int64 write_count


class SharedHrateRing:
    """Fixed-window ring buffer of timestamped samples in shared memory."""

    def __init__(self, shm: shared_memory.SharedMemory, window: int, dof: int,
                 capacity: int, owner: bool) -> None:
        """Wrap an existing shared-memory block. Use :meth:`create` / :meth:`attach`."""
        self._shm = shm
        self.window = int(window)
        self.dof = int(dof)
        self.capacity = int(capacity)
        self._owner = owner
        buf = shm.buf
        self._wc: NDArray | None = np.ndarray((1,), dtype=np.int64, buffer=buf, offset=0)
        self._times: NDArray | None = np.ndarray(
            (capacity,), dtype=np.float64, buffer=buf, offset=_HEADER)
        self._values: NDArray | None = np.ndarray(
            (capacity, dof), dtype=np.float64, buffer=buf, offset=_HEADER + capacity * 8)

    @staticmethod
    def _nbytes(capacity: int, dof: int) -> int:
        return _HEADER + capacity * 8 + capacity * dof * 8

    @classmethod
    def create(cls, window: int, dof: int, oversize: int = 8) -> "SharedHrateRing":
        """Allocate a new shared block sized ``window * oversize`` and zero it.

        Args:
            window: samples returned per :meth:`snapshot` (0 disables).
            dof: sample dimensionality (e.g. 6 for a wrench).
            oversize: capacity multiple of ``window`` to absorb writer/reader skew.
        """
        window = int(window)
        dof = int(dof)
        capacity = max(1, window * int(oversize))
        shm = shared_memory.SharedMemory(create=True, size=cls._nbytes(capacity, dof))
        ring = cls(shm, window, dof, capacity, owner=True)
        ring._wc[0] = 0
        ring._times[:] = 0.0
        ring._values[:] = 0.0
        return ring

    @classmethod
    def attach(cls, meta: dict) -> "SharedHrateRing":
        """Attach a second handle (e.g. in another process) to an existing block."""
        shm = shared_memory.SharedMemory(name=meta["name"])
        return cls(shm, meta["window"], meta["dof"], meta["capacity"], owner=False)

    @property
    def meta(self) -> dict:
        """Picklable descriptor to hand a child process for :meth:`attach`."""
        return {"name": self._shm.name, "window": self.window,
                "dof": self.dof, "capacity": self.capacity}

    @property
    def count(self) -> int:
        """Total samples appended since creation (monotonic; for rate diagnostics)."""
        return int(self._wc[0])

    def append(self, t: float, value) -> None:  # noqa: ANN001
        # Hot path (sensor native rate): single-slot write, publish count last.
        c = int(self._wc[0])
        i = c % self.capacity
        self._times[i] = t
        self._values[i] = value
        self._wc[0] = c + 1

    def snapshot(self, t_ref: float) -> tuple[NDArray, NDArray]:
        """Latest ``window`` samples as (values (N, dof), times (N,) - t_ref).

        Right-aligned with front zero-padding before the buffer fills. Retries a
        few times if the writer laps the window during the read (seqlock).
        """
        n = self.window
        if n == 0:
            return np.zeros((0, self.dof)), np.zeros((0,))
        values = np.zeros((n, self.dof))
        times = np.zeros(n)
        for _ in range(4):
            c0 = int(self._wc[0])
            values[:] = 0.0
            times[:] = 0.0
            m = min(c0, n)
            if m:
                idx = np.arange(c0 - m, c0) % self.capacity
                values[n - m:] = self._values[idx]
                times[n - m:] = self._times[idx] - t_ref
            c1 = int(self._wc[0])
            if c1 - c0 <= self.capacity - n:
                return values, times  # no overwrite of the read window
        return values, times  # best effort after retries

    def close(self) -> None:
        """Drop numpy views, close (and unlink if creator) the shared block."""
        self._wc = self._times = self._values = None
        try:
            self._shm.close()
            if self._owner:
                self._shm.unlink()
        except Exception:  # noqa: BLE001
            pass
