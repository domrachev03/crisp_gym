"""Reflected-wrench conditioning for the force-reflection return path.

A free-float leader (``k_task=0``) integrates any DC in the reflected force into
unbounded drift (no recentering spring). The F/T bias is imperfect and
pose-dependent, so the reflected force carries a slow ~1 N DC that drives the
leader away. ``FirstOrderHighPass`` strips that DC (passing contact transients);
``soft_deadband`` nulls the residual free-space noise floor. Both are opt-in:
``cutoff_hz <= 0`` / ``threshold <= 0`` make them pass-through.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class FirstOrderHighPass:
    """Per-component first-order high-pass (input minus a low-pass DC estimate)."""

    def __init__(self, cutoff_hz: float, dof: int, dt: float):
        self.enabled = cutoff_hz > 0.0
        self.dof = int(dof)
        # LP smoothing factor for the DC estimate: alpha = dt / (tau + dt), tau = 1/(2*pi*fc)
        if self.enabled:
            tau = 1.0 / (2.0 * np.pi * float(cutoff_hz))
            self.alpha = float(dt) / (tau + float(dt))
        else:
            self.alpha = 0.0
        self._dc = np.zeros(self.dof)

    def step(self, x: NDArray) -> NDArray:
        """Return the high-passed value of ``x`` (a copy of ``x`` when disabled)."""
        x = np.asarray(x, dtype=float)
        if not self.enabled:
            return x.copy()
        self._dc += self.alpha * (x - self._dc)
        return x - self._dc


def soft_deadband(vec: NDArray, threshold: float) -> NDArray:
    """Shrink a vector's magnitude by ``threshold``, preserving direction.

    Returns zeros when ``|vec| <= threshold`` (continuous at the threshold) and
    ``vec * (|vec| - threshold) / |vec|`` above it. ``threshold <= 0`` -> pass-through.
    """
    vec = np.asarray(vec, dtype=float)
    if threshold <= 0.0:
        return vec.copy()
    n = float(np.linalg.norm(vec))
    if n <= threshold:
        return np.zeros_like(vec)
    return vec * ((n - threshold) / n)
