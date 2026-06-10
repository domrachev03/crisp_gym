"""Time-Domain Passivity Approach (TDPA) for the position-force loop.

Port of ``MasterOnlyPOPC`` from franka_server_standalone (the deployed P-F mode);
see docs/design/tdpa_reference.md. Per-axis passivity observer + passivity
controller: the master tracks its own input energy (human force x velocity, gated
on contact) and the slave's output energy; when the reflected-force port would
generate energy (delay-induced), a variable damper is injected into the reflected
force. The revert-then-reaccumulate order preserves same-tick passivity.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class MasterOnlyPOPC:
    """Per-axis time-domain passivity controller modifying the reflected force."""

    def __init__(
        self,
        dof: int = 6,
        contact_threshold_n: float = 3.0,
        alpha_max: float = 1e6,
        eps_v: float = 1e-9,
    ):
        self.dof = dof
        self.contact_threshold_n = float(contact_threshold_n)
        self.alpha_max = float(alpha_max)
        self.eps_v = float(eps_v)
        self.e_m_in = np.zeros(dof)   # master input energy  >= 0
        self.e_m_out = np.zeros(dof)  # master output energy <= 0
        self.e_s_in = np.zeros(dof)   # slave input energy   >= 0
        self.e_s_out = np.zeros(dof)  # slave output energy  <= 0

    def update_master(self, Fh: NDArray, Vm: NDArray, Fe: NDArray, dt: float) -> None:
        """Accumulate master energies (gated on contact: |Fe| > threshold)."""
        for i in range(self.dof):
            if abs(Fe[i]) <= self.contact_threshold_n:
                continue
            power = Fh[i] * Vm[i]
            if power > 0:
                self.e_m_in[i] += power * dt
            elif power < 0:
                self.e_m_out[i] += power * dt

    def update_slave(self, F: NDArray, V: NDArray, dt: float) -> NDArray:
        """Accumulate slave energies; inject damping into F when the port is active."""
        F_res = np.array(F, dtype=float)
        for i in range(self.dof):
            power = F[i] * V[i]
            if power > 0:
                self.e_s_in[i] += power * dt
            elif power < 0:
                self.e_s_out[i] += power * dt  # tentative

            if (self.e_s_out[i] + self.e_m_in[i] < 0) and (V[i] ** 2 > self.eps_v):
                alpha = (self.e_s_out[i] + self.e_m_in[i]) / (V[i] ** 2 * dt)  # <= 0
                alpha = max(alpha, -self.alpha_max)
                self.e_s_out[i] -= F[i] * V[i] * dt          # revert tentative
                F_res[i] -= alpha * V[i]                      # inject damping
                self.e_s_out[i] += F_res[i] * V[i] * dt       # re-accumulate corrected
        return F_res

    def modify(self, reflected: NDArray, leader_velocity: NDArray, human_force: NDArray, dt: float) -> NDArray:
        """Passivate the reflected force before it is applied to the leader.

        Mirrors the master controller call order: update_master, then update_slave
        on the reflected port (F = -Fe, V = Vm); the force applied to the leader is
        ``-F_res`` (i.e. the passivated reflected wrench).
        """
        Fe = np.array(reflected, dtype=float)
        Vm = np.array(leader_velocity, dtype=float)
        Fh = np.array(human_force, dtype=float)
        self.update_master(Fh, Vm, Fe, dt)
        f_res = self.update_slave(-Fe, Vm, dt)
        return -f_res
