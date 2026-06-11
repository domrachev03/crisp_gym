"""The single live bilateral teleoperation control law.

One ``BilateralController.step()`` implements every shipped scheme (position,
P-F, P-F+TDPA, 4-channel PF-PF, PF-PF+TDPA, joint P-F); which channels are open is
decided entirely by the ``BilateralConfig`` flags. The same object runs against the
pure-python ``MSDRobot`` plants (for TDD) and, via a thin runtime adapter, against
real ``crisp_py`` robots — so the standalone runner and the structured recorder
share exactly one control law and recording captures whatever scheme is active.

The controller is axis-generic: it operates on a robot's *generalized position*
(end-effector pose vector or joint vector) and *wrench* through the ``RobotInterface``
protocol. Cartesian-specific framing (quaternion handling, home-anchored pose
mapping, wrench-frame rotation) lives in the cartesian adapter, which presents a
consistent same-dimension view to this law. ``step()`` only computes and commands;
the plant advances on its own (real time on hardware, an explicit ``plant.step`` in
dry-run tests).

Channels per scheme:
  - forward position : leader pos -> (delay) -> follower target              (all)
  - return force     : follower wrench -> (delay) -> leader feed-forward     (force)
  - forward force    : leader wrench  -> (delay) -> follower feed-forward    (force_fwd, 4-ch)
  - return position  : follower pos   -> (delay) -> leader position spring   (pos_spring, 4-ch)

TDPA passivates the reflected force; its master energy budget is driven by ``Fh``
(the human/leader force), which is the provided ``human_force`` in dry-run and the
measured leader wrench on hardware. Feeding a real ``Fh`` (instead of zero) is the
fix for the per-axis over-damping the old recorder path exhibited.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from crisp_gym.bilateral.bilateral_config import BilateralConfig
from crisp_gym.bilateral.filters import FirstOrderHighPass, soft_deadband
from crisp_gym.bilateral.tdpa import MasterOnlyPOPC
from crisp_gym.sim import DelayedChannel


@dataclass
class TeleopTelemetry:
    """Everything one control tick produced — enough to reconstruct the delay."""

    step: int
    live_leader_pos: NDArray
    live_follower_pos: NDArray
    commanded_follower_target: NDArray   # what the follower was told (DELAYED)
    leader_feedforward: NDArray          # total feed-forward force commanded to leader
    reflected_wrench: NDArray            # reflected-force component (post gain/filter/TDPA)
    spring_force: NDArray                # 4-ch return position-spring component
    forward_force: NDArray              # feed-forward commanded to follower (4-ch)
    follower_wrench: NDArray             # raw sensed
    leader_wrench: NDArray               # raw sensed
    delay_steps: int
    control_dt: float


class BilateralController:
    """Drives a leader/follower pair through one configurable bilateral scheme."""

    def __init__(self, leader, follower, config: BilateralConfig, dt: float | None = None,
                 wrench_bias: NDArray | None = None):
        """Wire up channels, TDPA and filters from ``config`` and anchor the homes.

        Args:
            leader, follower: objects satisfying ``RobotInterface`` (position,
                velocity, wrench, set_target_position, set_feedforward_force).
            config: the scheme to run.
            dt: control timestep; defaults to ``1 / config.control_frequency``.
            wrench_bias: optional follower-wrench bias to subtract (HW: capture at
                rest); defaults to zeros.
        """
        self.leader = leader
        self.follower = follower
        self.config = config
        self.dt = float(dt) if dt is not None else 1.0 / config.control_frequency

        self.dof = int(np.asarray(leader.position).shape[0])
        self.leader_home = np.asarray(leader.position, dtype=float).copy()
        self.follower_home = np.asarray(follower.position, dtype=float).copy()
        self.wrench_bias = (np.zeros(self.dof) if wrench_bias is None
                            else np.asarray(wrench_bias, dtype=float))

        d = config.delay_steps
        # absolute forward channel is primed with the leader home so an un-primed
        # channel commands the home target (no startup jump).
        fwd_fill = self.leader_home if config.coupling == "absolute" else None
        self.ch_fwd_pos = DelayedChannel(d, self.dof, fill=fwd_fill)
        self.ch_back_force = DelayedChannel(d, self.dof) if config.force else None
        self.ch_fwd_force = DelayedChannel(d, self.dof) if config.force_fwd else None
        self.ch_back_pos = (DelayedChannel(d, self.dof, fill=self.follower_home)
                            if config.pos_spring else None)

        self.tdpa = (MasterOnlyPOPC(dof=self.dof, contact_threshold_n=config.contact_threshold_n)
                     if config.tdpa else None)
        self.reflect_hp = FirstOrderHighPass(config.reflect_highpass_hz, self.dof, self.dt)

        self._prev_leader = self.leader_home.copy()         # relative increment anchor
        self._follower_target = self.follower_home.copy()   # relative integrated target
        self._step = 0

    def _clamp_force(self, f: NDArray) -> NDArray:
        """Clamp the translational (first <=3 axes) force magnitude to the max."""
        f = np.asarray(f, dtype=float).copy()
        k = min(3, self.dof)
        mag = float(np.linalg.norm(f[:k]))
        if mag > self.config.feedback_max_force > 0.0:
            f[:k] *= self.config.feedback_max_force / mag
        return f

    def step(self, dt: float | None = None, human_force: NDArray | None = None) -> TeleopTelemetry:
        """Compute and command one bilateral tick; return its telemetry.

        Args:
            dt: timestep override (defaults to the configured ``self.dt``).
            human_force: operator force on the leader. Provided in dry-run (drives
                the weightless leader plant and the TDPA budget). On hardware leave
                ``None`` — the human pushes physically and ``Fh`` is read from the
                measured leader wrench.
        """
        cfg = self.config
        dt = self.dt if dt is None else float(dt)

        leader_pos = np.asarray(self.leader.position, dtype=float)
        follower_pos = np.asarray(self.follower.position, dtype=float)
        leader_w = np.asarray(self.leader.wrench, dtype=float)
        follower_w = np.asarray(self.follower.wrench, dtype=float)

        # --- forward position: leader -> follower target ------------------------
        if cfg.coupling == "absolute":
            self.ch_fwd_pos.send(leader_pos)
            commanded = self.ch_fwd_pos.receive()
        else:  # relative: transmit increments, follower integrates (linear path)
            self.ch_fwd_pos.send(leader_pos - self._prev_leader)
            self._prev_leader = leader_pos
            self._follower_target = self._follower_target + self.ch_fwd_pos.receive()
            commanded = self._follower_target
        self.follower.set_target_position(commanded)

        # --- forward force (4-ch): leader wrench -> follower feed-forward --------
        # Clamped (parity with the reflected force) so a leader-wrench spike cannot
        # inject an unbounded force into the follower. NOTE: on a directly-held,
        # backdrivable master the NetFT reads the reaction to the leader's own
        # feed-forward, so combining force_fwd with pos_spring closes a positive
        # feedback loop (forward_force ~ spring -> drives follower apart -> grows
        # spring) that diverges with no contact. Prefer 'ppf' (spring, no force_fwd)
        # on such a rig; the clamp here only bounds the runaway, it does not cure it.
        forward_force = np.zeros(self.dof)
        if self.ch_fwd_force is not None:
            fwd_gain = cfg.force_fwd_gain if cfg.force_fwd_gain is not None else cfg.feedback_gain
            self.ch_fwd_force.send(fwd_gain * leader_w)
            forward_force = self._clamp_force(self.ch_fwd_force.receive())
            self.follower.set_feedforward_force(forward_force)

        # --- return force: follower wrench -> leader feed-forward ---------------
        reflected = np.zeros(self.dof)
        if self.ch_back_force is not None:
            fe = cfg.feedback_sign * cfg.feedback_gain * (follower_w - self.wrench_bias)
            fe = self.reflect_hp.step(fe)
            fe = soft_deadband(fe, cfg.reflect_deadband_n)
            self.ch_back_force.send(fe)
            reflected = self.ch_back_force.receive()
            Fh = human_force if human_force is not None else leader_w
            if self.tdpa is not None:
                reflected = self.tdpa.modify(reflected, self.leader.velocity, np.asarray(Fh, dtype=float), dt)
            reflected = self._clamp_force(reflected)

        # --- return position spring (4-ch): follower pos -> leader --------------
        spring = np.zeros(self.dof)
        if self.ch_back_pos is not None:
            self.ch_back_pos.send(follower_pos)
            fp = self.ch_back_pos.receive()
            spring = cfg.position_spring_k * (fp - leader_pos)

        leader_ff = reflected + spring
        if human_force is not None:
            leader_ff = leader_ff + np.asarray(human_force, dtype=float)
        self.leader.set_feedforward_force(leader_ff)

        tel = TeleopTelemetry(
            step=self._step,
            live_leader_pos=leader_pos.copy(),
            live_follower_pos=follower_pos.copy(),
            commanded_follower_target=np.asarray(commanded, dtype=float).copy(),
            leader_feedforward=leader_ff.copy(),
            reflected_wrench=reflected.copy(),
            spring_force=spring.copy(),
            forward_force=forward_force.copy(),
            follower_wrench=follower_w.copy(),
            leader_wrench=leader_w.copy(),
            delay_steps=cfg.delay_steps,
            control_dt=dt,
        )
        self._step += 1
        return tel
