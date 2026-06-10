"""Dry-run bilateral teleoperation loop (position-force) for testing.

Couples a leader and follower (any ``RobotInterface``, here MSD plants) through two
delayed channels: leader position -> follower target, follower contact wrench ->
leader feed-forward (force reflection). An optional ``passivity`` controller may
modify the reflected force each tick (step 2: TDPA). Pure-python, no ROS.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def run_bilateral(
    leader,
    follower,
    ch_pos,
    ch_force,
    human_force: NDArray,
    n_steps: int,
    dt: float,
    passivity=None,
    logger=None,
    coupling: str = "absolute",
) -> dict:
    """Run the dry-run bilateral loop and return logged trajectories.

    Args:
        leader: Leader plant (RobotInterface + ``step(dt)``); weightless, driven by
            the human force plus reflected follower force.
        follower: Follower plant; impedance-tracks the delayed leader position and
            reports its contact wrench.
        ch_pos: Delayed channel carrying leader position -> follower target.
        ch_force: Delayed channel carrying follower wrench -> leader.
        human_force: Constant operator force on the leader, shape (dof,).
        n_steps: Number of control steps.
        dt: Control timestep (s).
        passivity: Optional object with ``modify(reflected, leader_velocity, dt)``
            returning a passivated reflected force (TDPA). ``None`` = naive reflection.

    Returns:
        dict of (n_steps, dof) arrays: leader_pos, follower_pos, leader_vel, reflected.
    """
    if coupling not in ("absolute", "relative"):
        raise ValueError(f"coupling must be 'absolute' or 'relative', got {coupling!r}")
    dof = leader.position.shape[0]
    hf = np.array(human_force, dtype=float)
    out = {k: np.zeros((n_steps, dof)) for k in ("leader_pos", "follower_pos", "leader_vel", "reflected")}

    prev_leader_pos = leader.position  # for relative-increment coupling
    follower_target = follower.position  # relative anchor (offset from the leader)

    for i in range(n_steps):
        # Forward path: leader motion -> (delay) -> follower impedance target.
        if coupling == "absolute":
            ch_pos.send(leader.position)
            follower.set_target_position(ch_pos.receive())
        else:  # relative: transmit per-step increments, follower integrates them
            ch_pos.send(leader.position - prev_leader_pos)
            prev_leader_pos = leader.position
            follower_target = follower_target + ch_pos.receive()
            follower.set_target_position(follower_target)

        # Return path: follower contact wrench -> (delay) -> leader feed-forward.
        follower_wrench = follower.wrench
        ch_force.send(follower_wrench)
        reflected = ch_force.receive()
        if passivity is not None:
            reflected = passivity.modify(reflected, leader.velocity, hf, dt)

        leader.set_feedforward_force(hf + reflected)

        leader.step(dt)
        follower.step(dt)

        out["leader_pos"][i] = leader.position
        out["follower_pos"][i] = follower.position
        out["leader_vel"][i] = leader.velocity
        out["reflected"][i] = reflected

        if logger is not None:
            logger.log(
                step=i,
                t=i * dt,
                leader_pos=leader.position,
                follower_pos=follower.position,
                leader_vel=leader.velocity,
                follower_wrench=follower_wrench,
                reflected=reflected,
            )

    return out
