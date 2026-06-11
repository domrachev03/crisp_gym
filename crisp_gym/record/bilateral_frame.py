"""Fold a bilateral control tick's telemetry into recorded observation fields.

Pure numpy (no rclpy), so it is unit-tested off-hardware. The structured recorder
calls these to add, per frame:

    observation.follower_target   what the follower was commanded (the DELAYED target)
    observation.reflected_wrench  force reflected to the leader (post gain/filter/TDPA)
    observation.leader_feedforward total feed-forward force commanded to the leader
    observation.forward_force     feed-forward commanded to the follower (4-channel)
    observation.spring_force      4-channel return position-spring force

Recording ``follower_target`` alongside the live ``observation.leader_state`` makes
the channel delay recoverable directly (their offset), on top of the config
metadata and the high-rate pose windows.

The action recorded is the commanded follower target plus the gripper command, so
a policy learns exactly what the teleop scheme drove the follower with.
"""

from __future__ import annotations

import numpy as np

from crisp_gym.bilateral.controller import TeleopTelemetry

# obs key -> TeleopTelemetry attribute
_KEY_TO_ATTR = {
    "observation.follower_target": "commanded_follower_target",
    "observation.reflected_wrench": "reflected_wrench",
    "observation.leader_feedforward": "leader_feedforward",
    "observation.forward_force": "forward_force",
    "observation.spring_force": "spring_force",
}
BILATERAL_OBS_KEYS = tuple(_KEY_TO_ATTR)


def telemetry_obs_fields(tel: TeleopTelemetry) -> dict[str, np.ndarray]:
    """Map a telemetry record to its recorded observation fields (float32)."""
    return {key: np.asarray(getattr(tel, attr), dtype=np.float32)
            for key, attr in _KEY_TO_ATTR.items()}


def _component_names(dof: int) -> list[str]:
    if dof == 6:
        return ["x", "y", "z", "rx", "ry", "rz"]
    return [f"q{i}" for i in range(dof)]


def bilateral_feature_specs(dof: int) -> dict[str, dict]:
    """LeRobot feature specs for the bilateral telemetry fields, all shape ``(dof,)``."""
    names = _component_names(dof)
    return {key: {"dtype": "float32", "shape": (dof,), "names": names}
            for key in BILATERAL_OBS_KEYS}


def bilateral_action(tel: TeleopTelemetry, gripper_action: float) -> np.ndarray:
    """Recorded action = commanded follower target + gripper command (float32)."""
    return np.concatenate(
        [np.asarray(tel.commanded_follower_target, dtype=np.float32),
         np.asarray([gripper_action], dtype=np.float32)]
    ).astype(np.float32)
