"""Configuration for bilateral teleoperation schemes.

One ``BilateralConfig`` fully specifies a teleop scheme: which channels are open
(forward position, return force, forward force, return position spring), whether
TDPA passivates the reflected force, the gains/filters, the artificial channel
delay, and the leader/follower wiring. Schemes are shipped as one YAML per scheme
under ``config/teleop/bilateral/`` and selected by name, so a recording (or the
standalone runner) picks a teleop mode with a single ``--teleop-scheme`` argument
and the exact parameters live in one reviewable place.

``from_yaml`` takes an explicit path and depends only on PyYAML, keeping the
dataclass testable without crisp_py. ``make_bilateral_config`` resolves a scheme
*name* against the CRISP config paths and therefore imports crisp_py lazily.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass
class BilateralConfig:
    """All parameters of a single bilateral teleoperation scheme."""

    scheme: str = "position"
    mode: str = "cartesian"        # cartesian | joint
    coupling: str = "absolute"     # absolute | relative

    # channels
    force: bool = False            # return: follower wrench -> leader feed-forward
    force_fwd: bool = False        # forward: leader wrench -> follower feed-forward (4-ch)
    pos_spring: bool = False       # return: follower pos -> leader position spring (4-ch)
    tdpa: bool = False             # passivate the reflected force

    # timing
    control_frequency: float = 100.0
    delay_steps: int = 0           # artificial round-trip channel delay (control steps)

    # reflected-force shaping
    feedback_gain: float = 1.0
    feedback_sign: float = 1.0
    feedback_max_force: float = 30.0  # ~p99.9 of observed peg-insertion contact (pf-insert-10ep)
    contact_threshold_n: float = 1.0
    reflect_highpass_hz: float = 0.0
    reflect_deadband_n: float = 0.0

    # 4-channel forward-force gain (leader wrench -> follower ff); None = use feedback_gain
    force_fwd_gain: float | None = None

    # 4-channel position spring stiffness, leader pulled toward follower.
    # position_spring_k is translational (N/m); rot_spring_k is rotational (N*m/rad)
    # and MUST be far smaller -- a translational k applied to a rotvec difference is
    # an enormous yaw torque (resists rotation, diverges). 0 disables the rot spring.
    position_spring_k: float = 0.0
    rot_spring_k: float = 0.0

    # wiring
    follower_wrench_topic: str = "/right/netft_data_unbiased_tcp"
    leader_wrench_topic: str = "/left/netft_data_unbiased_tcp"
    leader_config: str = "left_leader_nogripper"
    leader_namespace: str = "left"
    follower_namespace: str = "right"
    follower_env_config: str = "panda_no_cam"

    @classmethod
    def from_yaml(cls, yaml_path: Path | str, **overrides) -> "BilateralConfig":
        """Build a config from a YAML file, then apply keyword overrides.

        Unknown keys in the YAML are ignored (with no silent typo-swallowing of
        the *flag* fields, which the scheme tests pin explicitly).
        """
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f) or {}
        data.update(overrides)
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known}
        return cls(**kwargs)


def make_bilateral_config(name: str, **overrides) -> BilateralConfig:
    """Resolve a scheme *name* to a ``BilateralConfig`` via the CRISP config paths.

    Imports crisp_py (through ``crisp_gym.config.path``) lazily so the dataclass
    and ``from_yaml`` remain usable in pure-python (no-ROS) test runs.
    """
    from crisp_gym.config.path import find_config, list_configs_in_folder

    path = find_config(f"teleop/bilateral/{name.lower()}.yaml")
    if path is None:
        available = sorted(
            p.stem for p in list_configs_in_folder("teleop/bilateral") if p.suffix == ".yaml"
        )
        raise ValueError(
            f"Unknown bilateral scheme {name!r}. Available: {available}"
        )
    return BilateralConfig.from_yaml(path.resolve(), **overrides)


def list_bilateral_schemes() -> list[str]:
    """List shipped bilateral scheme names (requires crisp_py for path resolution)."""
    from crisp_gym.config.path import list_configs_in_folder

    return sorted(
        p.stem for p in list_configs_in_folder("teleop/bilateral") if p.suffix == ".yaml"
    )
