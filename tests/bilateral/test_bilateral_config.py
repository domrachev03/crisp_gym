"""BilateralConfig dataclass + per-scheme YAML loading.

The scheme YAMLs are the single source of truth for which channels each
teleoperation mode opens. These tests pin the flag set every shipped scheme must
resolve to, so a typo in a YAML (or a renamed field) fails loudly rather than
silently recording the wrong teleop mode.

Runs without crisp_py: ``from_yaml`` takes an explicit path, so no config-path
resolution (which imports crisp_py) is exercised here.
"""

from pathlib import Path

import pytest

from crisp_gym.bilateral.bilateral_config import BilateralConfig

CONFIG_DIR = Path(__file__).parents[2] / "crisp_gym" / "config" / "teleop" / "bilateral"

# scheme -> (force, force_fwd, pos_spring, tdpa, mode)
EXPECTED = {
    "position":   (False, False, False, False, "cartesian"),
    "pf":         (True,  False, False, False, "cartesian"),
    "pf_tdpa":    (True,  False, False, True,  "cartesian"),
    "pfpf":       (True,  True,  True,  False, "cartesian"),
    "pfpf_tdpa":  (True,  True,  True,  True,  "cartesian"),
    "ppf":        (True,  False, True,  False, "cartesian"),
    "joint_pf":   (True,  False, False, False, "joint"),
}


@pytest.mark.parametrize("scheme", list(EXPECTED))
def test_scheme_yaml_loads_to_expected_flags(scheme):
    cfg = BilateralConfig.from_yaml(CONFIG_DIR / f"{scheme}.yaml")
    force, force_fwd, pos_spring, tdpa, mode = EXPECTED[scheme]
    assert cfg.scheme == scheme
    assert cfg.force is force
    assert cfg.force_fwd is force_fwd
    assert cfg.pos_spring is pos_spring
    assert cfg.tdpa is tdpa
    assert cfg.mode == mode


def test_joint_pf_has_no_tdpa():
    # explicit decision: joint mode ships P-F only this round
    cfg = BilateralConfig.from_yaml(CONFIG_DIR / "joint_pf.yaml")
    assert cfg.tdpa is False


def test_defaults_when_field_absent():
    cfg = BilateralConfig.from_yaml(CONFIG_DIR / "position.yaml")
    # delay defaults to 0 (no artificial channel delay unless a scheme asks for it)
    assert cfg.delay_steps == 0
    assert cfg.control_frequency > 0
    assert cfg.leader_wrench_topic.endswith("netft_data_unbiased_tcp")


def test_overrides_apply():
    cfg = BilateralConfig.from_yaml(CONFIG_DIR / "pf.yaml", delay_steps=40, feedback_gain=0.5)
    assert cfg.delay_steps == 40
    assert cfg.feedback_gain == 0.5
