"""Safety-gate tests for the mixed-version FR3 runner."""

from argparse import Namespace
from typing import Any
from unittest.mock import patch

import pytest

from crisp_gym.bilateral.bilateral_config import BilateralConfig
from crisp_gym.scripts.fr3_bilateral_teleop import (
    _apply_candidate_frame_limits,
    _validate_arm_mode,
    _warn_stale_streams,
)


def _args(**overrides: Any) -> Namespace:
    values = {
        "arm": False,
        "candidate_frame_check": False,
        "scheme": "position",
        "leader_base_to_common_quat": None,
        "transforms_verified": False,
        "frame_check_duration_s": 30.0,
    }
    values.update(overrides)
    return Namespace(**values)


def test_normal_arm_still_requires_verified_transform() -> None:
    """Normal motion cannot use a merely candidate transform."""
    with pytest.raises(SystemExit, match="Refusing to arm"):
        _validate_arm_mode(_args(arm=True, leader_base_to_common_quat=[0, 0, 0, 1]))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({}, "requires --arm"),
        ({"arm": True}, "provide the candidate"),
        (
            {"arm": True, "leader_base_to_common_quat": [0, 0, 0, 1], "scheme": "pf"},
            "only --scheme position",
        ),
        (
            {
                "arm": True,
                "leader_base_to_common_quat": [0, 0, 0, 1],
                "frame_check_duration_s": 0.0,
            },
            "must be positive",
        ),
    ],
)
def test_candidate_frame_check_rejects_unsafe_invocations(
    overrides: dict[str, Any], message: str
) -> None:
    """Candidate mode requires explicit, position-only, finite bounded arming."""
    with pytest.raises(SystemExit, match=message):
        _validate_arm_mode(_args(candidate_frame_check=True, **overrides))


def test_candidate_frame_check_has_tight_nonzero_limits() -> None:
    """Candidate mode overrides permissive configuration with conservative limits."""
    args = _args(
        arm=True,
        candidate_frame_check=True,
        leader_base_to_common_quat=[0, 0, 0, 1],
    )
    _validate_arm_mode(args)
    config = BilateralConfig(
        max_translation_m=1.0,
        max_rotation_rad=1.0,
        max_command_step_m=1.0,
        max_command_step_rad=1.0,
    )
    _apply_candidate_frame_limits(config)
    assert config.max_translation_m == pytest.approx(0.010)
    assert config.leader_max_translation_m == pytest.approx(0.010)
    assert config.max_rotation_rad == pytest.approx(0.010)
    assert config.max_command_step_m == pytest.approx(0.00025)
    assert config.max_command_step_rad == pytest.approx(0.001)


def test_stale_streams_warn_without_terminating_and_are_rate_limited() -> None:
    """Staleness should warn at 1 Hz and never raise from the runner helper."""
    stale = {"leader/pose": 0.2}
    with patch("crisp_gym.scripts.fr3_bilateral_teleop.logger.warning") as warning:
        last = _warn_stale_streams(stale, now=10.0, last_warning=float("-inf"))
        assert last == pytest.approx(10.0)
        warning.assert_called_once()

        warning.reset_mock()
        last = _warn_stale_streams(stale, now=10.5, last_warning=last)
        assert last == pytest.approx(10.0)
        warning.assert_not_called()

        last = _warn_stale_streams(stale, now=11.0, last_warning=last)
        assert last == pytest.approx(11.0)
        warning.assert_called_once()
