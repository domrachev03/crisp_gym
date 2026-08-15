"""Validation tests for structured trajectory replay."""

import numpy as np
import pytest

from crisp_gym.scripts.replay_structured_dataset import (
    _CARTESIAN_ACTION_NAMES,
    _next_replay_state,
    _qualified_topic,
    _validate_actions,
)


def _valid_actions() -> np.ndarray:
    actions = np.zeros((3, 7), dtype=float)
    actions[1, :3] = [0.01, -0.02, 0.03]
    actions[2, 3:6] = [0.1, 0.0, -0.1]
    return actions


def test_accepts_home_relative_cartesian_actions():
    """A finite trajectory beginning at the home anchor is replayable."""
    _validate_actions(_valid_actions(), _CARTESIAN_ACTION_NAMES, 0.01, 0.10)


def test_rejects_env_delta_or_joint_action_schema():
    """Replay must not silently reinterpret an incompatible action schema."""
    with pytest.raises(ValueError, match="action names"):
        _validate_actions(_valid_actions(), ["joint_0"] * 7, 0.01, 0.10)


def test_rejects_nonfinite_action():
    """NaN commands never reach the follower."""
    actions = _valid_actions()
    actions[1, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or infinity"):
        _validate_actions(actions, _CARTESIAN_ACTION_NAMES, 0.01, 0.10)


def test_rejects_startup_jump():
    """The first action must be close to the captured home anchor."""
    actions = _valid_actions()
    actions[0, 0] = 0.05
    with pytest.raises(ValueError, match="startup jump"):
        _validate_actions(actions, _CARTESIAN_ACTION_NAMES, 0.01, 0.10)


def test_qualifies_relative_follower_topic():
    """Publisher ownership checks use the same namespace resolution as ROS."""
    assert _qualified_topic("follower", "target_pose") == "/follower/target_pose"
    assert _qualified_topic("follower", "/absolute_target") == "/absolute_target"


def test_replay_reuses_record_transition_for_start_pause_and_resume():
    """The existing record command is a replay execution toggle."""
    assert _next_replay_state("is_waiting", "record") == "recording"
    assert _next_replay_state("recording", "record") == "paused"
    assert _next_replay_state("paused", "record") == "recording"


def test_replay_exit_requires_a_nonmoving_state():
    """Exit is ignored during motion but accepted before, after, or while paused."""
    assert _next_replay_state("recording", "exit") == "recording"
    assert _next_replay_state("is_waiting", "exit") == "exit"
    assert _next_replay_state("paused", "exit") == "exit"
    assert _next_replay_state("finished", "exit") == "exit"


def test_replay_ignores_record_only_save_delete_actions():
    """Replay cannot accidentally invoke recorder-only episode operations."""
    assert _next_replay_state("paused", "save") == "paused"
    assert _next_replay_state("paused", "delete") == "paused"
