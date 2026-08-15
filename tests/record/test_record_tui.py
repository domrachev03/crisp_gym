"""Tests for recording TUI state-dependent controls."""

from crisp_gym.scripts.record_tui import _action_allowed, _state_label


def test_record_toggles_only_waiting_or_recording():
    """Record toggles between waiting and active capture."""
    assert _action_allowed("record", "is_waiting")
    assert _action_allowed("record", "recording")
    assert not _action_allowed("record", "paused")


def test_save_and_delete_require_paused_episode():
    """An episode can be saved or discarded only after stopping it."""
    for action in ("save", "delete"):
        assert _action_allowed(action, "paused")
        assert not _action_allowed(action, "recording")


def test_exit_does_not_discard_active_recording():
    """Exit is rejected during capture so buffered frames are not lost."""
    assert _action_allowed("exit", "is_waiting")
    assert _action_allowed("exit", "paused")
    assert not _action_allowed("exit", "recording")


def test_unknown_state_has_clear_label():
    """The UI distinguishes a missing recorder from a normal state."""
    assert _state_label(None) == "NOT DETECTED"


def test_replay_record_key_starts_pauses_and_resumes():
    """Replay reuses the record key as a three-state execution toggle."""
    for state in ("is_waiting", "recording", "paused"):
        assert _action_allowed("record", state, mode="replay")
    assert not _action_allowed("save", "paused", mode="replay")


def test_replay_labels_and_safe_exit_states():
    """Replay has explicit labels and still requires pausing before exit."""
    assert _state_label("recording", mode="replay") == "REPLAYING"
    assert _state_label("finished", mode="replay") == "REPLAY COMPLETE"
    assert not _action_allowed("exit", "recording", mode="replay")
    assert _action_allowed("exit", "paused", mode="replay")
    assert _action_allowed("exit", "finished", mode="replay")
