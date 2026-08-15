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
