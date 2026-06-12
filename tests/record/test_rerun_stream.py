"""RerunStreamer: disabled = no-op, missing-rerun = graceful, logging maps obs fields.

No rerun install needed: the streamer lazy-imports ``rerun`` and a fake module is
injected to assert what would be logged. Mirrors the recorder's per-frame obs shape
(``observation.images.*`` / ``observation.<arm>_ft_hrate`` / ``observation.<arm>_state``).
"""

from __future__ import annotations

import sys
import threading
import time
import types

import numpy as np

from crisp_gym.record.rerun_stream import RerunStreamer


def _fake_obs() -> dict:
    return {
        "observation.images.wrist": np.zeros((4, 4, 3), dtype=np.uint8),
        "observation.images.front": np.zeros((4, 4, 3), dtype=np.uint8),
        "observation.follower_ft_hrate": np.ones((10, 6), dtype=np.float32),
        "observation.leader_ft_hrate": 2 * np.ones((10, 6), dtype=np.float32),
        "observation.follower_pose_hrate": np.arange(7, dtype=np.float32).reshape(1, 7),
        "observation.leader_pose_hrate": np.arange(7, dtype=np.float32).reshape(1, 7),
        "observation.follower_state": np.zeros((20,), dtype=np.float32),
        "observation.leader_state": np.zeros((20,), dtype=np.float32),
    }


def test_disabled_is_total_noop():
    s = RerunStreamer(enabled=False)
    assert s.active is False
    # must never raise / never import rerun
    s.log_frame(_fake_obs())
    s.log_frame(None)
    s.close()


def test_missing_rerun_degrades_gracefully(monkeypatch):
    # Force the lazy ``import rerun`` to fail.
    monkeypatch.setitem(sys.modules, "rerun", None)
    s = RerunStreamer(enabled=True, mode="web", web_port=9090, ws_port=9877)
    # serve never came up -> inactive, but no exception
    assert s.active is False
    s.log_frame(_fake_obs())  # no-op, no raise
    s.close()


def test_unknown_mode_disables_without_import(monkeypatch):
    fake = _FakeRerun()
    monkeypatch.setitem(sys.modules, "rerun", fake)
    s = RerunStreamer(enabled=True, mode="bogus")
    assert s.active is False
    assert fake.inited is None  # never imported/inited a bad mode
    s.log_frame(_fake_obs())
    s.close()


class _FakeRerun(types.ModuleType):
    """Minimal rerun stand-in capturing init/serve_web/spawn/save/log calls."""

    def __init__(self):
        super().__init__("rerun")
        self.inited = None
        self.served = None
        self.spawned = False
        self.saved = None
        self.compressed = 0  # count of Image.compress() calls
        self.logged: list[tuple[str, str]] = []  # (entity_path, kind)

    def init(self, app_id, spawn=False):  # noqa: ANN001
        self.inited = app_id

    def serve_web(self, *, open_browser, web_port, ws_port, server_memory_limit=None):  # noqa: ANN001
        self.served = (web_port, ws_port)

    def spawn(self, *a, **k):  # noqa: ANN002, ANN003
        self.spawned = True

    def save(self, path):  # noqa: ANN001
        self.saved = path

    # logging primitives -- record entity path + which archetype was used
    def log(self, entity_path, value, *, static=False, **kw):  # noqa: ANN001, ANN003
        self.logged.append((entity_path, type(value).__name__))

    def Image(self, arr):  # noqa: ANN001, N802
        fake = self

        def _compress(jpeg_quality=95):  # noqa: ANN001
            fake.compressed += 1
            return types.SimpleNamespace(_jpeg=jpeg_quality, __class__=type("EncodedImage", (), {}))

        return types.SimpleNamespace(_arr=arr, compress=_compress,
                                     __class__=type("Image", (), {}))

    def Scalar(self, v):  # noqa: ANN001, N802
        return types.SimpleNamespace(_v=v, __class__=type("Scalar", (), {}))

    def Points3D(self, pts):  # noqa: ANN001, N802
        return types.SimpleNamespace(_pts=pts, __class__=type("Points3D", (), {}))

    def set_time_seconds(self, *a, **k):  # noqa: ANN002, ANN003
        pass

    def set_time_sequence(self, *a, **k):  # noqa: ANN002, ANN003
        pass


def test_web_serve_and_log_with_fake_rerun(monkeypatch):
    fake = _FakeRerun()
    monkeypatch.setitem(sys.modules, "rerun", fake)
    s = RerunStreamer(enabled=True, mode="web", web_port=1234, ws_port=5678,
                      images_only=False)
    assert s.active is True
    assert fake.served == (1234, 5678)
    assert fake.inited == "crisp_record"

    s.log_frame(_fake_obs())
    s.flush()
    paths = [p for p, _ in fake.logged]
    # both camera images logged under their names
    assert any("wrist" in p and "camera" in p for p in paths)
    assert any("front" in p and "camera" in p for p in paths)
    # force magnitude scalars for both arms (images_only=False)
    assert any("follower" in p and "force" in p for p in paths)
    assert any("leader" in p and "force" in p for p in paths)
    s.close()


def test_spawn_mode_calls_spawn(monkeypatch):
    fake = _FakeRerun()
    monkeypatch.setitem(sys.modules, "rerun", fake)
    s = RerunStreamer(enabled=True, mode="spawn")
    assert s.active is True
    assert fake.spawned is True
    assert fake.served is None
    s.close()


def test_save_mode_calls_save(monkeypatch):
    fake = _FakeRerun()
    monkeypatch.setitem(sys.modules, "rerun", fake)
    s = RerunStreamer(enabled=True, mode="save", save_path="/tmp/x.rrd")
    assert s.active is True
    assert fake.saved == "/tmp/x.rrd"
    s.close()


def test_images_only_skips_force_and_pose(monkeypatch):
    fake = _FakeRerun()
    monkeypatch.setitem(sys.modules, "rerun", fake)
    s = RerunStreamer(enabled=True, mode="spawn", images_only=True)
    s.log_frame(_fake_obs())
    s.flush()
    paths = [p for p, _ in fake.logged]
    # images logged, but NO force/ee streams
    assert any("camera" in p for p in paths)
    assert not any("force" in p for p in paths)
    assert not any("/ee" in p for p in paths)
    s.close()


def test_log_frame_never_raises_on_bad_obs(monkeypatch):
    fake = _FakeRerun()

    def _boom(*a, **k):  # noqa: ANN002, ANN003
        raise RuntimeError("rerun exploded")

    fake.log = _boom
    monkeypatch.setitem(sys.modules, "rerun", fake)
    s = RerunStreamer(enabled=True, web_port=1, ws_port=2)
    assert s.active is True
    # an exception inside logging must be swallowed (warn once), never crash recording
    s.log_frame(_fake_obs())
    s.log_frame(_fake_obs())
    s.flush()
    s.close()


def test_log_frame_logs_async_after_flush(monkeypatch):
    fake = _FakeRerun()
    monkeypatch.setitem(sys.modules, "rerun", fake)
    s = RerunStreamer(enabled=True, mode="spawn", images_only=True)
    s.log_frame(_fake_obs())
    s.flush()  # worker logs on its own thread; flush waits for it
    assert any("camera" in p for p, _ in fake.logged)
    s.close()


def test_images_logged_jpeg_compressed(monkeypatch):
    fake = _FakeRerun()
    monkeypatch.setitem(sys.modules, "rerun", fake)
    s = RerunStreamer(enabled=True, mode="spawn", images_only=True, max_fps=0)
    s.log_frame(_fake_obs())
    s.flush()
    s.close()
    # _fake_obs has 2 camera images -> both JPEG-compressed before logging
    assert fake.compressed == 2
    assert [p for p, _ in fake.logged if "camera" in p]  # and they were logged


def test_sync_mode_logs_inline_without_worker(monkeypatch):
    fake = _FakeRerun()
    monkeypatch.setitem(sys.modules, "rerun", fake)
    s = RerunStreamer(enabled=True, mode="spawn", images_only=True, sync=True)
    assert s._worker is None  # no worker thread in sync mode
    s.log_frame(_fake_obs())  # logged immediately, no flush needed
    assert any("camera" in p for p, _ in fake.logged)
    s.close()


def test_downscale_caps_dimension_and_passes_small_through():
    s = RerunStreamer(enabled=False)  # disabled: exercise the pure helper (max_dim default 320)
    small = s._downscale(np.zeros((480, 640, 3), dtype=np.uint8))
    assert max(small.shape[:2]) <= 320
    tiny = np.zeros((8, 8, 3), dtype=np.uint8)
    assert s._downscale(tiny) is tiny  # already small -> no copy


def test_preview_downscale_keeps_recorded_frame_full_res(monkeypatch):
    fake = _FakeRerun()
    monkeypatch.setitem(sys.modules, "rerun", fake)
    s = RerunStreamer(enabled=True, mode="spawn", images_only=True, max_fps=0, max_dim=320)
    obs = {"observation.images.wrist": np.zeros((480, 640, 3), dtype=np.uint8)}
    s.log_frame(obs)
    s.flush()
    s.close()
    # the obs the writer process records is untouched -> dataset stays full resolution
    assert obs["observation.images.wrist"].shape == (480, 640, 3)


def test_max_fps_throttles_rapid_frames(monkeypatch):
    fake = _FakeRerun()
    monkeypatch.setitem(sys.modules, "rerun", fake)
    # max_fps=5 -> 200ms gate; 5 back-to-back calls accept only the first.
    s = RerunStreamer(enabled=True, mode="spawn", images_only=True, buffer=1, max_fps=5)
    for _ in range(5):
        s.log_frame(_fake_obs())
    s.flush()
    s.close()
    cam_logs = [p for p, _ in fake.logged if "camera" in p]
    assert len(cam_logs) == 2  # one accepted frame x 2 images; the rest throttled


def test_log_frame_nonblocking_and_bounded_under_backpressure(monkeypatch):
    # A wedged worker must not block the record loop, and the queue must stay bounded.
    gate = threading.Event()

    fake = _FakeRerun()
    real_log = fake.log

    def _blocking_log(path, value):  # noqa: ANN001
        gate.wait(2.0)  # first call wedges the worker until released
        real_log(path, value)

    fake.log = _blocking_log
    monkeypatch.setitem(sys.modules, "rerun", fake)

    # max_fps=0 disables the throttle so this exercises pure queue drop-oldest bounding.
    s = RerunStreamer(enabled=True, mode="spawn", images_only=True, buffer=3, max_fps=0)
    try:
        for _ in range(50):
            t0 = time.time()
            s.log_frame(_fake_obs())
            assert time.time() - t0 < 0.1  # never blocks, even with the worker wedged
        assert s._queue.qsize() <= 3  # drop-oldest keeps the queue bounded
    finally:
        gate.set()
        s.close()
