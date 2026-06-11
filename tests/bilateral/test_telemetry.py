"""Comprehensive teleop telemetry: structured per-step log + latency/drift analysis."""

import numpy as np

from crisp_gym.bilateral.telemetry import TeleopLogger, analyze


def test_logger_roundtrips_jsonl(tmp_path):
    logger = TeleopLogger()
    logger.log(step=0, t=0.0, leader_pos=np.array([0.1, 0.2]), wrench=np.array([1.0, 0.0]))
    logger.log(step=1, t=0.01, leader_pos=np.array([0.11, 0.2]), wrench=np.array([1.5, 0.0]))
    path = tmp_path / "run.jsonl"
    logger.to_jsonl(path)
    loaded = TeleopLogger.from_jsonl(path)
    assert len(loaded.records) == 2
    assert loaded.records[1]["step"] == 1
    assert np.allclose(loaded.records[0]["leader_pos"], [0.1, 0.2])


def test_to_arrays_stacks_columns():
    logger = TeleopLogger()
    for i in range(5):
        logger.log(step=i, t=i * 0.01, leader_pos=np.array([float(i)]))
    arr = logger.to_arrays()
    assert arr["leader_pos"].shape == (5, 1)
    assert np.allclose(arr["t"], [0.0, 0.01, 0.02, 0.03, 0.04])


def test_analyze_reports_loop_dt_latency_and_drift():
    logger = TeleopLogger()
    for i in range(100):
        t = i * 0.01
        logger.log(
            step=i, t=t,
            leader_pos=np.array([0.1 * i]),
            follower_pos=np.array([0.1 * i - 0.05 - 0.001 * i]),  # growing drift
            t_send=t, t_recv=t + 0.03,  # 30 ms one-way latency
        )
    rep = analyze(logger)
    assert np.isclose(rep["loop_dt_mean"], 0.01)
    assert np.isclose(rep["latency_mean"], 0.03, atol=1e-9)
    assert rep["max_drift"] > 0.1  # drift grew over the run
