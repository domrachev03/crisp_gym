"""Comprehensive teleoperation telemetry: structured per-step log + analysis.

The logger captures one record per control step (timestamps, poses, twists,
wrenches, channel send/recv stamps for latency, TDPA energies) and persists to
JSONL. ``analyze`` derives loop-rate, one-way latency, and leader/follower
position drift from a run -- the three things to diagnose on hardware (drift,
instability, delay).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _jsonable(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return v


class TeleopLogger:
    """Accumulates per-step records; serializes to/from JSONL."""

    def __init__(self) -> None:
        self.records: list[dict] = []

    def log(self, **fields: Any) -> None:
        """Append one step record. Caller supplies the fields (incl. timestamps)."""
        self.records.append(fields)

    def to_jsonl(self, path: str | Path) -> None:
        with open(path, "w") as f:
            for r in self.records:
                f.write(json.dumps({k: _jsonable(v) for k, v in r.items()}) + "\n")

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "TeleopLogger":
        logger = cls()
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                logger.records.append(
                    {k: (np.array(v) if isinstance(v, list) else v) for k, v in d.items()}
                )
        return logger

    def to_arrays(self) -> dict[str, np.ndarray]:
        """Stack each field across records into an array (n_steps, ...)."""
        if not self.records:
            return {}
        return {k: np.array([r[k] for r in self.records]) for k in self.records[0]}


def analyze(run: "TeleopLogger | list[dict]") -> dict:
    """Derive loop-rate, latency, and drift statistics from a logged run."""
    records = run.records if isinstance(run, TeleopLogger) else run
    rep: dict[str, float] = {}
    if not records:
        return rep

    if "t" in records[0]:
        t = np.array([r["t"] for r in records], dtype=float)
        if t.size > 1:
            dts = np.diff(t)
            rep["loop_dt_mean"] = float(dts.mean())
            rep["loop_dt_std"] = float(dts.std())
            rep["loop_dt_max"] = float(dts.max())
            rep["rate_hz"] = float(1.0 / dts.mean()) if dts.mean() > 0 else float("inf")

    if "t_send" in records[0] and "t_recv" in records[0]:
        lat = np.array([float(r["t_recv"]) - float(r["t_send"]) for r in records])
        rep["latency_mean"] = float(lat.mean())
        rep["latency_max"] = float(lat.max())

    if "leader_pos" in records[0] and "follower_pos" in records[0]:
        drift = np.array(
            [
                float(np.linalg.norm(np.atleast_1d(r["leader_pos"]) - np.atleast_1d(r["follower_pos"])))
                for r in records
            ]
        )
        rep["max_drift"] = float(drift.max())
        rep["final_drift"] = float(drift[-1])

    return rep
