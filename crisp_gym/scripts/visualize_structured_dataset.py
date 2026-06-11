"""Visualize a structured crisp_gym LeRobot dataset in Rerun.

The upstream ``lerobot-dataset-viz`` only logs the flat ``observation.state`` and
targets a newer Rerun API.  This viewer understands the structured schema
(observation.{follower,leader}_state / _ft / images.*) and uses the Rerun 0.22
API pinned by crisp_env.

Examples::

    crisp-visualize-structured --repo-id test/structured_verify --episode-index 0
    crisp-visualize-structured --repo-id robot-lev/foo --episode-index 0 \
        --save --output-dir /tmp/viz   # writes .rrd, then `rerun file.rrd`
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import rerun as rr
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset

logger = logging.getLogger(__name__)

_FT_COMPONENTS = ["fx", "fy", "fz", "tx", "ty", "tz"]


def _chw_to_hwc_u8(t: torch.Tensor) -> np.ndarray:
    if t.dtype == torch.float32:
        t = (t * 255).clamp(0, 255).to(torch.uint8)
    return t.permute(1, 2, 0).numpy()


def visualize_structured(
    repo_id: str,
    episode_index: int,
    save: bool = False,
    output_dir: str | None = None,
) -> Path | None:
    """Log one episode of a structured dataset to Rerun."""
    dataset = LeRobotDataset(repo_id, episodes=[episode_index])
    feats = dataset.features

    cam_keys = [k for k in feats if k.startswith("observation.images.")]
    state_keys = [k for k in feats if k.startswith("observation.") and k.endswith("_state")]
    ft_keys = [k for k in feats if k.startswith("observation.") and k.endswith("_ft")]
    logger.info(f"cameras={cam_keys} state={state_keys} ft={ft_keys}")

    rr.init(f"{repo_id}/episode_{episode_index}", spawn=not save)

    for idx in range(dataset.num_frames):
        frame = dataset[idx]
        rr.set_time_sequence("frame_index", idx)
        if "timestamp" in frame:
            rr.set_time_seconds("timestamp", float(frame["timestamp"]))

        for ck in cam_keys:
            rr.log(ck, rr.Image(_chw_to_hwc_u8(frame[ck])))

        for sk in state_keys:
            short = sk.split(".")[-1]
            names = feats[sk].get("names") or None
            vals = np.asarray(frame[sk]).flatten()
            names = names or [str(i) for i in range(len(vals))]
            for n, v in zip(names, vals):
                rr.log(f"{short}/{n}", rr.Scalar(float(v)))

        for fk in ft_keys:
            short = fk.split(".")[-1]
            arr = np.asarray(frame[fk])
            latest = arr[-1] if arr.ndim == 2 else arr  # most recent hrate sample
            for n, v in zip(_FT_COMPONENTS, latest):
                rr.log(f"{short}/{n}", rr.Scalar(float(v)))

        if "action" in frame:
            for i, v in enumerate(np.asarray(frame["action"]).flatten()):
                rr.log(f"action/{i}", rr.Scalar(float(v)))

    if save:
        out = Path(output_dir or ".")
        out.mkdir(parents=True, exist_ok=True)
        rrd = out / f"{repo_id.replace('/', '_')}_episode_{episode_index}.rrd"
        rr.save(str(rrd))
        logger.info(f"Saved {rrd}")
        return rrd
    return None


def main():
    """CLI entry for the structured dataset visualizer."""
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Visualize a structured crisp_gym LeRobot dataset")
    p.add_argument("--repo-id", type=str, required=True)
    p.add_argument("--episode-index", type=int, default=0)
    p.add_argument("--save", action="store_true", help="Write an .rrd instead of spawning a viewer.")
    p.add_argument("--output-dir", type=str, default=None)
    args = p.parse_args()
    visualize_structured(args.repo_id, args.episode_index, args.save, args.output_dir)


if __name__ == "__main__":
    main()
