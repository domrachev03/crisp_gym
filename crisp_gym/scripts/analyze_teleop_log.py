"""Offline analysis of a teleop telemetry JSONL log: latency, loop-rate, drift.

Usage:
    python -m crisp_gym.scripts.analyze_teleop_log run.jsonl [--plot]
"""

import argparse

from crisp_gym.bilateral.telemetry import TeleopLogger, analyze


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Path to the teleop telemetry .jsonl file")
    parser.add_argument("--plot", action="store_true", help="Plot trajectories (needs matplotlib)")
    args = parser.parse_args()

    logger = TeleopLogger.from_jsonl(args.path)
    rep = analyze(logger)
    print(f"steps: {len(logger.records)}")
    for key in ("rate_hz", "loop_dt_mean", "loop_dt_max", "latency_mean", "latency_max", "max_drift", "final_drift"):
        if key in rep:
            print(f"{key:>14}: {rep[key]:.6g}")

    if args.plot:
        import matplotlib.pyplot as plt

        arr = logger.to_arrays()
        t = arr.get("t")
        fig, ax = plt.subplots(2, 1, sharex=True)
        if "leader_pos" in arr and "follower_pos" in arr:
            ax[0].plot(t, arr["leader_pos"][:, 0], label="leader")
            ax[0].plot(t, arr["follower_pos"][:, 0], label="follower")
            ax[0].set_ylabel("pos [m]")
            ax[0].legend()
        if "reflected" in arr:
            ax[1].plot(t, arr["reflected"][:, 0], label="reflected force")
            ax[1].set_ylabel("force [N]")
            ax[1].set_xlabel("t [s]")
            ax[1].legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
