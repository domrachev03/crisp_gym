![crisp_gym](media/crisp_gym_logo.webp)

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![MIT Badge](https://img.shields.io/badge/MIT-License-blue?style=flat)
<a href="https://github.com/utiasDSL/crisp_gym/actions/workflows/ruff_ci.yml"><img src="https://github.com/utiasDSL/crisp_gym/actions/workflows/ruff_ci.yml/badge.svg"/></a>
<a href="https://utiasDSL.github.io/crisp_controllers/"><img alt="Static Badge" src="https://img.shields.io/badge/docs-passing-blue?style=flat&link=https%3A%2F%2FutiasDSL.github.io%2Fcrisp_controllers%2F"></a>
<a href="https://github.com/utiasDSL/crisp_gym/actions/workflows/pixi_ci.yml"><img src="https://github.com/utiasDSL/crisp_gym/actions/workflows/pixi_ci.yml/badge.svg"/></a>
<a href="https://utiasDSL.github.io/crisp_controllers#citing"><img alt="Static Badge" src="https://img.shields.io/badge/arxiv-cite-b31b1b?style=flat"></a>
<img width="60" alt="lerobot-tag" src="https://github.com/user-attachments/assets/441b1d03-43d4-4cb9-bc08-ef56f119933a" />

This repository contains Gymnasium environments to train and deploy high-level learning-based policies from [LeRobot](https://github.com/huggingface/lerobot) using [CRISP_PY](https://github.com/utiasDSL/crisp_py) and the [CRISP controllers](https://github.com/utiasDSL/crisp_controllers).

Check the [docs](https://utiasdsl.github.io/crisp_controllers/getting_started/#4-using-the-gym) to get started.

## Mixed-version dual FR3

The production IRIS rig uses a Jazzy/protocol-10 leader and a
Humble/protocol-7 follower. Controller lifecycle stays on each RT PC;
`inference_pc` uses two isolated ROS contexts and exchanges only stamped
pose/twist/wrench messages.

Install the pinned Jazzy client environment:

```bash
pixi install -e jazzy
```

The FR3 entry point is fail-closed. Without `--arm` it checks stream freshness,
frame IDs, and sole command ownership, then exits without publishing a command:

```bash
pixi run -e jazzy python crisp_gym/scripts/fr3_bilateral_teleop.py \
  --scheme position \
  --leader-base-to-common-quat <x> <y> <z> <w> \
  --transforms-verified
```

When the tilted leader's yaw is still a candidate, use the dedicated physical
axis-check mode instead of falsely passing `--transforms-verified`. It accepts
only the position scheme, limits the mapped workspace to 10 mm and rotation to
0.01 rad, limits translation command steps to 0.25 mm, and stops automatically:

```bash
pixi run -e jazzy python crisp_gym/scripts/fr3_bilateral_teleop.py \
  --scheme position --arm --candidate-frame-check \
  --frame-check-duration-s 30 \
  --leader-base-to-common-quat <candidate-x> <candidate-y> <candidate-z> <candidate-w>
```

Move one leader axis only a few millimeters and stop immediately on an axis swap
or sign error. A successful run is evidence to record; it does not automatically
promote the transform or relax the normal arming interlock.

Arming also requires the leader quaternion and an explicit confirmation that
both physical base-axis checks passed:

```bash
pixi run -e jazzy python crisp_gym/scripts/fr3_bilateral_teleop.py \
  --scheme pf --arm --feedback-gain 0.1 --feedback-ramp-s 3.0 \
  --leader-base-to-common-quat <x> <y> <z> <w> \
  --transforms-verified
```

The initial PF profile is bounded to 2 N reflected force, disables reflected
torque, limits workspace and command increments, validates all state with a
monotonic freshness lease, and commands zero wrench/current-pose hold during
shutdown. The RT controllers independently expire stale commands after 0.1 s.

Do not use `crisp_gym/scripts/bilateral_teleop.py` for this rig. It is the
Panda-era path, homes both robots, and uses inference-side controller services;
it now requires the explicit `--legacy-unsafe` acknowledgement for existing
Panda users.

The full calibration and commissioning ladder lives in the matching
`iris-panda-ros2/docs/fr3_dual_calibration_pf_runbook.md` branch.
