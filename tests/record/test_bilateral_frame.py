"""Pure telemetry -> recorded-observation mapping (no rclpy).

The structured recorder folds each controller tick's ``TeleopTelemetry`` into extra
observation fields so the dataset captures what the controller actually commanded
(the DELAYED follower target) and reflected — making channel delay recoverable.
This mapping is pure numpy and is pinned here; the rclpy frame assembly around it
is exercised on hardware.
"""

import numpy as np

from crisp_gym.bilateral.controller import TeleopTelemetry
from crisp_gym.record.bilateral_frame import (
    BILATERAL_OBS_KEYS,
    bilateral_action,
    bilateral_feature_specs,
    telemetry_obs_fields,
)


def _tel(dof=6, step=3):
    return TeleopTelemetry(
        step=step,
        live_leader_pos=np.arange(dof, dtype=float),
        live_follower_pos=np.zeros(dof),
        commanded_follower_target=np.full(dof, 0.5),
        leader_feedforward=np.full(dof, 1.0),
        reflected_wrench=np.full(dof, 2.0),
        spring_force=np.full(dof, 3.0),
        forward_force=np.full(dof, 4.0),
        follower_wrench=np.full(dof, 5.0),
        leader_wrench=np.full(dof, 6.0),
        delay_steps=7,
        control_dt=1e-2,
    )


def test_telemetry_obs_fields_keys_and_values():
    fields = telemetry_obs_fields(_tel(dof=6))
    assert set(fields) == set(BILATERAL_OBS_KEYS)
    assert np.allclose(fields["observation.follower_target"], 0.5)
    assert np.allclose(fields["observation.reflected_wrench"], 2.0)
    assert np.allclose(fields["observation.spring_force"], 3.0)
    assert np.allclose(fields["observation.forward_force"], 4.0)
    assert np.allclose(fields["observation.leader_feedforward"], 1.0)
    for v in fields.values():
        assert v.dtype == np.float32


def test_feature_specs_shapes_match_dof():
    specs = bilateral_feature_specs(6)
    assert set(specs) == set(BILATERAL_OBS_KEYS)
    for k, spec in specs.items():
        assert spec["shape"] == (6,)
        assert spec["dtype"] == "float32"
        assert len(spec["names"]) == 6


def test_feature_specs_joint_dof():
    specs = bilateral_feature_specs(7)
    assert specs["observation.follower_target"]["shape"] == (7,)


def test_action_is_commanded_target_plus_gripper():
    tel = _tel(dof=6)
    action = bilateral_action(tel, gripper_action=0.9)
    assert action.shape == (7,)
    assert np.allclose(action[:6], 0.5)
    assert action[6] == np.float32(0.9)
    assert action.dtype == np.float32
