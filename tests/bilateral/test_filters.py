"""Reflected-wrench conditioning: DC-removing high-pass + soft force deadband.

These tame the bias-driven divergence of the free-float leader (k_task=0): the
high-pass strips the slow F/T bias DC (~1.25 N pose-dependent residual) so only
contact transients reflect; the deadband nulls the residual free-space noise floor.
Both are opt-in (cutoff/threshold = 0 -> pass-through).
"""

import numpy as np

from crisp_gym.bilateral.filters import FirstOrderHighPass, soft_deadband


def test_highpass_disabled_is_passthrough():
    hp = FirstOrderHighPass(cutoff_hz=0.0, dof=3, dt=0.01)
    for x in ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]):
        assert np.allclose(hp.step(x), x)


def test_highpass_removes_constant_dc():
    hp = FirstOrderHighPass(cutoff_hz=1.0, dof=3, dt=0.01)
    c = np.array([1.0, -0.5, 0.3])
    out = np.zeros(3)
    for _ in range(2000):
        out = hp.step(c)
    assert np.linalg.norm(out) < 0.05 * np.linalg.norm(c)  # DC decayed away


def test_highpass_passes_transient():
    hp = FirstOrderHighPass(cutoff_hz=1.0, dof=1, dt=0.01)
    for _ in range(500):
        hp.step([0.0])  # settle at zero
    out = hp.step([10.0])  # sudden jump
    assert out[0] > 9.0  # transient passes ~unattenuated


def test_soft_deadband_zeros_below_threshold():
    assert np.allclose(soft_deadband([0.5, 0.0, 0.0], 2.0), [0.0, 0.0, 0.0])
    assert np.allclose(soft_deadband([0.0, 0.0, 0.0], 2.0), [0.0, 0.0, 0.0])  # no div-by-zero


def test_soft_deadband_subtracts_threshold_keeps_direction():
    assert np.allclose(soft_deadband([3.0, 0.0, 0.0], 2.0), [1.0, 0.0, 0.0])  # |3|-2 along +x
    assert np.allclose(soft_deadband([0.0, 5.0, 0.0], 2.0), [0.0, 3.0, 0.0])  # |5|-2 along +y


def test_soft_deadband_continuous_at_threshold():
    assert np.allclose(soft_deadband([2.0, 0.0, 0.0], 2.0), [0.0, 0.0, 0.0])


def test_soft_deadband_disabled_passthrough():
    assert np.allclose(soft_deadband([0.5, 0.1, 0.0], 0.0), [0.5, 0.1, 0.0])
