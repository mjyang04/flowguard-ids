"""Unit tests for Platt scaling (nids.evaluation.calibration)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nids.evaluation.calibration import (
    CalibrationResult,
    PlattCalibrator,
    _expected_calibration_error,
)


def _make_synthetic_logits(
    n: int = 2000, seed: int = 0, shift: float = 1.5, scale: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    """Generate miscalibrated logits with a controllable shift/scale.

    Ground-truth logit = `raw`; the reported model logit is `scale * raw +
    shift`, which biases sigmoid probabilities systematically. A correct
    Platt fit should roughly invert (scale, shift).
    """
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=n)
    # Ground-truth labels are drawn from the true sigmoid.
    true_probs = 1.0 / (1.0 + np.exp(-raw))
    labels = (rng.uniform(size=n) < true_probs).astype(np.int64)
    miscalibrated_logits = scale * raw + shift
    return miscalibrated_logits.astype(np.float64), labels


def test_platt_calibrator_reduces_ece_on_biased_logits():
    logits, labels = _make_synthetic_logits(n=4000, seed=1, shift=2.0, scale=0.5)

    pre_probs = 1.0 / (1.0 + np.exp(-logits))
    ece_raw = _expected_calibration_error(labels, pre_probs)

    calibrator = PlattCalibrator()
    result = calibrator.fit(logits, labels)

    assert isinstance(result, CalibrationResult)
    # Platt should reduce ECE or at worst keep it equal on biased logits.
    assert result.val_ece_after <= result.val_ece_before + 1e-6
    assert result.val_ece_before == pytest.approx(ece_raw, rel=1e-5)


def test_platt_calibrator_is_monotone_on_logits():
    logits, labels = _make_synthetic_logits(n=1500, seed=2)
    calibrator = PlattCalibrator()
    calibrator.fit(logits, labels)

    grid = np.linspace(logits.min(), logits.max(), 200)
    probs = calibrator.transform(grid)
    # A positive A corresponds to a monotone-increasing map; a negative A
    # means the model's logit is anti-correlated with labels (rare on real
    # data but Platt still yields a monotone decreasing map). Either way
    # the transform must be strictly monotone.
    diffs = np.diff(probs)
    assert np.all(diffs >= -1e-9) or np.all(diffs <= 1e-9)


def test_platt_transform_preserves_rank_order():
    logits = np.array([-3.0, -1.0, 0.5, 1.0, 2.5], dtype=np.float64)
    calibrator = PlattCalibrator()
    # Force known positive slope so ranks must be preserved.
    calibrator.A = 0.8
    calibrator.B = -0.2
    calibrator._fitted = True

    probs = calibrator.transform(logits)
    assert list(np.argsort(logits)) == list(np.argsort(probs))


def test_platt_transform_before_fit_raises():
    calibrator = PlattCalibrator()
    with pytest.raises(RuntimeError):
        calibrator.transform(np.array([0.1, -0.2]))


def test_platt_save_and_load_round_trip(tmp_path: Path):
    logits, labels = _make_synthetic_logits(n=500, seed=3)
    calibrator = PlattCalibrator()
    calibrator.fit(logits, labels)
    A_expected, B_expected = calibrator.A, calibrator.B

    path = tmp_path / "platt.npz"
    calibrator.save(path)

    restored = PlattCalibrator()
    restored.load(path)
    assert restored.A == pytest.approx(A_expected)
    assert restored.B == pytest.approx(B_expected)

    probs_orig = calibrator.transform(logits)
    probs_restored = restored.transform(logits)
    np.testing.assert_allclose(probs_orig, probs_restored, atol=1e-12)


def test_platt_handles_single_class_val_gracefully():
    # All-positive validation set — NLL gradient still well-defined but the
    # fit may not meaningfully reduce ECE. We only require that the call
    # returns without crashing and yields a valid CalibrationResult.
    logits = np.linspace(-2.0, 2.0, 100)
    labels = np.ones_like(logits, dtype=np.int64)
    calibrator = PlattCalibrator()
    result = calibrator.fit(logits, labels, max_iter=200)
    assert isinstance(result, CalibrationResult)
    assert np.isfinite(calibrator.A) and np.isfinite(calibrator.B)


def test_expected_calibration_error_perfect_calibration_is_low():
    rng = np.random.default_rng(7)
    # Draw labels from exactly the predicted probabilities.
    probs = rng.uniform(size=5000)
    labels = (rng.uniform(size=5000) < probs).astype(np.int64)
    ece = _expected_calibration_error(labels, probs, n_bins=20)
    assert ece < 0.05
