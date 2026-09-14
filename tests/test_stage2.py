from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from full_vectorial_waveguide_inverse_synthesis import (
    grad_x,
    grad_x_transpose,
    grad_y,
    grad_y_transpose,
    laplacian,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "gaussian_n110" / "gaussian_arrays.npz"


def test_cli_help_exposes_stage2() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "stage2_verify.py"), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Stage-2 sparse shift-invert eigenmode verification" in completed.stdout


def test_example_archive_contract() -> None:
    with np.load(EXAMPLE) as data:
        assert "u" in data
        assert "It" in data
        assert data["u"].shape == (110, 110)
        assert data["It"].shape == data["u"].shape


def test_laplacian_detects_grid_scale_oscillation() -> None:
    n = 24
    dx = 0.1
    yy, xx = np.indices((n, n))
    checkerboard = (-1.0) ** (xx + yy)
    response = laplacian(checkerboard, dx, dx)[2:-2, 2:-2]
    assert np.min(np.abs(response)) > 100.0


def test_first_derivative_transposes() -> None:
    rng = np.random.default_rng(7)
    a = rng.normal(size=(19, 19))
    b = rng.normal(size=(19, 19))
    dx = 0.07
    assert np.vdot(grad_x(a, dx), b) == pytest.approx(
        np.vdot(a, grad_x_transpose(b, dx)), rel=1e-12, abs=1e-12)
    assert np.vdot(grad_y(a, dx), b) == pytest.approx(
        np.vdot(a, grad_y_transpose(b, dx)), rel=1e-12, abs=1e-12)


@pytest.mark.slow
def test_gaussian_n110_stage2_regression(tmp_path: Path) -> None:
    prefix = tmp_path / "gaussian_n110"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "stage2_verify.py"),
            str(EXAMPLE),
            "--out-prefix",
            str(prefix),
        ],
        cwd=ROOT,
        check=True,
        timeout=240,
    )

    metrics = json.loads(
        Path(str(prefix) + "_metrics.json").read_text(encoding="utf-8"))
    assert metrics["n_eff"] == pytest.approx(1.5738488180, rel=2e-5)
    assert metrics["eps_I"] == pytest.approx(0.0163939733, rel=2e-3)
    assert metrics["SSIM"] == pytest.approx(0.9942304901, abs=5e-4)
    assert 0.5 <= metrics["polarization_purity"] <= 1.0

    with np.load(Path(str(prefix) + "_fields.npz")) as fields:
        assert fields["Ho_x"].shape == (110, 110)
        assert fields["Ho_y"].shape == (110, 110)
        assert np.all(np.isfinite(fields["Ho_x"]))
        assert np.all(np.isfinite(fields["Ho_y"]))

