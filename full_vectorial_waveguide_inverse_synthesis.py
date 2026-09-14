"""Full-vectorial prescribed-field inverse synthesis for waveguide cross sections.

Reference implementation accompanying the JSTQE manuscript

  "Full-Vectorial Prescribed-Field Inverse Synthesis of Optical Waveguides:
   A Classical Gradient-Based Approach"
  Manuscript ID JSTQE-CON-IDPAUQC2027-10493-2026

Running this script with its default settings reproduces the primary
Rayleigh-normalized rows of Table I of the manuscript to the precision
printed there.  The scalar, semi-vectorial, and lagged full-vectorial
baselines can be regenerated explicitly with their corresponding CLI flags.

Pipeline
--------
Stage 1  Build the prescribed TE amplitude target from an intensity map and
         minimize either the Rayleigh-normalized or lagged-normalizer
         full-vectorial operator residual with Adam under total-variation
         regularization and projected box constraints.
Stage 2  Assemble the sparse full-vectorial operator, extract its fundamental
         mode by shift-invert power iteration, and report the verified
         intensity metric of Eq. (13) and integrated-power polarization
         purity of Eq. (19).

Numerics
--------
* Fourth-order finite differences for both first and second derivatives, as in
  the manuscript.  A first-derivative operator applied twice is NOT used for
  the Laplacian: that produces a doubled-spacing stencil whose null space
  contains grid-scale oscillations, which admits spurious eigenmodes.
* Homogeneous Dirichlet boundary condition on the outermost two layers.  The
  target, normalization scale, data objective, residual metrics, and Adam
  update all use the same active interior domain. Objective gradients use the
  exact transpose of the implemented finite-difference operators, including
  their boundary closures (see --check-gradient).
* Stage 2 uses sparse inverse iteration about
  sigma = (0.98 n_max k0)^2 and returns the eigenpair nearest that shift.
  Independent multi-mode spectral scans establish the reported top-mode pair.
  SciPy is required for Stage 2; Stage 1 runs on NumPy alone.

Metric conventions follow the manuscript equations exactly; P_a denotes the
active-domain projector and both intensities are peak-normalized for eps_I:
  NFR   = ||P_a(H_o^N - H_t^N)|| / ||P_a H_t^N||                Eq. (12)
  eps_I = ||P_a(I_mode - I_t)|| / ||P_a I_t||                   Eq. (13)
  E_x   = ||P_a(q_x/M-h)||/||P_a h||, E_y=||P_a q_y/M||/||P_a h||
                                                                  Eq. (14)
  projected-gradient stationarity diagnostic                      Eq. (16)
  R_common, the common post-hoc full-vectorial Rayleigh residual   Eq. (18)
  integrated-power polarization purity                             Eq. (19)
  mode overlap in the known-rib benchmark                          Eq. (20)
  SSIM  is the windowed structural similarity of Wang et al. (7x7 uniform
        window), matching the scikit-image defaults used for the manuscript.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

Array = np.ndarray

# --------------------------------------------------------------------------
# Manuscript defaults (Section III, "Implementation details")
# --------------------------------------------------------------------------
MS_GRID = 110
MS_HALF_WIDTH = 3.0          # domain is [-3, 3] um in x and y
MS_WAVELENGTH = 1.15         # um
MS_N_MIN = 1.0
MS_N_MAX = 3.44
MS_ITERATIONS = 300
MS_LEARNING_RATE = 0.02
MS_LAMBDA_TV = 1e-6
MS_LAMBDA_BOUND = 0.0
MS_TV_DELTA = 1e-6
MS_TOL = 5e-3
MS_MIN_ITERATIONS = 80
MS_PLATEAU_WINDOW = 20
MS_PLATEAU_REL = 5e-4
MS_DELTA_U_TOL = 1e-4
MS_GPROJ_TOL = 5e-3
MS_SEED = 0

# Reference values for the primary Rayleigh-normalized rows of Table I.
TABLE_I_RAYLEIGH_NORMALIZED = {
    "gaussian":   {"NFR": 1.4410049e-2, "eps_I": 1.6393973e-2, "SSIM": 0.994, "n_eff": 1.5738488},
    "sun_like":   {"NFR": 4.3195352e-2, "eps_I": 4.0494186e-2, "SSIM": 0.956, "n_eff": 1.6257454},
    "ultra_flat": {"NFR": 5.5855307e-2, "eps_I": 1.2152283e-1, "SSIM": 0.930, "n_eff": 1.6991285},
}


@dataclass
class RunConfig:
    target: str
    input_npz: str | None
    method: str
    normalizer: str
    grid: int
    half_width: float
    wavelength: float
    n_min: float
    n_max: float
    iterations: int
    learning_rate: float
    lambda_tv: float
    lambda_bound: float
    tv_delta: float
    tol: float
    plateau_window: int
    plateau_rel: float
    min_iterations: int
    init: str
    seed: int
    verify: bool


# ==========================================================================
# Fourth-order finite differences
# ==========================================================================
def grad_x(a: Array, dx: float) -> Array:
    """Fourth-order interior first derivative along x (axis=1)."""
    g = np.zeros_like(a)
    g[:, 2:-2] = (-a[:, 4:] + 8.0 * a[:, 3:-1] - 8.0 * a[:, 1:-3] + a[:, :-4]) / (12.0 * dx)
    g[:, 1] = (a[:, 2] - a[:, 0]) / (2.0 * dx)
    g[:, -2] = (a[:, -1] - a[:, -3]) / (2.0 * dx)
    return g


def grad_y(a: Array, dy: float) -> Array:
    """Fourth-order interior first derivative along y (axis=0)."""
    g = np.zeros_like(a)
    g[2:-2, :] = (-a[4:, :] + 8.0 * a[3:-1, :] - 8.0 * a[1:-3, :] + a[:-4, :]) / (12.0 * dy)
    g[1, :] = (a[2, :] - a[0, :]) / (2.0 * dy)
    g[-2, :] = (a[-1, :] - a[-3, :]) / (2.0 * dy)
    return g


def grad_x_transpose(a: Array, dx: float) -> Array:
    """Euclidean transpose of :func:`grad_x` for the implemented stencil."""
    g = np.zeros_like(a)
    w = a[:, 2:-2] / (12.0 * dx)
    g[:, 4:] -= w
    g[:, 3:-1] += 8.0 * w
    g[:, 1:-3] -= 8.0 * w
    g[:, :-4] += w
    g[:, 2] += a[:, 1] / (2.0 * dx)
    g[:, 0] -= a[:, 1] / (2.0 * dx)
    g[:, -1] += a[:, -2] / (2.0 * dx)
    g[:, -3] -= a[:, -2] / (2.0 * dx)
    return g


def grad_y_transpose(a: Array, dy: float) -> Array:
    """Euclidean transpose of :func:`grad_y` for the implemented stencil."""
    g = np.zeros_like(a)
    w = a[2:-2, :] / (12.0 * dy)
    g[4:, :] -= w
    g[3:-1, :] += 8.0 * w
    g[1:-3, :] -= 8.0 * w
    g[:-4, :] += w
    g[2, :] += a[1, :] / (2.0 * dy)
    g[0, :] -= a[1, :] / (2.0 * dy)
    g[-1, :] += a[-2, :] / (2.0 * dy)
    g[-3, :] -= a[-2, :] / (2.0 * dy)
    return g


def laplacian(a: Array, dx: float, dy: float) -> Array:
    """Fourth-order five-point-per-axis Laplacian.

    This is a genuine second-derivative stencil.  Applying a first-derivative
    operator twice would instead give (a[i+2] - 2 a[i] + a[i-2]) / (4 h^2),
    which has four times the truncation error and is blind to grid-scale
    oscillation -- a common source of spurious modes.
    """
    out = np.zeros_like(a)
    out[:, 2:-2] += (
        -a[:, 4:] + 16.0 * a[:, 3:-1] - 30.0 * a[:, 2:-2] + 16.0 * a[:, 1:-3] - a[:, :-4]
    ) / (12.0 * dx * dx)
    out[2:-2, :] += (
        -a[4:, :] + 16.0 * a[3:-1, :] - 30.0 * a[2:-2, :] + 16.0 * a[1:-3, :] - a[:-4, :]
    ) / (12.0 * dy * dy)
    return out


def interior_mask(n: int) -> Array:
    """True on nodes where the update is applied (Dirichlet on two outer layers)."""
    m = np.ones((n, n), dtype=float)
    m[:2, :] = m[-2:, :] = m[:, :2] = m[:, -2:] = 0.0
    return m


def active_target_intensity(it: Array) -> Array:
    """Peak-normalized target restricted to the common Dirichlet domain."""
    normalized = normalize_intensity(it)
    active = normalized * interior_mask(normalized.shape[0])
    return normalize_intensity(active)


# ==========================================================================
# Grid and targets (manuscript definitions)
# ==========================================================================
def make_grid(n: int, half_width: float) -> tuple[Array, Array, float]:
    x = np.linspace(-half_width, half_width, n)
    xx, yy = np.meshgrid(x, x, indexing="xy")
    return xx, yy, float(x[1] - x[0])


def normalize_intensity(it: Array) -> Array:
    it = np.maximum(np.real(np.asarray(it, dtype=float)), 0.0)
    peak = float(np.max(it))
    if peak <= 0.0:
        raise ValueError("Target intensity has zero peak.")
    return it / peak


def synthetic_target(name: str, n: int, half_width: float) -> Array:
    """The three analytic targets of Section III-A."""
    xx, yy, _ = make_grid(n, half_width)

    if name == "gaussian":
        sigma = 0.6
        it = np.exp(-(xx**2 / (2.0 * sigma**2) + yy**2 / (2.0 * sigma**2)))

    elif name == "sun_like":
        sx, sy = 0.8, 0.3

        def rotated(angle_deg: float) -> Array:
            t = math.radians(angle_deg)
            c, s = math.cos(t), math.sin(t)
            xr = c * xx - s * yy
            yr = s * xx + c * yy
            return np.exp(-(xr**2 / (2.0 * sx**2) + yr**2 / (2.0 * sy**2)))

        horizontal = np.exp(-(xx**2 / (2.0 * sx**2) + yy**2 / (2.0 * sy**2)))
        vertical = np.exp(-(xx**2 / (2.0 * sy**2) + yy**2 / (2.0 * sx**2)))
        centre = np.exp(-(xx**2 / (2.0 * sy**2) + yy**2 / (2.0 * sy**2)))
        it = horizontal + vertical + rotated(-45.0) + rotated(45.0) - 2.5 * centre

    elif name == "ultra_flat":
        r = np.sqrt(xx**2 + yy**2)
        it = 1.0 / (1.0 + (r / 1.2) ** 8)

    else:
        raise ValueError(f"Unknown synthetic target: {name}")

    return active_target_intensity(np.abs(it))


def target_field_from_intensity(it: Array) -> tuple[Array, Array]:
    """Prescribed pure-TE ansatz H_t = (sqrt(I_t), 0), peak-normalized.

    Note the compatibility limitation discussed in Section II-A: for a
    localized target this field is not in general an exact eigenvector of the
    coupled operator, so the attainable residual is bounded away from zero.
    """
    amplitude = np.sqrt(active_target_intensity(it))
    peak = float(np.max(amplitude))
    return amplitude / peak, np.zeros_like(amplitude)


# ==========================================================================
# Operators
# ==========================================================================
def vector_operator_output(u: Array, hx: Array, hy: Array, k0: float, dx: float) -> tuple[Array, Array]:
    """q = L_v(u) H for the transverse magnetic-field components (Eqs. 1-3)."""
    u_safe = np.maximum(u, 1e-12)
    ax = grad_x(u_safe, dx) / u_safe          # d_x ln u
    ay = grad_y(u_safe, dx) / u_safe          # d_y ln u
    curl = grad_x(hy, dx) - grad_y(hx, dx)    # K = d_x H_y - d_y H_x
    qx = laplacian(hx, dx, dx) + (k0**2) * u_safe * hx + ay * curl
    qy = laplacian(hy, dx, dx) + (k0**2) * u_safe * hy - ax * curl
    return qx, qy


def scalar_operator_output(u: Array, h: Array, k0: float, dx: float) -> Array:
    """Scalar Helmholtz operator, for the baseline comparison of Section III-D."""
    return laplacian(h, dx, dx) + (k0**2) * np.maximum(u, 1e-12) * h


def semivectorial_operator_output(u: Array, h: Array, k0: float, dx: float) -> Array:
    """Semi-vectorial (x-dominant) operator, for the comparison of Section III-D."""
    u_safe = np.maximum(u, 1e-12)
    ax = grad_x(u_safe, dx) / u_safe
    return laplacian(h, dx, dx) + (k0**2) * u_safe * h + ax * grad_x(h, dx)


def scalar_operator_vjp_u(u: Array, h: Array, weight: Array,
                          k0: float, dx: float) -> Array:
    """Exact VJP ``(d q_scalar / d u)^T weight``."""
    del u, dx
    return (k0**2) * weight * h


def semivectorial_operator_vjp_u(u: Array, h: Array, weight: Array,
                                 k0: float, dx: float) -> Array:
    """Exact discrete VJP of :func:`semivectorial_operator_output`.

    For ``q = lap(h) + k0^2 u h + (D_x u/u) D_x h``, the coupling
    variation is

        (D_x du/u - (D_x u) du/u^2) D_x h.

    The first term must be accumulated with the true transpose of the
    implemented derivative stencil; replacing ``D_x^T`` by ``-D_x`` is not
    exact at the boundary-adjacent closures.
    """
    u_safe = np.maximum(u, 1e-12)
    du_x = grad_x(u_safe, dx)
    dh_x = grad_x(h, dx)
    coupled_weight = weight * dh_x
    grad = (k0**2) * weight * h
    grad += grad_x_transpose(coupled_weight / u_safe, dx)
    grad -= coupled_weight * du_x / (u_safe**2)
    return grad


# ==========================================================================
# Residual and metrics (manuscript conventions)
# ==========================================================================
def field_residual(u: Array, hx: Array, hy: Array, k0: float, dx: float) -> dict[str, Any]:
    """Relative lagged-normalizer residual on the active interior domain."""
    qx, qy = vector_operator_output(u, hx, hy, k0, dx)
    active = interior_mask(u.shape[0])
    hx_active, hy_active = active * hx, active * hy
    qx_active, qy_active = active * qx, active * qy
    m_lagged = float(np.max(np.sqrt(qx_active**2 + qy_active**2)))
    if m_lagged <= 0.0:
        raise ValueError("Lagged normalizer is zero.")
    rx = active * (qx / m_lagged - hx_active)
    ry = active * (qy / m_lagged - hy_active)
    ht_sq = float(np.sum(hx_active**2 + hy_active**2))
    nfr = float(np.sqrt(np.sum(rx**2 + ry**2) / max(ht_sq, 1e-30)))
    h_sq = float(np.sum(hx_active**2))
    return {
        "qx": qx, "qy": qy, "rx": rx, "ry": ry, "M": m_lagged, "NFR": nfr,
        "F": float(np.sum(rx**2 + ry**2) / max(ht_sq, 1e-30)),
        "ht_sq": ht_sq,
        "E_x": float(np.sqrt(np.sum(rx**2) / max(h_sq, 1e-30))),
        "E_y": float(np.sqrt(np.sum(ry**2) / max(h_sq, 1e-30))),
    }


def scalar_or_semi_residual(u: Array, h: Array, k0: float, dx: float,
                            method: str) -> dict[str, Any]:
    """Relative frozen-max residual on the active interior domain."""
    if method == "scalar":
        q = scalar_operator_output(u, h, k0, dx)
    elif method == "semi":
        q = semivectorial_operator_output(u, h, k0, dx)
    else:
        raise ValueError(f"Unsupported baseline method: {method}")
    active = interior_mask(u.shape[0])
    h_active, q_active = active * h, active * q
    m_lagged = float(np.max(np.abs(q_active)))
    if m_lagged <= 0.0:
        raise ValueError("Lagged normalizer is zero.")
    r = active * (q / m_lagged - h_active)
    h_sq = max(float(np.sum(h_active**2)), 1e-30)
    return {
        "q": q, "r": r, "M": m_lagged,
        "F": float(np.sum(r**2) / h_sq),
        "h_sq": h_sq,
        "NFR": float(np.sqrt(np.sum(r**2) / h_sq)),
        "E_x": float(np.sqrt(np.sum(r**2) / h_sq)),
        "E_y": 0.0,
    }


def rayleigh_residual(u: Array, hx: Array, hy: Array, k0: float, dx: float) -> dict[str, Any]:
    """Rayleigh-normalized scale-invariant residual.

    With q(u) = L_v(u) H_t the eigenvalue is eliminated analytically by

        alpha(u) = <H_t, q(u)> / <H_t, H_t>,

    and the objective is the scale-free relative residual

        F(u) = || q/alpha - H_t ||^2 / || H_t ||^2 .

    Writing g = q/alpha - H_t and s = ||H_t||^2, differentiation of alpha
    contributes the ``-F H_t`` term:

        dF = (2 / (alpha ||H_t||^2)) < g - F H_t , dq > ,

    The implementation below applies the exact transpose of the discrete
    fourth-order operator Jacobian, rather than substituting a continuum
    integration-by-parts identity at the grid level.
    """
    qx, qy = vector_operator_output(u, hx, hy, k0, dx)
    active = interior_mask(u.shape[0])
    hx_active, hy_active = active * hx, active * hy
    qx_active, qy_active = active * qx, active * qy
    ht_sq = float(np.sum(hx_active**2 + hy_active**2))
    alpha = float(np.sum(hx_active * qx_active + hy_active * qy_active)
                  / max(ht_sq, 1e-30))
    if abs(alpha) < 1e-30:
        raise ValueError("Rayleigh quotient is zero; H_t is orthogonal to L_v H_t.")
    gx_ = active * (qx / alpha - hx_active)
    gy_ = active * (qy / alpha - hy_active)
    objective = float(np.sum(gx_**2 + gy_**2) / max(ht_sq, 1e-30))
    return {
        "qx": qx, "qy": qy, "gx": gx_, "gy": gy_,
        "alpha": alpha, "F": objective,
        "NFR": math.sqrt(max(objective, 0.0)),         # same units as Eq. (12)
        "n_eff_estimate": math.sqrt(abs(alpha)) / k0,
        "ht_sq": ht_sq,
        "E_x": float(np.sqrt(np.sum(gx_**2) / max(float(np.sum(hx_active**2)), 1e-30))),
        "E_y": float(np.sqrt(np.sum(gy_**2) / max(float(np.sum(hx_active**2)), 1e-30))),
    }


def vector_operator_vjp_u(u: Array, hx: Array, hy: Array,
                          wx: Array, wy: Array, k0: float, dx: float) -> Array:
    """Exact discrete VJP ``(d q / d u)^T (wx, wy)`` for the code stencil."""
    u_safe = np.maximum(u, 1e-12)
    ax = grad_x(u_safe, dx) / u_safe
    ay = grad_y(u_safe, dx) / u_safe
    curl_t = grad_x(hy, dx) - grad_y(hx, dx)
    grad = (k0**2) * (wx * hx + wy * hy)
    grad += grad_y_transpose(wx * curl_t / u_safe, dx)
    grad -= wx * curl_t * ay / u_safe
    grad += grad_x_transpose(-wy * curl_t / u_safe, dx)
    grad += wy * curl_t * ax / u_safe
    return grad


def rayleigh_grad_u(u: Array, hx: Array, hy: Array, residual: dict[str, Any],
                    k0: float, dx: float) -> Array:
    """Exact discrete gradient of the Rayleigh-normalized objective."""
    gx_, gy_ = residual["gx"], residual["gy"]
    alpha, objective, ht_sq = residual["alpha"], residual["F"], residual["ht_sq"]
    active = interior_mask(u.shape[0])
    rx = gx_ - objective * active * hx
    ry = gy_ - objective * active * hy
    scale = 2.0 / (alpha * ht_sq)
    return vector_operator_vjp_u(
        u, hx, hy, scale * rx, scale * ry, k0, dx)


def epsilon_I(modal_intensity: Array, it: Array) -> float:
    """Verified-mode intensity error of Eq. (13); both maps peak-normalized."""
    a = active_target_intensity(modal_intensity)
    b = active_target_intensity(it)
    return float(np.linalg.norm(a - b) / max(float(np.linalg.norm(b)), 1e-30))


def _uniform_filter(a: Array, size: int) -> Array:
    """Separable running-mean filter (equivalent to scipy.ndimage.uniform_filter
    with mode='reflect'), NumPy only."""
    pad = size // 2
    out = np.pad(a, pad, mode="reflect")
    csum = np.cumsum(out, axis=0)
    csum = np.vstack([np.zeros((1, csum.shape[1])), csum])
    out = (csum[size:, :] - csum[:-size, :]) / size
    csum = np.cumsum(out, axis=1)
    csum = np.hstack([np.zeros((csum.shape[0], 1)), csum])
    return (csum[:, size:] - csum[:, :-size]) / size


def windowed_ssim(a: Array, b: Array, data_range: float = 1.0, win_size: int = 7) -> float:
    """Windowed SSIM (Wang et al. 2004), matching the scikit-image defaults used
    for the manuscript: uniform win_size x win_size window, sample covariance,
    border cropped by (win_size - 1) // 2."""
    a = active_target_intensity(a)
    b = active_target_intensity(b)
    npix = win_size**2
    cov_norm = npix / (npix - 1.0)                       # unbiased covariance
    ua = _uniform_filter(a, win_size)
    ub = _uniform_filter(b, win_size)
    uaa = _uniform_filter(a * a, win_size)
    ubb = _uniform_filter(b * b, win_size)
    uab = _uniform_filter(a * b, win_size)
    vaa = cov_norm * (uaa - ua * ua)
    vbb = cov_norm * (ubb - ub * ub)
    vab = cov_norm * (uab - ua * ub)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = (((2.0 * ua * ub + c1) * (2.0 * vab + c2))
                / ((ua**2 + ub**2 + c1) * (vaa + vbb + c2)))
    pad = (win_size - 1) // 2
    return float(np.mean(ssim_map[pad:-pad, pad:-pad]))


def polarization_purity(hx: Array, hy: Array) -> tuple[float, str]:
    """Integrated-power polarization purity of Eq. (19)."""
    px = float(np.sum(np.abs(hx) ** 2))
    py = float(np.sum(np.abs(hy) ** 2))
    total = px + py
    if total <= 0.0:
        return 0.0, "undefined"
    return max(px, py) / total, ("x" if px >= py else "y")


# ==========================================================================
# Regularizers and gradient
# ==========================================================================
def tv_value_n(n_map: Array, delta: float, dx: float) -> float:
    """Discrete smoothed isotropic TV functional used by the optimizer."""
    gx = grad_x(n_map, dx)
    gy = grad_y(n_map, dx)
    return float(np.sum(np.sqrt(gx**2 + gy**2 + delta)))


def tv_grad_n(n_map: Array, delta: float, dx: float) -> Array:
    """Exact discrete ``d TV / d n`` for the implemented stencils."""
    gx = grad_x(n_map, dx)
    gy = grad_y(n_map, dx)
    denom = np.sqrt(gx**2 + gy**2 + delta)
    return (grad_x_transpose(gx / denom, dx)
            + grad_y_transpose(gy / denom, dx))


def bound_grad_n(n_map: Array, n_min: float, n_max: float) -> Array:
    """d B / d n for the box penalty."""
    return -2.0 * np.maximum(0.0, n_min - n_map) + 2.0 * np.maximum(0.0, n_map - n_max)


def bound_value_n(n_map: Array, n_min: float, n_max: float) -> float:
    """Quadratic bound-penalty value (zero in the disclosed production run)."""
    return float(np.sum(np.maximum(0.0, n_min - n_map) ** 2
                        + np.maximum(0.0, n_map - n_max) ** 2))


def regularizer_value_u(u: Array, lambda_tv: float, lambda_bound: float,
                        delta: float, n_min: float, n_max: float,
                        dx: float) -> float:
    """Weighted regularizer value for the actual total objective."""
    n_map = np.sqrt(np.maximum(u, 1e-12))
    return (lambda_tv * tv_value_n(n_map, delta, dx)
            + lambda_bound * bound_value_n(n_map, n_min, n_max))


def regularizer_grad_u(u: Array, lambda_tv: float, lambda_bound: float,
                       delta: float, n_min: float, n_max: float, dx: float) -> Array:
    """Chain rule to u = n^2, applied once to the weighted sum.

    The weighted sum is divided by (2 n + 1e-12) as a single operation; doing
    the division term by term is algebraically identical but not identical in
    floating point, and the optimization trajectory is sensitive enough for
    that to matter (see the note on reproducibility in the README).
    """
    n_map = np.sqrt(np.maximum(u, 1e-12))
    return ((lambda_tv * tv_grad_n(n_map, delta, dx)
             + lambda_bound * bound_grad_n(n_map, n_min, n_max))
            / (2.0 * n_map + 1e-12))


def master_formula_grad_u(u: Array, hx: Array, hy: Array, residual: dict[str, Any],
                          k0: float, dx: float) -> Array:
    """Exact discrete gradient of the frozen-normalizer surrogate."""
    rx, ry, m_lagged = residual["rx"], residual["ry"], residual["M"]
    ht_sq = residual["ht_sq"]
    return vector_operator_vjp_u(
        u, hx, hy, (2.0 / (m_lagged * ht_sq)) * rx,
        (2.0 / (m_lagged * ht_sq)) * ry, k0, dx)


def scalar_or_semi_grad_u(u: Array, h: Array, residual: dict[str, Any],
                          method: str, k0: float, dx: float) -> Array:
    """Exact frozen-normalizer gradient for a scalar/semi baseline."""
    weight = (2.0 / (residual["M"] * residual["h_sq"])) * residual["r"]
    if method == "scalar":
        return scalar_operator_vjp_u(u, h, weight, k0, dx)
    if method == "semi":
        return semivectorial_operator_vjp_u(u, h, weight, k0, dx)
    raise ValueError(f"Unsupported baseline method: {method}")


def projected_gradient_norm(u: Array, grad: Array, eta: float,
                            n_min: float, n_max: float) -> float:
    """``||P_box(u - eta g) - u|| / eta`` gradient-mapping norm."""
    projected = np.clip(u - eta * grad, n_min**2, n_max**2)
    return float(np.linalg.norm(projected - u) / max(eta, 1e-30))


# ==========================================================================
# Initialization
# ==========================================================================
def initialize_u(it: Array, n_min: float, n_max: float, mode: str, seed: int) -> Array:
    rng = np.random.default_rng(seed)
    u_min = n_min**2
    u_max = n_max**2
    if mode == "target":                       # manuscript default
        return u_min + (u_max - u_min) * 0.3 * normalize_intensity(it)
    if mode == "homogeneous":
        n0 = float(np.clip(1.5, n_min, n_max))
        return np.full_like(it, n0**2, dtype=float)
    if mode == "random":
        return u_min + (u_max - u_min) * 0.5 * rng.random(it.shape)
    raise ValueError(f"Unknown initialization mode: {mode}")


# ========================================================================== 
# Stage 1
# ========================================================================== 
def stage1_state(cfg: RunConfig, u: Array, hx: Array, hy: Array) -> dict[str, Any]:
    """Evaluate the actual Stage-1 objective, exact gradient, and diagnostics."""
    n = u.shape[0]
    _, _, dx = make_grid(n, cfg.half_width)
    k0 = 2.0 * math.pi / cfg.wavelength

    if cfg.method == "full":
        if cfg.normalizer == "rayleigh":
            residual = rayleigh_residual(u, hx, hy, k0, dx)
            grad_data = rayleigh_grad_u(u, hx, hy, residual, k0, dx)
        else:
            residual = field_residual(u, hx, hy, k0, dx)
            grad_data = master_formula_grad_u(u, hx, hy, residual, k0, dx)
    else:
        residual = scalar_or_semi_residual(u, hx, k0, dx, cfg.method)
        grad_data = scalar_or_semi_grad_u(u, hx, residual, cfg.method, k0, dx)

    reg_value = regularizer_value_u(
        u, cfg.lambda_tv, cfg.lambda_bound, cfg.tv_delta,
        cfg.n_min, cfg.n_max, dx)
    grad_reg = regularizer_grad_u(
        u, cfg.lambda_tv, cfg.lambda_bound, cfg.tv_delta,
        cfg.n_min, cfg.n_max, dx)
    grad_total = grad_data + grad_reg
    grad_update = grad_total * interior_mask(n)
    return {
        "residual": residual,
        "F_data": float(residual["F"]),
        "F_regularizer": float(reg_value),
        "F_total": float(residual["F"] + reg_value),
        "NFR": float(residual["NFR"]),
        "E_x": float(residual["E_x"]),
        "E_y": float(residual["E_y"]),
        "grad_data": grad_data,
        "grad_regularizer": grad_reg,
        "grad_total": grad_total,
        "grad_update": grad_update,
        "data_gradient_norm": float(np.linalg.norm(grad_data)),
        "regularizer_gradient_norm": float(np.linalg.norm(grad_reg)),
        "projected_gradient": projected_gradient_norm(
            u, grad_update, cfg.learning_rate, cfg.n_min, cfg.n_max),
    }


def run_stage1(cfg: RunConfig, it: Array, hx: Array, hy: Array,
               u0: Array | None = None) -> dict[str, Any]:
    n = it.shape[0]
    u = initialize_u(it, cfg.n_min, cfg.n_max, cfg.init, cfg.seed) if u0 is None else u0.copy()

    m = np.zeros_like(u)
    v = np.zeros_like(u)
    b1, b2, eps_adam = 0.9, 0.999, 1e-10
    history: list[dict[str, Any]] = []
    stop_reason = "max_iterations"
    t0 = time.perf_counter()

    for t in range(1, cfg.iterations + 1):
        try:
            pre = stage1_state(cfg, u, hx, hy)
        except ValueError as exc:
            if "normalizer is zero" not in str(exc):
                raise
            stop_reason = "zero_normalizer"
            break
        grad = pre["grad_update"]

        m = b1 * m + (1.0 - b1) * grad
        v = b2 * v + (1.0 - b2) * grad * grad
        u_old = u
        u = u_old - cfg.learning_rate * (m / (1.0 - b1**t)) / (np.sqrt(v / (1.0 - b2**t)) + eps_adam)
        u = np.clip(u, cfg.n_min**2, cfg.n_max**2)

        delta_u = float(np.linalg.norm(u - u_old) / max(float(np.linalg.norm(u_old)), 1e-30))
        # History and stopping diagnostics are evaluated on the state that is
        # actually returned, eliminating the former one-step offset.
        post = stage1_state(cfg, u, hx, hy)
        nfr = post["NFR"]
        gproj = post["projected_gradient"]
        history.append({
            "iteration": t,
            "NFR": nfr,
            "delta_u": delta_u,
            "E_x": post["E_x"],
            "E_y": post["E_y"],
            "projected_gradient": gproj,
            "F_data": post["F_data"],
            "F_regularizer": post["F_regularizer"],
            "F_total": post["F_total"],
            "data_gradient_norm": post["data_gradient_norm"],
            "regularizer_gradient_norm": post["regularizer_gradient_norm"],
        })

        if t >= cfg.min_iterations:
            if nfr < cfg.tol:
                stop_reason = "tolerance"
                break
            if t > cfg.min_iterations + cfg.plateau_window:
                past = history[-1 - cfg.plateau_window]["NFR"]
                rel = abs(nfr - past) / max(past, 1e-30)
                if (rel < cfg.plateau_rel and delta_u < MS_DELTA_U_TOL
                        and gproj < MS_GPROJ_TOL):
                    stop_reason = "plateau"
                    break

    elapsed = time.perf_counter() - t0
    final = history[-1] if history else {}
    return {
        "u": u,
        "n": np.sqrt(np.maximum(u, 1e-12)),
        "history": history,
        "stop_reason": stop_reason,
        "iterations_completed": len(history),
        "elapsed_s": elapsed,
        "NFR": final.get("NFR"),
        "E_x": final.get("E_x"),
        "E_y": final.get("E_y"),
        "delta_u": final.get("delta_u"),
        "projected_gradient": final.get("projected_gradient"),
        "F_data": final.get("F_data"),
        "F_regularizer": final.get("F_regularizer"),
        "F_total": final.get("F_total"),
        "data_gradient_norm": final.get("data_gradient_norm"),
        "regularizer_gradient_norm": final.get("regularizer_gradient_norm"),
    }


# ==========================================================================
# Stage 2 -- sparse shift-invert eigenmode verification
# ==========================================================================
def verify_mode(u: Array, it: Array, cfg: RunConfig) -> dict[str, Any]:
    try:
        from scipy.sparse import diags, identity, kron, bmat
        from scipy.sparse.linalg import LinearOperator, factorized
    except ImportError:
        return {"available": False,
                "reason": "SciPy is required for Stage 2 eigenmode verification."}

    n = u.shape[0]
    _, _, dx = make_grid(n, cfg.half_width)
    k0 = 2.0 * math.pi / cfg.wavelength
    t0 = time.perf_counter()

    # Fourth-order 1-D operators
    d2 = diags([np.full(n - 2, -1 / 12), np.full(n - 1, 4 / 3), np.full(n, -2.5),
                np.full(n - 1, 4 / 3), np.full(n - 2, -1 / 12)],
               [-2, -1, 0, 1, 2]) / dx**2
    d1 = diags([np.full(n - 2, 1 / 12), np.full(n - 1, -2 / 3), np.zeros(n),
                np.full(n - 1, 2 / 3), np.full(n - 2, -1 / 12)],
               [-2, -1, 0, 1, 2]) / dx
    eye = identity(n)
    lap = kron(d2, eye) + kron(eye, d2)
    dx_op = kron(eye, d1)
    dy_op = kron(d1, eye)

    u_safe = np.maximum(u, 1e-12)
    ax = (grad_x(u_safe, dx) / u_safe).ravel()
    ay = (grad_y(u_safe, dx) / u_safe).ravel()
    diag_u = diags([(k0**2) * u_safe.ravel()], [0])
    dax, day = diags([ax], [0]), diags([ay], [0])

    operator = bmat([[lap + diag_u - day @ dy_op, day @ dx_op],
                     [dax @ dy_op, lap + diag_u - dax @ dx_op]], format="lil")

    boundary = np.zeros((n, n), dtype=bool)
    boundary[:2, :] = boundary[-2:, :] = boundary[:, :2] = boundary[:, -2:] = True
    for idx in np.where(boundary.ravel())[0]:
        for offset in (0, n * n):
            row = idx + offset
            operator.rows[row] = [row]
            operator.data[row] = [1.0]
    operator = operator.tocsr()

    sigma = (0.98 * cfg.n_max * k0) ** 2
    dof = 2 * n * n
    shifted = (operator - sigma * identity(dof)).tocsc()
    solve = factorized(shifted)
    op_inv = LinearOperator((dof, dof), matvec=solve)

    rng = np.random.default_rng(cfg.seed)
    vec = rng.random(dof)
    vec /= np.linalg.norm(vec)
    for _ in range(2000):
        nxt = op_inv @ vec
        nxt /= np.linalg.norm(nxt)
        if np.linalg.norm(nxt - vec) < 1e-8:
            vec = nxt
            break
        vec = nxt

    beta2 = float(vec @ (operator @ vec))
    n_eff = math.sqrt(abs(beta2)) / k0
    hx = vec[:n * n].reshape(n, n)
    hy = vec[n * n:].reshape(n, n)
    scale = max(float(np.max(np.abs(hx))), float(np.max(np.abs(hy))), 1e-30)
    hx, hy = hx / scale, hy / scale

    modal_i = hx**2 + hy**2
    purity, dominant = polarization_purity(hx, hy)
    return {
        "available": True,
        "n_eff": n_eff,
        "beta2": beta2,
        "eps_I": epsilon_I(modal_i, it),
        "SSIM": windowed_ssim(modal_i, it),
        "polarization_purity": purity,
        "dominant_component": dominant,
        "elapsed_s": time.perf_counter() - t0,
        "Ho_x": hx,
        "Ho_y": hy,
    }


# ==========================================================================
# Gradient audit
# ==========================================================================
def directional_gradient_check(u: Array, hx: Array, hy: Array, cfg: RunConfig,
                               samples: int = 3, step: float = 1e-6) -> dict[str, Any]:
    """Central-difference check of the discretized objective actually used.

    Results are reported separately for interior-supported directions and for
    directions that include boundary layers.  Both use the exact transpose of
    the implemented discrete operator Jacobian, so boundary terms are included
    rather than inferred from a continuum integration-by-parts formula.
    """
    n = u.shape[0]
    _, _, dx = make_grid(n, cfg.half_width)
    k0 = 2.0 * math.pi / cfg.wavelength
    state = stage1_state(cfg, u, hx, hy)
    grad = state["grad_total"]
    if cfg.method == "full" and cfg.normalizer == "rayleigh":
        res = rayleigh_residual(u, hx, hy, k0, dx)
        def data_objective(uu: Array) -> float:
            return float(rayleigh_residual(uu, hx, hy, k0, dx)["F"])
    elif cfg.method == "full":
        res = field_residual(u, hx, hy, k0, dx)
        m_frozen = res["M"]
        ht_sq = res["ht_sq"]
        active = interior_mask(n)
        def data_objective(uu: Array) -> float:
            qx, qy = vector_operator_output(uu, hx, hy, k0, dx)
            return float(np.sum((active * (qx / m_frozen - hx)) ** 2
                                + (active * (qy / m_frozen - hy)) ** 2)
                         / max(ht_sq, 1e-30))
    else:
        res = scalar_or_semi_residual(u, hx, k0, dx, cfg.method)
        m_frozen = res["M"]
        h_sq = res["h_sq"]
        active = interior_mask(n)
        def data_objective(uu: Array) -> float:
            if cfg.method == "scalar":
                q = scalar_operator_output(uu, hx, k0, dx)
            else:
                q = semivectorial_operator_output(uu, hx, k0, dx)
            return float(np.sum((active * (q / m_frozen - hx)) ** 2)
                         / max(h_sq, 1e-30))

    def total_objective(uu: Array) -> float:
        return (data_objective(uu)
                + regularizer_value_u(
                    uu, cfg.lambda_tv, cfg.lambda_bound, cfg.tv_delta,
                    cfg.n_min, cfg.n_max, dx))

    mask = interior_mask(n)
    rng = np.random.default_rng(cfg.seed)

    rows = []
    for scope, weight in (("interior", mask), ("with_boundary", np.ones_like(mask))):
        errors = []
        for s in range(samples):
            d = rng.standard_normal(u.shape) * weight
            fd = (total_objective(u + step * d)
                  - total_objective(u - step * d)) / (2.0 * step)
            analytic = float(np.sum(grad * d))
            rel = abs(fd - analytic) / max(abs(fd), abs(analytic), 1e-30)
            errors.append(rel)
            rows.append({"scope": scope, "sample": s, "finite_difference": fd,
                         "analytic": analytic, "relative_error": rel})
        print(f"  gradient check [{scope:13s}] median rel. error = {np.median(errors):.2e}")
    return {"step": step, "samples": samples,
            "objective": "data_plus_weighted_regularizer", "rows": rows}


# ==========================================================================
# I/O helpers
# ==========================================================================
def load_target_npz(path: Path) -> tuple[Array, Array | None]:
    data = dict(np.load(path, allow_pickle=False))
    for key in ("It", "I_t", "target_intensity", "intensity"):
        if key in data:
            it = active_target_intensity(np.asarray(data[key], dtype=float))
            break
    else:
        raise ValueError(f"No intensity array found in {path}.")
    u0 = None
    for key in ("u", "eps", "epsilon"):
        if key in data:
            u0 = np.asarray(data[key], dtype=float)
            break
    else:
        for key in ("n", "index"):
            if key in data:
                u0 = np.asarray(data[key], dtype=float) ** 2
                break
    return it, u0


def json_sanitize(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return None
    if isinstance(obj, dict):
        return {k: json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_sanitize(v) for v in obj]
    return obj


def write_history_csv(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


# ==========================================================================
# Self-test against Table I
# ==========================================================================
def self_test(cfg: RunConfig) -> int:
    """Reproduce the primary Rayleigh-normalized rows of Table I."""
    print("Reproducing the Rayleigh-normalized full-vectorial rows of Table I "
          f"(N={cfg.grid}, lambda={cfg.wavelength} um, {cfg.iterations} iterations)\n")
    header = (f"{'target':<12}{'NFR':>12}{'eps_I':>12}{'SSIM':>8}{'n_eff':>9}   status")
    print(header)
    print("-" * len(header))
    failures = 0
    for name, reference in TABLE_I_RAYLEIGH_NORMALIZED.items():
        it = synthetic_target(name, cfg.grid, cfg.half_width)
        hx, hy = target_field_from_intensity(it)
        stage1 = run_stage1(cfg, it, hx, hy)
        verified = verify_mode(stage1["u"], it, cfg)
        if not verified.get("available"):
            print(f"{name:<12}  Independent verification unavailable: {verified.get('reason')}")
            failures += 1
            continue

        got = {"NFR": stage1["NFR"], "eps_I": verified["eps_I"],
               "SSIM": verified["SSIM"], "n_eff": verified["n_eff"]}
        deviations = {k: abs(got[k] - reference[k]) / max(abs(reference[k]), 1e-30)
                      for k in reference}
        ok = max(deviations.values()) < 0.02
        failures += 0 if ok else 1
        if ok:
            status = f"OK (max dev {max(deviations.values()):.1%})"
        else:
            status = f"MISMATCH (max dev {max(deviations.values()):.1%})"
        print(f"{name:<12}{got['NFR']:>12.4e}{got['eps_I']:>12.4e}{got['SSIM']:>8.3f}"
              f"{got['n_eff']:>9.4f}   {status}")

    print()
    print("Reference (Table I):")
    for name, reference in TABLE_I_RAYLEIGH_NORMALIZED.items():
        print(f"{name:<12}{reference['NFR']:>12.4e}{reference['eps_I']:>12.4e}"
              f"{reference['SSIM']:>8.3f}{reference['n_eff']:>9.4f}")
    print()
    print("Note: the Rayleigh-normalized objective uses an exact discrete VJP.")
    print("Run --check-gradient to reproduce fixed-seed directional checks.")
    print("\nSelf-test", "PASSED" if failures == 0 else f"FAILED ({failures} case(s))")
    return failures


# ==========================================================================
# Pipeline
# ==========================================================================
def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    cfg = RunConfig(
        target=args.target, input_npz=str(args.input_npz) if args.input_npz else None,
        method=args.method, normalizer=args.normalizer, grid=args.grid, half_width=args.half_width,
        wavelength=args.wavelength, n_min=args.n_min, n_max=args.n_max,
        iterations=args.iterations, learning_rate=args.learning_rate,
        lambda_tv=args.lambda_tv, lambda_bound=args.lambda_bound,
        tv_delta=args.tv_delta, tol=args.tol, plateau_window=args.plateau_window,
        plateau_rel=args.plateau_rel, min_iterations=args.min_iterations,
        init=args.init, seed=args.seed, verify=args.verify,
    )

    u0 = None
    if args.input_npz is not None:
        it, u0 = load_target_npz(Path(args.input_npz))
        cfg.grid = it.shape[0]
    else:
        it = synthetic_target(cfg.target, cfg.grid, cfg.half_width)
    hx, hy = target_field_from_intensity(it)

    stage1 = run_stage1(cfg, it, hx, hy, u0)
    print(f"Stage 1: {stage1['iterations_completed']} iterations "
          f"({stage1['stop_reason']}), {stage1['elapsed_s']:.1f} s")
    print(f"  NFR = {stage1['NFR']:.4e}   E_x = {stage1['E_x']:.3e}   "
          f"E_y = {stage1['E_y']:.3e}")

    summary: dict[str, Any] = {"config": asdict(cfg), "stage1": {
        k: stage1[k] for k in ("stop_reason", "iterations_completed", "elapsed_s",
                               "NFR", "E_x", "E_y", "delta_u",
                               "projected_gradient", "F_data", "F_regularizer",
                               "F_total", "data_gradient_norm",
                               "regularizer_gradient_norm")}}

    verified: dict[str, Any] = {"available": False, "reason": "not_requested"}
    if cfg.verify:
        verified = verify_mode(stage1["u"], it, cfg)
        if verified.get("available"):
            print(f"Stage 2: n_eff = {verified['n_eff']:.4f}   "
                  f"eps_I = {verified['eps_I']:.4e}   SSIM = {verified['SSIM']:.3f}   "
                  f"purity = {verified['polarization_purity']:.3f} "
                  f"({verified['dominant_component']})")
        else:
            print(f"Stage 2 unavailable: {verified.get('reason')}")
    summary["verification"] = json_sanitize(verified)

    if args.check_gradient:
        # Audit at three intermediate iterates, as reported in the manuscript.
        print("Gradient audit (three intermediate iterates):")
        checks = []
        for frac in (0.0, 1 / 3, 2 / 3):
            t = int(frac * cfg.iterations)
            if t == 0:
                u_t = initialize_u(it, cfg.n_min, cfg.n_max, cfg.init, cfg.seed)
            else:
                cfg_t = RunConfig(**{**cfg.__dict__, "iterations": t})
                u_t = run_stage1(cfg_t, it, hx, hy, u0)["u"]
            print(f"  iterate t = {t}:")
            checks.append({"iterate": t,
                           **directional_gradient_check(u_t, hx, hy, cfg)})
        summary["gradient_check"] = checks

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix
    (out_dir / f"{prefix}_summary.json").write_text(
        json.dumps(json_sanitize(summary), indent=2), encoding="utf-8")
    write_history_csv(out_dir / f"{prefix}_history.csv", stage1["history"])
    arrays = {"It": it, "Ht_x": hx, "Ht_y": hy, "u": stage1["u"], "n": stage1["n"]}
    if verified.get("available"):
        arrays["Ho_x"] = verified["Ho_x"]
        arrays["Ho_y"] = verified["Ho_y"]
    np.savez_compressed(out_dir / f"{prefix}_arrays.npz", **arrays)
    print(f"\nWrote {out_dir / (prefix + '_summary.json')}")
    print(f"      {out_dir / (prefix + '_history.csv')}")
    print(f"      {out_dir / (prefix + '_arrays.npz')}")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", choices=["gaussian", "sun_like", "ultra_flat"], default="gaussian")
    p.add_argument("--method", choices=["full", "semi", "scalar"], default="full")
    p.add_argument("--normalizer", choices=["lagged", "rayleigh"], default="rayleigh",
                   help="lagged: max-magnitude stop-gradient baseline. rayleigh: "
                        "primary Rayleigh-normalized scale-invariant objective with "
                        "an exact discrete gradient.")
    p.add_argument("--input-npz", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("fv_waveguide_run"))
    p.add_argument("--prefix", default="fv_prescribed")
    p.add_argument("--grid", type=int, default=MS_GRID)
    p.add_argument("--half-width", type=float, default=MS_HALF_WIDTH)
    p.add_argument("--wavelength", type=float, default=MS_WAVELENGTH)
    p.add_argument("--n-min", type=float, default=MS_N_MIN)
    p.add_argument("--n-max", type=float, default=MS_N_MAX)
    p.add_argument("--iterations", type=int, default=MS_ITERATIONS)
    p.add_argument("--learning-rate", type=float, default=MS_LEARNING_RATE)
    p.add_argument("--lambda-tv", type=float, default=MS_LAMBDA_TV)
    p.add_argument(
        "--lambda-bound", type=float, default=MS_LAMBDA_BOUND,
        help="legacy audit option; zero in every disclosed run because bounds "
             "are enforced by projected clipping")
    p.add_argument("--tv-delta", type=float, default=MS_TV_DELTA)
    p.add_argument("--tol", type=float, default=MS_TOL)
    p.add_argument("--plateau-window", type=int, default=MS_PLATEAU_WINDOW)
    p.add_argument("--plateau-rel", type=float, default=MS_PLATEAU_REL)
    p.add_argument("--min-iterations", type=int, default=MS_MIN_ITERATIONS)
    p.add_argument("--init", choices=["target", "homogeneous", "random"], default="target")
    p.add_argument("--seed", type=int, default=MS_SEED)
    p.add_argument("--verify", action="store_true",
                   help="run Stage 2 sparse eigenmode verification (requires SciPy)")
    p.add_argument("--check-gradient", action="store_true",
                    help="audit the complete selected data objective plus TV regularization")
    p.add_argument("--self-test", action="store_true",
                    help="reproduce the primary Rayleigh-normalized rows of Table I")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        raise SystemExit(self_test(RunConfig(
            target=args.target, input_npz=None, method="full",
            normalizer=args.normalizer, grid=args.grid,
            half_width=args.half_width, wavelength=args.wavelength, n_min=args.n_min,
            n_max=args.n_max, iterations=args.iterations,
            learning_rate=args.learning_rate, lambda_tv=args.lambda_tv,
            lambda_bound=args.lambda_bound, tv_delta=args.tv_delta, tol=args.tol,
            plateau_window=args.plateau_window, plateau_rel=args.plateau_rel,
            min_iterations=args.min_iterations, init=args.init, seed=args.seed,
            verify=True)))
    run_pipeline(args)


if __name__ == "__main__":
    main()
