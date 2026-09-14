"""Run the manuscript's independent Stage-2 eigenmode verification only.

The input is a saved ``*_arrays.npz`` candidate produced by
``full_vectorial_waveguide_inverse_synthesis.py``.  The archive must contain
``u`` (squared refractive index) or ``n`` and a target-intensity array named
``It``, ``I_t``, ``target_intensity``, or ``intensity``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from full_vectorial_waveguide_inverse_synthesis import (
    MS_HALF_WIDTH,
    MS_ITERATIONS,
    MS_LAMBDA_BOUND,
    MS_LAMBDA_TV,
    MS_LEARNING_RATE,
    MS_MIN_ITERATIONS,
    MS_N_MAX,
    MS_N_MIN,
    MS_PLATEAU_REL,
    MS_PLATEAU_WINDOW,
    MS_SEED,
    MS_TOL,
    MS_TV_DELTA,
    MS_WAVELENGTH,
    RunConfig,
    verify_mode,
)


def first_array(data: np.lib.npyio.NpzFile, names: tuple[str, ...]) -> np.ndarray:
    for name in names:
        if name in data:
            return np.asarray(data[name], dtype=float)
    raise KeyError(f"none of {names} is present in the input archive")


def sidecar_config(arrays_path: Path, explicit: Path | None) -> dict:
    path = explicit
    if path is None and arrays_path.name.endswith("_arrays.npz"):
        path = arrays_path.with_name(arrays_path.name.replace("_arrays.npz", "_summary.json"))
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("config", {})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run independent Stage-2 sparse shift-invert eigenmode verification.")
    parser.add_argument("arrays_npz", type=Path, help="saved Stage-1 *_arrays.npz file")
    parser.add_argument("--summary-json", type=Path,
                        help="optional Stage-1 summary; auto-detected beside *_arrays.npz")
    parser.add_argument("--half-width", type=float)
    parser.add_argument("--wavelength", type=float)
    parser.add_argument("--n-min", type=float)
    parser.add_argument("--n-max", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out-prefix", type=Path,
                        help="output prefix (default: input stem plus _stage2)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.arrays_npz) as data:
        it = first_array(data, ("It", "I_t", "target_intensity", "intensity"))
        if "u" in data:
            u = np.asarray(data["u"], dtype=float)
        elif "n" in data:
            u = np.asarray(data["n"], dtype=float) ** 2
        else:
            raise KeyError("the input archive must contain u or n")

    if u.shape != it.shape or u.ndim != 2 or u.shape[0] != u.shape[1]:
        raise ValueError("u and target intensity must be equal square two-dimensional arrays")

    saved = sidecar_config(args.arrays_npz, args.summary_json)
    cfg = RunConfig(
        target=str(saved.get("target", "external")),
        input_npz=str(args.arrays_npz),
        method="full",
        normalizer="rayleigh",
        grid=u.shape[0],
        half_width=args.half_width if args.half_width is not None
        else float(saved.get("half_width", MS_HALF_WIDTH)),
        wavelength=args.wavelength if args.wavelength is not None
        else float(saved.get("wavelength", MS_WAVELENGTH)),
        n_min=args.n_min if args.n_min is not None else float(saved.get("n_min", MS_N_MIN)),
        n_max=args.n_max if args.n_max is not None else float(saved.get("n_max", MS_N_MAX)),
        iterations=int(saved.get("iterations", MS_ITERATIONS)),
        learning_rate=float(saved.get("learning_rate", MS_LEARNING_RATE)),
        lambda_tv=float(saved.get("lambda_tv", MS_LAMBDA_TV)),
        lambda_bound=float(saved.get("lambda_bound", MS_LAMBDA_BOUND)),
        tv_delta=float(saved.get("tv_delta", MS_TV_DELTA)),
        tol=float(saved.get("tol", MS_TOL)),
        plateau_window=int(saved.get("plateau_window", MS_PLATEAU_WINDOW)),
        plateau_rel=float(saved.get("plateau_rel", MS_PLATEAU_REL)),
        min_iterations=int(saved.get("min_iterations", MS_MIN_ITERATIONS)),
        init=str(saved.get("init", "target")),
        seed=args.seed if args.seed is not None else int(saved.get("seed", MS_SEED)),
        verify=True,
    )

    result = verify_mode(u, it, cfg)
    if not result.get("available"):
        raise RuntimeError(result.get("reason", "Stage-2 verification failed"))

    prefix = args.out_prefix or args.arrays_npz.with_name(args.arrays_npz.stem + "_stage2")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    metrics = {key: result[key] for key in (
        "n_eff", "beta2", "eps_I", "SSIM", "polarization_purity",
        "dominant_component", "elapsed_s")}
    Path(str(prefix) + "_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez_compressed(
        Path(str(prefix) + "_fields.npz"), Ho_x=result["Ho_x"], Ho_y=result["Ho_y"])

    print("Stage 2 complete")
    print(f"  n_eff = {metrics['n_eff']:.6f}")
    print(f"  eps_I = {metrics['eps_I']:.6e}")
    print(f"  SSIM = {metrics['SSIM']:.6f}")
    print(f"  polarization purity = {metrics['polarization_purity']:.6f} "
          f"({metrics['dominant_component']})")
    print(f"  wrote {prefix}_metrics.json")
    print(f"  wrote {prefix}_fields.npz")


if __name__ == "__main__":
    main()
