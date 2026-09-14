# Full-Vectorial Waveguide Stage-2 Verification

Reproducibility repository for the independent Stage-2 eigenmode verification
used in the JSTQE manuscript:

> *Full-Vectorial Prescribed-Field Inverse Synthesis of Optical Waveguides:
> A Classical Gradient-Based Approach*
>
> Manuscript ID: `JSTQE-CON-IDPAUQC2027-10493-2026`

The repository is a public-release layout of the code included with the
second-round minor revision. It preserves the validated numerical
implementation used for the reported results.

## What Stage 2 Does

Stage 2 takes a candidate squared-index profile `u = n^2` (or an index profile
`n`) and a target intensity map. It then:

1. assembles the sparse coupled two-component transverse magnetic-field
   operator with fourth-order finite differences;
2. applies the manuscript's two-layer homogeneous Dirichlet boundary;
3. extracts the eigenpair nearest
   `sigma = (0.98 * n_max * k0)^2` by sparse shift-invert iteration; and
4. reports independently verified `n_eff`, intensity error `epsilon_I`,
   windowed SSIM, and integrated-power polarization purity.

Stage 2 verifies the mode supported by a candidate profile. It does not claim
that a unique physical index profile can be recovered from one intensity
frame.

## Repository Layout

```text
.
|-- stage2_verify.py
|-- full_vectorial_waveguide_inverse_synthesis.py
|-- examples/
|   `-- gaussian_n110/
|       |-- gaussian_arrays.npz
|       `-- gaussian_summary.json
|-- tests/
|   `-- test_stage2.py
|-- docs/
|   |-- REPRODUCIBILITY.md
|   `-- STAGE2_METHOD.md
|-- requirements.txt
|-- requirements-dev.txt
|-- environment.yml
`-- .github/workflows/ci.yml
```

`stage2_verify.py` is the standalone Stage-2 command-line entry point.
`verify_mode()` in `full_vectorial_waveguide_inverse_synthesis.py` is the
underlying implementation. The main script is retained unchanged from the
validated supplementary archive so that the Stage-1-to-Stage-2 path remains
auditable.

## Installation

Python 3.10 or newer is required. The reported Windows validation used Python
3.13.9, NumPy 2.3.5, and SciPy 1.16.3.

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Linux or macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Alternatively, create the pinned Conda environment:

```bash
conda env create -f environment.yml
conda activate jstqe-stage2
```

## Quick Start

Run Stage 2 on the included N=110 Gaussian candidate:

```bash
python stage2_verify.py examples/gaussian_n110/gaussian_arrays.npz \
  --out-prefix outputs/gaussian_n110
```

Expected values, allowing small sparse-solver variation across platforms:

| Metric | Expected value |
|---|---:|
| `n_eff` | 1.573849 |
| `epsilon_I` | 1.639397e-2 |
| SSIM | 0.994230 |
| Polarization purity | 0.502232 |

The command writes:

```text
outputs/gaussian_n110_metrics.json
outputs/gaussian_n110_fields.npz
```

To execute both stages from the manuscript implementation:

```bash
python full_vectorial_waveguide_inverse_synthesis.py \
  --target gaussian --verify --out-dir outputs/full_run --prefix gaussian
```

## Input Contract

The positional input to `stage2_verify.py` must be a square 2-D NumPy archive
containing:

- one candidate array named `u` or `n`; and
- one target-intensity array named `It`, `I_t`, `target_intensity`, or
  `intensity`.

If the file is named `*_arrays.npz`, a neighboring `*_summary.json` is loaded
automatically for grid and material parameters. Command-line values such as
`--wavelength`, `--half-width`, `--n-min`, and `--n-max` override the sidecar.

## Tests

Install the development dependency and run:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The suite checks the CLI, input contract, fourth-order Laplacian, exact
first-derivative transpose, and the complete N=110 Stage-2 regression result.
The full regression normally takes tens of seconds.

## Reproducibility Notes

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for tolerances,
platform information, and output interpretation. The mapping between code and
manuscript equations is summarized in
[docs/STAGE2_METHOD.md](docs/STAGE2_METHOD.md).

## Citation

Please cite the final JSTQE article when its volume, pages, and DOI are
available. Until then, cite the manuscript title and ID shown above and state
the repository revision or commit used.

## License

This repository is distributed under the MIT License. See [LICENSE](LICENSE).

