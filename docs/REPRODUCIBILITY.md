# Reproducibility

## Validated Environment

The repository was validated on Windows with:

```text
Python 3.13.9
NumPy 2.3.5
SciPy 1.16.3
pytest 8.4.2
```

`requirements.txt` and `environment.yml` pin these versions. The GitHub
workflow rebuilds the environment and executes the test suite on Linux.

An additional Linux/WSL check using Python 3.13.13, NumPy 2.4.6, and SciPy
1.17.1 reproduced the displayed Stage-2 values exactly: `n_eff=1.573849`,
`epsilon_I=1.639397e-2`, and `SSIM=0.994230`.

## Reference Regression

Input:

```text
examples/gaussian_n110/gaussian_arrays.npz
```

Expected output:

| Metric | Reference |
|---|---:|
| `n_eff` | 1.5738488180006143 |
| `beta2` | 73.94181053286194 |
| `epsilon_I` | 0.01639397325911933 |
| `SSIM` | 0.9942304901308227 |
| polarization purity | 0.502231830704441 |

The regression test uses tight but non-bitwise tolerances because sparse
factorization and floating-point reduction order can differ by platform.

## Output Files

`stage2_verify.py` writes a JSON file containing scalar metrics and a
compressed NumPy archive containing `Ho_x` and `Ho_y`. Generated outputs are
ignored by Git so that only deliberate reference artifacts are versioned.

## Integrity Checks

The tests cover:

1. command-line discoverability;
2. the example archive contract;
3. sensitivity of the fourth-order Laplacian to grid-scale oscillation;
4. exact discrete transpose identities for first derivatives; and
5. the complete sparse Stage-2 regression.

The numerical source files in this repository are copied from the validated
minor-revision supplementary archive. Documentation and tests do not alter
the scientific implementation.
