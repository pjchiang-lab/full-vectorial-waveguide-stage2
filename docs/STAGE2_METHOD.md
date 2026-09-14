# Stage-2 Numerical Method

## Purpose

Stage 2 independently determines the full-vectorial mode supported by the
candidate refractive-index profile produced by Stage 1. The target-field
ansatz used during synthesis is not reused as the solved modal field.

## Operator and Discretization

The code assembles the coupled transverse magnetic-field operator in the
`(H_x, H_y)` basis. All four operator blocks are retained. First and second
derivatives use explicit fourth-order finite-difference stencils on a square
grid. The Laplacian is implemented as a genuine second-derivative stencil;
it is not formed by applying the centered first derivative twice.

The outermost two node layers impose homogeneous Dirichlet conditions,
matching the active-domain convention used for the reported metrics.

## Eigenpair Extraction

For free-space wavenumber `k0`, the sparse eigensystem is shifted by

```text
sigma = (0.98 * n_max * k0)^2.
```

Sparse factorization and inverse iteration select the eigenpair nearest this
shift. The returned propagation constant is converted to

```text
n_eff = sqrt(abs(beta^2)) / k0.
```

## Reported Metrics

- `epsilon_I`, manuscript Eq. (13): relative error between peak-normalized
  verified modal intensity and target intensity on the active domain.
- `SSIM`: 7x7-window structural similarity between the same intensity maps.
- Component residuals `E_x` and `E_y`, manuscript Eq. (14), are computed by
  Stage 1; they are not substitutes for independent verification.
- Polarization purity, manuscript Eq. (19): the larger integrated component
  power divided by total transverse magnetic-field power.
- Mode overlap, manuscript Eq. (20), is used for the known-rib diagnostic.

## Interpretation

High modal-intensity similarity establishes consistency between the candidate
and the prescribed intensity convention. It does not establish uniqueness of
the index profile, phase, or polarization from one intensity image.

