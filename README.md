[![DOI](https://zenodo.org/badge/1145589432.svg)](https://doi.org/10.5281/zenodo.19749064)

# DAMN

DAMN (Design A Matrix Now) is a fully-featured Python toolkit for building
time-aligned design matrices and fitting Poisson (and other) GLMs to neural data.

It provides:

- Design matrix construction for event-based (e.g. stimulus onset) and continuous (e.g. wheel velocity) regressors
- Automatic resampling of continuous regressors to master timebins
- Many basis functions
- Basis function truncation to avoid trial edge artifacts
- Trial-aligned or session-aligned matrix construction
- Easy dot-based access of individual regressors
- Tags to quickly access and modify (shuffle, zero-out, etc.) regressors of the same group (e.g. stimulus, choice and reward regressors can all be tagged as "task" and accessed together with that tag)
- Kernel reconstruction from weights and basis functions
- GPU-accelerated Poisson GLM fitting (PyTorch), with regularization
- Sklearn-compatible estimator API
- Convenience scoring metrics

## Installation

### 1. Clone and install into a fresh environment locally:

```bash
# create new python environment with 3.11
conda create --name damn python=3.11
conda activate damn
```
### 2. Next, follow the PyTorch install instructions for your machine, found at:
`
https://pytorch.org/get-started/locally/
`
### 3. Install the repo and dependencies
```
# clone and install repo
git clone https://github.com/mdmelin/DAMN.git
cd DAMN
pip install -e .
```


## Quick Start
See `examples` folder for more detailed examples of building and fitting models.
### 1. Build a design matrix

```python
import numpy as np
from damn.objects.design_matrix_objects import DesignMatrix
from damn.objects.regressor_objects import EventRegressor
from damn.objects.basis_function_objects import RaisedCosineBasis

BINWIDTH_S = 0.001
PRE_S = 0.1
POST_S = 0.4

stim_times = np.array([1.0, 2.0, 3.0])
stim_contrasts = np.array([1.0, 0.5, 1.0])

stim_reg = EventRegressor(
	name="stimulus",
	event_times=stim_times,
	event_values=stim_contrasts,
	binwidth_s=BINWIDTH_S,
	basis_objects=[RaisedCosineBasis(8, 0, 0.6, BINWIDTH_S, log_scale=True)],
)

dmat = DesignMatrix(
	master_alignment_times=stim_times,
	master_pre_s=PRE_S,
	master_post_s=POST_S,
	binwidth_s=BINWIDTH_S,
)
dmat.add_regressor(stim_reg)
dmat.build_matrix()

X = dmat.X
print(X.shape)
```

### 2. Align and bin spikes

```python
from damn.alignment import compute_spike_count

# spike_times is 1D array for one unit
Y = compute_spike_count(
	event_times=stim_times,
	spike_times=spike_times,
	pre_seconds=PRE_S,
	post_seconds=POST_S,
	binwidth_s=BINWIDTH_S,
)[0]

# flatten to (T, 1) if needed
Y = Y.reshape(-1, 1)
```

### 3. Fit Poisson GLM on CPU or GPU

```python
from damn.fit import fit_poisson_glm_lbfgs

W, b, train_loss_hist, val_loss_hist, train_bps_hist, val_bps_hist = fit_poisson_glm_lbfgs(
	X=X,
	Y=Y,
	alpha=1e-8,
	max_epochs=1000,
	val_fraction=0.1,
	early_stopping="train",
	patience=10,
)
```

Predicted rates:

```python
Yhat = np.exp(X @ W + b)
```

## Design philosophy

### Basis functions

Defined in `damn/basis_functions.py` and object wrappers in
`damn/objects/basis_function_objects.py`.

Implemented basis families include:

- `no_basis`
- `boxcar_smooth`
- `gaussian_smooth`
- `delta_basis`
- `raised_cosine_basis`
- `gaussian_basis`
- `bspline_basis`
- `fir_basis`

### Regressors and design matrix

Object model:

- `EventRegressor`: sparse events convolved with basis functions
- `ContinuousRegressor`: continuous signal resampled to matrix time bins
- `DesignMatrix`: container that aligns regressors and exposes concatenated `X` matrix

Useful workflow features:

- name/tag-based grouping
- hide/unhide regressors
- shuffle or zero regressors globally or by tag
- per-regressor kernel reconstruction after fitting coefficients

### Fitting utilities

Available in `damn/fit.py`:

- `fit_poisson_glm_lbfgs`: full-batch optimizer, generally fastest if data fits VRAM
- `fit_poisson_glm_adam`: minibatch optimizer for larger datasets
- `fit_poisson_glm_best_alpha`: choose one best L2 penalty from grid
- `fit_poisson_glm_best_alpha_per_target`: choose one alpha per output target
- `choose_optimizer`: heuristic helper for LBFGS vs Adam choice

### sklearn wrapper

`damn/sklearn.py` provides `PoissonGLM`, a `BaseEstimator`/`RegressorMixin` class.

```python
from damn.sklearn import PoissonGLM

model = PoissonGLM(
	optimizer_type="lbfgs",
	alpha="find",                 # automatic alpha search
	per_target_alpha=True,
	alpha_grid=np.logspace(-12, -8, 5),
	max_epochs=1000,
	val_fraction=0.1,
)
model.fit(X, Y)
Yhat = model.predict(X)
```

### Scoring metrics

Available in `damn/scoring.py`:

- `bits_per_spike`
- `bits_per_spike_multi_target`
- `r_squared`
- `r_squared_multi_target`

## Examples

Notebook examples:

- `examples/1_build_design_matrix.ipynb`: design matrix construction and kernel reconstruction.
- `examples/2_fit_poisson_glm_gpu.ipynb`: GPU fitting, alpha selection, and cross-validation.

## Notes and current behavior

- DAMN is designed around fixed bin width across regressors.
- Trialized matrices are typically preferred over non-trialized matrices.
- Some basis families are intentionally stubbed and raise `NotImplementedError`.
- For best GPU performance, prefer `fit_poisson_glm_lbfgs` when memory allows.

## Feedback and questions

For feedback, questions, or bug reports, please open an issue on GitHub or email.

## License

This project is licensed under GPL-3.0 (see `LICENSE`).
