"""
Poisson GLM fitting with PyTorch

This module provides functions to fit multi-neuron Poisson Generalized Linear Models (GLMs)
using PyTorch. It supports both full-batch (fit_poisson_glm_lbfgs) and minibatch (fit_poisson_glm_adam) optimization, optional
internal validation splits, and early stopping. In general, LBFGS is always recommended if the data will fit in VRAM. 
Adam optimization is preferred for very large datasets or when GPU memory is limited, but it will often converge much more slowly than LBFGS would.
Adam optimizing may also require more careful tuning of learning rate and early stopping parameters for your dataset

In general, there are a couple ways to get solutions to converge: 

- If you don't care about cross-validated performance, set val_fraction=0 and early_stopping='train'. This will just monitor the training loss and stop when it plateaus.
- If you care about cross-validated performance:
    - set val_fraction to something like 0.1 to hold out a validation set, and set early_stopping='val' to monitor the validation loss for early stopping.
        - This way is quickest in practice because it will stop as soon as the validation loss plateaus, but you are not guaranteed the optimal convergent solution given the supplied alpha penalty
        - You will want to monitor this closely and likely reduce 'patience' to stop training before val loss tails off too much.
    - Alternatively, you can set val_fraction > 0 but early_stopping='train' to monitor the training loss for early stopping, while still using the validation scores to select the best alpha.
        - fit_poisson_glm_best_alpha and fit_poisson_glm_best_alpha_per_target both can do this for you. It's somewhat analogous to sklearn.linear_model.RidgeCV, 
          where the optimal solution should be found given alpha and the training set, and then we evaluate performance on the val set.

Author: Max Melin, 2026
"""
import torch 
import numpy as np
from .optim.adam import fit_poisson_glm_adam as _fit_poisson_glm_adam_impl
from .optim.lbfgs import fit_poisson_glm_lbfgs as _fit_poisson_glm_lbfgs_impl
from .optim.common import resolve_torch_device
from .optim.common import CLAMP as CLAMP


def _sanitize_val_losses(val_losses, context):
    """Replace non-finite validation losses with +inf for safe argmin selection."""
    losses = np.asarray(val_losses, dtype=np.float64)
    finite_mask = np.isfinite(losses)
    if not np.all(finite_mask):
        bad_count = np.size(losses) - np.count_nonzero(finite_mask)
        print(
            f"WARNING: Ignoring {bad_count} non-finite validation loss values during {context}."
        )
    return np.where(finite_mask, losses, np.inf), finite_mask

def fit_poisson_glm_best_alpha_per_target(
    X,
    Y,
    optimizer_type="lbfgs",         # "lbfgs" or "adam"
    alpha_grid=None,                # list or array of candidate alphas
    max_epochs=1000,
    val_fraction=0.1,
    early_stopping='train',
    warm_start=False,
    patience=10,
    tol=1e-7,
    device=None,
    **fit_kwargs                    # extra kwargs to pass to the optimizer-specific fit function
):
    """
    Fit a Poisson GLM using either LBFGS or Adam and select the best alpha
    based on validation loss. Unlike fit_poisson_glm_best_alpha(), this function
    will find an array of best alphas, one per each target in the regrssion.

    Returns:
        best_W, best_b: parameters for best alpha
        best_alpha: selected alphas
        history: dict mapping alpha -> (train_loss_hist, val_loss_hist)
    """

    device = resolve_torch_device(device)
    
    assert val_fraction > 0, "val_fraction must be > 0 to select best alpha based on validation loss"
    # compute train inds and val inds from val_fraction
    val_inds = np.random.choice(X.shape[0], size=int(X.shape[0] * val_fraction), replace=False)

    if alpha_grid is None:
        alpha_grid = np.logspace(-3, 3, 7)
    alpha_grid = np.sort(alpha_grid)

    Ws, bs, val_losses = [],[],[]
    history = {}
    W, b = None, None # for warm starting across alphas
    for alpha in alpha_grid:
        print(f"\n--- Trying alpha = {alpha} ---")

        if optimizer_type.lower() == "lbfgs":
            result = fit_poisson_glm_lbfgs(
                X, Y,
                alpha=alpha,
                max_epochs=max_epochs,
                #val_fraction=val_fraction,
                val_inds=val_inds,
                early_stopping=early_stopping,
                patience=patience,
                tol=tol,
                device=device,
                per_target_loss=True,
                W_init=W,
                b_init=b,
                **fit_kwargs
            )
            W, b = result[0], result[1]
            train_loss_hist, val_loss_hist = result[2], result[3]
            train_bps_hist, val_bps_hist = result[4], result[5]
            val_loss_per_target = result[7] if len(result) > 7 else None
        elif optimizer_type.lower() == "adam":
            result = fit_poisson_glm_adam(
                X, Y,
                alpha=alpha,
                max_epochs=max_epochs,
                #val_fraction=val_fraction,
                val_inds=val_inds,
                early_stopping=early_stopping,
                patience=patience,
                tol=tol,
                device=device,
                per_target_loss=True,
                W_init=W,
                b_init=b,
                **fit_kwargs
            )
            W, b = result[0], result[1]
            train_loss_hist, val_loss_hist = result[2], result[3]
            train_bps_hist, val_bps_hist = result[4], result[5]
            val_loss_per_target = result[7] if len(result) > 7 else None
        else:
            raise ValueError("optimizer_type must be 'lbfgs' or 'adam'")

        if val_loss_per_target is None:
            raise ValueError("Expected per-target validation loss but received None.")

        history[alpha] = {
            "train_loss_hist": train_loss_hist,
            "val_loss_hist": val_loss_hist,
            "train_bps_hist": train_bps_hist,
            "val_bps_hist": val_bps_hist,
        }

        val_losses.append(val_loss_per_target)
        Ws.append(W)
        bs.append(b)
        if not warm_start:
            # don't warm start across alphas, re-initialize W and b for each alpha
            W, b = None, None

    val_losses = np.array(val_losses) # (num_alphas, N)
    val_losses_safe, finite_mask = _sanitize_val_losses(
        val_losses,
        context="per-target alpha selection",
    )
    invalid_targets = np.where(np.all(~finite_mask, axis=0))[0]
    if invalid_targets.size > 0:
        raise RuntimeError(
            "All alpha candidates produced non-finite validation losses "
            f"for targets {invalid_targets.tolist()}."
        )

    # check if finite losses are monotonically increasing or decreasing
    lossdiff = np.diff(val_losses_safe, axis=0)
    decreasing = np.all(lossdiff < 0, axis=0)
    increasing = np.all(lossdiff > 0, axis=0)

    if np.any(decreasing):
        print(f'WARNING: Validation loss decreases monotonically across the alpha grid for targets {np.where(decreasing)[0]}. Consider adding larger alpha values to the grid.')
    if np.any(increasing):
        print(f'WARNING: Validation loss increases monotonically across the alpha grid for targets {np.where(increasing)[0]}. Consider adding smaller alpha values to the grid.')

    # compute the best alpha per target
    best_alpha_idx = np.argmin(val_losses_safe, axis=0)
    best_alpha = alpha_grid[best_alpha_idx]
    best_W = np.stack([Ws[ind][:,i] for i,ind in enumerate(best_alpha_idx)]).T
    best_b = np.stack([bs[ind][i] for i,ind in enumerate(best_alpha_idx)])

    return best_W, best_b, best_alpha, history

def fit_poisson_glm_best_alpha(
    X,
    Y,
    optimizer_type="lbfgs",         # "lbfgs" or "adam"
    alpha_grid=None,                # list or array of candidate alphas
    max_epochs=100,
    val_fraction=0.1,
    early_stopping='train',
    patience=10,
    tol=1e-4,
    device=None,
    warm_start=False,
    **fit_kwargs                    # extra kwargs to pass to the optimizer-specific fit function
):
    """
    Fit a Poisson GLM using either LBFGS or Adam and select the best alpha
    based on validation loss.

    Returns:
        best_W, best_b: parameters for best alpha
        best_alpha: selected alpha
        history: dict mapping alpha -> (train_loss_hist, val_loss_hist)
    """

    device = resolve_torch_device(device)
    

    
    assert val_fraction > 0, "val_fraction must be > 0 to select best alpha based on validation loss"
    # compute train inds and val inds from val_fraction
    val_inds = np.random.choice(X.shape[0], size=int(X.shape[0] * val_fraction), replace=False)

    if alpha_grid is None:
        alpha_grid = np.logspace(-3, 3, 7)
    alpha_grid = np.sort(alpha_grid)

    Ws, bs, val_losses = [],[],[]
    history = {}
    W, b = None, None  # for warm starting across alphas

    for alpha in alpha_grid:
        print(f"\n--- Trying alpha = {alpha} ---")

        if optimizer_type.lower() == "lbfgs":
            result = fit_poisson_glm_lbfgs(
                X, Y,
                alpha=alpha,
                max_epochs=max_epochs,
                #val_fraction=val_fraction,
                val_inds=val_inds,
                early_stopping=early_stopping,
                patience=patience,
                tol=tol,
                device=device,
                per_target_loss=True,
                W_init=W,
                b_init=b,
                **fit_kwargs
            )
            W, b = result[0], result[1]
            train_loss_hist, val_loss_hist = result[2], result[3]
            train_bps_hist, val_bps_hist = result[4], result[5]
            val_loss_per_target = result[7] if len(result) > 7 else None
        elif optimizer_type.lower() == "adam":
            result = fit_poisson_glm_adam(
                X, Y,
                alpha=alpha,
                max_epochs=max_epochs,
                #val_fraction=val_fraction,
                val_inds=val_inds,
                early_stopping=early_stopping,
                patience=patience,
                tol=tol,
                device=device,
                per_target_loss=True,
                W_init=W,
                b_init=b,
                **fit_kwargs
            )
            W, b = result[0], result[1]
            train_loss_hist, val_loss_hist = result[2], result[3]
            train_bps_hist, val_bps_hist = result[4], result[5]
            val_loss_per_target = result[7] if len(result) > 7 else None
        else:
            raise ValueError("optimizer_type must be 'lbfgs' or 'adam'")

        if val_loss_per_target is None:
            raise ValueError("Expected per-target validation loss but received None.")

        history[alpha] = {
            "train_loss_hist": train_loss_hist,
            "val_loss_hist": val_loss_hist,
            "train_bps_hist": train_bps_hist,
            "val_bps_hist": val_bps_hist,
        }

        val_losses.append(np.sum(val_loss_per_target))
        Ws.append(W)
        bs.append(b)
        if not warm_start:
            # don't store solutions for warm starting across alphas, re-initialize W and b for each alpha
            W, b = None, None

    val_losses = np.array(val_losses) # (num_alphas,)
    val_losses_safe, finite_mask = _sanitize_val_losses(
        val_losses,
        context="global alpha selection",
    )
    if not np.any(finite_mask):
        raise RuntimeError(
            "All alpha candidates produced non-finite validation losses."
        )

    # check if finite losses are monotonically increasing or decreasing
    lossdiff = np.diff(val_losses_safe)
    decreasing = np.all(lossdiff < 0)
    increasing = np.all(lossdiff > 0)

    if np.any(decreasing):
        print(f'WARNING: Validation loss decreases monotonically across the alpha grid. Consider adding larger alpha values to the grid.')
    if np.any(increasing):
        print(f'WARNING: Validation loss increases monotonically across the alpha grid. Consider adding smaller alpha values to the grid.')

    # compute the best alpha across candidates
    best_alpha_idx = np.argmin(val_losses_safe, axis=0)
    best_alpha = alpha_grid[best_alpha_idx]
    best_W = Ws[best_alpha_idx]
    best_b = bs[best_alpha_idx]

    return best_W, best_b, best_alpha, history


# ============================================================
# -------------------- LBFGS Optimizer -----------------------
# ============================================================

def fit_poisson_glm_lbfgs(
    X,
    Y,
    alpha=0.0,
    max_epochs=1000,
    lbfgs_max_iter=20,
    line_search_fn="strong_wolfe",
    history_size=10,
    val_fraction=0.0,
    early_stopping=None, # 'train' or 'val' or None
    patience=10,
    tol=1e-8,
    print_every=1,
    seed=None,
    device=None,
    per_target_loss=False,
    val_inds=None,
    W_init=None,
    b_init=None
):
    return _fit_poisson_glm_lbfgs_impl(
        X,
        Y,
        alpha=alpha,
        max_epochs=max_epochs,
        lbfgs_max_iter=lbfgs_max_iter,
        line_search_fn=line_search_fn,
        history_size=history_size,
        val_fraction=val_fraction,
        early_stopping=early_stopping,
        patience=patience,
        tol=tol,
        print_every=print_every,
        seed=seed,
        device=device,
        per_target_loss=per_target_loss,
        val_inds=val_inds,
        W_init=W_init,
        b_init=b_init,
    )

# ============================================================
# -------------------- Adam Optimizer ------------------------
# ============================================================

def fit_poisson_glm_adam(
    X,
    Y,
    alpha=0.0,
    lr=1e-4,
    batch_size=2048,
    max_epochs=5000,
    val_fraction=0.0,
    early_stopping=None, # 'train' or 'val' or None
    patience=10,
    tol=1e-4,
    print_every=5,
    seed=None,
    device=None,
    eval_batch_size=None,
    per_target_loss=False,
    val_inds=None,
    W_init=None,
    b_init=None
):
    return _fit_poisson_glm_adam_impl(
        X,
        Y,
        alpha=alpha,
        lr=lr,
        batch_size=batch_size,
        max_epochs=max_epochs,
        val_fraction=val_fraction,
        early_stopping=early_stopping,
        patience=patience,
        tol=tol,
        print_every=print_every,
        seed=seed,
        device=device,
        eval_batch_size=eval_batch_size,
        per_target_loss=per_target_loss,
        val_inds=val_inds,
        W_init=W_init,
        b_init=b_init,
    )

def choose_optimizer(X, Y, buffer_factor=1.2,):
    raise NotImplementedError()
    """
    Decide whether to use LBFGS (full-batch) or Adam (minibatch) based on dataset size 
    and estimated memory requirements.

    Args:
        X (np.ndarray or torch.Tensor): Design matrix, shape (T, p)
        Y (np.ndarray or torch.Tensor): Response matrix, shape (T, N)
        buffer_factor (float): Safety factor for memory estimation. Defaults to 1.2.

    Returns:
        optimizer_choice (str): "lbfgs" for full-batch or "adam" for minibatch
        batch_size (int or None): None for LBFGS, recommended minibatch size for Adam
    """

    N = Y.shape[1]

    T,p = X.shape

    # convert float precision to bytes
    xbytes = X[0,0].nbytes
    ybytes = Y[0,0].nbytes
    # get datatype of X and Y to determine bytes per element
    X_mem = T * p * xbytes
    Y_mem = T * N * ybytes
    W_mem = p * N * xbytes
    b_mem = N * xbytes
    total_mem_needed = (X_mem + Y_mem + W_mem + b_mem) * buffer_factor
    print(f'Total memory needed for LBFGS: {total_mem_needed / 1e9:.2e} GB (X: {X_mem / 1e9:.2e} GB, Y: {Y_mem / 1e9:.2e} GB, W: {W_mem / 1e9:.2e} GB, b: {b_mem / 1e9:.2e} GB)')

    # Check CUDA memory if available
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory
        if total_mem_needed < gpu_mem:
            return "lbfgs", None

        batch_W_mem = p * N * xbytes
        batch_b_mem = N * xbytes
        batch_size = int((gpu_mem / buffer_factor - batch_W_mem - batch_b_mem) / (p * xbytes + N * ybytes))
        return "adam", max(1, batch_size)

    # MPS has no simple VRAM query via torch, so use conservative heuristic.
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and torch.backends.mps.is_available():
        if total_mem_needed < 8 * 1024**3:
            return "lbfgs", None
        batch_size = min(max(1, int(T * 0.01)), 4096)
        print("WARNING: MPS memory query unavailable; using heuristic batch-size recommendation.")
        return "adam", batch_size

    print('WARNING: NO GPU AVAILABLE; using CPU memory heuristic')
    cpu_mem_limit = 16 * 1024**3
    if total_mem_needed < cpu_mem_limit:
        return "lbfgs", None
    batch_size = min(max(1, int(T * 0.01)), 4096)
    return "adam", batch_size