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
import torch.nn.functional as F
import numpy as np
from .optim.adam import fit_poisson_glm_adam as _fit_poisson_glm_adam_impl
from .optim.lbfgs import fit_poisson_glm_lbfgs as _fit_poisson_glm_lbfgs_impl

CLAMP = 80 # effectively non-binding in most runs, but still below float32 exp overflow
# TODO: float64?


def _format_alpha(alpha, N, device, dtype=torch.float32):
    """Normalize alpha to shape (1, N) for consistent per-target regularization."""
    np_dtype = np.float64 if dtype == torch.float64 else np.float32
    alpha_arr = np.asarray(alpha, dtype=np_dtype)

    if alpha_arr.ndim == 0:
        alpha_arr = np.full((N,), float(alpha_arr), dtype=np_dtype)
    else:
        alpha_arr = alpha_arr.reshape(-1)
        if alpha_arr.size == 1:
            alpha_arr = np.full((N,), float(alpha_arr.item()), dtype=np_dtype)
        elif alpha_arr.size != N:
            raise ValueError(
                f"alpha must be a scalar or length-N array (N={N}), got shape {np.shape(alpha)}"
            )

    return torch.from_numpy(alpha_arr.reshape(1, N)).to(device=device, dtype=dtype)

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

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    assert val_fraction > 0, "val_fraction must be > 0 to select best alpha based on validation loss"
    # compute train inds and val inds from val_fraction
    val_inds = np.random.choice(X.shape[0], size=int(X.shape[0] * val_fraction), replace=False)

    if alpha_grid is None:
        alpha_grid = np.logspace(-3, 3, 7)
    alpha_grid = np.sort(alpha_grid)

    best_alpha = None
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
            train_loss_per_target = result[6] if len(result) > 6 else None
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
            train_loss_per_target = result[6] if len(result) > 6 else None
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
    # check if losses are monotonically increasing or decreaasing
    lossdiff = np.diff(val_losses, axis=0) 
    decreasing = np.all(lossdiff < 0, axis=0)
    increasing = np.all(lossdiff > 0, axis=0)

    if np.any(decreasing):
        print(f'WARNING: Validation loss decreases monotonically across the alpha grid for targets {np.where(decreasing)[0]}. Consider adding larger alpha values to the grid.')
    if np.any(increasing):
        print(f'WARNING: Validation loss increases monotonically across the alpha grid for targets {np.where(increasing)[0]}. Consider adding smaller alpha values to the grid.')

    # compute the best alpha per target
    best_alpha_idx = np.argmin(val_losses, axis=0)
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

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    

    
    assert val_fraction > 0, "val_fraction must be > 0 to select best alpha based on validation loss"
    # compute train inds and val inds from val_fraction
    val_inds = np.random.choice(X.shape[0], size=int(X.shape[0] * val_fraction), replace=False)

    if alpha_grid is None:
        alpha_grid = np.logspace(-3, 3, 7)
    alpha_grid = np.sort(alpha_grid)

    best_alpha = None
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
            train_loss_per_target = result[6] if len(result) > 6 else None
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
            train_loss_per_target = result[6] if len(result) > 6 else None
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

    val_losses = np.array(val_losses) # (num_alphas, N)
    # check if losses are monotonically increasing or decreaasing
    lossdiff = np.diff(val_losses)
    decreasing = np.all(lossdiff < 0)
    increasing = np.all(lossdiff > 0)

    if np.any(decreasing):
        print(f'WARNING: Validation loss decreases monotonically across the alpha grid. Consider adding larger alpha values to the grid.')
    if np.any(increasing):
        print(f'WARNING: Validation loss increases monotonically across the alpha grid. Consider adding smaller alpha values to the grid.')

    # compute the best alpha per target
    best_alpha_idx = np.argmin(val_losses, axis=0)
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


# ============================================================
# -------------------- Shared Utilities ----------------------
# ============================================================

def _check_for_bad_convergence(loss_hist, tol=1e-4, patience=5):
    # if loss tanks way down at the end, raise an error
    loss_diff = np.diff(loss_hist[-patience:])
    if np.any(loss_diff < -tol):
        raise RuntimeError(
            "Warning: Loss decreased by more than "
            f"{tol} in the last {patience} epochs. "
            "This may indicate bad convergence. "
            "Consider increasing max_epochs or adjusting optimizer parameters."
        )

def _prepare_data(X, Y, val_fraction, val_inds=None, seed=None):
    if val_inds is not None and val_fraction > 0:
        raise ValueError("Only one of val_inds or val_fraction should be provided.")

    if val_inds is not None:
        # assert not a boolean mask
        if isinstance(val_inds, np.ndarray) and val_inds.dtype == bool:
            raise ValueError("val_inds should be an array of indices, not a boolean mask.")
        assert np.all((val_inds >= 0) & (val_inds < X.shape[0])), "val_inds must be valid indices for X"
        mask = np.ones(X.shape[0], dtype=bool)
        mask[val_inds] = False
        X_train = X[mask]
        Y_train = Y[mask]
        X_val = X[val_inds]
        Y_val = Y[val_inds]
        has_val = True
    else:
        rng = np.random.default_rng(seed)
        T = X.shape[0]
        idx = np.arange(T)
        rng.shuffle(idx)
        if val_fraction > 0:
            split = int(T * (1 - val_fraction))
            train_idx, val_idx = idx[:split], idx[split:]
            X_train, Y_train = X[train_idx], Y[train_idx]
            X_val, Y_val = X[val_idx], Y[val_idx]
            has_val = True
        else:
            X_train, Y_train = X, Y
            X_val, Y_val = None, None
            has_val = False

    return X_train, Y_train, X_val, Y_val, has_val


def _initialize_params(p, N, mean_rates, device, dtype=torch.float32):
    #b = torch.zeros(N, device=device, requires_grad=True)
    #W = 0.01 * torch.randn(p, N, device=device, requires_grad=True)
    W = torch.randn(p, N, device=device, dtype=dtype) * 0.01
    W.requires_grad_(True)
    b = torch.log(mean_rates.to(dtype=dtype) + 1e-8).to(device=device, dtype=dtype).requires_grad_()
    
    return W, b

def _poisson_loss(W, b, X, Y, alpha=None):
    eta = torch.clamp(X @ W + b, max=CLAMP)
    #exp_eta = torch.exp(eta)
    #eta = X @ W + b
    # apply alpha per target
    data_loss = torch.nn.functional.poisson_nll_loss(
                                                    input=eta,        # NOTE: log-rate
                                                    target=Y,
                                                    log_input=True,
                                                    full=False,
                                                    reduction="mean")
    #data_loss2 = torch.sum(exp_eta - Y * eta)
    if alpha is not None:
        # with penalty (used for fitting)
        #return torch.sum(exp_eta - Y * eta) + torch.sum(alpha * torch.sum(W**2, dim=0))
        return data_loss + torch.sum(alpha * torch.sum(W**2, dim=0))
    else:
        # raw nll
        #return torch.sum(exp_eta - Y * eta)
        return data_loss

def _poisson_loss_per_target(W, b, X, Y, alpha=None):
    eta = torch.clamp(X @ W + b, max=CLAMP)  # (T, N)
    #eta = X @ W + b                  # (T, N)
    exp_eta = torch.exp(eta)         # (T, N)
    data_loss = torch.sum(exp_eta - Y * eta, dim=0)
    if alpha is None:
        # raw nll
        return data_loss
    else:
        # with penalty (used for fitting)
        l2_per_target = torch.sum(W**2, dim=0)
        reg_loss = alpha * l2_per_target
        return data_loss + reg_loss
    

#def _poisson_deviance_loss(W, b, X, Y, alpha):
#    """
#    Poisson deviance loss with L2 regularization.
#    Loss is normalized by number of samples, but 
#    the gradient is more complex to compute.
#    """
#    N = X.shape[0]
#    eta = torch.clamp(X @ W + b, max=20)
#    mu = torch.exp(eta)
#    # Poisson deviance per sample
#    deviance = 2 * (Y * (torch.log((Y + 1e-8) / mu) - 1) + mu)
#    return torch.sum(deviance) / N + alpha * torch.sum(W**2)


def _evaluate_streamed(W, b, X_cpu, Y_cpu, alpha, device, eval_batch_size):
    with torch.no_grad():
        log2 = torch.log(torch.tensor(2.0, device=device))
        eps = 1e-12

        total_nll = 0.0
        logL_model = 0.0
        logL_null = 0.0
        total_spikes = 0.0

        mean_rate = torch.mean(Y_cpu, dim=0, keepdim=True).to(device)

        for start in range(0, X_cpu.shape[0], eval_batch_size):
            end = min(start + eval_batch_size, X_cpu.shape[0])

            Xb = X_cpu[start:end].to(device, non_blocking=True)
            Yb = Y_cpu[start:end].to(device, non_blocking=True)

            #eta = Xb @ W + b   # NO clamp
            eta = torch.clamp(Xb @ W + b, max=CLAMP)

            # ✅ PyTorch NLL (correct loss)
            total_nll += F.poisson_nll_loss(
                eta, Yb,
                log_input=True,
                full=False,
                reduction="sum"
            )

            # still need log-likelihood for BPS
            exp_eta = torch.exp(eta)

            logL_model += torch.sum(Yb * eta - exp_eta)
            logL_null += torch.sum(
                Yb * torch.log(mean_rate + eps) - mean_rate
            )

            total_spikes += torch.sum(Yb)

            del Xb, Yb, eta, exp_eta

        if alpha is not None:
            total_nll += torch.sum(alpha * torch.sum(W**2, dim=0))

        bps = (logL_model - logL_null) / (total_spikes * log2)

    return total_nll, bps


def _evaluate_full_gpu(W, b, X, Y, alpha):
    log2 = torch.log(torch.tensor(2.0, device=X.device))
    eps = 1e-12

    with torch.no_grad():
        loss = _poisson_loss(W, b, X, Y, alpha)

        eta = torch.clamp(X @ W + b, max=CLAMP)
        exp_eta = torch.exp(eta)

        mean_rate = torch.mean(Y, dim=0, keepdim=True)
        logL_model = torch.sum(Y * eta - exp_eta)
        logL_null = torch.sum(Y * torch.log(mean_rate + eps) - mean_rate)

        bps = (logL_model - logL_null) / (torch.sum(Y) * log2)

    return loss, bps

    
def _print_progress(epoch, train_loss, train_bps,
                    has_val, val_loss, val_bps,
                    print_every):
    if epoch % print_every == 0:
        msg = (
            f"Epoch {epoch:4d} | "
            f"Train Loss: {train_loss:.5e} | "
            f"Train BPS: {train_bps:.5f}"
        )
        if has_val:
            msg += (
                f" | Val Loss: {val_loss:.5e} | "
                f"Val BPS: {val_bps:.5f}"
            )
        print(msg) 

#####


def choose_optimizer(X, Y, buffer_factor=1.2,):
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

    # Check GPU memory
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory
        if total_mem_needed < gpu_mem:
            return "lbfgs", None
        else:
            # pick minibatch size so that it would fit in GPU memory
            # We can estimate the memory for a single batch as:
            batch_W_mem = p * N * xbytes
            batch_b_mem = N * xbytes
            batch_size = int((gpu_mem / buffer_factor - batch_W_mem - batch_b_mem) / (p * xbytes + N * ybytes))
            return "adam", batch_size
    else:
        print('WARNING: NO GPU AVAILIBLE')
        return None, None
        # CPU fallback: assume ~16GB available, same logic
        cpu_mem_limit = 16 * 1024**3
        if total_mem_needed < cpu_mem_limit:
            return "lbfgs", None
        else:
            batch_size = min(max(1, int(T * 0.01)), 4096)
            return "adam", batch_size