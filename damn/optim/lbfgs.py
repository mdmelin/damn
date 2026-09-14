import numpy as np
import torch

from .common import (
    evaluate_full_gpu,
    format_alpha,
    initialize_params,
    poisson_loss,
    poisson_loss_per_target,
    prepare_data,
    print_progress,
)


def fit_poisson_glm_lbfgs(
    X,
    Y,
    alpha=0.0,
    max_epochs=1000,
    lbfgs_max_iter=20,
    line_search_fn="strong_wolfe",
    history_size=10,
    val_fraction=0.0,
    early_stopping=None,
    patience=10,
    tol=1e-8,
    print_every=1,
    seed=None,
    device=None,
    per_target_loss=False,
    val_inds=None,
    W_init=None,
    b_init=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        torch.cuda.empty_cache()

    bad_cols = np.where(
        np.all(X == 0, axis=0)
        | np.all(np.isnan(X), axis=0)
        | np.all(X == X[0, :], axis=0)
    )[0]
    good_cols = np.where(
        ~(
            np.all(X == 0, axis=0)
            | np.all(np.isnan(X), axis=0)
            | np.all(X == X[0, :], axis=0)
        )
    )[0]
    num_cols = X.shape[1]
    print(f"Removing {len(bad_cols)} bad columns with all zeros, all nans, or all the same value")
    X = np.delete(X, bad_cols, axis=1)

    X_train, Y_train, X_val, Y_val, has_val = prepare_data(
        X, Y, val_fraction, val_inds, seed
    )

    dtype = torch.float32
    X_train = torch.from_numpy(X_train).to(device=device, dtype=dtype)
    Y_train = torch.from_numpy(Y_train).to(device=device, dtype=dtype)

    N = Y_train.shape[1]
    alpha = format_alpha(alpha, N, device, dtype=dtype)

    if has_val:
        X_val = torch.from_numpy(X_val).to(device=device, dtype=dtype)
        Y_val = torch.from_numpy(Y_val).to(device=device, dtype=dtype)

    _, p = X_train.shape

    if W_init is None and b_init is None:
        mean_rates = torch.mean(Y_train, dim=0)
        W, b = initialize_params(p, N, mean_rates, device, dtype=dtype)
    else:
        if len(bad_cols) > 0 and W_init is not None:
            W_init = np.delete(W_init, bad_cols, axis=0)
        W = torch.from_numpy(W_init).to(device=device, dtype=dtype)
        b = torch.from_numpy(b_init).to(device=device, dtype=dtype)
        with torch.no_grad():
            W += 0.0001 * torch.randn_like(W)
            b += 0.0001 * torch.randn_like(b)
        W.requires_grad_()
        b.requires_grad_()

    optimizer = torch.optim.LBFGS(
        [W, b],
        max_iter=lbfgs_max_iter,
        line_search_fn=line_search_fn,
        history_size=history_size,
    )

    train_loss_hist, val_loss_hist = [], []
    train_bps_hist, val_bps_hist = [], []

    best_monitor = float("inf")
    epochs_no_improve = 0
    val_loss = None
    val_bps = None

    for epoch in range(max_epochs):

        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss = poisson_loss(W, b, X_train, Y_train, alpha)
            loss.backward()
            return loss

        optimizer.step(closure)

        train_loss, train_bps = evaluate_full_gpu(W, b, X_train, Y_train, alpha)
        train_loss_hist.append(float(train_loss))
        train_bps_hist.append(float(train_bps))

        if has_val:
            val_loss, val_bps = evaluate_full_gpu(W, b, X_val, Y_val, alpha)
            val_loss_hist.append(float(val_loss))
            val_bps_hist.append(float(val_bps))

        val_loss_value = float(val_loss) if (has_val and val_loss is not None) else None
        val_bps_value = float(val_bps) if (has_val and val_bps is not None) else None

        print_progress(
            epoch,
            float(train_loss),
            float(train_bps),
            has_val,
            val_loss_value,
            val_bps_value,
            print_every,
        )

        if early_stopping is not None:
            if early_stopping == "val" and not has_val:
                raise ValueError("Early stopping on validation loss requested but no validation set provided.")
            if early_stopping == "val":
                if val_loss is None:
                    raise ValueError("Validation loss is unavailable for early stopping.")
                monitor = float(val_loss)
            elif early_stopping == "train":
                monitor = float(train_loss)
            else:
                raise ValueError("early_stopping must be 'train', 'val', or None.")

            if best_monitor - monitor > tol:
                best_monitor = monitor
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(
                        f"Early stopping triggered at epoch {epoch}. "
                        f"No improvement greater than {tol} "
                        f"for {patience} consecutive epochs."
                    )
                    break
    else:
        print(f"Warning: Reached max_epochs ({max_epochs})")

    Wcpu = W.detach().cpu().numpy()
    bcpu = b.detach().cpu().numpy()

    if device == "cuda":
        torch.cuda.empty_cache()

    if len(bad_cols) > 0:
        Wcpu_full = np.zeros((num_cols, N), dtype=Wcpu.dtype)
        Wcpu_full[good_cols, :] = Wcpu
        Wcpu = Wcpu_full

    if not per_target_loss:
        return (
            Wcpu,
            bcpu,
            train_loss_hist,
            val_loss_hist,
            train_bps_hist,
            val_bps_hist,
        )

    train_per_target_loss = poisson_loss_per_target(W, b, X_train, Y_train)
    val_per_target_loss = poisson_loss_per_target(W, b, X_val, Y_val) if has_val else None
    val_per_target_loss_np = (
        val_per_target_loss.detach().cpu().numpy() if val_per_target_loss is not None else None
    )

    return (
        Wcpu,
        bcpu,
        train_loss_hist,
        val_loss_hist,
        train_bps_hist,
        val_bps_hist,
        train_per_target_loss.detach().cpu().numpy(),
        val_per_target_loss_np,
    )
