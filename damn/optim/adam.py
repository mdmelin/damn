import torch

from .common import (
    evaluate_streamed,
    format_alpha,
    initialize_params,
    poisson_loss,
    poisson_loss_per_target,
    prepare_data,
    print_progress,
)


def fit_poisson_glm_adam(
    X,
    Y,
    alpha=0.0,
    lr=1e-4,
    batch_size=2048,
    max_epochs=5000,
    val_fraction=0.0,
    early_stopping=None,
    patience=10,
    tol=1e-4,
    print_every=5,
    seed=None,
    device=None,
    eval_batch_size=None,
    per_target_loss=False,
    val_inds=None,
    W_init=None,
    b_init=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        torch.cuda.empty_cache()

    X_train, Y_train, X_val, Y_val, has_val = prepare_data(
        X, Y, val_fraction, val_inds, seed
    )

    X_train_cpu = torch.from_numpy(X_train).float().pin_memory()
    Y_train_cpu = torch.from_numpy(Y_train).float().pin_memory()

    N = Y_train_cpu.shape[1]
    alpha = format_alpha(alpha, N, device, dtype=torch.float32)

    X_val_cpu = None
    Y_val_cpu = None
    if has_val:
        X_val_cpu = torch.from_numpy(X_val).float().pin_memory()
        Y_val_cpu = torch.from_numpy(Y_val).float().pin_memory()

    T_train, p = X_train_cpu.shape

    if W_init is None and b_init is None:
        mean_rates = torch.mean(Y_train_cpu, dim=0)
        W, b = initialize_params(p, N, mean_rates, device, dtype=torch.float32)
    else:
        W = torch.from_numpy(W_init).float().to(device)
        b = torch.from_numpy(b_init).float().to(device)
        with torch.no_grad():
            W += 0.01 * torch.randn_like(W)
            b += 0.01 * torch.randn_like(b)
        W.requires_grad_()
        b.requires_grad_()

    optimizer = torch.optim.Adam([W, b], lr=lr)

    if eval_batch_size is None:
        eval_batch_size = batch_size

    train_loss_hist, val_loss_hist = [], []
    train_bps_hist, val_bps_hist = [], []

    best_monitor = float("inf")
    epochs_no_improve = 0

    train_loss, train_bps = evaluate_streamed(
        W, b, X_train_cpu, Y_train_cpu, alpha, device, eval_batch_size
    )
    val_loss = None
    val_bps = None
    if has_val:
        val_loss, val_bps = evaluate_streamed(
            W, b, X_val_cpu, Y_val_cpu, alpha, device, eval_batch_size
        )

    for epoch in range(max_epochs):
        perm = torch.randperm(T_train)
        for start in range(0, T_train, batch_size):
            end = min(start + batch_size, T_train)
            idx = perm[start:end]

            xb = X_train_cpu[idx].to(device, non_blocking=True)
            yb = Y_train_cpu[idx].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            loss = poisson_loss(W, b, xb, yb, alpha)
            loss.backward()
            optimizer.step()

            del xb, yb, loss

        if epoch % print_every == 0 or epoch == max_epochs - 1:
            train_loss, train_bps = evaluate_streamed(
                W, b, X_train_cpu, Y_train_cpu, alpha, device, eval_batch_size
            )
            train_loss_hist.append(float(train_loss))
            train_bps_hist.append(float(train_bps))

            if has_val:
                val_loss, val_bps = evaluate_streamed(
                    W, b, X_val_cpu, Y_val_cpu, alpha, device, eval_batch_size
                )
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

    if not per_target_loss:
        return (
            Wcpu,
            bcpu,
            train_loss_hist,
            val_loss_hist,
            train_bps_hist,
            val_bps_hist,
        )

    train_per_target_loss = poisson_loss_per_target(W, b, X_train_cpu.to(device), Y_train_cpu.to(device))
    val_per_target_loss = (
        poisson_loss_per_target(W, b, X_val_cpu.to(device), Y_val_cpu.to(device))
        if has_val and X_val_cpu is not None and Y_val_cpu is not None
        else None
    )
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
