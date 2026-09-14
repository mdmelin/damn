import numpy as np
import torch
import torch.nn.functional as F

CLAMP = 80  # effectively non-binding in most runs, but still below float32 exp overflow


def format_alpha(alpha, n_targets, device, dtype=torch.float32):
    """Normalize alpha to shape (1, N) for consistent per-target regularization."""
    np_dtype = np.float64 if dtype == torch.float64 else np.float32
    alpha_arr = np.asarray(alpha, dtype=np_dtype)

    if alpha_arr.ndim == 0:
        alpha_arr = np.full((n_targets,), float(alpha_arr), dtype=np_dtype)
    else:
        alpha_arr = alpha_arr.reshape(-1)
        if alpha_arr.size == 1:
            alpha_arr = np.full((n_targets,), float(alpha_arr.item()), dtype=np_dtype)
        elif alpha_arr.size != n_targets:
            raise ValueError(
                f"alpha must be a scalar or length-N array (N={n_targets}), got shape {np.shape(alpha)}"
            )

    return torch.from_numpy(alpha_arr.reshape(1, n_targets)).to(device=device, dtype=dtype)


def prepare_data(x, y, val_fraction, val_inds=None, seed=None):
    if val_inds is not None and val_fraction > 0:
        raise ValueError("Only one of val_inds or val_fraction should be provided.")

    if val_inds is not None:
        if isinstance(val_inds, np.ndarray) and val_inds.dtype == bool:
            raise ValueError("val_inds should be an array of indices, not a boolean mask.")
        if not np.all((val_inds >= 0) & (val_inds < x.shape[0])):
            raise ValueError("val_inds must be valid indices for X")
        mask = np.ones(x.shape[0], dtype=bool)
        mask[val_inds] = False
        x_train = x[mask]
        y_train = y[mask]
        x_val = x[val_inds]
        y_val = y[val_inds]
        has_val = True
    else:
        rng = np.random.default_rng(seed)
        t = x.shape[0]
        idx = np.arange(t)
        rng.shuffle(idx)
        if val_fraction > 0:
            split = int(t * (1 - val_fraction))
            train_idx, val_idx = idx[:split], idx[split:]
            x_train, y_train = x[train_idx], y[train_idx]
            x_val, y_val = x[val_idx], y[val_idx]
            has_val = True
        else:
            x_train, y_train = x, y
            x_val, y_val = None, None
            has_val = False

    return x_train, y_train, x_val, y_val, has_val


def initialize_params(n_features, n_targets, mean_rates, device, dtype=torch.float32):
    w = torch.randn(n_features, n_targets, device=device, dtype=dtype) * 0.01
    w.requires_grad_(True)
    b = torch.log(mean_rates.to(dtype=dtype) + 1e-8).to(device=device, dtype=dtype).requires_grad_()
    return w, b


def poisson_loss(w, b, x, y, alpha=None):
    eta = torch.clamp(x @ w + b, max=CLAMP)
    data_loss = F.poisson_nll_loss(
        input=eta,
        target=y,
        log_input=True,
        full=False,
        reduction="mean",
    )
    if alpha is not None:
        return data_loss + torch.sum(alpha * torch.sum(w**2, dim=0))
    return data_loss


def poisson_loss_per_target(w, b, x, y, alpha=None):
    eta = torch.clamp(x @ w + b, max=CLAMP)
    exp_eta = torch.exp(eta)
    data_loss = torch.sum(exp_eta - y * eta, dim=0)
    if alpha is None:
        return data_loss
    l2_per_target = torch.sum(w**2, dim=0)
    return data_loss + alpha * l2_per_target


def evaluate_streamed(w, b, x_cpu, y_cpu, alpha, device, eval_batch_size):
    with torch.no_grad():
        log2 = torch.log(torch.tensor(2.0, device=device))
        eps = 1e-12

        total_nll = 0.0
        logl_model = 0.0
        logl_null = 0.0
        total_spikes = 0.0

        mean_rate = torch.mean(y_cpu, dim=0, keepdim=True).to(device)

        for start in range(0, x_cpu.shape[0], eval_batch_size):
            end = min(start + eval_batch_size, x_cpu.shape[0])

            xb = x_cpu[start:end].to(device, non_blocking=True)
            yb = y_cpu[start:end].to(device, non_blocking=True)

            eta = torch.clamp(xb @ w + b, max=CLAMP)

            total_nll += F.poisson_nll_loss(
                eta,
                yb,
                log_input=True,
                full=False,
                reduction="sum",
            )

            exp_eta = torch.exp(eta)
            logl_model += torch.sum(yb * eta - exp_eta)
            logl_null += torch.sum(yb * torch.log(mean_rate + eps) - mean_rate)
            total_spikes += torch.sum(yb)

            del xb, yb, eta, exp_eta

        if alpha is not None:
            total_nll += torch.sum(alpha * torch.sum(w**2, dim=0))

        bps = (logl_model - logl_null) / (total_spikes * log2)

    return total_nll, bps


def evaluate_full_gpu(w, b, x, y, alpha):
    log2 = torch.log(torch.tensor(2.0, device=x.device))
    eps = 1e-12

    with torch.no_grad():
        loss = poisson_loss(w, b, x, y, alpha)

        eta = torch.clamp(x @ w + b, max=CLAMP)
        exp_eta = torch.exp(eta)

        mean_rate = torch.mean(y, dim=0, keepdim=True)
        logl_model = torch.sum(y * eta - exp_eta)
        logl_null = torch.sum(y * torch.log(mean_rate + eps) - mean_rate)

        bps = (logl_model - logl_null) / (torch.sum(y) * log2)

    return loss, bps


def print_progress(epoch, train_loss, train_bps, has_val, val_loss, val_bps, print_every):
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
