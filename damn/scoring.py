import numpy as np

def bits_per_spike(y_true, y_pred):
    '''an implementation of bits per spike from
    Gerstner, W., Kistler, W. M., Naud, R., & Paninski, L. Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition. Chapter 10. “Evaluating Goodness-of-fit”. 2014.
    '''
    eps = 1e-12
    # Keep predicted rates finite and strictly positive for stable log-likelihood math.
    max_rate = 1e12
    y_pred = np.nan_to_num(y_pred, nan=eps, posinf=max_rate, neginf=eps)
    y_pred = np.clip(y_pred, eps, max_rate)

    # Model log-likelihood
    ll_model = np.sum(y_true * np.log(y_pred) - y_pred)

    # Null model (mean firing rate)
    mean_rate = np.mean(y_true)
    ll_null = np.sum(y_true * np.log(mean_rate + eps) - mean_rate)

    total_spikes = np.sum(y_true)

    return (ll_model - ll_null) / (np.log(2) * total_spikes)

def bits_per_spike_multi_target(y_true, y_pred):
    scores = []
    for i in range(y_true.shape[1]):
        scores.append(bits_per_spike(y_true[:, i], y_pred[:, i]))
    return np.array(scores)

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def r_squared_multi_target(y_true, y_pred):
    scores = []
    for i in range(y_true.shape[1]):
        scores.append(r_squared(y_true[:, i], y_pred[:, i]))
    return np.array(scores)