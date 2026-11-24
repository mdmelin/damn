import numpy as np
from .alignment import construct_timebins

'''Bases functions and other tools for building GLM design matrices.'''

###########################################
#### Basis functions with no time lags ####
#### Most useful for continuous signals ###
#### These functions preserve the overall #
#### magnitude of the original signal #####
###########################################

def no_basis():
    # No basis function, just return a single delta function at time zero
    return np.ones([1,1]), np.zeros((1,))

def boxcar_smooth(pre_s, post_s, binwidth_s):
    # Boxcar moving average basis function
    t, edges,_ = construct_timebins(pre_s, post_s, binwidth_s)
    T = len(t)
    basis = np.ones((T, 1)) / T
    return basis, t

def gaussian_smooth(pre_s, post_s, binwidth_s):
    # Gaussian moving average basis function
    t, edges,_ = construct_timebins(pre_s, post_s, binwidth_s)
    T = len(t)
    sigma = (post_s + pre_s) / 6  # 99.7% of Gaussian within the window
    basis = np.exp(-0.5 * (t / sigma) ** 2)
    basis /= basis.sum()  # Normalize
    basis = basis.reshape(-1, 1)
    return basis, t

def polynomial(degree):
    raise NotImplementedError("Polynomial basis function not implemented here. You can achieve the same effect by adding another regressor with your regressor multiplied by the desired polynomial degree.")
    # Would be trickly to implement here, basis functions right now are agnostic to the input signal, 
    # but a polynomial is not.
    pass
    
    
###########################################
#### Basis functions with lags availible ###
###########################################

def delta_basis(pre_s, post_s, binwidth_s):
    """
    Construct a delta (impulse) basis over time.

    Each basis function is 1 at a single time bin and 0 elsewhere.
    This can be thought of as a special case of the FIR filter, 
    where the width of each boxcar is just one time bin.

    Parameters
    ----------
    pre_seconds : float
        Time before zero (positive value).
    post_seconds : float
        Time after zero.
    binwidth_s : float
        Width of time bins in seconds.

    Returns
    -------
    basis : ndarray, shape (T, T)
        Delta basis matrix (identity).
    t : ndarray, shape (T,)
        Time vector spanning [-pre_seconds, post_seconds].
    """
    # Time axis
    t = construct_timebins(pre_s, post_s, binwidth_s)[0]
    T = len(t)

    # Delta basis = identity matrix
    basis = np.eye(T)
    # flip the basis so the first index is earliest in time

    return basis, t
        
def raised_cosine_basis(n_funcs, pre_s, post_s, binwidth_s, log_scale=False):
    """
    Construct a raised cosine basis over time.

    Parameters
    ----------
    n_basis_funcs : int
        Number of basis functions.
    pre_seconds : float
        Time before zero (positive value).
    post_seconds : float
        Time after zero.
    binwidth : float
        Width of time bins in seconds.
    log_scale : bool, default False
        If True, basis centers are spaced logarithmically (compressed near zero).

    Returns
    -------
    basis : ndarray, shape (T, n_basis_funcs)
        Raised cosine basis matrix.
    """
    # Time axis
    t, edges,_ = construct_timebins(pre_s, post_s, binwidth_s)
    T = len(t)

    if log_scale:
        if pre_s > 0 and post_s > 0:
            # warn the user that this is probably a bad idea
            raise ValueError("Log-scale basis with both pre and post times > 0 is not supported.")
            
        eps = binwidth_s
        t_pos = t + pre_s + eps

        # Log-time
        log_t = np.log(t_pos)

        centers = np.linspace(log_t.min(), log_t.max(), n_funcs)
        width = centers[1] - centers[0]

        basis = np.zeros((T, n_funcs))
        for k, c in enumerate(centers):
            x = (log_t - c) * np.pi / width
            basis[:, k] = 0.5 * (1 + np.cos(np.clip(x, -np.pi, np.pi)))
            basis[np.abs(x) >= np.pi, k] = 0.0
        if pre_s > 0:
            basis = basis[::-1, :]

    else:
        # Linear spacing
        centers = np.linspace(t[0], t[-1], n_funcs)
        width = centers[1] - centers[0]

        basis = np.zeros((T, n_funcs))
        for k, c in enumerate(centers):
            x = (t - c) * np.pi / width
            basis[:, k] = 0.5 * (1 + np.cos(np.clip(x, -np.pi, np.pi)))
            basis[np.abs(x) >= np.pi, k] = 0.0

    return basis,t

def gaussian_basis(n_funcs, pre_s, post_s, binwidth_s, sigma=None):
    """
    Construct a Gaussian basis over time.

    Parameters
    ----------
    n_basis_funcs : int
        Number of basis functions.
    pre_seconds : float
        Time before zero (positive value).
    post_seconds : float
        Time after zero.
    binwidth : float
        Width of time bins in seconds.
    sigma : float, optional
        Standard deviation of the Gaussians. If None, set to spacing between centers.

    Returns
    -------
    basis : ndarray, shape (T, n_basis_funcs)
        Gaussian basis matrix.
    """
    # Time axis
    t, edges,_ = construct_timebins(pre_s, post_s, binwidth_s)
    T = len(t)

    centers = np.linspace(t[0], t[-1], n_funcs)
    if sigma is None:
        sigma = (centers[1] - centers[0])  # Default sigma to spacing between centers

    basis = np.zeros((T, n_funcs))
    for k, c in enumerate(centers):
        basis[:, k] = np.exp(-0.5 * ((t - c) / sigma) ** 2)

    return basis, t


def bspline_basis(n_funcs, pre_s, post_s, binwidth_s, degree=3):
    """
    Construct a B-spline basis over time.

    Parameters
    ----------
    n_basis_funcs : int
        Number of basis functions.
    pre_seconds : float
        Time before zero (positive value).
    post_seconds : float
        Time after zero.
    binwidth : float
        Width of time bins in seconds.
    degree : int, default 3
        Degree of the B-splines.

    Returns
    -------
    basis : ndarray, shape (T, n_basis_funcs)
        B-spline basis matrix.
    """
    from scipy.interpolate import BSpline

    # Time axis
    t, edges,_ = construct_timebins(pre_s, post_s, binwidth_s)
    T = len(t)

    # Knot vector
    n_knots = n_funcs + degree + 1
    knots = np.linspace(t[0], t[-1], n_knots - 2 * degree)
    knots = np.concatenate(([t[0]] * degree, knots, [t[-1]] * degree))

    basis = np.zeros((T, n_funcs))
    for i in range(n_funcs):
        coeffs = np.zeros(n_funcs)
        coeffs[i] = 1
        spline = BSpline(knots, coeffs, degree)
        basis[:, i] = spline(t)

    return basis, t

def fir_basis(impulse_binsize_s, pre_s, post_s, binwidth_s):
    '''
    Construct a Finite Impulse Response (FIR) basis over time.
    This filter looks like a series of boxcars, so it smooths and includes lags.
    Basically the same as boxcar_smooth() but now takes an argument for the number of basis functions.
    '''
    if pre_s > 0 and post_s > 0:
        print('Warning: FIR basis with both pre and post times > 0 may have a poorly defined zero-time, its best to create two FIR bases moving forward and backward from the event.')
    # Time axis
    t, edges,_ = construct_timebins(pre_s, post_s, binwidth_s)
    T = len(t)
    n_funcs = int(np.ceil((pre_s + post_s) / impulse_binsize_s))
    basis = np.zeros((T, n_funcs))
    for k in range(n_funcs):
        start_time = k * impulse_binsize_s - pre_s
        end_time = start_time + impulse_binsize_s
        basis[:, k] = ((t >= start_time) & (t < end_time)).astype(float)
    return basis, t


##############################################
#### Basis functions for periodic signals ####
##############################################

def von_mises_basis():
    raise NotImplementedError("Von Mises basis functions not implemented yet.")

def morelet_wavelet_basis():
    raise NotImplementedError("Morlet wavelet basis not implemented yet.")

def fourier_basis():
    raise NotImplementedError("Fourier basis functions not implemented yet.")

def circular_spline_basis():
    raise NotImplementedError("Circular spline basis functions not implemented yet.")

#############################################
##### Spatial basis functions, used #########
##### for estimating spatially dependent ####
##### things like receptive/place fields ####
#############################################

# TODO: figure out the bes way to do this. Maybe these aren't needed at all, we can just make multiple of the 
# above basis functions and apply them to spatial dimensions. SptatialRegressor?

def radial_basis():
    raise NotImplementedError("Spatial radial basis functions not implemented yet.")

def spatial_gaussian_basis():
    raise NotImplementedError("Spatial Gaussian basis functions not implemented yet.")
##########################################
#### Helper funcitons to create bases ####
##########################################

def _make_morlet_wavelet(frequency, srate, n_cycles=7):
    """
    Create a Morlet wavelet.

    Parameters
    ----------
    frequency : float
        Center frequency of the wavelet in Hz.
    srate : float
        Sampling rate in Hz.
    n_cycles : int, default 7
        Number of cycles in the wavelet.

    Returns
    -------
    wavelet : ndarray
        Morlet wavelet.
    t : ndarray
        Time vector corresponding to the wavelet.
    """
    # Time vector
    sigma_t = n_cycles / (2 * np.pi * frequency)
    t_max = 3 * sigma_t
    t = np.arange(-t_max, t_max, 1 / srate)

    # Morlet wavelet
    A = 1 / (sigma_t * np.sqrt(2 * np.pi))
    wavelet = A * np.exp(-t**2 / (2 * sigma_t**2)) * np.exp(1j * 2 * np.pi * frequency * t)

    return wavelet, t

basis_functions = dict(
    raised_cosine=raised_cosine_basis,
    delta=delta_basis,
    bspline=bspline_basis,
    no_basis=no_basis,
    boxcar_smooth=boxcar_smooth,
    gaussian_smooth=gaussian_smooth,
    fir=fir_basis,
    gaussian_basis=gaussian_basis
)
