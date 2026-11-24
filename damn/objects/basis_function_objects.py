import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod
from ..basis_functions import *

# =========================
# Basis function objects
# =========================

class BasisFunction(ABC):
    '''
    An abstract base class to handle basis functions that are employed by the Regressor objects.
    Importantly, we don't store anything about the design matrix, or the coefficients for a particular basis function.
    This is because the same basis function can be provided to multiple Regressors, while allowing each Regressor to have
    its own set of coefficients for that basis function.
    '''
    def __init__(self, pre_s, post_s, binwidth_s):
        self.pre_s = pre_s
        self.post_s = post_s
        self.binwidth_s = binwidth_s
        self.basis = None
        self.basis_time = None
        self.t0 = None
        if self.binwidth_s is not None:
            self.basis, self.basis_time = self.construct_basis()
            self.t0 = np.searchsorted(self.basis_time, 0.0)

    @abstractmethod
    def construct_basis(self) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def set_binwidth(self, binwidth_s):
        if binwidth_s is None:
            raise ValueError("binwidth_s cannot be None")

        if self.binwidth_s is None:
            self.binwidth_s = binwidth_s
            self.basis, self.basis_time = self.construct_basis()
            self.t0 = np.searchsorted(self.basis_time, 0.0)
            return

        if not np.isclose(self.binwidth_s, binwidth_s):
            raise ValueError(
                f"Basis binwidth_s={self.binwidth_s} does not match requested {binwidth_s}"
            )

        if self.basis is None or self.basis_time is None:
            self.basis, self.basis_time = self.construct_basis()
            self.t0 = np.searchsorted(self.basis_time, 0.0)

    def ensure_constructed(self, binwidth_s=None):
        if self.basis is not None and self.basis_time is not None:
            return

        if self.binwidth_s is None:
            if binwidth_s is None:
                raise ValueError(
                    f"{self.__class__.__name__} needs binwidth_s before it can be constructed"
                )
            self.set_binwidth(binwidth_s)
            return

        self.basis, self.basis_time = self.construct_basis()
        self.t0 = np.searchsorted(self.basis_time, 0.0)

    def reconstruct_kernel(self, coeffs):
        self.ensure_constructed()
        if self.basis is None or self.basis_time is None:
            raise RuntimeError("Basis is not constructed")
        if len(coeffs) != self.basis.shape[1]:
            raise ValueError(
                f"Coefficient length {len(coeffs)} does not match "
                f"number of basis functions {self.basis.shape[1]}"
            )

        kernel = self.basis @ coeffs
        return kernel, self.basis_time

    def __getitem__(self, key):
        self.ensure_constructed()
        if self.basis is None:
            raise RuntimeError("Basis is not constructed")
        return self.basis[key]

    def __str__(self):
        if self.basis is None:
            shape = "<unconstructed>"
        else:
            shape = self.basis.shape
        return (
            f"{self.__class__.__name__} with shape {shape}. "
            f"T0={self.t0}, pre_s={self.pre_s}, post_s={self.post_s}, "
            f"binwidth_s={self.binwidth_s}"
        )

    def plot(self):
        pass

class NoBasis(BasisFunction):
    def __init__(self, pre_s=None, post_s=None, binwidth_s=None):
        super().__init__(pre_s=0, post_s=0, binwidth_s=None)
    
    def construct_basis(self):
        return no_basis()

class BoxcarSmooth(BasisFunction):
    def __init__(self, pre_s, post_s, binwidth_s):
        super().__init__(pre_s, post_s, binwidth_s)

    def construct_basis(self):
        return boxcar_smooth(self.pre_s, self.post_s, self.binwidth_s)

class GaussianSmooth(BasisFunction):
    def __init__(self, pre_s, post_s, binwidth_s):
        super().__init__(pre_s, post_s, binwidth_s)
    
    def construct_basis(self):
        return gaussian_smooth(self.pre_s, self.post_s, self.binwidth_s)


class DeltaBasis(BasisFunction):
    def __init__(self, pre_s, post_s, binwidth_s):
        super().__init__(pre_s, post_s, binwidth_s)

    def construct_basis(self):
        return delta_basis(self.pre_s, self.post_s, self.binwidth_s)

class FIRBasis(BasisFunction):
    def __init__(self, impulse_binsize_s, pre_s, post_s, binwidth_s):
        self.impulse_binsize_s = impulse_binsize_s
        super().__init__(pre_s, post_s, binwidth_s)

    def construct_basis(self):
        return fir_basis(
            self.impulse_binsize_s,
            self.pre_s,
            self.post_s,
            self.binwidth_s,
        )

class GaussianBasis(BasisFunction):
    def __init__(self, n_funcs, pre_s, post_s, binwidth_s, sigma=None):
        self.n_funcs = n_funcs
        self.sigma = sigma
        super().__init__(pre_s, post_s, binwidth_s)

    def construct_basis(self):
        return gaussian_basis(
            self.n_funcs,
            self.pre_s,
            self.post_s,
            self.binwidth_s,
            self.sigma,
        )

class RaisedCosineBasis(BasisFunction):
    def __init__(self, n_funcs, pre_s, post_s, binwidth_s, log_scale=False):
        self.n_funcs = n_funcs
        self.log_scale = log_scale
        super().__init__(pre_s, post_s, binwidth_s)

    def construct_basis(self):
        return raised_cosine_basis(
            self.n_funcs,
            self.pre_s,
            self.post_s,
            self.binwidth_s,
            self.log_scale,
        )

class BsplineBasis(BasisFunction):
    def __init__(self, n_funcs, pre_s, post_s, binwidth_s, degree=3):
        self.n_funcs = n_funcs
        self.degree = degree
        super().__init__(pre_s, post_s, binwidth_s)
        
    def construct_basis(self):
        return bspline_basis(
            self.n_funcs,
            self.pre_s,
            self.post_s,
            self.binwidth_s,
            self.degree,
        )
        


class CustomBasisFunction(BasisFunction):
    def __init__(self, basis_array, pre_s, post_s, binwidth_s):
        self._basis_array = np.asarray(basis_array)
        super().__init__(pre_s=pre_s, post_s=post_s, binwidth_s=binwidth_s)

    def construct_basis(self):
        basis_time, _, _ = construct_timebins(
            self.pre_s, self.post_s, self.binwidth_s
        )
        return self._basis_array, basis_time


basis_function_classes = {
    "no_basis" : NoBasis,
    "delta": DeltaBasis,
    "raised_cosine": RaisedCosineBasis,
    "bspline" : BsplineBasis,
    "gaussian_basis" : GaussianBasis,
    "boxcar_smooth" : BoxcarSmooth,
    "gaussian_smooth" : GaussianSmooth,
    "fir" : FIRBasis,
}

def _is_basis_function_like(obj):
    return all(hasattr(obj, attr) for attr in ("construct_basis", "__getitem__"))