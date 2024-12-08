from numba import njit, prange, types, set_num_threads, vectorize
import numpy as np
from numba.experimental import jitclass
import os

set_num_threads(os.cpu_count())

spec = [
    ('x', types.float64[:]),
    ('y', types.float64[:]),
    ('a', types.float64[:]),
    ('b', types.float64[:]),
    ('c', types.float64[:]),
    ('d', types.float64[:]),
    ('n', types.int64),
    ('loglog', types.boolean),
    ('eps', types.float64),
]

@jitclass(spec)
class CubicSpline1D:
    def __init__(self, x, y, loglog=False, eps=1e-10):
        """
        Natural cubic spline interpolator
        """
        x = x.astype(np.float64)
        y = y.astype(np.float64)
        self.n = len(x) - 1
        self.loglog = loglog
        self.eps = float(eps)
        if self.loglog:
            x_safe = np.maximum(x, 0.0) + self.eps
            y_safe = np.maximum(y, 0.0) + self.eps
            self.x = np.log10(x_safe)
            self.y = np.log10(y_safe)
        else:
            self.x = x
            self.y = y
            
        self.a = np.zeros(self.n, dtype=np.float64)
        self.b = np.zeros(self.n, dtype=np.float64)
        self.c = np.zeros(self.n + 1, dtype=np.float64)
        self.d = np.zeros(self.n, dtype=np.float64)
        self._compute_coefficients()
    
    def _compute_coefficients(self):
        n = self.n
        x = self.x
        y = self.y
        h = np.zeros(n, dtype=np.float64)
        alpha = np.zeros(n, dtype=np.float64)
        l = np.zeros(n + 1, dtype=np.float64)
        mu = np.zeros(n + 1, dtype=np.float64)
        z = np.zeros(n + 1, dtype=np.float64)
    
        # Step 1: Calculate h and alpha
        for i in range(n):
            h[i] = x[i+1] - x[i]
        for i in range(1, n):
            alpha[i] = (3.0/h[i]) * (y[i+1] - y[i]) - (3.0/h[i-1]) * (y[i] - y[i-1])
    
        # Step 2: Set up the tridiagonal system
        l[0] = 1.0
        mu[0] = 0.0
        z[0] = 0.0
        for i in range(1, n):
            l[i] = 2.0*(x[i+1] - x[i-1]) - h[i-1]*mu[i-1]
            mu[i] = h[i]/l[i]
            z[i] = (alpha[i] - h[i-1]*z[i-1])/l[i]
    
        l[n] = 1.0
        z[n] = 0.0
        self.c[n] = 0.0
    
        # Step 3: Back substitution
        for j in range(n-1, -1, -1):
            self.c[j] = z[j] - mu[j]*self.c[j+1]
            self.b[j] = (y[j+1] - y[j])/h[j] - h[j]*(self.c[j+1] + 2.0*self.c[j])/3.0
            self.d[j] = (self.c[j+1] - self.c[j])/(3.0*h[j])
            self.a[j] = y[j]
    
    def evaluate(self, xi):
        return evaluate_point(xi, self)

@njit
def evaluate_point(xi, spline):
    x = spline.x
    n = spline.n
    a = spline.a
    b = spline.b
    c = spline.c
    d = spline.d
    loglog = spline.loglog
    eps = spline.eps

    if loglog:
        xi_safe = np.maximum(xi, 0.0) + eps
        xi_eval = np.log10(xi_safe)
    else:
        xi_eval = xi

    idx = np.searchsorted(x, xi_eval) - 1

    # Handle extrapolation (we will extrapolate linearly)
    if idx < 0:
        idx = 0
        dx = xi_eval - x[0]
        spline_value = a[idx] + b[idx]*dx
    elif idx >= n:
        idx = n - 1
        dx = xi_eval - x[n]
        spline_value = a[idx] + b[idx]*dx
    else:
        dx = xi_eval - x[idx]
        spline_value = a[idx] + b[idx]*dx + c[idx]*dx**2 + d[idx]*dx**3
        
    if loglog:
        result = np.power(10.0, spline_value)
    else:
        result = spline_value
    return result

@njit(parallel=True)
def evaluate_spline_parallel(spline, xi_array):
    n_points = xi_array.size
    result = np.empty_like(xi_array)
    for i in prange(n_points):
        xi = xi_array.flat[i]
        result.flat[i] = evaluate_point(xi, spline)
    return result


class spline1d:
    def __init__(self, x_arr, y_arr, loglog=False):
        self.spline = CubicSpline1D(x_arr, y_arr, loglog)
    def __call__(self, xi_arr):
        return evaluate_spline_parallel(self.spline, xi_arr)        
    
@njit(parallel=True)
def check_bounds_incl(x, lb, rb):
    result = 1
    for i in prange(x.size):
        if (x.flat[i] < lb) or (x.flat[i] > rb):
            result *= 0
    return result

@vectorize([types.float32(types.float32), 
            types.float64(types.float64)], nopython=True, target='parallel')
def _exp(arr):
    return np.exp(arr)