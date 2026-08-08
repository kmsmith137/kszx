from numba import njit, prange, types, set_num_threads, vectorize
import numpy as np
from numba.experimental import jitclass
import os
from math import erfc
set_num_threads(os.cpu_count())

# ================================================================
# Cubic & Bicubic Spline Interpolators (Numba/NumPy, unified + fixes)
# - 1D natural cubic spline (log_x / log_y, linear tails at ends)
# - 2D bicubic spline (log_x / log_y / log_z), with:
#     * optional linear tails via first-order (with mixed-term) extrapolation
# - Parallel/Sequential evaluation kernels
# ================================================================


# =========================
# Shared low-level helpers
# =========================

@njit
def _cubic_spline_coeffs_1d(x, y):
    """
    Natural cubic spline coefficients on [x[i], x[i+1]]:
        s_i(t) = a[i] + b[i]*t + c[i]*t^2 + d[i]*t^3,  t = (X - x[i])
    Returns a(n), b(n), c(n+1), d(n) with n = len(x)-1
    """
    n = x.size - 1
    a = np.empty(n, dtype=np.float64)
    b = np.empty(n, dtype=np.float64)
    c = np.zeros(n + 1, dtype=np.float64)
    d = np.empty(n, dtype=np.float64)

    h = np.empty(n, dtype=np.float64)
    alpha = np.zeros(n, dtype=np.float64)
    l = np.empty(n + 1, dtype=np.float64)
    mu = np.empty(n + 1, dtype=np.float64)
    z = np.empty(n + 1, dtype=np.float64)

    for i in range(n):
        h[i] = x[i+1] - x[i]
    for i in range(1, n):
        alpha[i] = (3.0/h[i])*(y[i+1]-y[i]) - (3.0/h[i-1])*(y[i]-y[i-1])

    l[0] = 1.0
    mu[0] = 0.0
    z[0] = 0.0
    for i in range(1, n):
        l[i] = 2.0*(x[i+1]-x[i-1]) - h[i-1]*mu[i-1]
        mu[i] = h[i]/l[i]
        z[i] = (alpha[i] - h[i-1]*z[i-1]) / l[i]

    l[n] = 1.0
    z[n] = 0.0
    c[n] = 0.0

    for j in range(n-1, -1, -1):
        c[j] = z[j] - mu[j]*c[j+1]
        b[j] = (y[j+1]-y[j])/h[j] - h[j]*(c[j+1] + 2.0*c[j]) / 3.0
        d[j] = (c[j+1] - c[j]) / (3.0*h[j])
        a[j] = y[j]
    return a, b, c, d

@njit
def _node_slopes_from_coeffs(b):
    """Derive node slopes from segment b-coefficients (for bicubic Hermite data)."""
    n = b.size
    slopes = np.empty(n+1, dtype=np.float64)
    slopes[0] = b[0]
    for i in range(1, n):
        slopes[i] = 0.5*(b[i-1] + b[i])
    slopes[n] = b[n-1]
    return slopes

@njit
def _locate_cell(x, xi):
    """Clamp-search for cell index (works for 1D and as building block for 2D)."""
    idx = np.searchsorted(x, xi) - 1
    if idx < 0:
        idx = 0
    elif idx > x.size - 2:
        idx = x.size - 2
    return idx


# ======================
# 1D natural cubic spline
# ======================

spec1d = [
    ('x',      types.float64[:]),
    ('y',      types.float64[:]),
    ('a',      types.float64[:]),
    ('b',      types.float64[:]),
    ('c',      types.float64[:]),
    ('d',      types.float64[:]),
    ('n',      types.int64),
    ('log_x',  types.boolean),
    ('log_y',  types.boolean),
    ('eps',    types.float64),
]

@jitclass(spec1d)
class CubicSpline1D:
    def __init__(self, x, y, log_x=False, log_y=False, eps=1e-16):
        """
        Natural cubic spline interpolator with optional log10 transforms:

          - if log_x: x -> log10(max(x,0)+eps)
          - if log_y: y -> log10(max(y,0)+eps) for fitting, and output is 10**(spline(x))

        Notes:
          * x must be strictly increasing after any transform.
        """
        x = x.astype(np.float64)
        y = y.astype(np.float64)

        self.log_x = log_x
        self.log_y = log_y
        self.eps   = float(eps)

        if self.log_x:
            x = np.log10(np.maximum(x, 0.0) + self.eps)
        if self.log_y:
            y = np.log10(np.maximum(y, 0.0) + self.eps)

        self.x = x
        self.y = y
        self.n = x.size - 1

        a, b, c, d = _cubic_spline_coeffs_1d(self.x, self.y)
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def evaluate(self, xi):
        return evaluate_point_1d(xi, self)

@njit
def evaluate_point_1d(spline, xi):
    if spline.log_x:
        xi_eval = np.log10(max(xi, 0.0) + spline.eps)
    else:
        xi_eval = xi

    x = spline.x
    a, b, c, d = spline.a, spline.b, spline.c, spline.d
    n = spline.n

    # Linear tails (explicit), cubic inside
    if xi_eval <= x[0]:
        dx  = xi_eval - x[0]
        val = a[0] + b[0]*dx
    elif xi_eval >= x[n]:
        dx  = xi_eval - x[n]
        val = a[n-1] + b[n-1]*dx
    else:
        i   = _locate_cell(x, xi_eval)
        dx  = xi_eval - x[i]
        val = a[i] + b[i]*dx + c[i]*dx*dx + d[i]*dx*dx*dx

    if spline.log_y:
        return 10.0**val
    else:
        return val

@njit(parallel=True)
def evaluate_cubic_parallel(spline, xi_arr):
    out = np.empty_like(xi_arr, dtype=np.float64)
    xf = xi_arr.ravel()
    of = out.ravel()
    n = xf.size
    for k in prange(n):
        of[k] = evaluate_point_1d(spline, xf[k])
    return out

@njit
def evaluate_cubic_sequential(spline, xi_arr):
    out = np.empty_like(xi_arr, dtype=np.float64)
    xf = xi_arr.ravel()
    of = out.ravel()
    n = xf.size
    for k in range(n):
        of[k] = evaluate_point_1d(spline, xf[k])
    return out

class spline1d:
    """
    Convenience wrapper.

    Call mode
    ---------
    f(xi) -> pointwise/broadcast evaluation; returns same shape as xi.
    """
    def __init__(self, x, y, log_x=False, log_y=False, eps=1e-16):
        self.spline = CubicSpline1D(x, y, log_x=log_x, log_y=log_y, eps=eps)

    def __call__(self, xi_arr):
        xi_arr = np.asarray(xi_arr, dtype=np.float64)
        if xi_arr.size <= int(1e4):
            return evaluate_cubic_sequential(self.spline, xi_arr)
        else:
            return evaluate_cubic_parallel(self.spline, xi_arr)


# ===================
# 2D bicubic spline
# ===================

@njit
def _compute_fx_grid(x, z_row):
    Ny, Nx = z_row.shape
    fx = np.empty((Ny, Nx), dtype=np.float64)
    for j in range(Ny):
        _, b, _, _ = _cubic_spline_coeffs_1d(x, z_row[j, :])
        fx[j, :] = _node_slopes_from_coeffs(b)
    return fx

@njit
def _compute_fy_grid(y, z_col):
    Ny, Nx = z_col.shape
    fy = np.empty((Ny, Nx), dtype=np.float64)
    for i in range(Nx):
        _, b, _, _ = _cubic_spline_coeffs_1d(y, z_col[:, i])
        fy[:, i] = _node_slopes_from_coeffs(b)
    return fy

@njit
def _compute_fxy_grid(y, fx_col):
    Ny, Nx = fx_col.shape
    fxy = np.empty((Ny, Nx), dtype=np.float64)
    for i in range(Nx):
        _, b, _, _ = _cubic_spline_coeffs_1d(y, fx_col[:, i])
        fxy[:, i] = _node_slopes_from_coeffs(b)
    return fxy

@njit
def _bicubic_eval_cell(xi, yi, x, y, z, fx, fy, fxy, i, j):
    x0 = x[i]; x1 = x[i+1]
    y0 = y[j]; y1 = y[j+1]
    dx = x1 - x0
    dy = y1 - y0

    t = (xi - x0) / dx
    u = (yi - y0) / dy

    f00 = z[j,   i  ]; f10 = z[j,   i+1]
    f01 = z[j+1, i  ]; f11 = z[j+1, i+1]

    fx00 = fx[j,   i  ]; fx10 = fx[j,   i+1]
    fx01 = fx[j+1, i  ]; fx11 = fx[j+1, i+1]

    fy00 = fy[j,   i  ]; fy10 = fy[j,   i+1]
    fy01 = fy[j+1, i  ]; fy11 = fy[j+1, i+1]

    fxy00 = fxy[j,   i  ]; fxy10 = fxy[j,   i+1]
    fxy01 = fxy[j+1, i  ]; fxy11 = fxy[j+1, i+1]

    # Hermite data matrix (scale partials to unit cell)
    G = np.empty((4,4), dtype=np.float64)
    G[0,0] = f00;        G[0,1] = f01;        G[0,2] = dy*fy00;        G[0,3] = dy*fy01
    G[1,0] = f10;        G[1,1] = f11;        G[1,2] = dy*fy10;        G[1,3] = dy*fy11
    G[2,0] = dx*fx00;    G[2,1] = dx*fx01;    G[2,2] = dx*dy*fxy00;    G[2,3] = dx*dy*fxy01
    G[3,0] = dx*fx10;    G[3,1] = dx*fx11;    G[3,2] = dx*dy*fxy10;    G[3,3] = dx*dy*fxy11

    H = np.array([[ 1.0,  0.0,  0.0,  0.0],
                  [ 0.0,  0.0,  1.0,  0.0],
                  [-3.0,  3.0, -2.0, -1.0],
                  [ 2.0, -2.0,  1.0,  1.0]], dtype=np.float64)
    H_T = H.T

    HG = np.empty((4,4), dtype=np.float64)
    for r in range(4):
        for c in range(4):
            s = 0.0
            for k in range(4):
                s += H[r, k] * G[k, c]
            HG[r, c] = s

    A = np.empty((4,4), dtype=np.float64)
    for r in range(4):
        for c in range(4):
            s = 0.0
            for k in range(4):
                s += HG[r, k] * H_T[k, c]
            A[r, c] = s

    T = np.array([1.0, t, t*t, t*t*t], dtype=np.float64)
    U = np.array([1.0, u, u*u, u*u*u], dtype=np.float64)

    Au = np.empty(4, dtype=np.float64)
    for r in range(4):
        s = 0.0
        for c in range(4):
            s += A[r, c] * U[c]
        Au[r] = s

    val = 0.0
    for r in range(4):
        val += T[r] * Au[r]
    return val

# --- value + derivatives at arbitrary point inside a cell
@njit
def _bicubic_eval_cell_with_derivs(xi, yi, x, y, z, fx, fy, fxy, i, j):
    x0 = x[i]; x1 = x[i+1]
    y0 = y[j]; y1 = y[j+1]
    dx = x1 - x0
    dy = y1 - y0

    t = (xi - x0) / dx
    u = (yi - y0) / dy

    f00 = z[j,   i  ]; f10 = z[j,   i+1]
    f01 = z[j+1, i  ]; f11 = z[j+1, i+1]

    fx00 = fx[j,   i  ]; fx10 = fx[j,   i+1]
    fx01 = fx[j+1, i  ]; fx11 = fx[j+1, i+1]

    fy00 = fy[j,   i  ]; fy10 = fy[j,   i+1]
    fy01 = fy[j+1, i  ]; fy11 = fy[j+1, i+1]

    fxy00 = fxy[j,   i  ]; fxy10 = fxy[j,   i+1]
    fxy01 = fxy[j+1, i  ]; fxy11 = fxy[j+1, i+1]

    # Hermite data matrix (scale partials to unit cell)
    G = np.empty((4,4), dtype=np.float64)
    G[0,0] = f00;        G[0,1] = f01;        G[0,2] = dy*fy00;        G[0,3] = dy*fy01
    G[1,0] = f10;        G[1,1] = f11;        G[1,2] = dy*fy10;        G[1,3] = dy*fy11
    G[2,0] = dx*fx00;    G[2,1] = dx*fx01;    G[2,2] = dx*dy*fxy00;    G[2,3] = dx*dy*fxy01
    G[3,0] = dx*fx10;    G[3,1] = dx*fx11;    G[3,2] = dx*dy*fxy10;    G[3,3] = dx*dy*fxy11

    H = np.array([[ 1.0,  0.0,  0.0,  0.0],
                  [ 0.0,  0.0,  1.0,  0.0],
                  [-3.0,  3.0, -2.0, -1.0],
                  [ 2.0, -2.0,  1.0,  1.0]], dtype=np.float64)
    H_T = H.T

    # A = H @ G @ H^T
    HG = np.empty((4,4), dtype=np.float64)
    for r in range(4):
        for c in range(4):
            s = 0.0
            for k in range(4):
                s += H[r, k] * G[k, c]
            HG[r, c] = s

    A = np.empty((4,4), dtype=np.float64)
    for r in range(4):
        for c in range(4):
            s = 0.0
            for k in range(4):
                s += HG[r, k] * H_T[k, c]
            A[r, c] = s

    # Bases and derivatives
    T  = np.array([1.0, t, t*t, t*t*t], dtype=np.float64)
    dT = np.array([0.0, 1.0, 2.0*t, 3.0*t*t], dtype=np.float64)
    U  = np.array([1.0, u, u*u, u*u*u], dtype=np.float64)
    dU = np.array([0.0, 1.0, 2.0*u, 3.0*u*u], dtype=np.float64)

    Au = np.empty(4, dtype=np.float64)
    for r in range(4):
        s = 0.0
        for c in range(4):
            s += A[r, c] * U[c]
        Au[r] = s

    f = 0.0
    for r in range(4):
        f += T[r] * Au[r]

    dAu_u = np.empty(4, dtype=np.float64)
    for r in range(4):
        s = 0.0
        for c in range(4):
            s += A[r, c] * dU[c]
        dAu_u[r] = s

    df_dt = 0.0
    for r in range(4):
        df_dt += dT[r] * Au[r]

    df_du = 0.0
    for r in range(4):
        df_du += T[r] * dAu_u[r]

    # chain rule
    fx_val  = df_dt / dx
    fy_val  = df_du / dy

    # mixed derivative: d^2 f / (dt du) then scale
    d2 = 0.0
    for r in range(4):
        d2 += dT[r] * dAu_u[r]
    fxy_val = d2 / (dx*dy)

    return f, fx_val, fy_val, fxy_val

spec2d = [
    ('x',      types.float64[:]),
    ('y',      types.float64[:]),
    ('z',      types.float64[:, :]),
    ('fx',     types.float64[:, :]),
    ('fy',     types.float64[:, :]),
    ('fxy',    types.float64[:, :]),
    ('nx',     types.int64),
    ('ny',     types.int64),
    ('log_x',  types.boolean),
    ('log_y',  types.boolean),
    ('log_z',  types.boolean),
    ('eps',    types.float64),
    ('linear_tails', types.boolean),
]

@jitclass(spec2d)
class BicubicSpline2D:
    def __init__(self, x, y, z, log_x=False, log_y=False, log_z=False, eps=1e-16, linear_tails=False):
        """
        Bicubic spline with optional log10 transforms and optional linear tails:
          - if log_x: x -> log10(max(x,0)+eps)
          - if log_y: y -> log10(max(y,0)+eps)
          - if log_z: z -> log10(max(z,0)+eps) for fitting, output exponentiated
          - linear_tails: if True, use first-order (with mixed) extrapolation outside domain
                          else clamp and use cubic extrapolation (default)
        """
        x = x.astype(np.float64)
        y = y.astype(np.float64)
        z = z.astype(np.float64)

        self.log_x = log_x
        self.log_y = log_y
        self.log_z = log_z
        self.eps   = float(eps)
        self.linear_tails = linear_tails

        if self.log_x:
            x = np.log10(np.maximum(x, 0.0) + self.eps)
        if self.log_y:
            y = np.log10(np.maximum(y, 0.0) + self.eps)
        if self.log_z:
            z = np.log10(np.maximum(z, 0.0) + self.eps)

        self.x = x
        self.y = y
        self.z = z

        self.nx = self.x.size - 1
        self.ny = self.y.size - 1

        self.fx  = _compute_fx_grid(self.x, self.z)
        self.fy  = _compute_fy_grid(self.y, self.z)
        self.fxy = _compute_fxy_grid(self.y, self.fx)

    def evaluate(self, xi, yi):
        return evaluate_point_2d(xi, yi, self)

@njit
def evaluate_point_2d(spline, xi, yi):
    xi_eval = xi
    yi_eval = yi
    if spline.log_x:
        xi_eval = np.log10(max(xi, 0.0) + spline.eps)
    if spline.log_y:
        yi_eval = np.log10(max(yi, 0.0) + spline.eps)

    x = spline.x; y = spline.y

    inside_x = (xi_eval >= x[0]) and (xi_eval <= x[-1])
    inside_y = (yi_eval >= y[0]) and (yi_eval <= y[-1])

    # cubic (clamped) inside or when linear tails disabled
    if not spline.linear_tails or (inside_x and inside_y):
        i = _locate_cell(x, xi_eval)
        j = _locate_cell(y, yi_eval)
        val = _bicubic_eval_cell(xi_eval, yi_eval, x, y, spline.z, spline.fx, spline.fy, spline.fxy, i, j)
        if spline.log_z:
            return 10.0**val
        else:
            return val

    # linear tails: project to boundary and first-order expand (with mixed term)
    xb = xi_eval
    if xi_eval < x[0]:
        xb = x[0]
    elif xi_eval > x[-1]:
        xb = x[-1]

    yb = yi_eval
    if yi_eval < y[0]:
        yb = y[0]
    elif yi_eval > y[-1]:
        yb = y[-1]

    i = _locate_cell(x, min(max(xb, x[0]), np.nextafter(x[-1], -np.inf)))
    j = _locate_cell(y, min(max(yb, y[0]), np.nextafter(y[-1], -np.inf)))

    f0, fx0, fy0, fxy0 = _bicubic_eval_cell_with_derivs(xb, yb, x, y, spline.z, spline.fx, spline.fy, spline.fxy, i, j)

    dx_tail = xi_eval - xb
    dy_tail = yi_eval - yb

    # include mixed term; drop it if you want strictly separable linear tails
    val = f0 + dx_tail*fx0 + dy_tail*fy0 + dx_tail*dy_tail*fxy0

    if spline.log_z:
        return 10.0**val
    else:
        return val

@njit(parallel=True)
def evaluate_bicubic_grid_parallel_xy(spline, xi_arr, yi_arr):
    Ny = yi_arr.size
    Nx = xi_arr.size
    out = np.empty((Ny, Nx), dtype=np.float64)
    for j in prange(Ny):
        yj = yi_arr[j]
        for i in range(Nx):
            out[j, i] = evaluate_point_2d(spline, xi_arr[i], yj)
    return out

@njit
def evaluate_bicubic_grid_sequential_xy(spline, xi_arr, yi_arr):
    Ny = yi_arr.size
    Nx = xi_arr.size
    out = np.empty((Ny, Nx), dtype=np.float64)
    for j in range(Ny):
        yj = yi_arr[j]
        for i in range(Nx):
            out[j, i] = evaluate_point_2d(spline, xi_arr[i], yj)
    return out

@njit(parallel=True)
def evaluate_bicubic_grid_parallel_ij(spline, xi_arr, yi_arr):
    Nx = xi_arr.size
    Ny = yi_arr.size
    out = np.empty((Nx, Ny), dtype=np.float64)
    for i in prange(Nx):
        xi = xi_arr[i]
        for j in range(Ny):
            out[i, j] = evaluate_point_2d(spline, xi, yi_arr[j])
    return out

@njit
def evaluate_bicubic_grid_sequential_ij(spline, xi_arr, yi_arr):
    Nx = xi_arr.size
    Ny = yi_arr.size
    out = np.empty((Nx, Ny), dtype=np.float64)
    for i in range(Nx):
        xi = xi_arr[i]
        for j in range(Ny):
            out[i, j] = evaluate_point_2d(spline, xi, yi_arr[j])
    return out

@njit(parallel=True)
def evaluate_bicubic_parallel(spline, xi_arr, yi_arr):
    out = np.empty_like(xi_arr, dtype=np.float64)
    n = out.size
    xf = xi_arr.ravel()
    yf = yi_arr.ravel()
    of = out.ravel()
    for k in prange(n):
        of[k] = evaluate_point_2d(spline, xf[k], yf[k])
    return out

@njit
def evaluate_bicubic_sequential(spline, xi_arr, yi_arr):
    out = np.empty_like(xi_arr, dtype=np.float64)
    n = out.size
    xf = xi_arr.ravel()
    yf = yi_arr.ravel()
    of = out.ravel()
    for k in range(n):
        of[k] = evaluate_point_2d(spline, xf[k], yf[k])
    return out

class spline2d:
    """
    Convenience wrapper.

    Call modes
    ----------
    f(xi, yi, meshgrid=False)     -> pointwise evaluation (xi, yi broadcasted)
    f(xi, yi, meshgrid=True|'xy') -> outer grid, shape (len(yi), len(xi))
    f(xi, yi, meshgrid='ij')      -> outer grid, shape (len(xi), len(yi))
    """
    def __init__(self, x, y, z, log_x=False, log_y=False, log_z=False, eps=1e-16, linear_tails=False):
        self.spline = BicubicSpline2D(
            x, y, z, log_x=log_x, log_y=log_y, log_z=log_z, eps=eps, linear_tails=linear_tails
        )

    def __call__(self, xi_arr, yi_arr, meshgrid=False):
        xi_arr = np.asarray(xi_arr, dtype=np.float64)
        yi_arr = np.asarray(yi_arr, dtype=np.float64)

        # Grid (outer product) modes
        size = xi_arr.size * yi_arr.size
        if meshgrid is True or meshgrid == 'xy':
            if size <= int(1e4):
                return evaluate_bicubic_grid_sequential_xy(self.spline, xi_arr, yi_arr)
            else:
                return evaluate_bicubic_grid_parallel_xy(self.spline, xi_arr, yi_arr)

        if meshgrid == 'ij':
            if size <= int(1e4):
                return evaluate_bicubic_grid_sequential_ij(self.spline, xi_arr, yi_arr)
            else:
                return evaluate_bicubic_grid_parallel_ij(self.spline, xi_arr, yi_arr)

        xi_b, yi_b = np.broadcast_arrays(xi_arr, yi_arr)
        if xi_b.size <= int(1e4):
            return evaluate_bicubic_sequential(self.spline, xi_b, yi_b)
        else:
            return evaluate_bicubic_parallel(self.spline, xi_b, yi_b)

    
#@njit(parallel=True)
#def _parallel_check_bounds_incl(x, lb, rb):
#    for i in prange(x.size):
#        if (x.flat[i] < lb) or (x.flat[i] > rb):
#            return False
#        return True
    
@njit
def check_bounds_incl(x, lb, rb):
    for i in prange(x.size):
        if (x.flat[i] < lb) or (x.flat[i] > rb):
            return False
        return True
    
#def check_bounds_incl(x, lb, rb):
#    if x.size > int(1e4):
#        return _parallel_check_bounds_incl(x, lb, rb)
#    return _check_bounds_incl(x, lb, rb)

@vectorize([types.float32(types.float32), 
            types.float64(types.float64)], nopython=True, target='parallel')
def _exp(arr):
    return np.exp(arr)


@vectorize([types.float32(types.float32), 
            types.float64(types.float64)], nopython=True, target='parallel')
def _sqrt(arr):
    return np.sqrt(arr)

@njit(parallel=True)
def _random_normal(out, scale):
    for i in prange(out.size):
        out.flat[i] = np.random.normal(scale=scale, loc=0.)
    return out    

def random_normal(size, scale):
    scale = float(scale)
    out = np.empty(size, dtype=np.float64)
    return _random_normal(out, scale)
    
    
@njit(parallel=True)    
def multiply_inplace(arr_a, arr_b):
    for i in prange(arr_a.size):
        arr_a.flat[i] *= arr_b.flat[i]
        
        
@njit
def get_k_value(i, j, k_index, dim, L):
    """
    Given grid indices (i, j, k_index), grid resolution 'dim', and box size 'L',
    returns the corresponding k value in the Fourier grid.

    Parameters:
    - i, j: Grid indices along the x and y axes (0 <= i, j < dim)
    - k_index: Grid index along the z-axis (0 <= k_index <= dim // 2)
               (since we're using a reduced FFT along this axis)
    - dim: Grid resolution along each axis
    - L: Physical size of the box along each axis

    Returns:
    - k_value: The corresponding k value
    """
    n = dim
    if i < n // 2:
        freq_i = 2 * np.pi * i / L
    else:
        freq_i = 2 * np.pi * (i - n) / L

    if j < n // 2:
        freq_j = 2 * np.pi * j / L
    else:
        freq_j = 2 * np.pi * (j - n) / L

    freq_k = 2 * np.pi * k_index / L

    k_squared = freq_i**2 + freq_j**2 + freq_k**2

    return np.sqrt(k_squared)

@njit(parallel=True)
def get_k_3D_box(dim, L):
    """ Generates a grid of fourier k in a 3d box with 'dim' pixels per side 'L'
    """
    out = np.empty((dim,dim,dim//2+1),dtype=np.float64)
    for i in prange(dim):
        for j in range(dim):
            for k in range(dim//2+1):
                out[i,j,k] = get_k_value(i,j,k, dim, L).item()
    return out



@njit(inline='always', fastmath=True)
def _cic_ker(x):
    return 1. - (2./3.)*x**2

@njit(inline='always', fastmath=True)
def _cubic_ker(x):
    return 1. - (22./45.)*(x**4) - (124./945.)*(x**6)

@njit(parallel=True)
def _apply_kernel_compensation(arr, ker_f, exponent):
    
    imax, jmax, kmax = arr.shape
    rdim = imax
    for i in prange(imax):
        for j in range(jmax):
            for k in range(kmax):
                
                si = np.sin(np.pi * i / rdim)
                sj = np.sin(np.pi * j / rdim)
                sk = np.sin(np.pi * k / rdim)
                
                compensation = ker_f(si)*ker_f(sj)*ker_f(sk)
                
                arr[i,j,k] *= np.power(compensation, exponent) 
    
    

@vectorize([types.float32(types.float32),
            types.float64(types.float64)],
           nopython=True,
            target='parallel')
def _P(x):
    return 0.5*erfc(x/2**0.5)      