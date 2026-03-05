#include <iostream>
#include "cpp_kernels.hpp"

using namespace std;

// FIXME there is some cut-and-paste between estimate_power_spectrum() and kbin_average(),
// that could be cleaned up by defining some helper functions.


inline double square(double x) { return x*x; }


static void _kbin_average(double *out_fk, long *out_bcounts,
              const double *in_fk, const long *fk_strides,
              long nkbins, const double *k2_delim, double curr_k2, long curr_bin,
              long ndim, const long *np, const double *kf)
{
    long np0 = np[0];
    double kf0 = kf[0];
    long s0 = fk_strides[0];

    if (ndim == 1) {
    for (long ik = 0; 2*ik <= np0; ik++) {
        // Update curr_bin.
        double k2 = curr_k2 + square(ik*kf0);
        while (k2 >= k2_delim[curr_bin+1])
        if (++curr_bin == nkbins)
            return;
    
        if (curr_bin >= 0) {
        long m = ((ik==0) || (2*ik==np0)) ? 1 : 2;
        out_fk[curr_bin] += m * in_fk[0];
        out_bcounts[curr_bin] += m;
        }
        
        in_fk += s0;
    }
    }
    else {
    for (long ik0 = 0; ik0 < np0; ik0++) {
        long ik = std::min(ik0, np0-ik0);;
        double k2 = curr_k2 + square(ik*kf0);

        // Update curr_bin
        while ((curr_bin >= 0) && (k2 < k2_delim[curr_bin]))
        curr_bin--;   // search down
        while ((curr_bin < nkbins) && (k2 >= k2_delim[curr_bin+1]))
        curr_bin++;   // search up

        // Call _kbin_average() recursively, with ndim -> (ndim-1).
        if (curr_bin < nkbins)
        _kbin_average(out_fk, out_bcounts,
                  in_fk + ik0*s0, fk_strides+1,
                  nkbins, k2_delim, k2, curr_bin,
                  ndim-1, np+1, kf+1);
    }
    }
}


py::tuple kbin_average(py::array_t<const double> &fk, py::array_t<const double> &k_delim, py::array_t<const long> &npix, py::array_t<const double> &kf)
{
    if (k_delim.ndim() != 1)
    throw runtime_error("kbin_average: expected k_delim.ndim == 1");
    if (npix.ndim() != 1)
    throw runtime_error("kbin_average: expected npix.ndim == 1");
    if (kf.ndim() != 1)
    throw runtime_error("kbin_average: expected kf.ndim == 1");
    if (npix.shape(0) != kf.shape(0))
    throw runtime_error("kbin_average: expected len(npix) == len(kf)");
    if (get_stride(npix,0) != 1)
    throw runtime_error("kbin_average: expected npix to be contiguous array");
    if (get_stride(kf,0) != 1)
    throw runtime_error("kbin_average: expected kf to be contiguous array");
    
    long nkbins = get_shape(k_delim,0) - 1;    // note -1 here
    int ndim = get_shape(npix,0);

    if (ndim < 1)
    throw runtime_error("kbin_average: expected len(npix) >= 1");
    if (nkbins < 1)
    throw runtime_error("kbin_average: expected len(k_delim) >= 2");
    
    vector<long> np(ndim);
    vector<double> k2_delim(nkbins+1);
    vector<long> fk_strides(ndim);

    // Error-check 'kbin_delim' argument, and populate 'k2_delim' vector (squaring)

    const double *kd = k_delim.data();
    long ks = get_stride(k_delim, 0);
    
    if (kd[0] < 0.)
    throw runtime_error("kbin_average: expected k_delim[0] >= 0");

    for (long b = 0; b < nkbins; b++)
    if (kd[b*ks] >= kd[(b+1)*ks])
        throw runtime_error("kbin_average: expected k_delim array to be sorted");

    for (long b = 0; b < nkbins+1; b++)
    k2_delim[b] = square(kd[b*ks]);
    
    // Error-check 'npix' argument, and populate 'np' vector.
    
    for (int axis = 0; axis < ndim; axis++) {
    long n = npix.data()[axis];
    if (n <= 0)
        throw runtime_error("kbin_average: expected npix > 0");
    np[axis] = n;
    }

    // Error-check 'fk' argument, and populate 'fk_strides'

    if (fk.ndim() != ndim)
    throw runtime_error("kbin_average: expected map.ndim == len(npix)");
    
    for (int axis = 0; axis < ndim; axis++) {
    long n_expected = (axis < (ndim-1)) ? np[axis] : ((np[axis]/2) + 1);
    if (fk.shape(axis) != n_expected)
        throw runtime_error("kbin_average: map shape is inconsistent with 'npix'");
    
    fk_strides[axis] = get_stride(fk, axis);
    }

    // Run PSE kernel.

    py::array_t<double> ret_fk({nkbins});
    py::array_t<long> ret_bcounts({nkbins});

    double *out_fk = ret_fk.mutable_data();
    long *out_bcounts = ret_bcounts.mutable_data();

    memset(out_fk, 0, nkbins * sizeof(*out_fk));
    memset(out_bcounts, 0, nkbins * sizeof(*out_bcounts));
    long curr_bin = (k2_delim[0] > 0.) ? -1 : 0;

    _kbin_average(out_fk, out_bcounts,
          fk.data(), &fk_strides[0],
          nkbins, &k2_delim[0], 0.0, curr_bin,
          ndim, &np[0], kf.data());

    // Apply normalization.
    
    for (long b = 0; b < nkbins; b++) {
    long n = out_bcounts[b];
    double fk_norm = (n > 0) ? (1.0 / double(n)) : 0.0;
    out_fk[b] *= fk_norm;
    }

    return py::make_tuple(ret_fk, ret_bcounts);
}
