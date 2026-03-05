// multiply_xli_{real,fourier}_space():
//
// Multiply a 3-d grid (real-space or Fourier-space) by X_{li}(hat{n}),
// the real spherical harmonics used in the decomposition:
//
//   P_l(khat . rhat) = sum_{i=0}^{2l} X_{li}(khat) X_{li}(rhat)
//
// This decomposition allows spin-l FFTs to be written as a sum of
// (2l+1) ordinary FFTs. See "FFT implementation notes" in fft.rst.
//
// FIXME the functions multiply_xli_{real,fourier}_space() aren't
// super well optimized.
//
// If you run:
//   OMP_NUM_THREADS=1 python -m kszx time
//
// then you'll see that it's a lot slower than it should be. (To
// avoid multiply_xli being a bottleneck in single-threaded FFTs,
// I'd like to make it close to memory bandwidth limited.)
//
// I haven't tried diagnosing the slowness, but I bet it would help
// to introduce some length-4 (or length-8?) arrays, to help the
// compiler emit simd instructions.

#include <omp.h>
#include <cmath>
#include "cpp_kernels.hpp"

using namespace std;

static constexpr int Lmax = 8;


// Precomputes normalization and recurrence coefficients for evaluating
// X_{li}(nhat), the real spherical harmonic at degree l, index i.
//
// Index i maps to azimuthal order m = (i+1)/2. The angular part is
// Re((x+iy)^m) if i = 2m-1, or Im((x+iy)^m) if i = 2m (for m >= 1).
// For m = 0 (i = 0), the angular part is just 1.

struct xlm_helper
{
    int l;   // target degree, 0 <= l <= Lmax
    int i;   // real spherical harmonic index, 0 <= i < 2l+1
    int m;   // azimuthal order, m = (i+1)/2
    bool reim;  // false = Re((x+iy)^m), true = Im((x+iy)^m)

    double C = 0.0;                // normalization for starting the recurrence at l = m
    double alpha_rec[Lmax+1];      // alpha_rec[l] = 1 / alpha_l
    double alpha_rat[Lmax+1];      // alpha_rat[l] = alpha_{l-1} / alpha_l

    xlm_helper(int l_, int i_)
    {
        l = l_;
        i = i_;

        // FIXME throw exceptions
        assert(l >= 0);
        assert(l <= Lmax);
        assert(i >= 0);
        assert(i < 2*l+1);

        m = (i+1)/2;
        reim = (m > 0) && (i == 2*m);

        // Compute normalization constant C such that, on the unit sphere:
        //   C * Re/Im((x+iy)^m) = X_{l=m, i}(nhat)
        //
        // Since the three-term recurrence is linear and preserves
        // the normalization, this initial value at l=m propagates
        // correctly to give X_{l,i} at the target l.
        //
        // C^2 = [m>0 ? 2 : 1] / (2l+1) * prod_{j=1}^{m} (2j+1)/(2j)
        // Sign: (-1)^m (Condon-Shortley phase).

        C = (m > 0) ? 2 : 1;
        C /= (2*l+1);

        for (int j = 1; j <= m; j++)
            C *= (1.0 + 1.0/(2*j));

        C = sqrt(C);
        C = (m & 1) ? (-C) : C;

        // Precompute recurrence coefficients alpha_l = sqrt((l^2 - m^2) / (4l^2 - 1)).
        // alpha_l = 0 for l <= m (recurrence starts at l = m).

        for (int j = 0; j <= m; j++)
            alpha_rec[j] = alpha_rat[j] = 0.0;

        double alpha_prev = 0.0;
        for (int ll = m+1; ll <= Lmax; ll++) {
            double num = ll*ll - m*m;
            double den = 4*ll*ll - 1;
            double alpha = sqrt(num/den);

            alpha_rec[ll] = 1.0 / alpha;
            alpha_rat[ll] = alpha_prev / alpha;
            alpha_prev = alpha;
        }
    }
    
    // Evaluate X_{li}(nhat) where nhat = (x,y,z)/|(x,y,z)|.
    inline double get(double x, double y, double z)
    {
        // Step 1: normalize (x,y,z) to a unit vector.
        // FIXME does this compile to a fast x86 rsqrt instruction?
        double t = 1.0 / sqrt(x*x + y*y + z*z);
        x *= t;
        y *= t;
        z *= t;

        // Step 2: compute e = (x+iy)^m.
        double ere = 1.0;
        double eim = 0.0;

        for (int mm = 0; mm < m; mm++) {
            double new_ere = ere*x - eim*y;
            double new_eim = ere*y + eim*x;
            ere = new_ere;
            eim = new_eim;
        }

        // Step 3: initialize recurrence at l = m.
        // On the unit sphere, Re/Im((x+iy)^m) = sin^m(theta) * cos/sin(m*phi).
        // C is chosen so that C * Re/Im((x+iy)^m) = X_{m,i}(nhat).
        double xli = C * (reim ? eim : ere);
        double xli_prev = 0.0;

        // Step 4: three-term recurrence from l=m up to the target l.
        //
        //   X_{l+1} = (z / alpha_{l+1}) * X_l - (alpha_l / alpha_{l+1}) * X_{l-1}
        //
        // where alpha_l = sqrt((l^2 - m^2) / (4l^2 - 1)).
        // This is the standard recurrence for normalized associated
        // Legendre functions, applied to X_{li}(nhat).
        //
        // FIXME renormalization (for numerical stability at large l)

        for (int ll = m; ll < l; ll++) {
            double xli_next = (alpha_rec[ll+1] * z * xli) - (alpha_rat[ll+1] * xli_prev);
            xli_prev = xli;
            xli = xli_next;
        }

        return xli;
    }
};


// Can be used either for real-space maps (T=double) or Fourier-space (T=complex<double>).
template<typename T>
struct grid_helper
{
    T *data;          // grid data
    long n0, n1, n2;  // grid shape
    long s0, s1, s2;  // grid strides

    grid_helper(py::array_t<T> &grid)
    {
        if (grid.ndim() != 3)
            throw std::runtime_error("expected 'grid' to be a 3-d array");
        
        if constexpr (std::is_const<T>::value)
            data = grid.data();
        else
            data = grid.mutable_data();
        
        n0 = get_shape(grid, 0);
        n1 = get_shape(grid, 1);
        n2 = get_shape(grid, 2);
        
        s0 = get_stride(grid, 0);
        s1 = get_stride(grid, 1);
        s2 = get_stride(grid, 2);
        
        if ((n0 < 2) || (n1 < 2) || (n2 < 2))
            throw std::runtime_error("expected all grid dimensions >= 2");
    }
};


// -------------------------------------------------------------------------------------------------


// Special case for l=0: X_{00}(nhat) = 1, so this just copies src to
// dst with a scalar coefficient. If Accum=true, adds to dst instead.
template<bool Accum, typename T>
inline void _multiply_x00(grid_helper<T> &dst, grid_helper<const T> &src, T coeff)
{
#pragma omp parallel for
    for (long i0 = 0; i0 < dst.n0; i0++) {
        for (long i1 = 0; i1 < dst.n1; i1++) {
            T *dp = dst.data + (i0 * dst.s0) + (i1 * dst.s1);
            const T *sp = src.data + (i0 * src.s0) + (i1 * src.s1);
            
            for (long i2 = 0; i2 < dst.n2; i2++) {
                T v = coeff * sp[i2 * src.s2];
                
                if constexpr (Accum)
                    dp[i2 * dst.s2] += v;
                else
                    dp[i2 * dst.s2] = v;
            }
        }
    }
}


// Multiply a real-space grid by coeff * X_{li}(rhat), where rhat is the
// unit vector from the observer to each grid point. Observer position
// is implicitly at (lpos0, lpos1, lpos2) - (i0,i1,i2)*pixsize.
// If Accum=true, adds to dst; otherwise overwrites dst.
template<bool Accum>
inline void _multiply_xli_real_space(grid_helper<double> &dst, grid_helper<const double> &src, xlm_helper &h, double lpos0, double lpos1, double lpos2, double pixsize, double coeff)
{
    if (h.l == 0) {
        _multiply_x00<Accum> (dst, src, coeff);
        return;
    }
    
#pragma omp parallel for
    for (long i0 = 0; i0 < dst.n0; i0++) {
        double x = lpos0 + (i0 * pixsize);
        for (long i1 = 0; i1 < dst.n1; i1++) {
            double y = lpos1 + (i1 * pixsize);
            double *dp = dst.data + (i0 * dst.s0) + (i1 * dst.s1);
            const double *sp = src.data + (i0 * src.s0) + (i1 * src.s1);
            
            for (long i2 = 0; i2 < dst.n2; i2++) {
                double z = lpos2 + (i2 * pixsize);
                double xli = h.get(x, y, z);
                double v = coeff * xli * sp[i2 * src.s2];
                
                if constexpr (Accum)
                    dp[i2 * dst.s2] += v;
                else
                    dp[i2 * dst.s2] = v;
            }
        }
    }
}


// Python-facing wrapper: multiply a real-space grid by coeff * X_{li}(rhat).
// (lpos0, lpos1, lpos2) define the observer-relative position of grid point (0,0,0).
void multiply_xli_real_space(py::array_t<double> &dst_, py::array_t<const double> &src_, int l, int i, double lpos0, double lpos1, double lpos2, double pixsize, double coeff, bool accum)
{
    grid_helper<double> dst(dst_);
    grid_helper<const double> src(src_);
    xlm_helper h(l,i);

    if ((dst.n0 != src.n0) || (dst.n1 != src.n1) || (dst.n2 != src.n2))
        throw std::runtime_error("expected dst/src maps to have the same shapes");
    if (pixsize <= 0)
        throw std::runtime_error("expected pixsize > 0");    
    
    if (accum)
        _multiply_xli_real_space<true> (dst, src, h, lpos0, lpos1, lpos2, pixsize, coeff);
    else
        _multiply_xli_real_space<false> (dst, src, h, lpos0, lpos1, lpos2, pixsize, coeff);
}


// Multiply a Fourier-space grid by coeff * X_{li}(khat), where khat is the
// direction of the wavevector at each grid point. The grid has rfft
// layout: shape (n0, n1, nz/2+1), where nz is the real-space size
// along the last axis. DC and Nyquist modes are zeroed for l > 0
// (X_{li} is undefined at k=0 and ill-defined at Nyquist).
// If Accum=true, adds to dst; otherwise overwrites dst.
template<bool Accum>
inline void _multiply_xli_fourier_space(grid_helper<complex<double>> &dst, grid_helper<const complex<double>> &src, xlm_helper &h, long nz, complex<double> coeff)
{
    if (h.l == 0) {
        // For l=0, X_{00} = 1 everywhere, so just scale. No DC/Nyquist zeroing.
        _multiply_x00<Accum> (dst, src, coeff);
        return;
    }

    double rec_nz = 1.0 / nz;

    // Map grid indices to wavevector components (proportional to k).
    // Axes 0,1: wrap indices > n/2 to negative frequencies.
    // Axis 2 (rfft): indices are non-negative, range [0, nz/2].
#pragma omp parallel for
    for (long i0 = 0; i0 < dst.n0; i0++) {
        double x = (2*i0 > dst.n0) ? (i0 - dst.n0) : (i0);
        x /= dst.n0;
        
        for (long i1 = 0; i1 < dst.n1; i1++) {
            complex<double> *dp = dst.data + (i0 * dst.s0) + (i1 * dst.s1);
            const complex<double> *sp = src.data + (i0 * src.s0) + (i1 * src.s1);
            double y = (2*i1 > dst.n1) ? (i1 - dst.n1) : (i1);
            y /= dst.n1;
            
            for (long i2 = 0; i2 < dst.n2; i2++) {
                double z = rec_nz * i2;
                bool dc = (i0+i1+i2) == 0;
                bool nyq = (2*i0 == dst.n0) || (2*i1 == dst.n1) || (2*i2 == nz);
                
                double xli = (nyq || dc) ? 0.0 : h.get(x, y, z);
                complex<double> v = coeff * xli * sp[i2 * src.s2];
                
                if constexpr (Accum)
                    dp[i2 * dst.s2] += v;
                else
                    dp[i2 * dst.s2] = v;
            }
        }
    }    
}


// Python-facing wrapper: multiply a Fourier-space grid by coeff * X_{li}(khat).
// nz is the real-space grid size along the last axis (needed because the
// rfft output shape nz/2+1 doesn't uniquely determine nz).
//
// The coeff must be purely real for even l, or purely imaginary for odd l,
// consistent with the epsilon_l convention (epsilon_l = i for odd l, 1 for even l).
void multiply_xli_fourier_space(py::array_t<complex<double>> &dst_, py::array_t<const complex<double>> &src_, int l, int i, long nz, complex<double> coeff, bool accum)
{
    grid_helper<complex<double>> dst(dst_);
    grid_helper<const complex<double>> src(src_);
    xlm_helper h(l,i);

    if ((dst.n0 != src.n0) || (dst.n1 != src.n1) || (dst.n2 != src.n2))
        throw std::runtime_error("expected dst/src maps to have the same shapes");
    if (dst.n2 != (nz/2)+1)
        throw std::runtime_error("dst/src map shape is inconsistent with 'nz' argument");

    // Validate that coeff respects the epsilon_l parity constraint.
    double x = (l & 1) ? coeff.real() : coeff.imag();

    if (x != 0.0) {
        std::stringstream ss;
        ss << "multiply_xli_fourier_space(l=" << l << "): expected coeff."
           << ((l & 1) ? "real" : "imag")
           << "=0, got " << x;
        throw std::runtime_error(ss.str());
    }
    
    if (accum)
        _multiply_xli_fourier_space<true> (dst, src, h, nz, coeff);
    else
        _multiply_xli_fourier_space<false> (dst, src, h, nz, coeff);
}
