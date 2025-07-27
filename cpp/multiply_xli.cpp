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
//
// For more info on what these functions compute, see "FFT implementation"
// in the sphinx docs:
//
//   https://kszx.readthedocs.io/en/latest/fft.html#fft-implementation
//
// or the reference implementation in kszx/tests/test_fft.py.

#include <omp.h>
#include <cmath>
#include "cpp_kernels.hpp"

using namespace std;

static constexpr int Lmax = 8;


struct xlm_helper
{
    int l;   // 0 <= l <= lmax
    int i;
    int m;
    bool reim;

    double C = 0.0;
    double eps_rec[Lmax+1];   // (1 / eps_l)
    double eps_rat[Lmax+1];   // (eps_{l-1} / eps_l)

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

	C = (m > 0) ? 2 : 1;
	C /= (2*l+1);
	
	for (int j = 1; j <= m; j++)
	    C *= (1.0 + 1.0/(2*j));

	C = sqrt(C);
	C = (m & 1) ? (-C) : C;
	   
	for (int l = 0; l <= m; l++)
	    eps_rec[l] = eps_rat[l] = 0.0;

	double eps_prev = 0.0;
	for (int l = m+1; l <= Lmax; l++) {
	    double num = l*l - m*m;
	    double den = 4*l*l - 1;
	    double eps = sqrt(num/den);
	    
	    eps_rec[l] = 1.0 / eps;
	    eps_rat[l] = eps_prev / eps;
	    eps_prev = eps;
	}
    }

    inline double get(double x, double y, double z)
    {
	// FIXME does this compile to a fast x86 rsqrt instruction?
	double t = 1.0 / sqrt(x*x + y*y + z*z);
	x *= t;
	y *= t;
	z *= t;

	// e = (x+iy)^m
	
	double ere = 1.0;
	double eim = 0.0;

	for (int mm = 0; mm < m; mm++) {
	    double new_ere = ere*x - eim*y;
	    double new_eim = ere*y + eim*x;
	    ere = new_ere;
	    eim = new_eim;	    
	}

	double xli = C * (reim ? eim : ere);
	double xli_prev = 0.0;

	// FIXME renormalization
	
	for (int ll = m; ll < l; ll++) {
	    // ll -> (ll+1)
	    double xli_next = (eps_rec[ll+1] * z * xli) - (eps_rat[ll+1] * xli_prev);
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


// Called for l=0.
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


template<bool Accum>
inline void _multiply_xli_fourier_space(grid_helper<complex<double>> &dst, grid_helper<const complex<double>> &src, xlm_helper &h, long nz, complex<double> coeff)
{
    if (h.l == 0) {
	// Note that for l=0, we don't zero the DC/nyquist modes.
	_multiply_x00<Accum> (dst, src, coeff);
	return;
    }

    double rec_nz = 1.0 / nz;
    
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


void multiply_xli_fourier_space(py::array_t<complex<double>> &dst_, py::array_t<const complex<double>> &src_, int l, int i, long nz, complex<double> coeff, bool accum)
{
    grid_helper<complex<double>> dst(dst_);
    grid_helper<const complex<double>> src(src_);
    xlm_helper h(l,i);

    if ((dst.n0 != src.n0) || (dst.n1 != src.n1) || (dst.n2 != src.n2))
	throw std::runtime_error("expected dst/src maps to have the same shapes");
    if (dst.n2 != (nz/2)+1)
	throw std::runtime_error("dst/src map shape is inconsistent with 'nz' argument");

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
