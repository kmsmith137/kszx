#include <omp.h>
#include <cmath>
#include "cpp_kernels.hpp"


static constexpr int Lmax = 8;


struct xlm_helper
{
    int l;   // 0 <= l <= lmax
    int i;
    int m;
    bool reim;

    double C = 0.0;
    double eps[Lmax+1];

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
	C = (m & 2) ? C : (-C);
	   
	for (int l = 0; l <= m; l++)
	    eps[l] = 0.0;

	for (int l = m+1; l <= Lmax; l++) {
	    double num = l*l - m*m;
	    double den = 4*l*l - 1;
	    eps[l] = sqrt(num/den);
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
	    double xli_next = (z * xli) - (eps[ll] * xli_prev);
	    xli_prev = xli;
	    xli = xli_next;
	}

	return xli;
    }
};


// -------------------------------------------------------------------------------------------------


// Use either T=double or T=(const double).
template<typename T>
struct rs_helper
{
    T *data;            // grid data
    long n0, n1, n2;  // grid shape
    long s0, s1, s2;  // grid strides

    rs_helper(py::array_t<T> &grid)
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


void multiply_xli_real_space(py::array_t<double> &dst_, py::array_t<const double> &src_, int l, int i, double lpos0, double lpos1, double lpos2, double pixsize)
{
    rs_helper<double> dst(dst_);
    rs_helper<const double> src(src_);
    xlm_helper h(l,i);

    if ((dst.n0 != src.n0) || (dst.n1 != src.n1) || (dst.n2 != src.n2))
	throw std::runtime_error("expected dst/src maps to have the same shapes");
    if (pixsize <= 0)
	throw std::runtime_error("expected pixsize > 0");    

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
		
		dp[i2 * dst.s2] = xli * sp[i2 * src.s2];
	    }
	}
    }
}
