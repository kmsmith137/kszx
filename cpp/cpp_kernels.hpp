#ifndef _KSZX_CPP_KERNELS_HPP
#define _KSZX_CPP_KERNELS_HPP

#include <climits>
#include <complex>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// Hardware branch predictor hint.
#ifndef _unlikely
#define _unlikely(x)  (__builtin_expect(!!(x), 0))
#endif

namespace py = pybind11;


// -------------------------------------------------------------------------------------------------
//
// These functions are exported to python (see cpp_kernels.cpp for pybind11 boilerplate).


extern py::array_t<double> cic_interpolate_3d(py::array_t<const double> &grid, py::array_t<const double> &points,
					      double lpos0, double lpos1, double lpos2, double pixsize, bool periodic);

extern py::array_t<double> cubic_interpolate_3d(py::array_t<const double> &grid, py::array_t<const double> &points,
						double lpos0, double lpos1, double lpos2, double pixsize, bool periodic);


extern void cic_grid_3d(py::array_t<double> &grid, py::array_t<const double> &points, py::array_t<const double> &weights,
			double wscal, double lpos0, double lpos1, double lpos2, double pixsize, bool periodic);

extern void cubic_grid_3d(py::array_t<double> &grid, py::array_t<const double> &points, py::array_t<const double> &weights,
			  double wscal, double lpos0, double lpos1, double lpos2, double pixsize, bool periodic);


extern py::tuple estimate_power_spectrum(py::list map_list, py::array_t<const double> &k_delim,
					 py::array_t<const long> &npix, py::array_t<const double> &kf,
					 double box_volume);

extern py::tuple kbin_average(py::array_t<const double> &fk, py::array_t<const double> &k_delim,
			      py::array_t<const long> &npix, py::array_t<const double> &kf);


extern void multiply_xli_real_space(py::array_t<double> &dst_, py::array_t<const double> &src_, int l, int i,
				    double lpos0, double lpos1, double lpos2, double pixsize, double coeff,
				    bool accum);

extern void multiply_xli_fourier_space(py::array_t<std::complex<double>> &dst_,
				       py::array_t<const std::complex<double>> &src_,
				       int l, int i, long nz, std::complex<double> coeff,
				       bool accum);


// -------------------------------------------------------------------------------------------------
//
// These inline functions are used internally.


// Like fmod(), but the return value is in the range [0,y).
inline double xfmod(double x, double y)
{
    x = fmod(x,y);
    return (x < 0) ? (x+y) : x;
}
    

template<typename T>
inline long get_shape(const py::array_t<T> &arr, int d)
{
    return arr.shape(d);
}


template<typename T>
inline long get_stride(const py::array_t<T> &arr, int d)
{
    long s = arr.strides(d);
    long ss = s / sizeof(T);

    if (_unlikely(s != (ss * sizeof(T))))
	throw std::runtime_error("kszx.cpp_kernels: unaligned array stride?!");

    return ss;
}

    
#endif  // _KSZX_CPP_KERNELS_HPP
