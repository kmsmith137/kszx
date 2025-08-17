#include <omp.h>
#include <sstream>
#include <iostream>
#include "cpp_kernels.hpp"
#include "interpolation_args.hpp"


using namespace std;


inline long wrap_hi(long i, long n) { return (i < n) ? i : (i-n); }
    

struct cic_axis
{
    double w0, w1;
    long i0, i1;  // includes strides, may wrap around

    inline cic_axis(double x, long n, long stride, bool periodic)
    {
	x = periodic ? xfmod(x,n) : x;
	
	long i = long(x);
	bool good = periodic || ((x >= 0.0) && (i <= n-2));
	
	if (_unlikely(!good))
	    throw runtime_error("kszx: point is out of bounds in interpolate_points() or grid_points()");

	i = std::max(i, 0L);   // unnecessary?
	i = std::min(i, n-1);  // unnecessary?
	double f = x - (double)i;
	
	this->w0 = 1.0 - f;
	this->w1 = f;
	
	this->i0 = stride * i;
	this->i1 = stride * wrap_hi(i+1,n);
    }
};


// -------------------------------------------------------------------------------------------------
//
// CIC interpolation.


inline double cic_interp_1d(cic_axis &ax, const double *arr)
{
    return (ax.w0 * arr[ax.i0]) + (ax.w1 * arr[ax.i1]);
}

inline double cic_interp_2d(cic_axis &ax0, cic_axis &ax1, const double *arr)
{
    double f0 = cic_interp_1d(ax1, arr + ax0.i0);
    double f1 = cic_interp_1d(ax1, arr + ax0.i1);
    return (ax0.w0 * f0) + (ax0.w1 * f1);
}

inline double cic_interp_3d(cic_axis &ax0, cic_axis &ax1, cic_axis &ax2, const double *arr)
{
    double f0 = cic_interp_2d(ax1, ax2, arr + ax0.i0);
    double f1 = cic_interp_2d(ax1, ax2, arr + ax0.i1);
    return (ax0.w0 * f0) + (ax0.w1 * f1);
}


py::array_t<double> cic_interpolate_3d(py::array_t<const double> &grid, py::array_t<const double> &points, double lpos0, double lpos1, double lpos2, double pixsize, bool periodic)
{
    interpolation_args<const double> args(grid, points, lpos0, lpos1, lpos2, pixsize);
    
    if ((args.gn0 < 2) || (args.gn1 < 2) || (args.gn2 < 2))
	throw runtime_error("kszx.interpolate_points('cic'): all grid dimensions must be >= 2");

    py::array_t<double> ret({args.npoints});
    double *rdata = ret.mutable_data();

#pragma omp parallel for schedule(guided,32)
    for (long i = 0; i < args.npoints; i++) {
	double x, y, z;
	args.get_xyz(i, x, y, z);
	
	cic_axis ax0(x, args.gn0, args.gs0, periodic);
	cic_axis ax1(y, args.gn1, args.gs1, periodic);
	cic_axis ax2(z, args.gn2, args.gs2, periodic);

	rdata[i] = cic_interp_3d(ax0, ax1, ax2, args.gdata);
    }

    return ret;
}


// -------------------------------------------------------------------------------------------------
//
// CIC gridding.


inline void cic_grid_1d(cic_axis &ax, double *arr, double val)
{
    arr[ax.i0] += ax.w0 * val;
    arr[ax.i1] += ax.w1 * val;
}

inline void cic_grid_2d(cic_axis &ax0, cic_axis &ax1, double *arr, double val)
{
    cic_grid_1d(ax1, arr + ax0.i0, ax0.w0 * val);
    cic_grid_1d(ax1, arr + ax0.i1, ax0.w1 * val);
}

inline void cic_grid_3d(cic_axis &ax0, cic_axis &ax1, cic_axis &ax2, double *arr, double val)
{
    cic_grid_2d(ax1, ax2, arr + ax0.i0, ax0.w0 * val);
    cic_grid_2d(ax1, ax2, arr + ax0.i1, ax0.w1 * val);
}


void cic_grid_3d(py::array_t<double> &grid, py::array_t<const double> &points, py::array_t<const double> &weights, double wscal, double lpos0, double lpos1, double lpos2, double pixsize, bool periodic)
{
    interpolation_args<double> args(grid, points, weights, wscal, lpos0, lpos1, lpos2, pixsize);

    if ((args.gn0 < 2) || (args.gn1 < 2) || (args.gn2 < 2))
	throw runtime_error("kszx.grid_points('cic'): all grid dimensions must be >= 2");

    for (long i = 0; i < args.npoints; i++) {
	double x, y, z, w;
	args.get_xyzw(i, x, y, z, w);
	
	cic_axis ax0(x, args.gn0, args.gs0, periodic);
	cic_axis ax1(y, args.gn1, args.gs1, periodic);
	cic_axis ax2(z, args.gn2, args.gs2, periodic);

	cic_grid_3d(ax0, ax1, ax2, args.gdata, w);
    }
}
