#include <iostream>
#include "cpp_kernels.hpp"
#include "interpolation_args.hpp"

using namespace std;


inline long wrap_lo(long i, long n) { return (i >= 0) ? i : (i+n); }
inline long wrap_hi(long i, long n) { return (i < n) ? i : (i-n); }


struct cubic_axis
{
    double w0, w1, w2, w3;
    long i0, i1, i2, i3;  // includes strides, may wrap around

    inline cubic_axis(double x, long n, long stride, bool periodic)
    {
	x = periodic ? xfmod(x,n) : x;
	
	long i = long(x);
	bool good = periodic || ((i >= 1) && (i <= n-3));
	
	if (_unlikely(!good))
	    throw runtime_error("kszx: point is out of bounds in interpolate_points() or grid_points()");
	
	i = std::max(i, 0L);   // unnecessary?
	i = std::min(i, n-1);  // unnecessary?
	double f = x - (double)i;
	
	this->w0 = -(f)*(f-1)*(f-2) / 6.0;
	this->w1 = (f+1)*(f-1)*(f-2) / 2.0;
	this->w2 = -(f+1)*(f)*(f-2) / 2.0;		
	this->w3 = (f+1)*(f)*(f-1) / 6.0;
	
	this->i0 = stride * wrap_lo(i-1,n);
	this->i1 = stride * i;
	this->i2 = stride * wrap_hi(i+1,n);
	this->i3 = stride * wrap_hi(i+2,n);
    }
};


// -------------------------------------------------------------------------------------------------
//
// Cubic interpolation.


inline double cubic_interp_1d(cubic_axis &ax, const double *arr)
{
    return (ax.w0 * arr[ax.i0]) + (ax.w1 * arr[ax.i1]) + (ax.w2 * arr[ax.i2]) + (ax.w3 * arr[ax.i3]);
}

inline double cubic_interp_2d(cubic_axis &ax0, cubic_axis &ax1, const double *arr)
{
    double f0 = cubic_interp_1d(ax1, arr + ax0.i0);
    double f1 = cubic_interp_1d(ax1, arr + ax0.i1);
    double f2 = cubic_interp_1d(ax1, arr + ax0.i2);
    double f3 = cubic_interp_1d(ax1, arr + ax0.i3);
    return (ax0.w0 * f0) + (ax0.w1 * f1) + (ax0.w2 * f2) + (ax0.w3 * f3);
}

inline double cubic_interp_3d(cubic_axis &ax0, cubic_axis &ax1, cubic_axis &ax2, const double *arr)
{
    double f0 = cubic_interp_2d(ax1, ax2, arr + ax0.i0);
    double f1 = cubic_interp_2d(ax1, ax2, arr + ax0.i1);
    double f2 = cubic_interp_2d(ax1, ax2, arr + ax0.i2);
    double f3 = cubic_interp_2d(ax1, ax2, arr + ax0.i3);
    return (ax0.w0 * f0) + (ax0.w1 * f1) + (ax0.w2 * f2) + (ax0.w3 * f3);
}


py::array_t<double> cubic_interpolate_3d(py::array_t<const double> &grid, py::array_t<const double> &points, double lpos0, double lpos1, double lpos2, double pixsize, bool periodic)
{
    interpolation_args<const double> args(grid, points, lpos0, lpos1, lpos2, pixsize);
    
    if ((args.gn0 < 4) || (args.gn1 < 4) || (args.gn2 < 4))
	throw runtime_error("kszx.interpolate_points('cubic'): all grid dimensions must be >= 4");

    py::array_t<double> ret({args.npoints});
    double *rdata = ret.mutable_data();

#pragma omp parallel for schedule(guided,32)
    for (long i = 0; i < args.npoints; i++) {
	double x, y, z;
	args.get_xyz(i, x, y, z);

	cubic_axis ax0(x, args.gn0, args.gs0, periodic);
	cubic_axis ax1(y, args.gn1, args.gs1, periodic);
	cubic_axis ax2(z, args.gn2, args.gs2, periodic);

	rdata[i] = cubic_interp_3d(ax0, ax1, ax2, args.gdata);
    }

    return ret;
}


// -------------------------------------------------------------------------------------------------
//
// Cubic gridding.


inline void cubic_grid_1d(cubic_axis &ax, double *arr, double val)
{
    arr[ax.i0] += ax.w0 * val;
    arr[ax.i1] += ax.w1 * val;
    arr[ax.i2] += ax.w2 * val;
    arr[ax.i3] += ax.w3 * val;
}

inline void cubic_grid_2d(cubic_axis &ax0, cubic_axis &ax1, double *arr, double val)
{
    cubic_grid_1d(ax1, arr + ax0.i0, ax0.w0 * val);
    cubic_grid_1d(ax1, arr + ax0.i1, ax0.w1 * val);
    cubic_grid_1d(ax1, arr + ax0.i2, ax0.w2 * val);
    cubic_grid_1d(ax1, arr + ax0.i3, ax0.w3 * val);
}

inline void cubic_grid_3d(cubic_axis &ax0, cubic_axis &ax1, cubic_axis &ax2, double *arr, double val)
{
    cubic_grid_2d(ax1, ax2, arr + ax0.i0, ax0.w0 * val);
    cubic_grid_2d(ax1, ax2, arr + ax0.i1, ax0.w1 * val);
    cubic_grid_2d(ax1, ax2, arr + ax0.i2, ax0.w2 * val);
    cubic_grid_2d(ax1, ax2, arr + ax0.i3, ax0.w3 * val);
}


void cubic_grid_3d(py::array_t<double> &grid, py::array_t<const double> &points, py::array_t<const double> &weights, double wscal, double lpos0, double lpos1, double lpos2, double pixsize, bool periodic)
{
    interpolation_args<double> args(grid, points, weights, wscal, lpos0, lpos1, lpos2, pixsize);
    
    if ((args.gn0 < 4) || (args.gn1 < 4) || (args.gn2 < 4))
	throw runtime_error("kszx.grid_points('cubic'): all grid dimensions must be >= 4");

    for (long i = 0; i < args.npoints; i++) {
	double x, y, z, w;
	args.get_xyzw(i, x, y, z, w);
	
	cubic_axis ax0(x, args.gn0, args.gs0, periodic);
	cubic_axis ax1(y, args.gn1, args.gs1, periodic);
	cubic_axis ax2(z, args.gn2, args.gs2, periodic);
	
	cubic_grid_3d(ax0, ax1, ax2, args.gdata, w);
    }
}
