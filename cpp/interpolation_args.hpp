#ifndef _KSZX_INTERPOLATION_ARGS_HPP
#define _KSZX_INTERPOLATION_ARGS_HPP

#include "cpp_kernels.hpp"


// -------------------------------------------------------------------------------------------------
//
// interpolation_args: this helper class handles the arguments (grid, points, lpos, pixsize)
// that are common to all interpolation/gridding kernels. Used in cic.cpp, cubic.cpp, etc.
//
// Use T=(const double) for interpolation, and T=(double) for gridding.
template<typename T>
struct interpolation_args
{
    T *gdata;            // grid data
    long gn0, gn1, gn2;  // grid shape (gn2=0 for 2D)
    long gs0, gs1, gs2;  // grid strides (gs2=0 for 2D)

    const double *pdata;  // points data
    long npoints;         // points array has shape (npoints, ndim)
    long ps0, ps1;        // strides in 2-d points array

    // Used to translate 'points' to grid_coords in get_xy() / get_xyz().
    double lpos0, lpos1, lpos2, rec_ps;

    // Optional: weights array (only for gridding, not interpolation)
    const double *wdata = nullptr;
    long ws = 0;      // stride in 1-d weights array
    double w0 = 0.0;  // overall constant (= wscal / pixel_volume)


    // ----- 3D constructors -----

    // 3D interpolation constructor (no 'weights' array).
    interpolation_args(py::array_t<T> &grid, py::array_t<const double> &points, double lpos0_, double lpos1_, double lpos2_, double pixsize)
    {
        if (grid.ndim() != 3)
            throw std::runtime_error("expected 'grid' to be a 3-d array");
        if (points.ndim() != 2)
            throw std::runtime_error("expected 'points' to be a 2-d array");
        if (points.shape(1) != 3)
            throw std::runtime_error("expected 'points' to be a shape (N,3) array");
        if (pixsize <= 0)
            throw std::runtime_error("expected pixsize > 0");

        if constexpr (std::is_const<T>::value)
            gdata = grid.data();
        else
            gdata = grid.mutable_data();

        gn0 = get_shape(grid, 0);
        gn1 = get_shape(grid, 1);
        gn2 = get_shape(grid, 2);
        gs0 = get_stride(grid, 0);
        gs1 = get_stride(grid, 1);
        gs2 = get_stride(grid, 2);

        if ((gn0 < 2) || (gn1 < 2) || (gn2 < 2))
            throw std::runtime_error("expected all grid dimensions >= 2");

        pdata = points.data();
        npoints = points.shape(0);
        ps0 = get_stride(points, 0);
        ps1 = get_stride(points, 1);

        lpos0 = lpos0_;
        lpos1 = lpos1_;
        lpos2 = lpos2_;
        rec_ps = 1.0 / pixsize;
    }

    // 3D gridding constructor (with 'weights' array).
    interpolation_args(py::array_t<T> &grid, py::array_t<const double> &points, py::array_t<const double> &weights, double wscal, double lpos0_, double lpos1_, double lpos2_, double pixsize)
        : interpolation_args(grid, points, lpos0_, lpos1_, lpos2_, pixsize)
    {
        if (weights.ndim() == 0) {
            wdata = weights.data();
            ws = 0;
        }
        else if ((weights.ndim() == 1) && (weights.shape(0) == npoints)) {
            wdata = weights.data();
            ws = get_stride(weights, 0);
        }
        else
            throw std::runtime_error("expected 'weights' array to be to be either 0-d, or shape (npoints,)");

        w0 = wscal * rec_ps * rec_ps * rec_ps;
    }


    // ----- 2D constructors -----

    // 2D interpolation constructor (no 'weights' array).
    interpolation_args(py::array_t<T> &grid, py::array_t<const double> &points, double lpos0_, double lpos1_, double pixsize)
    {
        if (grid.ndim() != 2)
            throw std::runtime_error("expected 'grid' to be a 2-d array");
        if (points.ndim() != 2)
            throw std::runtime_error("expected 'points' to be a 2-d array");
        if (points.shape(1) != 2)
            throw std::runtime_error("expected 'points' to be a shape (N,2) array");
        if (pixsize <= 0)
            throw std::runtime_error("expected pixsize > 0");

        if constexpr (std::is_const<T>::value)
            gdata = grid.data();
        else
            gdata = grid.mutable_data();

        gn0 = get_shape(grid, 0);
        gn1 = get_shape(grid, 1);
        gn2 = 0;
        gs0 = get_stride(grid, 0);
        gs1 = get_stride(grid, 1);
        gs2 = 0;

        if ((gn0 < 2) || (gn1 < 2))
            throw std::runtime_error("expected all grid dimensions >= 2");

        pdata = points.data();
        npoints = points.shape(0);
        ps0 = get_stride(points, 0);
        ps1 = get_stride(points, 1);

        lpos0 = lpos0_;
        lpos1 = lpos1_;
        lpos2 = 0.0;
        rec_ps = 1.0 / pixsize;
    }

    // 2D gridding constructor (with 'weights' array).
    interpolation_args(py::array_t<T> &grid, py::array_t<const double> &points, py::array_t<const double> &weights, double wscal, double lpos0_, double lpos1_, double pixsize)
        : interpolation_args(grid, points, lpos0_, lpos1_, pixsize)
    {
        if (weights.ndim() == 0) {
            wdata = weights.data();
            ws = 0;
        }
        else if ((weights.ndim() == 1) && (weights.shape(0) == npoints)) {
            wdata = weights.data();
            ws = get_stride(weights, 0);
        }
        else
            throw std::runtime_error("expected 'weights' array to be to be either 0-d, or shape (npoints,)");

        w0 = wscal * rec_ps * rec_ps;
    }


    // ----- 2D accessors -----

    // Get (x,y) in "grid coordinates".
    inline void get_xy(long i, double &x, double &y)
    {
        x = rec_ps * (pdata[i*ps0] - lpos0);
        y = rec_ps * (pdata[i*ps0 + ps1] - lpos1);
    }

    // Get (x,y) in "grid coordinates", plus associated gridding weight.
    inline void get_xyw(long i, double &x, double &y, double &w)
    {
        get_xy(i, x, y);
        w = w0 * wdata[i*ws];  // note factor w0 = wscal / (pixel volume)
    }

    // ----- 3D accessors -----

    // Get (x,y,z) in "grid coordinates".
    inline void get_xyz(long i, double &x, double &y, double &z)
    {
        x = rec_ps * (pdata[i*ps0] - lpos0);
        y = rec_ps * (pdata[i*ps0 + ps1] - lpos1);
        z = rec_ps * (pdata[i*ps0 + 2*ps1] - lpos2);
    }

    // Get (x,y,z) in "grid coordinates", plus associated gridding weight.
    inline void get_xyzw(long i, double &x, double &y, double &z, double &w)
    {
        get_xyz(i, x, y, z);
        w = w0 * wdata[i*ws];  // note factor w0 = wscal / (pixel volume)
    }
};


#endif  // _KSZX_INTERPOLATION_ARGS_HPP
