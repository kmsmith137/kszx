#include "cpp_kernels.hpp"
#include <omp.h>


PYBIND11_MODULE(cpp_kernels, m)
{
    // m.doc = "C++ kernels";

    m.def("omp_get_max_threads", omp_get_max_threads);
    m.def("omp_set_num_threads", omp_set_num_threads, py::arg("nthreads"));
    
    m.def("cic_interpolate_3d", cic_interpolate_3d,
	  py::arg("grid"), py::arg("points"),
	  py::arg("lpos0"), py::arg("lpos1"), py::arg("lpos2"),
	  py::arg("pixsize"), py::arg("periodic"));

    m.def("cubic_interpolate_3d", cubic_interpolate_3d,
	  py::arg("grid"), py::arg("points"),
	  py::arg("lpos0"), py::arg("lpos1"), py::arg("lpos2"),
	  py::arg("pixsize"), py::arg("periodic"));

    m.def("cic_grid_3d", cic_grid_3d,
	  py::arg("grid"), py::arg("points"), py::arg("weights"),
	  py::arg("wscal"), py::arg("lpos0"), py::arg("lpos1"),
	  py::arg("lpos2"), py::arg("pixsize"), py::arg("periodic"));

    m.def("cubic_grid_3d", cubic_grid_3d,
	  py::arg("grid"), py::arg("points"), py::arg("weights"),
	  py::arg("wscal"), py::arg("lpos0"), py::arg("lpos1"),
	  py::arg("lpos2"), py::arg("pixsize"), py::arg("periodic"));

    m.def("estimate_power_spectrum", estimate_power_spectrum,
	  py::arg("map_list"), py::arg("k_delim"),
	  py::arg("npix"), py::arg("kf"),
	  py::arg("box_volume"));

    m.def("kbin_average", kbin_average,
	  py::arg("fk"), py::arg("k_delim"),
	  py::arg("npix"), py::arg("kf"));

    m.def("multiply_xli_real_space", multiply_xli_real_space,
	  py::arg("dst"), py::arg("src"), py::arg("l"), py::arg("i"),
	  py::arg("lpos0"), py::arg("lpos1"), py::arg("lpos2"),
	  py::arg("pixsize"), py::arg("coeff"), py::arg("accum"));

    m.def("multiply_xli_fourier_space", multiply_xli_fourier_space,
	  py::arg("dst"), py::arg("src"), py::arg("l"), py::arg("i"),
	  py::arg("nz"), py::arg("coeff"), py::arg("accum"));
}
