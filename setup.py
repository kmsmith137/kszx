# Reference: https://github.com/pybind/python_example/blob/master/setup.py

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

cpp_source_files = [
    "cpp/multiply_xli.cpp",
    "cpp/cpp_kernels.cpp",
    "cpp/cic.cpp",
    "cpp/cubic.cpp",
    "cpp/estimate_power_spectrum.cpp",
    "cpp/kbin_average.cpp"
]

ext_module = Pybind11Extension(
    "kszx.cpp_kernels",
    cpp_source_files,
    extra_compile_args = ['-O3','-fopenmp'],
    extra_link_args = ['-fopenmp']
)

setup(
    name = 'kszx',
    version = '0.0.1',
    packages = [ 'kszx', 'kszx.tests', 'kszx.timing', 'kszx.retirement_home' ],
    ext_modules = [ ext_module ],
    cmdclass = {"build_ext": build_ext},
    zip_safe = False,
    python_requires = ">=3.11"
)
