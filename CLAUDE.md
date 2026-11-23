# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

All shell commands must be run in the `kszx` conda environment. Prefix every Bash command with
`conda run -n kszx --no-capture-output` or activate the environment first.

The package has a compiled C++ extension (`kszx.cpp_kernels`, built via pybind11 from `cpp/*.cpp`).
After editing any file under `cpp/`, rebuild with an editable install:

```
conda run -n kszx --no-capture-output pip install -v -e .
```

## Commands

```
# Run the full unit test suite (defined in kszx/tests/__init__.py:run_all_tests)
conda run -n kszx --no-capture-output python -m kszx test

# Run a single test module/function directly
conda run -n kszx --no-capture-output python -c "from kszx.tests import test_fft; test_fft.test_xli()"

# Run timing benchmarks (kszx/timing/)
conda run -n kszx --no-capture-output python -m kszx time

# Build sphinx docs locally
conda run -n kszx --no-capture-output sphinx-build docs/source docs/_build
```

Note: `pytest` is not used here — tests are plain functions wired together by hand in
`kszx/tests/__init__.py:run_all_tests()`. When adding a new test function, register it there.

## Documentation

Useful documentation is in `docs/source/*.rst` (in addition to docstrings). `docs/source/index.rst`
is the toctree and shows how the package is organized conceptually (core classes / core functions /
high-level classes / datasets) — worth checking when deciding where new code belongs.

## Architecture

kszx is a framework for kSZ (kinetic Sunyaev-Zel'dovich) cross-correlation pipelines using a
"3-d Cartesian" approach. It's organized in layers:

- **Core representation (`Box.py`, `core.py`)**: There is no `Map` class. A real-space map is just
  a `(box, arr)` pair where `arr` is a float numpy array, and a Fourier-space map is a `(box, arr)`
  pair where `arr` is complex. `Box` defines the pixelization (grid shape, pixel size, observer
  location) and array shape/dtype conventions; `core.py` has the stateless functions that operate on
  these pairs (FFTs with spin, interpolation/gridding, power spectrum estimation, simulating Gaussian
  fields, etc.). Read `Box.py`'s class docstring before touching array-shape logic.

- **`kszx/numba/`**: A parallel, numba-accelerated reimplementation of `core.py` and `Cosmology.py`
  (with its own `numba_utils.py` for jitted kernels). This is a separate code path, not a shared
  helper — keep changes to the plain and numba versions in sync when both need the same fix.

- **C++ kernels (`cpp/*.cpp`, `cpp/*.hpp`)**: Performance-critical gridding/interpolation/power-spectrum
  kernels, wrapped with pybind11 into `kszx.cpp_kernels` (declared in `setup.py`). `core.py` calls into
  these for the hot paths (CIC/cubic gridding, `estimate_power_spectrum`, `kbin_average`,
  `multiply_xli`). Changing kernel behavior requires a rebuild (see Environment above), and the
  behavior should stay consistent with the pure-Python fallback paths where they coexist.

- **Core classes**: `Box`/`BoundingBox` (pixelization/geometry), `Cosmology` (cosmological params,
  used across the pipeline), `Catalog` (galaxy/random catalogs — the standard data container passed
  into gridding and pipeline code).

- **High-level classes build on the core layer**: `KszPipe` is the main end-to-end kSZ analysis
  pipeline (reads `params.yml` + catalogs from an input dir, writes power spectra to an output dir;
  runnable via `python -m kszx kszpipe_run <input_dir> <output_dir>` or from a script/notebook).
  `KszPipeOutdir` loads its outputs. `SurrogateFactory` generates surrogate sims used by KszPipe to
  characterize window functions and assign error bars. `Likelihood`/`CmbClFitter`/
  `RegulatedDeconvolver` are other high-level analysis classes layered on top of the core functions.
  See `docs/source/kszpipe.rst` for the pipeline's file-format and algorithm details.

- **Dataset readers (`act.py`, `desi.py`, `desils_lrg.py`, `desils_main.py`, `planck.py`, `sdss.py`)**:
  Each exposes reader functions (e.g. `sdss.read_galaxies()`) with a `download=False` arg that, when
  set, auto-downloads the file. Auto-downloaded files are cached under `$KSZX_DATA_DIR` (falls back to
  `$HOME/kszx_data`) — see `docs/source/intro.rst` for the data-directory convention. Also reachable
  via `python -m kszx download_{act,desi,planck,sdss}`.

- **`mlhack.py` / `wahack.py`**: Experimental code for specific papers (ML paper, wide-angle paper).
  Interfaces here are explicitly unstable/subject to change — don't treat them as part of the stable
  API.

- **`retirement_home/`**: Deprecated code kept around but no longer part of the maintained API.

- Notebooks live in a separate sibling repo, `kszx_notebooks` (not in this repo) — this repo is kept
  small on purpose and contains only the installable package's source.
