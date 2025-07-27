.. _fft wrappers:

:mod:`FFTs`
===========

.. autofunction:: kszx.fft_r2c

.. autofunction:: kszx.fft_c2r

.. _fft_conventions:

Conventions and "spin"
----------------------

In this section, we document our conventions/normalizations for FFTs, and the meaning of the
``spin`` argument to :func:`~kszx.fft_c2r()` and :func:`~kszx.fft_r2c()`.

Real-space and Fourier-space maps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

 - A real-space map is represented as a pair ``(box, numpy_array)``, where ``box`` is an instance
   of class :class:`~kszx.Box`. The numpy array has ``float`` dtype and shape:

   $$(\mbox{real-space shape}) = {\tt \mbox{box.real_space_shape}} = (n_0, n_1, \cdots, n_{d-1})$$

 - A Fourier-space map is represented as a pair ``(box, numpy_array)``, where ``box`` is an instance
   of class :class:`~kszx.Box`. The numpy array has ``complex`` dtype and shape:

   $$(\mbox{Fourier-space shape}) = {\tt \mbox{box.fourier_space_shape}} = (n_0, n_1, \cdots, \lfloor n_{d-1}/2 \rfloor + 1)$$

 - Note that we define a Box class, but not a Map class (instead, we represent maps by ``(box,arr)`` pairs).
   For now, ``arr`` must be an ordinary numpy array, but in the future, we might support more fun possibilities
   (e.g. mpi/cupy/jax arrays).

Fourier conventions and normalization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

 - In :func:`~kszx.fft_c2r()` and :func:`~kszx.fft_r2c()`, we use the following Fourier conventions:

   $$\begin{align}
   f(k) &= V_{pix} \sum_x f(x) e^{-ik\cdot x} \\
   f(x) &= V_{box}^{-1} \sum_k f(k) e^{ik\cdot x}
   \end{align}$$

 - In :func:`~kszx.simulate_gaussian_field()` and :func:`~kszx.estimate_power_spectrum()`, we use
   the following normalization for the power spectrum $P(k)$:

   $$\langle f(k) f(k')^* \rangle = V_{\rm box} P(k) \delta_{kk'}$$

 - Idea behind these conventions: in a finite pixelized box, these conventions are as similar
   as possible to the following infinite-volume continuous conventions:

   $$\begin{align}
   f(k) &= \int d^nx\, f(x) e^{-ik\cdot x} \\
   f(x) &= \int \frac{d^nk}{(2\pi)^n} \, f(k) e^{ik\cdot x} \\
   \langle f(k) f(k')^* \rangle &= P(k) (2\pi)^n \delta^n(k-k')
   \end{align}$$

.. _ffts_with_spin:
   
FFTs with nonzero "spin"
^^^^^^^^^^^^^^^^^^^^^^^^

 - We define spin-1 Fourier transforms by inserting an extra factor
   $\epsilon P_l({\hat k} \cdot {\hat r})$:

   $$\begin{align}
   f(k) &= V_{pix} \sum_x \epsilon^* P_l({\hat k} \cdot {\hat r}) f(x) e^{-ik\cdot x} \\
   f(x) &= V_{box}^{-1} \sum_k \epsilon P_l({\hat k} \cdot {\hat r}) f(k) e^{ik\cdot x}
   \end{align}$$

   where the line-of-sight direction $\hat r$ is defined in "observer coordinates"
   (see :class:`~kszx.Box` for more info), and our convention for the phase $\epsilon$ is:
   
   $$\epsilon = \begin{cases}
   i & \mbox{if $l$ is odd} \\
   1 & \mbox{if $l$ is even}
   \end{cases}$$

   (Note that $\epsilon$ must be real for even $l$, and imaginary for odd $l$, in order
   for the spin-$l$ FFT to preserve the real-valued conditions $f(x)^* = f(x)$ and
   $f(k)^* = f(-k)$.)
   
   Spin-$l$ transforms are useful because they are building blocks for "natural" applications
   such as radial velocities, RSDs, and anisotropic power spectrum estimators. In the bullet
   points below, we give a few applications.

 - Application 1: ``kszx.fft_c2r(..., spin=1)`` can be used to compute the radial velocity
   field from the density field (with a factor $faH/k$). In equations, we have (in a
   constant-time "snapshot" without lightcone evolution):

   $$v_r(x) = \int \frac{d^3k}{(2\pi)^3} \, \frac{faH}{k} (i {\hat k} \cdot {\hat r}) \delta_m(k) e^{ik\cdot x}$$
   
   Code might look like this::

    box = kszx.Box(...)
    cosmo = kszx.Cosmology('planck18+bao')

    # delta_m = Fourier-space density field at z=0
    delta_m = kszx.simulate_gaussian_field(box, lambda k: cosmo.Plin_z0(k))

    # vr = Real-space radial velocity field at z=0
    f = cosmo.frsd(z=0)
    H = cosmo.H(z=0)
    vr = kszx.multiply_kfunc(box, delta, lambda k: f*H/k, dc=0)
    vr = kszx.fft_c2r(box, vr, spin=1)

 - Application 2: the spin-1 r2c transform can be used to estimate $P_{gv}(k)$
   or $P_{vv}(k)$ from the radial velocity field (or the kSZ velocity reconstruction),
   by calling :func:`~kszx.fft_r2c()` followed by :func:`~kszx.estimate_power_spectrum()`.
   
   Code might look like this::

     box = kszx.Box(...)
     kbin_edges = np.linspace(0, 0.1, 11)   # kmax=0.1, nkbins=10

     # Assume we have real-space maps delta_g (galaxy field) and vr (kSZ velocity reconstruction)
     delta_g = ....  # real-space map (dtype float, shape box.real_space_shape)
     vr = ...        # real-space map (dtype float, shape box.real_space_shape)

     # Real space -> Fourier space
     delta_g = kszx.fft_r2c(box, delta_g)   # spin=0 is the default
     vr = kszx.fft_r2c(box, vr, spin=1)     # note spin=1 for radial velocity!

     # Returns a shape (2,2,nkbins) array, containing P_{gg}, P_{g,vr}, P_{vr,vr}.
     # Note that power spectra are unnormalized -- see estimate_power_spectrum() docstring.
     pk = kszx.estimate_power_spectrum(box, [delta_g,vr], kbin_edges)
     
   (In a real pipeline, you'd want to apply power spectrum normalization -- see for example
   :func:`kszx.wfunc_utils.compute_wapprox()`. This exmaple code is intended to illustrate
   low-level building blocks: :func:`~kszx.fft_r2c()` and :func:`~kszx.estimate_power_spectrum()`.)

 - Application 3: the spin-2 c2r transform can be used to simulate redshift space
   distortions (to leading order in $k$.) In equations, we have:

   $$\delta_g(x) = \int \frac{d^3k}{(2\pi)^3} \,
   (b_g + f ({\hat k} \cdot {\hat r})^2)
   \delta_{\rm lin}(k) e^{ik\cdot x}$$

   which we can also write as:
   
   $$\delta_g(x) = \int \frac{d^3k}{(2\pi)^3} \,
   \left( b_g + \frac{f}{3} + \frac{2f}{3} P_2({\hat k} \cdot {\hat r}) \right)
   \delta_{\rm lin}(k) e^{ik\cdot x}$$

   Code might look like this::

    box = kszx.Box(...)
    cosmo = kszx.Cosmology('planck18+bao')
    bg = 2.0

    # delta_m = Fourier-space density field at z=0
    delta_m = kszx.simulate_gaussian_field(box, lambda k: cosmo.Plin_z0(k))

    # delta_g = Real-space radial velocity field at z=0
    f = cosmo.frsd(z=0)
    delta_g = (bg + f/3.) * kszx.fft_c2r(box, delta_m, spin=0)
    delta_g += (2*f/3.) * kszx.fft_c2r(box, delta_m, spin=2)

 - Application 4: anisotropic power spectrum estimation.
   For example, consider the spin-$l$ anisotropic power spectrum estimator $\hat P_l(k)$
   defined in Hand, Li, Slepian, and Seljak (1704.02357). To implement this estimator,
   we cross-correlate the spin-0 and spin-$l$ FFTs. Code might look like this::

     box = kszx.Box(...)
     kbin_edges = np.linspace(0, 0.1, 11)   # kmax=0.1, nkbins=10
     l = 2   # for example

     # Real-space field whose anisotropic power spectrum we want to estimate
     f = np.zeros(box.real_space_shape)
     f[:,:,:] = ...  # initialize somehow

     # Real space -> Fourier space
     f0 = kszx.fft_r2c(box, f, spin=0)
     fl = kszx.fft_r2c(box, f, spin=l)

     # Hand et al P_l(k) estimator
     pk = kszx.estimate_power_spectrum(box, [f0,fl], kbin_edges)
     pk = pk[0,1,:]   # shape (2,2,nkbins) -> (1-d length-nkbins array)
     
   (In a real pipeline, you'd want to apply power spectrum normalization -- see for example
   :func:`kszx.wfunc_utils.compute_wapprox()`. This exmaple code is intended to illustrate
   low-level building blocks: :func:`~kszx.fft_r2c()` and :func:`~kszx.estimate_power_spectrum()`.
   Additionally, our normalization differs from 1704.02357 by the factor $\epsilon$ defined
   above.)
   
 - In addition to :func:`~kszx.fft_r2c()` and :func:`~kszx.fft_c2r()`, the ``spin``
   optional argument also occurs in scattered functions which include an FFT step.
   For example, :func:`~kszx.grid_points()` with ``fft=True``.

.. _fft_implementation

FFT implementation notes
^^^^^^^^^^^^^^^^^^^^^^^^

We implement spin-$l$ FFTs by writing:

$$P_l({\hat k} \cdot {\hat r}) = \sum_{i=0}^{2l} X_{li}({\hat k}) X_{li}({\hat r})$$

where we define real spherical harmonics $\{ X_{li} \}_{0 \le i < 2l+1}$ by:

$$X_{li}({\hat r}) = \begin{cases}
\sqrt{4\pi/(2l+1)} \, Y_{l0}({\hat r}) & \mbox{if $i=0$} \\
\sqrt{8\pi/(2l+1)} \, \mbox{Re} \, Y_{lm}({\hat r}) & \mbox{if $i=2m-1$ where $m\ge 1$} \\
\sqrt{8\pi/(2l+1)} \, \mbox{Im} \, Y_{lm}({\hat r}) & \mbox{if $i=2m$ where $m\ge 1$}
\end{cases}$$

Then the spin-$l$ FFT can be written as a sum of $(2l+1)$ ordinary (spin-0) FFTs.
We write this out explicitly for the c2r transform:

$$\begin{align}
f(x) &= V_{box}^{-1} \sum_k \epsilon P_l({\hat k} \cdot {\hat r}) f(k) e^{ik\cdot x} \\
&= \epsilon V_{box}^{-1} \sum_{i=0}^{2l} X_{li}(x) \sum_k X_{li}(k) f(k) e^{ik\cdot x}
\end{align}$$
