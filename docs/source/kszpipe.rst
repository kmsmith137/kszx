:mod:`KszPipe`
==============

.. autoclass:: kszx.KszPipe
    :members:

.. autoclass:: kszx.KszPipeOutdir
    :members:

.. _kszpipe_details:

KszPipe details
---------------

High-level organization
^^^^^^^^^^^^^^^^^^^^^^^

Our KSZ pipelines can be organized into three stages:

- "Pre-KszPipe": a collection of jupyter notebooks which filter the CMB maps,
  apply cuts/weighting to the galaxy catalog, and write a few output files
  (``params.yml``, ``galaxies.h5``, ``randoms.h5``). These output files are
  specified below under :ref:`kszpipe_input_files`.

  For this stage, we've been using ad hoc jupyter notebooks, rather than
  "polished" pipeline code in git. We often want to run experiments where
  we change an aspect of the processing, so the "hackability" of jupyter
  notebooks is really convenient.

- "KszPipe": takes the galaxy/random catalogs (see :ref:`kszpipe_input_files`),
  and creates files containing power spectra of the data, and power spectra of
  simulated "surrogate" fields. These output files are specified below under
  :ref:`kszpipe_output_files`.

  In this stage, the heavy lifting can be done with :class:`kszx.KszPipe`,
  which calls :class:`kszx.SurrogateFactory`.

- "Post-KszPipe": plot power spectra, fit parameters $(f_{NL}, b_v)$,
  and run MCMCs. Some useful helper classes: :class:`kszx.KszPipeOutdir`,
  :class:`kszx.PgvLikelihood`.
   
Data and surrogate fields
^^^^^^^^^^^^^^^^^^^^^^^^^

The KszPipe computes power spectra involving a galaxy density field $\rho_g$,
one or more kSZ velocity reconstructions $\hat v_r$, and surrogate fields $S_g, S_v$.
Definitions of these fields are given in the overleaf, and can be summarized as follows:

$$W_L(x) &= \mbox{Large-scale galaxy weight function (e.g. FKP)}  \\
W_S(x) &= \mbox{Small-scale weight function, applied to galaxies during velocity reconstruction}  \\
\rho_g(x) &= \bigg( \sum_{i\in \rm gal} W_i^L \, \delta^3(x-x_i) \bigg) - \frac{N_g}{N_r} \bigg( \sum_{j\in \rm rand} W_j^L \, \delta^3(x-x_j) \bigg) \\
\hat v_r(x) &= \sum_{i\in \rm gal} W_i^S \, \tilde T(\theta_i) \, \delta^3(x-x_i) \\
S_g(x) &= \sum_{j\in \rm rand} \frac{N_g}{N_r} W_j^L \big( b_j^G \delta_m(x_j) + f_{NL} b_j^{NG} \phi(x_j) + \eta_j \big) \delta^3(x-x_j) \\
S_v(x) &= \sum_{j\in\rm rand} \bigg( \frac{N_g}{N_r} W_j^S b_j^v v_r(x_j) + M_j W_j^S \tilde T(\theta_j) \bigg)  \delta^3(x-x_j) \\
\big\langle \eta_j^2 \big\rangle &= \frac{N_r}{N_g} - \big\langle \delta_G(z_j)^2 \big\rangle \\
M_j &= \begin{cases}
1 & \mbox{if $j$ is in a randomly selected subset of size } N_{\rm gal} \\
0 & \mbox{otherwise}
\end{cases}$$

To mitigate foregrounds, the KszPipe also includes mean-subtraction steps in $v_r$ and $S_v$ which
are described in the overleaf.

The bias $b_v$ which appears in these equations is a per-object fiducial bias, which must
be estimated in an earlier pipeline stage, before the KszPipe is run.
The fiducial bias $b_v$ will depend on a choice of fiducial $P_{ge}(k)$, as well as the filter
applied to $T_{CMB}$. (Since different CMB frequency channels can use different CMB filters,
$b_v$ can depend on CMB frequency channel.)
One reasonable way of initializing $b_v$ is to use the following approximate expression
from the overleaf:
$$b_j^v &\approx B_v(z_j) \, W_{\rm CMB}(\theta_j) \\
B_v(\chi) &\equiv \frac{K(\chi)}{\chi^2} \int \frac{d^2L}{(2\pi)^2} \, b_L F_L \, P_{ge}^{\rm true}(k,\chi)_{k=L/\chi}$$

.. _kszpipe_input_files:

KszPipe input files
^^^^^^^^^^^^^^^^^^^

The KszPipe constructor has an argument ``input_dir``.
This is the name of a directory containing the following files:

 - ``params.yml``: this file is easiest to describe by example::

     # Version of the KszPipe params.yml format.
     # (In case we change the file format in the future.)
     version: 1

     # Number of surrogate sims
     nsurr: 400

     # Galaxy bias used in surrogate sims
     surr_bg: 2.1

     # Number of redshift bins used for field-level mean-subtraction.
     # Passed as 'nbins' argument to kszx.utils.subtract_binned_means().
     # If zero, then kszx.utils.subtract_binned_means() is not called.

     nzbins_gal: 25  # mean-subtraction for galaxy field
     nzbins_vr: 25   # mean-subtraction for galaxy field

     # k-binning for power spectrum estimation
     # Note: bins are generated from (nkbins, kmax) as follows.
     #
     #  kbin_edges = np.linspace(0, kmax, nkbins+1)
     #  kbin_centers = (kbin_edges[1:] + kbin_edges[:-1]) / 2.

     nkbins: 25
     kmax: 0.05

 - ``galaxies.h5``: galaxy catalog, written in HDF5 format by
   :meth:`~kszx.Catalog.write_h5()`. The galaxy catalog should
   contain the following columns.

     - ``ra_deg``, ``dec_deg``:  angular locations of galaxies.
     - ``bv_90``, ``bv_150``: per-object KSZ velocity bias $b_v^i$, defined by the
       equation $\tilde T(\theta_i) = b_v^i v_r(x_i) + (\mbox{noise})$.
     - ``tcmb_90``, ``tcmb_150``: per-object filtered CMB temperatures $\tilde T(\theta_i)$.
     - ``weight_gal``, ``weight_vr``: per-object weightings used when constructing the
       galaxy density field $\rho_g$ and kSZ velocity reconstruction $\hat v_r$. (These
       weightings are denoted $W_i^L$ and $W_i^S$ in the overleaf.)
     - ``z``: observed redshift.

 - ``randoms.h5`` random catalog. Contains the columns above, except that the ``z``
   column (redshift) is replaced by two columns ``zobs``, ``ztrue``. (For a photometric
   survey, the ``zobs`` and ``ztrue`` columns will differ -- for a spectroscopic survey
   they will be copies of each other.)

 - ``bounding_box.pkl``: object of class :class:`~kszx.BoundingBox`, saved in pickle format.

.. _kszpipe_output_files:

KszPipe output files
^^^^^^^^^^^^^^^^^^^^

The KszPipe constructor has an argument ``output_dir``.
This is the name of a directory which the KszPipe will populate with the following files:

 - ``params.yml``: this file is copied from the input directory to the output directory
   unmodified.

 - ``pk_data.npy``: array of shape ``(3,3,nkbins)``, containing auto and cross power
   spectra of the following fields:

     - 0: galaxy overdensity $\delta_g$
     - 1: kSZ velocity reconstruction $v_r^{90}$
     - 2: kSZ velocity reconstruction $v_r^{150}$

 - ``pk_surrogates.npy``: array of shape ``(nsurr,6,6,nkbins)``, containing auto and
   cross power spectra of the following fields:

     - 0: surrogate galaxy field $S_g$ with $f_{NL}=0$.
     - 1: derivative $dS_g/df_{NL}$.
     - 2: surrogate kSZ velocity reconstruction $S_v^{90}$, with $b_v=0$ (i.e. noise only).
     - 3: derivative $dS_v^{90}/db_v$.
     - 4: surrogate kSZ velocity reconstruction $S_v^{150}$, with $b_v=0$ (i.e. noise only).
     - 5: derivative $dS_v^{150}/db_v$.
