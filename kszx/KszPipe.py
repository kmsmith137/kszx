import os
import yaml
import shutil
import functools
import numpy as np

from . import core
from . import utils
from . import io_utils
from . import wfunc_utils

from .Catalog import Catalog
from .Cosmology import Cosmology
from .SurrogateFactory import SurrogateFactory


class KszPipe:
    def __init__(self, input_dir, output_dir):
        r"""This is the main kSZ analysis pipeline, which computes data/surrogate power spectra from catalogs.

        There are two ways to run a KszPipe pipeline. The first way is to create a KszPipe instance and
        call the ``run()`` method. This can be done either in a script or a jupyter notebook (if you're
        using jupyter, you should keep in mind that the KszPipe may take hours to run, and you'll need to
        babysit the connection to the jupyterhib). The second way is to run from the command line with::
        
           python -m kszx kszpipe_run [-p NUM_PROCESSES] <input_dir> <output_dir>
        
        The ``input_dir`` contains a parameter file ``params.yml`` and galaxy/random catalogs.
        The ``output_dir`` will be populated with power spectra.
        For details of what KszPipe computes, and documentation of file formats, see the sphinx docs:

          https://kszx.readthedocs.io/en/latest/kszpipe.html#kszpipe-details

        After running the pipeline, you may want to load pipeline outputs using the helper class
        :class:`~kszx.KszPipeOutdir`, or do parameter estimation using :class:`~kszx.PgvLikelihood`.
        
        High-level features:
        
          - Runs "surrogate" sims (see overleaf) to characterize the survey window function,
            determine dependence of power spectra on $(f_{NL}, b_v)$, and assign error bars
            to power spectra.
 
          - Velocity reconstruction noise is included in surrogate sims via a bootstrap procedure,
            using the observed CMB realization. This automatically incorporates noise inhomogeneity
            and "striping", and captures correlations e.g. between 90 and 150 GHz.

          - The galaxy catalog can be spectroscopic or photometric (via the ``ztrue_col`` and 
            ``zobs_col`` constructor args). Surrogate sims will capture the effect of photo-z errors.

          - The windowed power spectra $P_{gg}$, $P_{gv}$, $P_{vv}$ use a normalization which
            should be approximately correct. The normalization is an ansatz which is imperfect,
            especially on large scales, so surrogate sims should still be used to compare power
            spetra to models. Eventually, we'll implement a precise calculation of the window
            function.

          - Currently assumes one galaxy field, and two velocity reconstructions labelled
            "90" and "150" (with ACT in mind).

          - Currently, there is not much implemented for CMB foregrounds. Later, I'd like
            to include foreground clustering terms in the surrogate model (i.e. terms of the
            form $b_\delta \delta(x)$, in addition to the kSZ term $b_v v_r(x)$), and estimate
            the $b_\delta$ biases by estimating the spin-zero $P_{gv}$ power spectrum.
        """
        
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Create output_dir and 'tmp' subdir
        os.makedirs(f'{output_dir}/tmp', exist_ok=True)

        # Read files from pipeline input_dir.
        # Reference: https://kszx.readthedocs.io/en/latest/kszpipe.html#kszpipe-details
        
        with open(f'{input_dir}/params.yml', 'r') as f:
            params = yaml.safe_load(f)

            self.version = params['version']
            assert self.version == 1   # for now

            self.nsurr = params['nsurr']
            self.surr_bg = params['surr_bg']
            self.surr_bfg = 0 if 'surr_bfg' not in params else params['surr_bfg']
            self.nzbins_gal = params['nzbins_gal']
            self.nzbins_vr = params['nzbins_vr']

            self.kmax = params['kmax']
            self.nkbins = params['nkbins']            
            self.kbin_edges = np.linspace(0, self.kmax, self.nkbins+1)
            self.kbin_centers = (self.kbin_edges[1:] + self.kbin_edges[:-1]) / 2.

        self.box = io_utils.read_pickle(f'{input_dir}/bounding_box.pkl')
        self.kernel = 'cubic'   # hardcoded for now
        self.deltac = 1.68      # hardcoded for now

        self.pk_data_filename = f'{output_dir}/pk_data.npy'
        self.pk_surr_filename = f'{output_dir}/pk_surrogates.npy'
        self.pk_single_surr_filenames = [ f'{output_dir}/tmp/pk_surr_{i}.npy' for i in range(self.nsurr) ]


    @functools.cached_property
    def cosmo(self):
        return Cosmology('planck18+bao')

    @functools.cached_property
    def gcat(self):
        return Catalog.from_h5(f'{self.input_dir}/galaxies.h5')

    @functools.cached_property
    def rcat(self):
        return Catalog.from_h5(f'{self.input_dir}/randoms.h5')

    @functools.cached_property
    def rcat_xyz_obs(self):
        return self.rcat.get_xyz(self.cosmo, 'zobs')

    @functools.cached_property
    def sum_rcat_gweights(self):
        return np.sum(self.rcat.weight_gal) if hasattr(self.rcat, 'weight_gal') else self.rcat.size

    @functools.cached_property
    def sum_rcat_vr_weights(self):
        return np.sum(self.rcat.weight_vr) if hasattr(self.rcat, 'weight_vr') else self.rcat.size

    
    @functools.cached_property
    def window_function(self):
        r"""
        3-by-3 matrix $W_{ij}$ containing the window function for power spectra on three spatial footprints:
        
          - footprint 0: random catalog weighted by ``weight_gal`` column
          - footprint 1: random catalog weighted by product of columns ``weight_vr * bv_90``
          - footprint 2: random catalog weighted by product of columns ``weight_vr * bv_150``

        These spatial weightings are appropriate for the $\delta_g$, $v_r^{90}$, and $v_r^{150}$ fields.

        Window functions are computed with ``wfunc_utils.compute_wcrude()`` and are crude approximations
        (for more info see :func:`~kszx.wfunc_utils.compute_wcrude()` docstring), but this is okay since
        surrogate fields are treated consistently.
        """
        
        print('Initializing KszPipe.window_function')
        
        nrand = self.rcat.size
        rweights = getattr(self.rcat, 'weight_gal', np.ones(nrand))
        vweights = getattr(self.rcat, 'weight_vr', np.ones(nrand))

        # Cache these properties for later use (not logically necessary, but may help later pipeline stages run faster)
        self.sum_rcat_gweights
        self.sum_rcat_vr_weights
        
        # Fourier-space maps representing footprints.
        footprints = [
            core.grid_points(self.box, self.rcat_xyz_obs, rweights, kernel=self.kernel, fft=True, compensate=True),
            core.grid_points(self.box, self.rcat_xyz_obs, vweights * self.rcat.bv_90, kernel=self.kernel, fft=True, compensate=True),
            core.grid_points(self.box, self.rcat_xyz_obs, vweights * self.rcat.bv_150, kernel=self.kernel, fft=True, compensate=True)
        ]

        # Compute window function using wfunc_utils.compute_wcrude().
        wf = wfunc_utils.compute_wcrude(self.box, footprints)
        
        print('KszPipe.window_function initialized')
        return wf

    
    @functools.cached_property
    def surrogate_factory(self):
        """
        Returns an instance of class SurrogateFactory, a helper class for simulating the
        density and radial velocity fields at locations of randoms.
        """
        
        print('Initializing KszPipe.surrogate_factory')

        surr_ngal_mean = self.gcat.size
        surr_ngal_rms = 4 * np.sqrt(self.gcat.size)  # 4x Poisson        
        sf = SurrogateFactory(self.box, self.cosmo, self.rcat, surr_ngal_mean, surr_ngal_rms, 'ztrue')

        print('KszPipe.surrogate_factory initialized')
        return sf

    
    def get_pk_data(self, run=False, force=False):
        r"""Returns a shape (5,5,nkbins) array, and saves it in ``pipeline_outdir/pk_data.npy``.

        The returned array contains auto and cross power spectra of the following fields:

          - 0: galaxy overdensity, spin 0
          - 1: kSZ velocity reconstruction $v_r^{90}$, spin 0
          - 2: kSZ velocity reconstruction $v_r^{150}$, spin 0  
          - 3: kSZ velocity reconstruction $v_r^{90}$, spin 1
          - 4: kSZ velocity reconstruction $v_r^{150}$, spin 1

        Flags:
        
          - If ``run=False``, then this function expects the $P(k)$ file to be on disk from a previous pipeline run.
          - If ``run=True``, then the $P(k)$ file will be computed if it is not on disk.
          - If ``force=True``, then this function recomputes $P(k)$, even if it is on disk from a previous pipeline run.
        """
        
        if (not force) and os.path.exists(self.pk_data_filename):
            return io_utils.read_npy(self.pk_data_filename)

        if not (run or force):
            raise RuntimeError(f'KszPipe.get_pk_data2(): run=force=False was specified, and file {self.pk_data_filename} not found')
        
        print('get_pk_data(): running\n', end='')
        
        gweights = getattr(self.gcat, 'weight_gal', np.ones(self.gcat.size))
        rweights = getattr(self.rcat, 'weight_gal', np.ones(self.rcat.size))
        vweights = getattr(self.gcat, 'weight_vr', np.ones(self.gcat.size))
        gcat_xyz = self.gcat.get_xyz(self.cosmo)

        # To mitigate CMB foregrounds, we apply mean-subtraction to the vr fields.
        # (Note that we perform the same mean-subtraction to surrogate fields, in get_pk_surrogate().)
        
        coeffs_v90 = utils.subtract_binned_means(vweights * self.gcat.tcmb_90, self.gcat.z, self.nzbins_vr)
        coeffs_v150 = utils.subtract_binned_means(vweights * self.gcat.tcmb_150, self.gcat.z, self.nzbins_vr)

        # Note spin is 0 and 1 on the vr FFTs.
        fourier_space_maps = [
            core.grid_points(self.box, gcat_xyz, gweights, self.rcat_xyz_obs, rweights, kernel=self.kernel, fft=True, compensate=True),
            core.grid_points(self.box, gcat_xyz, coeffs_v90,  kernel=self.kernel, fft=True, spin=0, compensate=True),
            core.grid_points(self.box, gcat_xyz, coeffs_v150, kernel=self.kernel, fft=True, spin=0, compensate=True),
            core.grid_points(self.box, gcat_xyz, coeffs_v90,  kernel=self.kernel, fft=True, spin=1, compensate=True),
            core.grid_points(self.box, gcat_xyz, coeffs_v150, kernel=self.kernel, fft=True, spin=1, compensate=True)
        ]

        # Rescale window function (by roughly a factor Ngal/Nrand in each footprint).
        w = np.zeros(5)
        w[0] = np.sum(gweights) / self.sum_rcat_gweights
        w[1] = w[2] = w[3] = w[4] = np.sum(vweights) / self.sum_rcat_vr_weights
        wf = self.window_function[[0,1,2,1,2],:][:,[0,1,2,1,2]] * w[:, None] * w[None, :]

        # Estimate power spectra. and normalize by dividing by window function.
        pk = core.estimate_power_spectrum(self.box, fourier_space_maps, self.kbin_edges)
        pk /= wf[:, :, None]

        # Save 'pk_data.npy' to disk. Note that the file format is specified here:
        # https://kszx.readthedocs.io/en/latest/kszpipe.html#kszpipe-details
        io_utils.write_npy(self.pk_data_filename, pk)
        
        return pk


    def get_pk_surrogate(self, isurr, run=False, force=False):
        r"""Returns a shape (14,14,nkbins) array, and saves it in ``pipeline_outdir/tmp/pk_surr_{isurr}.npy``.
        
        The returned array contains auto and cross power spectra of the following fields, for a single surrogate:
        
          - 0: surrogate galaxy field $S_g$ with $f_{NL}=0$.
          - 1: derivative $dS_g/df_{NL}$.
          - 2: surrogate kSZ velocity reconstruction $S_v^{90}$, with $b_v=0$ (i.e. noise only) with spin = 0.
          - 3: same as 2 with spin = 1.
          - 4: derivative $dS_v^{90}/db_v$ with spin = 0.
          - 5: same as 4 with spin = 1.
          - 6: derivative $dS_v^{90}/db_fg$ with spin = 0.
          - 7: same as 6 with spin = 1.
          - 8: surrogate kSZ velocity reconstruction $S_v^{150}$, with $b_v=0$ (i.e. noise only) with spin = 0.
          - 9: same as 9 with spin = 1.
          - 10: derivative $dS_v^{150}/db_v$ with spin = 0.
          - 11: same as 10 with spin = 1.
          - 12: derivative $dS_v^{150}/db_fg$ with spin = 0.
          - 13: same as 12 with spin = 1.
        
        Flags:

          - If ``run=False``, then this function expects the $P(k)$ file to be on disk from a previous pipeline run.
          - If ``run=True``, then the $P(k)$ file will be computed if it is not on disk.
          - If ``force=True``, then this function recomputes $P(k)$, even if it is on disk from a previous pipeline run.
        """

        fname = self.pk_single_surr_filenames[isurr]
        
        if (not force) and os.path.exists(fname):
            return io_utils.read_npy(fname)

        if not (run or force):
            raise RuntimeError(f'KszPipe.get_pk_surrogate2(): run=False was specified, and file {fname} not found')

        print(f'get_pk_surrogate({isurr}): running\n', end='')

        zobs = self.rcat.zobs
        nrand = self.rcat.size
        rweights = getattr(self.rcat, 'weight_gal', np.ones(nrand))
        vweights = getattr(self.rcat, 'weight_vr', np.ones(nrand))

        # The SurrogateFactory simulates LSS fields (delta, phi, vr) sampled at random catalog locations.
        self.surrogate_factory.simulate_surrogate()
        ngal = self.surrogate_factory.ngal

        # Noise realization for Sg (see overleaf).
        eta_rms = np.sqrt((nrand/ngal) - (self.surr_bg**2 * self.surrogate_factory.sigma2) * self.surrogate_factory.D**2)
        if np.min(eta_rms) < 0:
            raise RuntimeError('Noise RMS went negative! This is probably a symptom of not enough randoms (note {(ngal/nrand)=})')
        eta = np.random.normal(scale=eta_rms)

        # Each surrogate field is a sum (with coefficients) over the random catalog.
        # First we compute the coefficient arrays (6 fields total).
        # For more info, see the overleaf, or the sphinx docs:
        #  https://kszx.readthedocs.io/en/latest/kszpipe.html#kszpipe-details
        Sg = (ngal/nrand) * rweights * (self.surr_bg * self.surrogate_factory.delta + eta)
        dSg_dfnl = (ngal/nrand) * rweights * (2 * self.deltac) * (self.surr_bg-1) * self.surrogate_factory.phi
        Sv90_noise = vweights * self.surrogate_factory.M * self.rcat.tcmb_90
        Sv150_noise = vweights * self.surrogate_factory.M * self.rcat.tcmb_150
        Sv90_signal = (ngal/nrand) * vweights * self.rcat.bv_90 * self.surrogate_factory.vr
        Sv150_signal = (ngal/nrand) * vweights * self.rcat.bv_150 * self.surrogate_factory.vr
        Sv90_fg = (ngal/nrand) * vweights * self.surr_bfg * self.surrogate_factory.delta
        Sv150_fg = (ngal/nrand) * vweights * self.surr_bfg * self.surrogate_factory.delta

        # Mean subtraction for the surrogate field Sg.
        # This is intended to make the surrogate field more similar to the galaxy overdensity delta_g
        # which satisfies "integral constraints" since the random z-distribution is inferred from the
        # galaxies. (In practice, the effect seems to be small.)
        if self.nzbins_gal > 0:
            Sg = utils.subtract_binned_means(Sg, zobs, self.nzbins_gal)
            dSg_dfnl = utils.subtract_binned_means(dSg_dfnl, zobs, self.nzbins_gal)

        # Mean subtraction for the surrogate fields Sv.
        # This is intended to mitgate foregrounds.( Note that we perform the same
        # mean subtraction to the vr arrays, in get_pk_data()).

        if self.nzbins_vr > 0:
            Sv90_noise = utils.subtract_binned_means(Sv90_noise, zobs, self.nzbins_vr)
            Sv90_signal = utils.subtract_binned_means(Sv90_signal, zobs, self.nzbins_vr)
            Sv150_noise = utils.subtract_binned_means(Sv150_noise, zobs, self.nzbins_vr)
            Sv150_signal = utils.subtract_binned_means(Sv150_signal, zobs, self.nzbins_vr)
            Sv90_fg = utils.subtract_binned_means(Sv90_fg, zobs, self.nzbins_vr)
            Sv150_fg = utils.subtract_binned_means(Sv150_fg, zobs, self.nzbins_vr)

        # (Coefficient arrays) -> (Fourier-space fields).
        fourier_space_maps = []
        for i, field in enumerate([Sg, dSg_dfnl, Sv90_noise, Sv90_signal, Sv90_fg, Sv150_noise, Sv150_signal, Sv150_fg]):
            spins = [0] if  i < 2 else [0, 1]
            for spin in spins:
                fourier_space_maps += [core.grid_points(self.box, self.rcat_xyz_obs, field, kernel=self.kernel, fft=True, spin=spin, compensate=True)]

        # Rescale window function, by a factor (ngal/nrand) in each footprint.
        wf = (ngal/nrand)**2 * self.window_function

        # Expand window function from shape (3,3) to shape (14,14).
        wf = wf[[0,0,1,1,1,1,1,1,2,2,2,2,2,2],:][:,[0,0,1,1,1,1,1,1,2,2,2,2,2,2]]

        # Estimate power spectra. and normalize by dividing by window function.
        pk = core.estimate_power_spectrum(self.box, fourier_space_maps, self.kbin_edges)
        pk /= wf[:,:,None]

        io_utils.write_npy(fname, pk)
        return pk


    def get_pk_surrogates(self):
        """Returns a shape (nsurr,6,6,nkins) array, and saves it in ``pipeline_outdir/pk_surrogates.npy``.
        
        The returned array contains auto and cross power spectra of the following fields, for all surrogates:

          - 0: surrogate galaxy field $S_g$ with $f_{NL}=0$.
          - 1: derivative $dS_g/df_{NL}$.
          - 2: surrogate kSZ velocity reconstruction $S_v^{90}$, with $b_v=0$ (i.e. noise only).
          - 3: derivative $dS_v^{90}/db_v$.
          - 4: surrogate kSZ velocity reconstruction $S_v^{150}$, with $b_v=0$ (i.e. noise only).
          - 5: derivative $dS_v^{150}/db_v$.

        This function only reads files from disk -- it does not run the pipeline.
        To run the pipeline, use :meth:`~kszx.KszPipe.run()`.
        """
        
        if os.path.exists(self.pk_surr_filename):
            return io_utils.read_npy(self.pk_surr_filename)

        if not all(os.path.exists(f) for f in self.pk_single_surr_filenames):
            raise RuntimeError(f'KszPipe.read_pk_surrogates(): necessary files do not exist; you need to call KszPipe.run()')

        pk = np.array([ io_utils.read_npy(f) for f in self.pk_single_surr_filenames ])

        # Save 'pk_surrogates.npy' to disk. Note that the file format is specified here:
        # https://kszx.readthedocs.io/en/latest/kszpipe.html#kszpipe-details
        
        io_utils.write_npy(self.pk_surr_filename, pk)
        
        return pk

        
    def run(self, processes):
        """Runs pipeline and saves results to disk, skipping results already on disk from previous runs.

        Implementation: creates a multiprocessing Pool, and calls :meth:`~kszx.KszPipe.get_pk_data()`
        and :meth:`~kszx.KszPipe.get_pk_surrogates()` in worker processes.

        Can be run from the command line with::
        
           python -m kszx kszpipe_run [-p NUM_PROCESSES] <input_dir> <output_dir>

        The ``processes`` argument is the number of worker processes. Currently I don't have a good way
        of setting this automatically -- the caller must adjust the number of processes, based on the
        size of the datasets, and amount of memory available.
        """
        
        # Copy yaml file from input to output dir.
        if not os.path.exists(f'{self.output_dir}/params.yml'):
            shutil.copyfile(f'{self.input_dir}/params.yml', f'{self.output_dir}/params.yml')
                
        have_data = os.path.exists(self.pk_data_filename)
        have_surr = os.path.exists(self.pk_surr_filename)
        missing_surrs = [ ] if have_surr else [ i for (i,f) in enumerate(self.pk_single_surr_filenames) if not os.path.exists(f) ]

        if (not have_surr) and (len(missing_surrs) == 0):
            self.get_pk_surrogates()   # creates "top-level" file
            have_surr = True
            
        if have_data and have_surr:
            print(f'KszPipe.run(): pipeline has already been run, exiting early')
            return
        
        # Initialize window function and SurrogateFactory before creating multiprocessing Pool.
        self.window_function
        self.surrogate_factory

        # FIXME currently I don't have a good way of setting the number of processes automatically --
        # caller must adjust the number of processes to the amount of memory in the node.
        
        with utils.Pool(processes) as pool:
            l = [ ]
                
            if not have_data:
                l += [ pool.apply_async(self.get_pk_data, (True,False)) ]   # (run,force)=(True,False)
            for i in missing_surrs:
                l += [ pool.apply_async(self.get_pk_surrogate, (i,True)) ]  # (run,force)=(True,False)

            for x in l:
                x.get()

        if not have_surr:
            # Consolidates all surrogates into one file
            self.get_pk_surrogates()


####################################################################################################


class KszPipeOutdir:
    def __init__(self, dirname, nsurr=None):
        r"""A helper class for loading and processing output files from ``class KszPipe``.

        Note: for MCMCs and parameter fits, there is a separate class :class:`~kszx.PgvLikelihood`.
        The KszPipeOutdir class is more minimal (the main use case is plot scripts!)

        The constructor reads the files ``{dirname}/params.yml``, ``{dirname}/pk_data.npy``,
        ``{dirname}/pk_surrogates.npy`` which are generated by :meth:`~kszx.KszPipe.run()`.
        For more info on these files, and documentation of file formats, see the sphinx docs:

           https://kszx.readthedocs.io/en/latest/kszpipe.html#kszpipe-details
          
        Constructor arguments:

          - ``dirname`` (string): name of pipeline output directory.
         
          - ``nsurr`` (integer or None): this is a hack for running on an incomplete
            pipeline. If specified, then ``{dirname}/pk_surr.npy`` is not read.
            Instead we read files of the form ``{dirname}/tmp/pk_surr_{i}.npy``.
        """

        filename = f'{dirname}/params.yml'
        print(f'Reading {filename}')
        
        with open(filename, 'r') as f:
            params = yaml.safe_load(f)

        kmax = params['kmax']
        nkbins = params['nkbins']
        
        kbin_edges = np.linspace(0, kmax, nkbins+1)
        kbin_centers = (kbin_edges[1:] + kbin_edges[:-1]) / 2.

        pk_data = io_utils.read_npy(f'{dirname}/pk_data.npy')
        if pk_data.shape != (5, 5, nkbins):
            raise RuntimeError(f'Got {pk_data.shape=}, expected (5,5,nkbins) where {nkbins=}')

        if nsurr is None:
            pk_surr = io_utils.read_npy(f'{dirname}/pk_surrogates.npy')
            if (pk_surr.ndim != 4) or (pk_surr.shape[1:] != (14,14,nkbins)):
                raise RuntimeError(f'Got {pk_surr.shape=}, expected (nsurr,14,14,nkbins) where {nkbins=}')

        else:
            pk_surr = [ ]
            
            for i in range(nsurr):
                pk_surrogate = io_utils.read_npy(f'{dirname}/tmp/pk_surr_{i}.npy')
                if pk_surrogate.shape != (14, 14, nkbins):
                    raise RuntimeError(f'Got {pk_surrogate.shape=}, expected (14,14,nkbins) where {nkbins=}')
                pk_surr.append(pk_surrogate)
                
            pk_surr = np.array(pk_surr)
            pk_surr = np.reshape(pk_surr, (nsurr, 14, 14, nkbins))   # needed if nsurr==0

        self.k = kbin_centers
        self.nkbins = nkbins
        self.kmax = kmax
        self.dk = kmax / nkbins
        
        self.pk_data = pk_data
        self.pk_surr = np.array(pk_surr)
        self.nsurr = self.pk_surr.shape[0]
        self.surr_bg = params['surr_bg']
        self.surr_bfg = 0 if 'surr_bfg' not in params else params['surr_bfg']


    def pgg_data(self):
        r"""Returns shape ``(nkbins,)`` array containing $P_{gg}^{data}(k)$."""
        return self.pk_data[0,0,:]
        
    def pgg_mean(self, fnl=0):
        r"""Returns shape ``(nkbins,)`` array, containing $\langle P_{gg}^{surr}(k) \rangle$."""
        assert self.nsurr >= 1
        return np.mean(self._pgg_surr(fnl), axis=0)

    def pgg_rms(self, fnl=0):
        r"""Returns shape ``(nkbins,)`` array, containing sqrt(Var($P_{gg}^{surr}(k)$))."""
        assert self.nsurr >= 2
        return np.sqrt(np.var(self._pgg_surr(fnl), axis=0))

    def pgv_data(self, field, ell=1):
        r"""Returns shape ``(nkbins,)`` array containing $P_{gv}^{ell=1}^{data}(k)$.

        The ``field`` argument is a length-2 vector, selecting a linear combination
        of the 90+150 GHz velocity reconstructions. For example:
        
           - ``field=[1,0]`` for 90 GHz reconstruction
           - ``field=[0,1]`` for 150 GHz reconstruction
           - ``field=[1,-1]`` for null (90-150) GHz reconstruction.
        """
        assert ell in [0, 1]
        field = self._check_field(field)
        if ell == 0:
            return field[0]*self.pk_data[0,1] + field[1]*self.pk_data[0,2]
        elif ell == 1:
            return field[0]*self.pk_data[0,3] + field[1]*self.pk_data[0,4]

    def pgv_mean(self, field, fnl, bv, bfg=0, ell=1):
        r"""Returns shape ``(nkbins,)`` array containing $\langle P_{gv}^{surr}(k) \rangle$.

        The ``field`` argument is a length-2 vector, selecting a linear combination
        of the 90+150 GHz velocity reconstructions. For example:
        
           - ``field=[1,0]`` for 90 GHz reconstruction
           - ``field=[0,1]`` for 150 GHz reconstruction
           - ``field=[1,-1]`` for null (90-150) GHz reconstruction.
        """
        assert ell in [0, 1]
        assert self.nsurr >= 1
        return np.mean(self._pgv_surr(field, fnl, bv, bfg, ell=ell), axis=0)

    def pgv_rms(self, field, fnl, bv, bfg=0, ell=1):
        r"""Returns shape ``(nkbins,)`` array containing sqrt(Var($P_{gv}^{surr}(k)$)).

        The ``field`` argument is a length-2 vector, selecting a linear combination
        of the 90+150 GHz velocity reconstructions. For example:
        
           - ``field=[1,0]`` for 90 GHz reconstruction
           - ``field=[0,1]`` for 150 GHz reconstruction
           - ``field=[1,-1]`` for null (90-150) GHz reconstruction.
        """
        assert ell in [0, 1]
        assert self.nsurr >= 2
        return np.sqrt(np.var(self._pgv_surr(field, fnl, bv, bfg, ell=ell), axis=0))

    def pvv_data(self, field, ell=1):
        r"""Returns shape ``(nkbins,)`` array containing $P_{vv}^{data}(k)$.

        The ``field`` argument is a length-2 vector, selecting a linear combination
        of the 90+150 GHz velocity reconstructions. For example:
        
           - ``field=[1,0]`` for 90 GHz reconstruction
           - ``field=[0,1]`` for 150 GHz reconstruction
           - ``field=[1,-1]`` for null (90-150) GHz reconstruction.
        """
        assert ell in [0, 1]
        field = self._check_field(field)
        if ell == 0:
            t = field[0]*self.pk_data[1,1:3] + field[1]*self.pk_data[2,1:3]  # shape (5,)
        elif ell == 1:
            t = field[0]*self.pk_data[3,3:5] + field[1]*self.pk_data[4,3:5]  # shape (5,)
        return field[0]*t[0] + field[1]*t[1]
        
    def pvv_mean(self, field, bv, bfg=0, ell=1):
        r"""Returns shape ``(nkbins,)`` array containing $\langle P_{vv}^{data}(k) \rangle$.

        The ``field`` argument is a length-2 vector, selecting a linear combination
        of the 90+150 GHz velocity reconstructions. For example:
        
           - ``field=[1,0]`` for 90 GHz reconstruction
           - ``field=[0,1]`` for 150 GHz reconstruction
           - ``field=[1,-1]`` for null (90-150) GHz reconstruction.
        """
        assert ell in [0, 1]
        assert self.nsurr >= 1
        return np.mean(self._pvv_surr(field, bv, bfg, ell=ell), axis=0)
        
    def pvv_rms(self, field, bv, bfg=0, ell=1):
        r"""Returns shape ``(nkbins,)`` array containing sqrt(var($P_{vv}^{data}(k)$)).

        The ``field`` argument is a length-2 vector, selecting a linear combination
        of the 90+150 GHz velocity reconstructions. For example:
        
           - ``field=[1,0]`` for 90 GHz reconstruction
           - ``field=[0,1]`` for 150 GHz reconstruction
           - ``field=[1,-1]`` for null (90-150) GHz reconstruction.
        """
        assert ell in [0, 1]
        assert self.nsurr >= 2
        return np.sqrt(np.var(self._pvv_surr(field, bv, bfg, ell=ell), axis=0))

    def _check_field(self, field):
        """Checks that 'field' is a 1-d array of length 2."""
        field = np.array(field, dtype=float)
        if field.shape != (2,):
            raise RuntimeError(f"Expected 'field' argument to be a 1-d array of length 2, got {field.shape=}")
        return field
    
    def _pgg_surr(self, fnl=0):
        """Returns shape (nsurr, nkbins) array, containing P_{gg} for each surrogate"""
        pgg = self.pk_surr[:,:2,:2,:]             # shape (nsurr, 2, 2, nkbins)
        pgg = pgg[:,0,:,:] + fnl * pgg[:,1,:,:]   # shape (nsurr, 2, nkbins)
        return pgg[:,0,:] + fnl * pgg[:,1,:]      # shape (nsurr, nkbins)

    def _pgv_surr(self, field, fnl, bv, bfg=0, ell=1):
        """Returns shape (nsurr, nkbins) array, containing P_{gv} for each surrogate"""
        field = self._check_field(field)
        pgv = self.pk_surr[:,:2,2:,:]                                 # shape (nsurr, 2, 12, nkbins)
        # this is the g term:
        pgv = pgv[:,0,:,:] + fnl * pgv[:,1,:,:]                       # shape (nsurr, 12, nkbins)
        # keep only 90 or 150 and the correct ell:
        pgv = field[0] * pgv[:,0+ell:6:2,:] + field[1] * pgv[:,6+ell:12:2,:]      # shape (nsurr, 6, nkbins)
        return pgv[:,0,:] + bv * pgv[:,1,:] + bfg * pgv[:,2,:]   # shape (nsurr, nkbins)

    def _pvv_surr(self, field, bv, bfg=0, ell=1):
        """Returns shape (nsurr, nkbins) array, containing P_{vv} for each surrogate"""
        field = self._check_field(field)
        pvv = self.pk_surr[:,2:,2:,:]                                  # shape (nsurr, 12, 12, nkbins)
        pvv = field[0] * pvv[:,0+ell:6:2,0+ell:6:2,:] + field[1] * pvv[:,6+ell:12:2,6+ell:12:2,:] # shape (nsurr, 3, 3, nkbins)
        pvv = pvv[:,0,:,:] + bv * pvv[:,1,:,:] + bfg * pvv[:,2,:,:]    # shape (nsurr, 3, nkbins)
        return pvv[:,0,:] + bv * pvv[:,1,:] + bfg * pvv[:,2,:]       # shape (nsurr, nkbins)
