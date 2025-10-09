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

        June 2025: This is version == 2 that introduces backwards-incompatible changes vs. version == 1.

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
            assert self.version == 2

            self.cmb_fields = params['cmb_fields']
            self.spin_gal = params['estimate_spin_gal']
            self.spin_vr = params['estimate_spin_vr']

            self.nsurr = params['nsurr']
            self.surr_bg = params['surr_bg']
            self.sim_surr_fg = params['simulate_surrogate_foregrounds']
            self.nzbins_gal = params['nzbins_gal']
            self.nzbins_vr = params['nzbins_vr']

            kmin, kmax, kstep = params['kmin'], params['kmax'], params['kstep']
            self.kbin_edges = [np.arange(kmin[i], kmax[i], kstep[i]) for i in range(len(kmin))]

        self.box = io_utils.read_pickle(f'{input_dir}/bounding_box.pkl')
        self.kernel = 'cubic'   # hardcoded for now
        self.deltac = 1.68      # hardcoded for now

        self.pk_data_filename = [f'{output_dir}/pk_data_binning{i}.npy' for i in range(len(self.kbin_edges))]
        self.pk_surr_filename = [f'{output_dir}/pk_surrogates_binning{i}.npy' for i in range(len(self.kbin_edges))]
        self.pk_single_surr_filenames = [[f'{output_dir}/tmp/pk_surr_{j}_binning{i}.npy' for i in range(len(self.kbin_edges))] for j in range(self.nsurr)]

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
        1+len(cmb_fields)-by-1+len(cmb_fields) matrix $W_{ij}$ containing the window function for power spectra on spatial footprints:
        
          - footprint 0: random catalog weighted by ``weight_gal`` column
          - footprint x: random catalog weighted by product of columns ``weight_vr * bv_freq``

        These spatial weightings are appropriate for the $\delta_g$, $v_r^{freq}$ fields.

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
        footprints = [core.grid_points(self.box, self.rcat_xyz_obs, rweights, kernel=self.kernel, fft=True, compensate=True)] + [core.grid_points(self.box, self.rcat_xyz_obs, vweights * self.rcat.get_column(f'bv_{freq}'), kernel=self.kernel, fft=True, compensate=True) for freq in self.cmb_fields]
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
    
    def get_pk_data(self, run=False, force=False, save_fourier_maps=False):
        r"""Returns the computed pk, and saves it in ``pipeline_outdir/pk_data.npy``.

        For spin_gal = [0], spin_vr = [0, 1] and cmb_fields = ['90', '150']. The returned array contains auto and cross power spectra of the following fields:
          - 0: galaxy overdensity, spin 0
          - 1: kSZ velocity reconstruction $v_r^{90}$, spin 0
          - 2: kSZ velocity reconstruction $v_r^{150}$, spin 0  
          - 3: kSZ velocity reconstruction $v_r^{90}$, spin 1
          - 4: kSZ velocity reconstruction $v_r^{150}$, spin 1
        The returned array has shape (5, 5, nkbins), where nkbins is the number of k bins. General form is of shape (#spin_gal + #spin_vr*#cmb_fields, #spin_gal + #spin_vel*#cmb_fields, nkbins).

        Flags:
          - If ``run=False``, then this function expects the $P(k)$ file to be on disk from a previous pipeline run.
          - If ``run=True``, then the $P(k)$ file will be computed if it is not on disk.
          - If ``force=True``, then this function recomputes $P(k)$, even if it is on disk from a previous pipeline run.
        """
        
        if (not force) and all([os.path.exists(fn) for fn in self.pk_data_filename]):
            pks = [io_utils.read_npy(fn) for fn in self.pk_data_filename]
            if len(self.pk_data_filename) == 1: pks = pks[0]
            return pks

        if not (run or force):
            raise RuntimeError(f'KszPipe.get_pk_data(): run=force=False was specified, and file {self.pk_data_filename} not found')
        
        print('get_pk_data(): running\n', end='')
        
        gweights = getattr(self.gcat, 'weight_gal', np.ones(self.gcat.size))
        rweights = getattr(self.rcat, 'weight_gal', np.ones(self.rcat.size))
        vweights = getattr(self.gcat, 'weight_vr', np.ones(self.gcat.size))
        gcat_xyz = self.gcat.get_xyz(self.cosmo)

        # To mitigate CMB foregrounds, we apply mean-subtraction to the vr fields.
        # (Note that we perform the same mean-subtraction to surrogate fields, in get_pk_surrogate().)
        coeffs = {freq: utils.subtract_binned_means(vweights * self.gcat.get_column(f'tcmb_{freq}'), self.gcat.z, self.nzbins_vr) for freq in self.cmb_fields}

        # Compute the different FFT:
        fourier_space_maps, w, idx = [], [], []
        for spin in self.spin_gal:
            fourier_space_maps += [core.grid_points(self.box, gcat_xyz, gweights, self.rcat_xyz_obs, rweights, kernel=self.kernel, fft=True, spin=spin,compensate=True)]
            w += [np.sum(gweights) / self.sum_rcat_gweights]
            idx += [0]
        for spin in self.spin_vr:
            for i, freq in enumerate(self.cmb_fields): 
                fourier_space_maps += [core.grid_points(self.box, gcat_xyz, coeffs[freq],  kernel=self.kernel, fft=True, spin=spin, compensate=True)]
                w += [np.sum(vweights) / self.sum_rcat_vr_weights]
                idx += [i + 1]

        # save fourier maps to perform field level inference later on if needed.
        if save_fourier_maps:
            for i, kbin_edges in enumerate(self.kbin_edges):
                # rebin the fourier space maps:
                to_save = [core.kbin_average(self.box, mp, kbin_edges) for mp in fourier_space_maps]
                io_utils.write_npy(f'{self.output_dir}/fourier_maps_{i}.npy', to_save)

        # Rescale window function (by roughly a factor Ngal/Nrand in each footprint).
        w = np.array(w)
        wf = self.window_function[idx,:][:,idx] * w[:, None] * w[None, :]

        print('WARNING It lacks here the term in (2l+1) for p_gv(ell=1) !!!!! (no problem because it is the same between data and surrogate !!)')

        pks = []
        for i, kbin_edges in enumerate(self.kbin_edges):
            # Estimate power spectra. and normalize by dividing by window function.
            pk = core.estimate_power_spectrum(self.box, fourier_space_maps, kbin_edges)
            pk /= wf[:, :, None]
            # Save 'pk_data.npy' to disk. Note that the file format is specified here:
            # https://kszx.readthedocs.io/en/latest/kszpipe.html#kszpipe-details
            io_utils.write_npy(self.pk_data_filename[i], pk)
            pks += [pk]
        
        if len(self.pk_data_filename): pks = pks[0]
        return pks

    def get_pk_surrogate(self, isurr, run=False, force=False):
        r"""Returns a shape (A,A,nkbins) array, and saves it in ``pipeline_outdir/tmp/pk_surr_{isurr}.npy``,
        where A = #spin_gal * 3 + #spin_vr * #term * #cmb_fields and # term = 2 if no forgeround and 3 if foregrounds are simulated.
        
        The returned array contains auto and cross power spectra of the following fields, for a single surrogate:
          - surrogate galaxy field $S_g$ (contains only the shotnoise), $dS_g/db_1$ (contains delta_field) and $dS_fnl/dfnl$ (contains phi field) with spin in self.spin_gal.
          - derivative $dS_g/df_{NL}$ with spin in self.spin_gal.
          - surrogate kSZ velocity reconstruction $S_v^{freq}$, with $b_v=0$ (i.e. noise only) with spin in self.spin_vr.
          - derivative $dS_v^{freq}/db_v$ with spin in self.spin_vr.
          - derivative $dS_v^{freq}/db_fg$ with spin in self.spin_vr.

        Flags:
          - If ``run=False``, then this function expects the $P(k)$ file to be on disk from a previous pipeline run.
          - If ``run=True``, then the $P(k)$ file will be computed if it is not on disk.
          - If ``force=True``, then this function recomputes $P(k)$, even if it is on disk from a previous pipeline run.
        """

        fname = self.pk_single_surr_filenames[isurr]
        if (not force) and all([os.path.exists(fn) for fn in fname]):
            pks = [io_utils.read_npy(fn) for fn in fname]
            if len(fname) == 1: pks = pks[0]
            return pks

        if not (run or force):
            raise RuntimeError(f'KszPipe.get_pk_surrogate(): run=False was specified, and file {fname} not found')

        import time 
        start = time.time()

        print(f'[{isurr=}] get_pk_surrogate(): start')

        zobs = self.rcat.zobs
        nrand = self.rcat.size
        rweights = getattr(self.rcat, 'weight_gal', np.ones(nrand))
        vweights = getattr(self.rcat, 'weight_vr', np.ones(nrand))

        # The SurrogateFactory simulates LSS fields (delta, phi, vr, rsd) sampled at random catalog locations.
        self.surrogate_factory.simulate_surrogate()
        ngal = self.surrogate_factory.ngal

        # Noise realization for Sg (see overleaf).
        eta_rms = np.sqrt((nrand/ngal) - (self.surr_bg**2 * self.surrogate_factory.sigma2) * self.surrogate_factory.D**2)
        if np.min(eta_rms) < 0:
            raise RuntimeError('Noise RMS went negative! This is probably a symptom of not enough randoms (note {(ngal/nrand)=})')
        eta = np.random.normal(scale=eta_rms)

        print(f'[{isurr=}] simulate_surrogate done', time.time() - start)

        # Each surrogate field is a sum (with coefficients) over the random catalog.
        # First we compute the coefficient arrays.
        # For more info, see the overleaf, or the sphinx docs: https://kszx.readthedocs.io/en/latest/kszpipe.html#kszpipe-details
        Sg_noise = (ngal/nrand) * rweights * eta
        dSg_db1 = (ngal/nrand) * rweights * self.surrogate_factory.delta
        dSg_df = (ngal/nrand) * rweights * self.surrogate_factory.rsd
        dSg_dfnl = (ngal/nrand) * rweights * (2 * self.deltac)  * self.surrogate_factory.phi
        Sv_noise = {f'{freq}': vweights * self.surrogate_factory.M * self.rcat.get_column(f'tcmb_{freq}') for freq in self.cmb_fields}
        Sv_signal = {f'{freq}': (ngal/nrand) * vweights * self.rcat.get_column(f'bv_{freq}') * self.surrogate_factory.vr for freq in self.cmb_fields}
        if self.sim_surr_fg:
            Sv_fg = {f'{freq}': (ngal/nrand) * vweights * self.rcat.get_column(f'bv_{freq}') * self.surrogate_factory.delta for freq in self.cmb_fields}

        # Mean subtraction for the surrogate field Sg.
        # This is intended to make the surrogate field more similar to the galaxy overdensity delta_g
        # which satisfies "integral constraints" since the random z-distribution is inferred from the
        # galaxies. (In practice, the effect seems to be small.)
        if self.nzbins_gal > 0:
            Sg_noise = utils.subtract_binned_means(Sg_noise, zobs, self.nzbins_gal)
            dSg_db1 = utils.subtract_binned_means(dSg_db1, zobs, self.nzbins_gal)
            dSg_df = utils.subtract_binned_means(dSg_df, zobs, self.nzbins_gal)
            dSg_dfnl = utils.subtract_binned_means(dSg_dfnl, zobs, self.nzbins_gal)

        # Mean subtraction for the surrogate fields Sv.
        # This is intended to mitgate foregrounds.( Note that we perform the same
        # mean subtraction to the vr arrays, in get_pk_data()).
        if self.nzbins_vr > 0:
            for freq in self.cmb_fields:
                Sv_noise[freq] = utils.subtract_binned_means(Sv_noise[freq], zobs, self.nzbins_vr)
                Sv_signal[freq] = utils.subtract_binned_means(Sv_signal[freq], zobs, self.nzbins_vr)
                if self.sim_surr_fg:
                    Sv_fg[freq] = utils.subtract_binned_means(Sv_fg[freq], zobs, self.nzbins_vr)

        # (Coefficient arrays) -> (Fourier-space fields).
        fourier_space_maps, idx = [], []
        for spin in self.spin_gal:
            for term in [Sg_noise, dSg_db1, dSg_df, dSg_dfnl]:
                fourier_space_maps += [core.grid_points(self.box, self.rcat_xyz_obs, term, kernel=self.kernel, fft=True, spin=spin, compensate=True)]
                idx += [0]

            print(f'[{isurr=}] Galaxy {spin=} fields done', time.time() - start)
        
        for spin in self.spin_vr:
            for i, freq in enumerate(self.cmb_fields): 
                terms = [Sv_noise, Sv_signal, Sv_fg] if self.sim_surr_fg else [Sv_noise, Sv_signal]
                for term in terms:
                    fourier_space_maps += [core.grid_points(self.box, self.rcat_xyz_obs, term[freq], kernel=self.kernel, fft=True, spin=spin, compensate=True)]
                    idx += [i + 1]

            print(f'[{isurr=}] Velocity {spin=} fields done', time.time() - start)

        # Rescale window function, by a factor (ngal/nrand) in each footprint.
        wf = (ngal/nrand)**2 * self.window_function
        # Expand window function from shape (3,3) to shape (15,15).
        wf = wf[idx,:][:,idx]

        pks = []
        for i, kbin_edges in enumerate(self.kbin_edges):
            # Estimate power spectra. and normalize by dividing by window function.
            pk = core.estimate_power_spectrum(self.box, fourier_space_maps, kbin_edges)
            pk /= wf[:,:,None]
            # save pk:
            io_utils.write_npy(fname[i], pk)
            pks += [pk]

        if len(fname): pks = pks[0]
        return pks

    def get_pk_surrogates(self):
        """Returns the concatenation of all surrogates generated with get_pk_surrogate(), and saves it in ``pipeline_outdir/pk_surrogates.npy``.

        This function only reads files from disk -- it does not run the pipeline.
        To run the pipeline, use :meth:`~kszx.KszPipe.run()`.
        """

        if all([os.path.exists(fn) for fn in self.pk_surr_filename]):
            pks = [io_utils.read_npy(fn) for fn in self.pk_surr_filename]
            if len(self.pk_surr_filename) == 1: pks = pks[0]
            return pks

        if not all([all([os.path.exists(ff) for ff in fn_single_surr]) for fn_single_surr in self.pk_single_surr_filenames]):
            raise RuntimeError(f'KszPipe.read_pk_surrogates(): necessary files do not exist; you need to call KszPipe.run()')

        pks = []
        for i, fn_surr in enumerate(self.pk_surr_filename):
            pk = np.array([io_utils.read_npy(f[i]) for f in self.pk_single_surr_filenames])
            # Save 'pk_surrogates.npy' to disk. Note that the file format is specified here:
            # https://kszx.readthedocs.io/en/latest/kszpipe.html#kszpipe-details
            io_utils.write_npy(fn_surr, pk)
            pks += [pk]
        if len(self.pk_surr_filename) == 1: pks = pks[0]
        return pks

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
        shutil.copyfile(f'{self.input_dir}/params.yml', f'{self.output_dir}/params.yml')

        # Add information into the output_dir/params.yml 
        with open(f'{self.output_dir}/params.yml', 'a') as f:
            print(file=f)
            print('# structure of pk_data.npy', file=f)
            print("data_fields:", file=f)
            idx = 0
            for spin in self.spin_gal:
                print(f"  gal-{spin}: {idx}  # galaxy overdensity delta_g with spin={spin}", file=f)
                idx += 1
            for spin in self.spin_vr:
                for freq in self.cmb_fields:
                    print(f"  {freq}-{spin}: {idx}  # KSZ velocity reconstruction from cmb_field={freq} with spin={spin}", file=f)
                    idx += 1
            print(file=f)
            print('# structure of pk_surr.npy', file=f)
            print("surrogate_fields:", file=f)
            idx = 0
            for spin in self.spin_gal:
                print(f"  gal-{spin}-null: {idx}  # surrogate galaxy field S_g with spin={spin}", file=f)
                idx += 1
                print(f"  gal-{spin}-b1: {idx}  # derivative (dS_g/db1) with spin={spin}", file=f)
                idx += 1
                print(f"  gal-{spin}-f: {idx}  # derivative (dS_g/df) with spin={spin}", file=f)
                idx += 1
                print(f"  gal-{spin}-fnl: {idx}  # derivative (dS_g/dfNL) with spin={spin}", file=f)
                idx += 1
            for spin in self.spin_vr:
                for freq in self.cmb_fields:
                    print(f"  {freq}-{spin}-null: {idx}  # Sv_freq, with bv=0, bfg=0, ... with spin={spin}", file=f)
                    idx += 1
                    print(f"  {freq}-{spin}-bv: {idx}  # derivative (dSv_freq/dbv) with spin={spin}", file=f)
                    idx += 1
                    if self.sim_surr_fg:
                        print(f"  {freq}-{spin}-bfg: {idx}  # derivative (dSv_freq/dbfg) with spin={spin}", file=f)
                        idx += 1

        have_data = all([os.path.exists(fn) for fn in self.pk_data_filename])
        have_surr = all([os.path.exists(fn) for fn in self.pk_surr_filename])
        missing_surrs = [] if have_surr else [i for (i,f) in enumerate(self.pk_single_surr_filenames) if not all([os.path.exists(fn) for fn in f])]

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
            # before going to the surrogate do the data and wait that they finish:
            if not have_data:
                x = pool.apply_async(self.get_pk_data, (True, False))   # (run,force)=(True,False)
                x.get()

            # to avoid the pool launch new surrogate on the same process before the completion of the previous one, we need to make batch and wait that all the job in the batch are done:
            # I'm surprised that there is no function in multiporcess to do this ... 
            nbatchs = len(missing_surrs) // processes
            for nn in range(nbatchs):
                print(f'Start batch {nn}/{nbatchs}')
                l = [pool.apply_async(self.get_pk_surrogate, (i, True)) for i in missing_surrs[nn*processes: (nn+1)*processes]]  # (run,force)=(True,False)
                # .get() block the code until receveing the answer of the pool.apply_async job:
                for x in l: x.get()
                print(f'Batch {nn} done!')

        if not have_surr:
            # Consolidates all surrogates into one file
            self.get_pk_surrogates()


####################################################################################################

class KszPipeOutdir:
    def __init__(self, dirname, nsurr=None, p=1.0, f=None, binning={'gg':0, 'gv':0, 'vv':0}, spin={'g': [0], 'v': [0,1]}):
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

        - ``p`` (float): value used to describe b_phi (p=1 for LRG, 1.4/1.6 for QSO)

        - ``f`` (float or None): the growth rate. If None, it is computed from the cosmology.

        - ``binning`` (dict): the binning scheme to load for the different power spectra. It does not allow different binning for the same type of power spectrum for now.

        - ``spin`` (dict): the spin components to load for galaxies (key 'g') and velocity (key 'v'). Useful to reduce the size of the covariance matrix pre-loaded if you want to ignore some spin components.
        """

        filename = f'{dirname}/params.yml'
        print(f'Reading {filename}')
        
        with open(filename, 'r') as ff:
            params = yaml.safe_load(ff)

        kmin, kmax, kstep = params['kmin'], params['kmax'], params['kstep']
        kbin_edges = [np.arange(kmin[i], kmax[i], kstep[i]) for i in range(len(kmin))]
        kbin_centers = [(kbin_edges[i][1:] + kbin_edges[i][:-1]) / 2 for i in range(len(kmin))]
        nkbins = [kbin_centers[i].size for i in range(len(kmin))]

        data_fields = params['data_fields']
        pk_data = [io_utils.read_npy(f'{dirname}/pk_data_binning{i}.npy') for i in range(len(kbin_edges))]
        for i, pk_d in enumerate(pk_data):
            if pk_d.shape != (len(data_fields), len(data_fields), nkbins[i]):
                raise RuntimeError(f'Got {pk_d.shape=}, expected ({len(data_fields)},{len(data_fields)},nkbins) where {nkbins[i]=}')

        surr_fields = params['surrogate_fields']
        if nsurr is None:
            pk_surr = [io_utils.read_npy(f'{dirname}/pk_surrogates_binning{i}.npy') for i in range(len(kbin_edges))]
            for i, pk_s in enumerate(pk_surr):
                if (pk_s.ndim != 4) or (pk_s.shape[1:] != (len(surr_fields),len(surr_fields),nkbins[i])):
                    raise RuntimeError(f'Got {pk_s.shape=}, expected (nsurr,{len(surr_fields)},{len(surr_fields)},nkbins) where {nkbins[i]=}')
        else:
            pk_surr = []
            print(f'Reading {nsurr} surrogate files from {dirname}/tmp/pk_surr_*_binning*.npy')
            for j in range(len(kbin_edges)):
                pk_surr_tmp = []
                for i in range(nsurr):
                    pk_surrogate = io_utils.read_npy(f'{dirname}/tmp/pk_surr_{i}_binning{j}.npy', verbose=False)
                    if pk_surrogate.shape != (len(surr_fields), len(surr_fields), nkbins[j]):
                        raise RuntimeError(f'Got {pk_surrogate.shape=}, expected ({len(surr_fields)},{len(surr_fields)},nkbins) where {nkbins[j]=}')
                    pk_surr_tmp += [pk_surrogate]
                pk_surr_tmp = np.array(pk_surr_tmp)
                pk_surr_tmp = np.reshape(pk_surr_tmp, (nsurr, len(surr_fields), len(surr_fields), nkbins[j]))   # needed if nsurr==0
                pk_surr += [pk_surr_tmp]
        
        self.binning = binning 
        self.k = {key: kbin_centers[val] for key, val in binning.items()}
        self.nkbins = nkbins
        self.kmin, self.kmax, self.kstep = kmin, kmax, kstep

        self.cmb_fields = params['cmb_fields']
        self.spin_gal = params['estimate_spin_gal']
        self.spin_vr = params['estimate_spin_vr']

        self.data_fields = data_fields
        self.pk_data = pk_data

        self.surr_fields = surr_fields
        self.sim_surr_fg = params['simulate_surrogate_foregrounds']
        self.suff_gal_list = ['null', 'b1', 'f', 'fnl']
        self.suff_vel_list = ['null', 'bv', 'bfg'] if self.sim_surr_fg else ['null', 'bv']
        self.p = p
        if f is not None:
            print(f'We use {f=} as growth rate!')
            self.f = f
        else:
            self.f = self.cosmo.frsd(z=params['zeff'])

        # Restrict to the requested spin components: 
        if spin is not None:
            print(f'Restricting surrogate fields to spin={spin}...')
            to_test = []
            for kk in spin.keys():
                if kk == 'g':
                    to_test += [f'gal-{ss}-{suff}' for ss in spin[kk] for suff in self.suff_gal_list]
                if kk == 'v':
                    for ff in self.cmb_fields:
                        to_test += [f'{ff}-{ss}-{suff}' for ss in spin[kk] for suff in self.suff_vel_list]
            idx_to_keep = [self.surr_fields[tt] for tt in to_test if tt in self.surr_fields.keys()]

            pk_surr = [pk_surr[i][:,idx_to_keep][:,:,idx_to_keep] for i in range(len(self.nkbins))]
            self.surr_fields = {tt: i for i, tt in enumerate(to_test)}

        self.pk_surr = pk_surr
        self.nsurr = self.pk_surr[0].shape[0]
        self.nsurr_fields = len(self.surr_fields)

        self._precompute_mean_and_cov()

        print('KszPipeOutdir initialized.')

    def _precompute_mean_and_cov(self):
        """Precomputes mean and covariance of surrogate power spectra, for speedup purposes."""

        # Precompute for each surrogate field dedicated mean and covariance for speed up purpose.
        print('Computing surrogate means ...')
        self.surr_mean = [np.ascontiguousarray(np.mean(self.pk_surr[i], axis=0)) for i in range(len(self.nkbins))] # make contiguous

        print('Computing surrogate covariances ...')
        cov = {}
        to_compute = np.unique([val for val in self.binning.values()])
        for i in to_compute:
            for j in to_compute:
                if j == i:
                    cov_tmp = np.cov(self.pk_surr[i].reshape(self.nsurr, self.nsurr_fields*self.nsurr_fields*self.nkbins[i]), rowvar=False)
                else:
                    # DO NOT USE np.cov(X,Y) since it will computes the covariance of the concatenated array --> super memory intensive...)
                    cov_tmp = utils.cross_covariance(self.pk_surr[i].reshape(self.nsurr, self.nsurr_fields*self.nsurr_fields*self.nkbins[i]), self.pk_surr[j].reshape(self.nsurr, self.nsurr_fields*self.nsurr_fields*self.nkbins[j]))

                cov_tmp = cov_tmp.reshape((self.nsurr_fields*self.nsurr_fields, self.nkbins[i], self.nsurr_fields*self.nsurr_fields, self.nkbins[j]))
                cov_tmp = cov_tmp.transpose(0, 2, 1, 3)         # reorder axes to have cov matrix in the last two indices for each field.
                # Can be optimized here, we only need the upper / lower triangle part of this matrix...
                cov[f'{i}-{j}'] = np.ascontiguousarray(cov_tmp)  # make contiguous
        self.surr_cov = cov

    @functools.cached_property
    def cosmo(self):
        return Cosmology('planck18+bao')

    def _check_field(self, field):
        """Checks that 'field' is a 1-d array of length 2."""
        field = np.array(field, dtype=float)
        if field.shape != (2,):
            raise RuntimeError(f"Expected 'field' argument to be a 1-d array of length 2, got {field.shape=}")
        return field

    def D_g(self, k, sigmag):
        """Returns the damping factor for delta_g. It will be applied after the convolution with the window function i.e. direclty to the surrogate."""
        return 1 / (1 + (k * sigmag)**2 / 2)

    def D_v(self, k, sigmav):
        """Returns the damping factor for radial velocity. It will be applied after the convolution with the window function i.e direclty to the surrogate."""
        return np.sinc(k*sigmav)

    # def g_data(self, ell=0):
    #     r"""Returns shape ``(nkbins,)`` array containing $P_{gg}^{data}(k)$ for galaxy overdensity with spin ``ell``."""
    #     assert ell in self.spin_gal
    #     idx = self.data_fields[f'gal-{ell}']
    #     return self.pk_data[self.binning['gg']][idx, idx, :]

    def pgg_data(self, ell=[0, 0]):
        r"""Returns shape ``(nkbins,)`` array containing $P_{gg}^{data}(k)$."""
        assert ell[0] in self.spin_gal
        assert ell[1] in self.spin_gal
        idx1 = self.data_fields[f'gal-{ell[0]}']
        idx2 = self.data_fields[f'gal-{ell[1]}']
        return self.pk_data[self.binning['gg']][idx1, idx2, :]
        
    def pgv_data(self, freq=['90','150'], field=[1,0], ell=[0, 1]):
        r"""Returns shape ``(nkbins,)`` array containing $P_{gv}^{ell=1}^{data}(k)$.

        The ``field`` argument is a length-2 vector, selecting a linear combination
        of the 90+150 GHz velocity reconstructions. For example:
        
           - ``field=[1,0]`` for 90 GHz reconstruction
           - ``field=[0,1]`` for 150 GHz reconstruction
           - ``field=[0.5,0.5]`` for mean (90+150) GHz reconstruction **not optimal**.
           - ``field=[1,-1]`` for null (90-150) GHz reconstruction.
        """
        field = self._check_field(field)
        assert ell[0] in self.spin_gal
        assert ell[1] in self.spin_vr

        idx_gal = self.data_fields[f'gal-{ell[0]}']
        idx_vr_1, idx_vr_2 = self.data_fields[f'{freq[0]}-{ell[1]}'], self.data_fields[f'{freq[1]}-{ell[1]}']

        return field[0]*self.pk_data[self.binning['gv']][idx_gal, idx_vr_1] + field[1]*self.pk_data[self.binning['gv']][idx_gal, idx_vr_2]

    def pvv_data(self, freq=[['90','150'], ['90','150']], field=[[1,0], [1,0]], ell=[1,1]):
        r"""Returns shape ``(nkbins,)`` array containing $P_{vv}^{data}(k)$.

        The ``field`` argument is a length-2 vector, selecting a linear combination
        of the 90+150 GHz velocity reconstructions. For example:
        
           - ``field=[1,0]`` for 90 GHz reconstruction
           - ``field=[0,1]`` for 150 GHz reconstruction
           - ``field=[0.5,0.5]`` for mean (90+150) GHz reconstruction **not optimal**.
           - ``field=[1,-1]`` for null (90-150) GHz reconstruction.
        """
        assert ell[0] in self.spin_vr
        assert ell[1] in self.spin_vr

        idx_vr_11, idx_vr_12 = self.data_fields[f'{freq[0][0]}-{ell[0]}'], self.data_fields[f'{freq[0][1]}-{ell[0]}']
        idx_vr_21, idx_vr_22 = self.data_fields[f'{freq[1][0]}-{ell[1]}'], self.data_fields[f'{freq[1][1]}-{ell[1]}']

        t = field[0][0]*self.pk_data[self.binning['vv']][idx_vr_11,:] + field[0][1]*self.pk_data[self.binning['vv']][idx_vr_12,:]
        return field[1][0]*t[idx_vr_21] + field[1][1]*t[idx_vr_22]

    def pgg_mean(self, b1=1, fnl=0, sn=1, sigmag=0, ell=[0,0]):
        r"""Returns shape ``(nkbins,)`` array, containing $\langle P_{gg}^{surr}(k) \rangle$."""
        assert ell[0] in self.spin_gal
        assert ell[1] in self.spin_gal

        params_gal = np.array([sn, b1, self.f, fnl*(b1 - self.p)])
        # Add RSD damping factor term (Warning: shotnoise is not damped !!) --> params_gal will be (4, nkbins)
        params_gal = params_gal[:,None] * [np.ones_like(self.k['gg']), self.D_g(self.k['gg'], sigmag), self.D_g(self.k['gg'], sigmag), self.D_g(self.k['gg'], sigmag)]
        idx_gal_1 = [self.surr_fields[f'gal-{ell[0]}-{suff}'] for suff in self.suff_gal_list]
        idx_gal_2 = [self.surr_fields[f'gal-{ell[1]}-{suff}'] for suff in self.suff_gal_list]

        pgg = self.surr_mean[self.binning['gg']][idx_gal_1, :, :]       # shape (2, #fields, nkbins)
        pgg = np.sum(params_gal[:, None, :] * pgg, axis=0)  # shape (nsurr,  #fields, nkbins)
        return np.sum(params_gal[:, :] * pgg[idx_gal_2, :], axis=0)     # shape (nsurr, nkbins)

    def pgv_mean(self, b1=1, fnl=0, sn=1, bv=1, snv=1, bfg=0, sigmag=0, sigmav=0, freq=['90','150'], field=[1,0], ell=[0, 1]):
        r"""Returns shape ``(nkbins,)`` array containing $\langle P_{gv}^{surr}(k) \rangle$.

        The ``field`` argument is a length-2 vector, selecting a linear combination
        of the 90+150 GHz velocity reconstructions. For example:
        
           - ``field=[1,0]`` for 90 GHz reconstruction
           - ``field=[0,1]`` for 150 GHz reconstruction
           - ``field=[0.5,0.5]`` for mean (90+150) GHz reconstruction **not optimal**.
           - ``field=[1,-1]`` for null (90-150) GHz reconstruction.
        """
        field = self._check_field(field)
        assert ell[0] in self.spin_gal
        assert ell[1] in self.spin_vr

        params_gal = np.array([sn, b1, self.f, fnl*(b1 - self.p)])
        # Add RSD damping factor term: (Warning: only b1 / fnl terms are damped, not the shotnoise term) --> params_gal will be (4, nkbins)
        params_gal = params_gal[:,None] * [np.ones_like(self.k['gv']), self.D_g(self.k['gv'], sigmag), self.D_g(self.k['gv'], sigmag), self.D_g(self.k['gv'], sigmag)]
        idx_gal = [self.surr_fields[f'gal-{ell[0]}-{suff}'] for suff in self.suff_gal_list]

        if self.sim_surr_fg:
            params_vel = np.array([snv, bv, bfg])[:,None] * [np.ones_like(self.k['gv']), self.D_v(self.k['gv'], sigmav), np.ones_like(self.k['gv'])]  # Add RSD damping factor term --> shape (3, nkbins)
        else:
            params_vel = np.array([snv, bv])[:,None] * [np.ones_like(self.k['gv']), self.D_v(self.k['gv'], sigmav)]  # Add RSD damping factor term --> shape (2, nkbins)
        idx_vr_1 = [self.surr_fields[f'{freq[0]}-{ell[1]}-{suff}'] for suff in self.suff_vel_list]
        idx_vr_2 = [self.surr_fields[f'{freq[1]}-{ell[1]}-{suff}'] for suff in self.suff_vel_list]

        pgv = self.surr_mean[self.binning['gv']][idx_gal,:,:] # shape (2, #fields, nkbins)
        # this is the g term:
        pgv = np.sum(params_gal[:, None, :] * pgv, axis=0) # shape (#fields, nkbins)
        # combine or not the different frequency fields:
        pgv = field[0] * pgv[idx_vr_1,:] + field[1] * pgv[idx_vr_2,:]      # shape (2 or 3, nkbins)
        return np.sum(params_vel[:, :] * pgv, axis=0) # shape (nkbins)

    def pvv_mean(self, bv=1, snv=1, bfg=0, sigmav=0, freq=[['90','150'], ['90','150']], field=[[1,0], [1,0]], ell=[1,1]):
        r"""Returns shape ``(nkbins,)`` array containing $\langle P_{vv}^{data}(k) \rangle$.

        The ``field`` argument is a length-2 vector, selecting a linear combination
        of the 90+150 GHz velocity reconstructions. For example:
        
           - ``field=[1,0]`` for 90 GHz reconstruction
           - ``field=[0,1]`` for 150 GHz reconstruction
           - ``field=[0.5,0.5]`` for mean (90+150) GHz reconstruction **not optimal**.
           - ``field=[1,-1]`` for null (90-150) GHz reconstruction.
        """
        assert ell[0] in self.spin_vr
        assert ell[1] in self.spin_vr

        if self.sim_surr_fg:
            params_vel = np.array([snv, bv, bfg])[:,None] * [np.ones_like(self.k['vv']), self.D_v(self.k['vv'], sigmav), np.ones_like(self.k['vv'])]  # Add RSD damping factor term --> shape (3, nkbins)
        else:
            params_vel = np.array([snv, bv])[:,None] * [np.ones_like(self.k['vv']), self.D_v(self.k['vv'], sigmav)]  # Add RSD damping factor term --> shape (2, nkbins)

        idx_vr_11 = [self.surr_fields[f'{freq[0][0]}-{ell[0]}-{suff}'] for suff in self.suff_vel_list]
        idx_vr_12 = [self.surr_fields[f'{freq[0][1]}-{ell[0]}-{suff}'] for suff in self.suff_vel_list]
        idx_vr_21 = [self.surr_fields[f'{freq[1][0]}-{ell[1]}-{suff}'] for suff in self.suff_vel_list]
        idx_vr_22 = [self.surr_fields[f'{freq[1][1]}-{ell[1]}-{suff}'] for suff in self.suff_vel_list]

        pvv = field[0][0] * self.surr_mean[self.binning['vv']][idx_vr_11,:,:] + field[0][1] * self.surr_mean[self.binning['vv']][idx_vr_12,:,:] # shape (#params_vel, #fields, nkbins)
        pvv = np.sum(params_vel[:, None, :] * pvv, axis=0) # shape (#fields, nkbins)
        pvv = field[1][0] * pvv[idx_vr_21,:] + field[1][1] * pvv[idx_vr_22,:]      # shape (#params_vel, nkbins)
        return np.sum(params_vel[:, :] * pvv, axis=0) # shape (nkbins)

    def pggxpgg_cov(self, b11=1, fnl1=0, sn1=1, sigmag1=0, b12=1, fnl2=0, sn2=1, sigmag2=0,
                    ell1=[0, 0], ell2=[0, 0]):
        r"""Returns shape ``(nkbins, nkbins)`` covariance matrix of $P_{gg}^{surr}(k) x P_{gg}^{surr}(k)$."""
        idx_gal_11 = [self.surr_fields[f'gal-{ell1[0]}-{suff}'] for suff in self.suff_gal_list]
        idx_gal_12 = [self.surr_fields[f'gal-{ell1[1]}-{suff}'] for suff in self.suff_gal_list]
        idx_gal_21 = [self.surr_fields[f'gal-{ell2[0]}-{suff}'] for suff in self.suff_gal_list]
        idx_gal_22 = [self.surr_fields[f'gal-{ell2[1]}-{suff}'] for suff in self.suff_gal_list]

        ## TODO: write a function to compute these coeffs to avoid code duplication in pggxpgg, pggxpgv, pgvxpgv, pgvxpvv, pvvxpvv!!!
        coeffs1 = np.array([sn1, b11, self.f, fnl1*(b11 - self.p)])
        coeffs1 = np.ravel(coeffs1[:,None]*coeffs1[None,:]) # shape (6, ) ie (len(idx_gal_11)*len(idx_gal_12), )
        # Add RSD damping factor term: (Warning: only b1 / fnl terms are damped, not the shotnoise term) --> shape (6, nkbins)
        rsd1 = np.array([np.ones_like(self.k['gg']), self.D_g(self.k['gg'], sigmag1), self.D_g(self.k['gg'], sigmag1), self.D_g(self.k['gg'], sigmag1)])
        rsd1 = rsd1[:,None] * rsd1[None,:]
        rsd1 = rsd1.reshape(rsd1.shape[0]*rsd1.shape[1], rsd1.shape[2])
        coeffs1 = coeffs1[:,None] * rsd1 # shape (6, nkbins)

        coeffs2 = np.array([1, b12, self.f, fnl2*(b12 - self.p)])
        coeffs2 = np.ravel(coeffs2[:,None]*coeffs2[None,:])
        # Add RSD damping factor term:
        rsd2 = np.array([np.ones_like(self.k['gg']), self.D_g(self.k['gg'], sigmag2), self.D_g(self.k['gg'], sigmag2), self.D_g(self.k['gg'], sigmag2)])
        rsd2 = rsd2[:,None] * rsd2[None,:]
        rsd2 = rsd2.reshape(rsd2.shape[0]*rsd2.shape[1], rsd2.shape[2])
        coeffs2 = coeffs2[:,None] * rsd2 # shape (6, nkbins)

        # array with all the coefficiens !
        # shape (len(idx_gal_21)*len(idx_gal_22)**2, nkbins, nkbins)
        coeff_cov = (coeffs1[:,None,:,None] * coeffs2[None,:,None,:]).reshape(coeffs1.shape[0]*coeffs2.shape[0], coeffs1.shape[1], coeffs2.shape[1])

        # keep only the indices that are needed for the covariance matrix:
        # shape (len(idx_gal_11)*len(idx_gal_12), )
        idx1 = np.array([i*self.nsurr_fields + j for i in idx_gal_11 for j in idx_gal_12])  
        idx2 = np.array([ii*self.nsurr_fields + jj for ii in idx_gal_21 for jj in idx_gal_22]) 

        surr_cov = self.surr_cov[f"{self.binning['gg']}-{self.binning['gg']}"]  # select the covariance for the right binning
        cov = coeff_cov * surr_cov[idx1][:, idx2, :, :].reshape(len(idx1)*len(idx2), self.nkbins[self.binning['gg']], self.nkbins[self.binning['gg']])  

        return np.sum(cov, axis=0)  # shape (nkbins, nkbins)

    def pgvxpgv_cov(self, b11=1, fnl1=0, sn1=1, bv1=1, snv1=1, bfg1=0, b12=1, sigmag1=0, sigmav1=0, fnl2=0, sn2=1, bv2=1, snv2=1, bfg2=0, sigmag2=0, sigmav2=0,
                    freq1=['90','150'], field1=[1,0], ell1=[0, 1],
                    freq2=['90','150'], field2=[1,0], ell2=[0, 1]):
        r"""Returns shape ``(nkbins, nkbins)`` covariance matrix of $P_{gv}^{surr}(k) x P_{gv}^{surr}(k)$.
        The ``field`` argument is a length-2 vector, selecting a linear combination
        of the 90+150 GHz velocity reconstructions. For example:
           - ``field=[1,0]`` for 90 GHz reconstruction
           - ``field=[0,1]`` for 150 GHz reconstruction
           - ``field=[0.5,0.5]`` for mean (90+150) GHz reconstruction **not optimal**.
           - ``field=[1,-1]`` for null (90-150) GHz reconstruction.
        """
        field1, field2 = self._check_field(field1), self._check_field(field2)

        idx_gal1 = [self.surr_fields[f'gal-{ell1[0]}-{suff}'] for suff in self.suff_gal_list]
        idx_gal2 = [self.surr_fields[f'gal-{ell2[0]}-{suff}'] for suff in self.suff_gal_list]

        idx_vr1_1 = [self.surr_fields[f'{freq1[0]}-{ell1[1]}-{suff}'] for suff in self.suff_vel_list]
        idx_vr1_2 = [self.surr_fields[f'{freq1[1]}-{ell1[1]}-{suff}'] for suff in self.suff_vel_list]
        idx_vr2_1 = [self.surr_fields[f'{freq2[0]}-{ell2[1]}-{suff}'] for suff in self.suff_vel_list]
        idx_vr2_2 = [self.surr_fields[f'{freq2[1]}-{ell2[1]}-{suff}'] for suff in self.suff_vel_list]       

        coeffs1 = np.ravel(np.array([sn1, b11, self.f, fnl1*(b11 - self.p)])[:,None] * (np.array([snv1, bv1, bfg1]) if self.sim_surr_fg else np.array([snv2, bv1]))[None,:])
        rsd11 = np.array([np.ones_like(self.k['gv']), self.D_g(self.k['gv'], sigmag1), self.D_g(self.k['gv'], sigmag1), self.D_g(self.k['gv'], sigmag1)])
        rsd12 = np.array([np.ones_like(self.k['gv']), self.D_v(self.k['gv'], sigmav1), np.ones_like(self.k['gv'])]) if self.sim_surr_fg else np.array([np.ones_like(self.k['gv']), self.D_v(self.k['gv'], sigmav1)])
        rsd1 = rsd11[:,None] * rsd12[None,:]
        rsd1 = rsd1.reshape(rsd1.shape[0]*rsd1.shape[1], rsd1.shape[2])
        coeffs1 = coeffs1[:,None] * rsd1 

        coeffs2 = np.ravel(np.array([sn2, b12, self.f, fnl2*(b12 - self.p)])[:,None] * (np.array([snv2, bv2, bfg2]) if self.sim_surr_fg else np.array([snv2, bv2]))[None,:])
        rsd21 = np.array([np.ones_like(self.k['gv']), self.D_g(self.k['gv'], sigmag2), self.D_g(self.k['gv'], sigmag2), self.D_g(self.k['gv'], sigmag2)])
        rsd22 = np.array([np.ones_like(self.k['gv']), self.D_v(self.k['gv'], sigmav2), np.ones_like(self.k['gv'])]) if self.sim_surr_fg else np.array([np.ones_like(self.k['gv']), self.D_v(self.k['gv'], sigmav2)])
        rsd2 = rsd21[:,None] * rsd22[None,:]
        rsd2 = rsd2.reshape(rsd2.shape[0]*rsd2.shape[1], rsd2.shape[2])
        coeffs2 = coeffs2[:,None] * rsd2 

        # shape (..., nkbins, nkbins)
        coeff_cov = (coeffs1[:,None,:,None] * coeffs2[None,:,None,:]).reshape(coeffs1.shape[0]*coeffs2.shape[0], coeffs1.shape[1], coeffs2.shape[1])

        cov = []
        # for speed up reason, we do not want to compute covariance matrices that are not needed (for instance if fields=[1,0]) 
        for i, idx_vr1 in enumerate([idx_vr1_1, idx_vr1_2]):
            for j, idx_vr2 in enumerate([idx_vr2_1, idx_vr2_2]):
                if field1[i] == 0 or field2[j] == 0:
                    cov_tmp = np.zeros((self.nkbins[self.binning['gv']], self.nkbins[self.binning['gv']]))  # shape (nkbins, nkbins)
                else:
                    idx1 = np.array([ii*self.nsurr_fields + jj for ii in idx_gal1 for jj in idx_vr1])  # shape (len(idx_gal_1)*len(idx_vr1), ) 
                    idx2 = np.array([ii*self.nsurr_fields + jj for ii in idx_gal2 for jj in idx_vr2])  # shape (len(idx_gal_2)*len(idx_vr2), )
                    surr_cov = self.surr_cov[f"{self.binning['gv']}-{self.binning['gv']}"] # select the covariance for the right binning
                    cov_tmp = np.sum(coeff_cov * surr_cov[idx1][:, idx2, :, :].reshape(len(idx1)*len(idx2), self.nkbins[self.binning['gv']], self.nkbins[self.binning['gv']]), axis=0)  # shape (nkbins, nkbins)
                cov += [cov_tmp]

        field = np.ravel(field1[:, None] * field2[None, :])
        return np.sum(field[:, None, None] * cov, axis=0)  # shape (nkbins, nkbins)

    def pvvxpvv_cov(self, bv1=1, snv1=1, bfg1=0, sigmav1=0, bv2=1, snv2=1, bfg2=0, sigmav2=0,
                    freq1=[['90','150'], ['90','150']], field1=[[1,0], [1,0]], ell1=[1,1], 
                    freq2=[['90','150'], ['90','150']], field2=[[1,0], [1,0]], ell2=[1,1]):
        r"""Returns shape ``(nkbins, nkbins)`` covariance matrix of $P_{vv}^{surr}(k)$."""

        idx_vr1_1_1 = [self.surr_fields[f'{freq1[0][0]}-{ell1[0]}-{suff}'] for suff in self.suff_vel_list]
        idx_vr1_1_2 = [self.surr_fields[f'{freq1[0][1]}-{ell1[0]}-{suff}'] for suff in self.suff_vel_list]
        idx_vr1_2_1 = [self.surr_fields[f'{freq1[1][0]}-{ell1[1]}-{suff}'] for suff in self.suff_vel_list]
        idx_vr1_2_2 = [self.surr_fields[f'{freq1[1][1]}-{ell1[1]}-{suff}'] for suff in self.suff_vel_list]    

        idx_vr2_1_1 = [self.surr_fields[f'{freq2[0][0]}-{ell2[0]}-{suff}'] for suff in self.suff_vel_list]
        idx_vr2_1_2 = [self.surr_fields[f'{freq2[0][1]}-{ell2[0]}-{suff}'] for suff in self.suff_vel_list]
        idx_vr2_2_1 = [self.surr_fields[f'{freq2[1][0]}-{ell2[1]}-{suff}'] for suff in self.suff_vel_list]
        idx_vr2_2_2 = [self.surr_fields[f'{freq2[1][1]}-{ell2[1]}-{suff}'] for suff in self.suff_vel_list]       

        coeffs1 = np.array([snv1, bv1, bfg1]) if self.sim_surr_fg else np.array([snv1, bv1])
        coeffs1 = np.ravel(coeffs1[:,None] * coeffs1[None,:])
        # Add RSD damping factor term:
        rsd1 = np.array([np.ones_like(self.k['vv']), self.D_v(self.k['vv'], sigmav1), np.ones_like(self.k['vv'])]) if self.sim_surr_fg else np.array([np.ones_like(self.k['vv']), self.D_v(self.k['vv'], sigmav1)])
        rsd1 = rsd1[:,None] * rsd1[None,:]
        rsd1 = rsd1.reshape(rsd1.shape[0]*rsd1.shape[1], rsd1.shape[2])
        coeffs1 = coeffs1[:,None] * rsd1

        coeffs2 = np.array([snv2, bv2, bfg2]) if self.sim_surr_fg else np.array([snv2, bv2])
        coeffs2 = coeffs2 = np.ravel(coeffs2[:,None]*coeffs2[None,:])
        # Add RSD damping factor term:
        rsd2 = np.array([np.ones_like(self.k['vv']), self.D_v(self.k['vv'], sigmav2), np.ones_like(self.k['vv'])]) if self.sim_surr_fg else np.array([np.ones_like(self.k['vv']), self.D_v(self.k['vv'], sigmav2)])
        rsd2 = rsd2[:,None] * rsd2[None,:]
        rsd2 = rsd2.reshape(rsd2.shape[0]*rsd2.shape[1], rsd2.shape[2])
        coeffs2 = coeffs2[:,None] * rsd2

        # shape (..., nkbins, nkbins)
        coeff_cov = (coeffs1[:,None,:,None] * coeffs2[None,:,None,:]).reshape(coeffs1.shape[0]*coeffs2.shape[0], coeffs1.shape[1], coeffs2.shape[1])

        cov = []  # it will be a 16 x nkbins x nkbins array to cover any possible combination of fields1 / fields2.
        # for speed up reason, we do not want to compute covariance matrices that are not needed (for instance if fields=[1,0]) 
        for i, idx_vr1_1 in enumerate([idx_vr1_1_1, idx_vr1_1_2]):
            for j, idx_vr1_2 in enumerate([idx_vr1_2_1, idx_vr1_2_2]):
                for k, idx_vr2_1 in enumerate([idx_vr2_1_1, idx_vr2_1_2]):
                    for l, idx_vr2_2 in enumerate([idx_vr2_2_1, idx_vr2_2_2]):
                        if field1[0][i] == 0 or field1[1][j] == 0 or field2[0][k] == 0 or field2[1][l] == 0:
                            cov_tmp = np.zeros((self.nkbins[self.binning['vv']], self.nkbins[self.binning['vv']]))  # shape (nkbins, nkbins)
                        else:
                            idx1 = np.array([ii*self.nsurr_fields + jj for ii in idx_vr1_1 for jj in idx_vr2_1])  # shape (len(idx_gal_1)*len(idx_vr1), ) 
                            idx2 = np.array([ii*self.nsurr_fields + jj for ii in idx_vr1_2 for jj in idx_vr2_2])  # shape (len(idx_gal_2)*len(idx_vr2), )
                            surr_cov = self.surr_cov[f"{self.binning['vv']}-{self.binning['vv']}"] # select the covariance for the right binning
                            cov_tmp = np.sum(coeff_cov * surr_cov[idx1][:, idx2, :, :].reshape(len(idx1)*len(idx2), self.nkbins[self.binning['vv']], self.nkbins[self.binning['vv']]), axis=0)  # shape (nkbins, nkbins)
                        cov += [cov_tmp]

        field = np.ravel(np.ravel(np.array(field1[0])[:, None] * np.array(field1[1])[None, :])[:, None] * np.ravel(np.array(field2[0])[:, None] * np.array(field2[1])[None, :])[None, :])
        return np.sum(field[:, None, None] * cov, axis=0)  # shape (nkbins, nkbins)

    def pggxpgv_cov(self, b11=1, fnl1=0, sn1=1, sigmag1=0, b12=1, fnl2=0, sn2=1, bv2=1, snv2=1, bfg2=0, sigmag2=0, sigmav2=0,
                    ell1=[0, 0], 
                    freq2=['90','150'], field2=[1,0], ell2=[0, 1]):
        r"""Returns shape ``(nkbins, nkbins)`` cross-covariance matrix of $P_{gg}^{surr}(k) \times P_{gv}^{surr}(k)$."""
        
        field2 = self._check_field(field2)

        idx_gal1 = [self.surr_fields[f'gal-{ell1[0]}-{suff}'] for suff in self.suff_gal_list]
        idx_gal2 = [self.surr_fields[f'gal-{ell1[1]}-{suff}'] for suff in self.suff_gal_list]
        idx_gal3 = [self.surr_fields[f'gal-{ell2[0]}-{suff}'] for suff in self.suff_gal_list]

        idx_vr1_1 = [self.surr_fields[f'{freq2[0]}-{ell2[1]}-{suff}'] for suff in self.suff_vel_list]
        idx_vr1_2 = [self.surr_fields[f'{freq2[1]}-{ell2[1]}-{suff}'] for suff in self.suff_vel_list]

        coeffs1 = np.array([sn1, b11, self.f, fnl1*(b11 - self.p)])
        coeffs1 = np.ravel(coeffs1[:,None]*coeffs1[None,:]) # shape (6, ) ie (len(idx_gal_11)*len(idx_gal_12), )
        # Add RSD damping factor term: (Warning: only b1 / fnl terms are damped, not the shotnoise term) --> shape (6, nkbins)
        rsd1 = np.array([np.ones_like(self.k['gg']), self.D_g(self.k['gg'], sigmag1), self.D_g(self.k['gg'], sigmag1), self.D_g(self.k['gg'], sigmag1)])
        rsd1 = rsd1[:,None] * rsd1[None,:]
        rsd1 = rsd1.reshape(rsd1.shape[0]*rsd1.shape[1], rsd1.shape[2])
        coeffs1 = coeffs1[:,None] * rsd1 # shape (6, nkbins)

        coeffs2 = np.ravel(np.array([sn2, b12, self.f, fnl2*(b12 - self.p)])[:,None] * (np.array([snv2, bv2, bfg2]) if self.sim_surr_fg else np.array([snv2, bv2]))[None,:])
        rsd21 = np.array([np.ones_like(self.k['gv']), self.D_g(self.k['gv'], sigmag2), self.D_g(self.k['gv'], sigmag2), self.D_g(self.k['gv'], sigmag2)])
        rsd22 = np.array([np.ones_like(self.k['gv']), self.D_v(self.k['gv'], sigmav2), np.ones_like(self.k['gv'])]) if self.sim_surr_fg else np.array([np.ones_like(self.k['gv']), self.D_v(self.k['gv'], sigmav2)])
        rsd2 = rsd21[:,None] * rsd22[None,:]
        rsd2 = rsd2.reshape(rsd2.shape[0]*rsd2.shape[1], rsd2.shape[2])
        coeffs2 = coeffs2[:,None] * rsd2 

        # shape (..., nkbins, nkbins)
        coeff_cov = (coeffs1[:,None,:,None] * coeffs2[None,:,None,:]).reshape(coeffs1.shape[0]*coeffs2.shape[0], coeffs1.shape[1], coeffs2.shape[1])
        
        cov = []
        # for speed up reason, we do not want to compute covariance matrices that are not needed (for instance if fields=[1,0]) 
        for i, idx_vr in enumerate([idx_vr1_1, idx_vr1_2]):
            if field2[i] == 0:
                cov_tmp = np.zeros((self.nkbins[self.binning['gg']], self.nkbins[self.binning['gv']]))  # shape (nkbins, nkbins)
            else:
                idx1 = np.array([ii*self.nsurr_fields + jj for ii in idx_gal1 for jj in idx_gal2]) 
                idx2 = np.array([ii*self.nsurr_fields + jj for ii in idx_gal3 for jj in idx_vr]) 
                surr_cov = self.surr_cov[f"{self.binning['gg']}-{self.binning['gv']}"] # select the covariance for the right binning
                cov_tmp = np.sum(coeff_cov * surr_cov[idx1][:, idx2, :, :].reshape(len(idx1)*len(idx2), self.nkbins[self.binning['gg']], self.nkbins[self.binning['gv']]), axis=0)  # shape (nkbins, nkbins)
            cov += [cov_tmp]

        return np.sum(field2[:, None, None] * cov, axis=0)  # shape (nkbins, nkbins)

    def pgvxpgg_cov(self, b11=1, fnl1=0, bv1=1, sn1=1, snv1=1, bfg1=0, sigmag1=0, sigmav1=0, b12=1, fnl2=0, sn2=1, sigmag2=0,
                    freq1=['90','150'], field1=[1,0], ell1=[0, 0], 
                    ell2=[0, 1]):
        r"""Returns shape ``(nkbins, nkbins)`` cross-covariance matrix of $P_{gv}^{surr}(k) \times P_{gg}^{surr}(k)$."""
        
        return self.pggxpgv_cov(b11=b12, fnl1=fnl2, sn1=sn2, sigmag1=sigmag2, b12=b11, fnl2=fnl1, bv2=bv1, sn2=sn1, snv2=snv1, bfg2=bfg1, sigmag2=sigmag1, sigmav2=sigmav1, ell1=ell2, freq2=freq1, field2=field1, ell2=ell1).T

    def pgvxpvv_cov(self, b11=1, fnl1=0, sn1=1, bv1=1, snv1=1, bfg1=0, sigmag1=0, sigmav1=0, bv2=1, snv2=1, bfg2=0, sigmav2=0,
                    freq1=['90','150'], field1=[1,0], ell1=[0,1], 
                    freq2=[['90','150'], ['90','150']], field2=[[1,0], [1,0]], ell2=[1,1]):
        r"""Returns shape ``(nkbins, nkbins)`` cross-covariance matrix of $P_{gv}^{surr}(k) \times P_{vv}^{surr}(k)$."""
        idx_gal1 = [self.surr_fields[f'gal-{ell1[0]}-{suff}'] for suff in self.suff_gal_list]

        idx_vr1_1 = [self.surr_fields[f'{freq1[0]}-{ell1[1]}-{suff}'] for suff in self.suff_vel_list]
        idx_vr1_2 = [self.surr_fields[f'{freq1[1]}-{ell1[1]}-{suff}'] for suff in self.suff_vel_list]
        idx_vr2_1_1 = [self.surr_fields[f'{freq2[0][0]}-{ell2[0]}-{suff}'] for suff in self.suff_vel_list]
        idx_vr2_1_2 = [self.surr_fields[f'{freq2[0][1]}-{ell2[0]}-{suff}'] for suff in self.suff_vel_list]
        idx_vr2_2_1 = [self.surr_fields[f'{freq2[1][0]}-{ell2[1]}-{suff}'] for suff in self.suff_vel_list]
        idx_vr2_2_2 = [self.surr_fields[f'{freq2[1][1]}-{ell2[1]}-{suff}'] for suff in self.suff_vel_list]       

        coeffs1 = np.ravel(np.array([sn1, b11, self.f, fnl1*(b11 - self.p)])[:,None] * (np.array([snv1, bv1, bfg1]) if self.sim_surr_fg else np.array([snv2, bv1]))[None,:])
        rsd11 = np.array([np.ones_like(self.k['gv']), self.D_g(self.k['gv'], sigmag1), self.D_g(self.k['gv'], sigmag1), self.D_g(self.k['gv'], sigmag1)])
        rsd12 = np.array([np.ones_like(self.k['gv']), self.D_v(self.k['gv'], sigmav1), np.ones_like(self.k['gv'])]) if self.sim_surr_fg else np.array([np.ones_like(self.k['gv']), self.D_v(self.k['gv'], sigmav1)])
        rsd1 = rsd11[:,None] * rsd12[None,:]
        rsd1 = rsd1.reshape(rsd1.shape[0]*rsd1.shape[1], rsd1.shape[2])
        coeffs1 = coeffs1[:,None] * rsd1 

        coeffs2 = np.array([snv2, bv2, bfg2]) if self.sim_surr_fg else np.array([snv2, bv2])
        coeffs2 = coeffs2 = np.ravel(coeffs2[:,None]*coeffs2[None,:])
        # Add RSD damping factor term:
        rsd2 = np.array([np.ones_like(self.k['vv']), self.D_v(self.k['vv'], sigmav2), np.ones_like(self.k['vv'])]) if self.sim_surr_fg else np.array([np.ones_like(self.k['vv']), self.D_v(self.k['vv'], sigmav2)])
        rsd2 = rsd2[:,None] * rsd2[None,:]
        rsd2 = rsd2.reshape(rsd2.shape[0]*rsd2.shape[1], rsd2.shape[2])
        coeffs2 = coeffs2[:,None] * rsd2

        # shape (..., nkbins, nkbins)
        coeff_cov = (coeffs1[:,None,:,None] * coeffs2[None,:,None,:]).reshape(coeffs1.shape[0]*coeffs2.shape[0], coeffs1.shape[1], coeffs2.shape[1])

        cov = []
        # for speed up reason, we do not want to compute covariance matrices that are not needed (for instance if fields=[1,0]) 
        for i, idx_vr1 in enumerate([idx_vr1_1, idx_vr1_2]):
            for j, idx_vr2_1 in enumerate([idx_vr2_1_1, idx_vr2_1_2]):
                for k, idx_vr2_2 in enumerate([idx_vr2_2_1, idx_vr2_2_2]):
                    if field1[i] == 0 or field2[0][j] == 0 or field2[1][k] == 0:
                        cov_tmp = np.zeros((self.nkbins[self.binning['gv']], self.nkbins[self.binning['vv']]))  # shape (nkbins, nkbins)
                    else:
                        idx1 = np.array([ii*self.nsurr_fields + jj for ii in idx_gal1 for jj in idx_vr1]) 
                        idx2 = np.array([ii*self.nsurr_fields + jj for ii in idx_vr2_1 for jj in idx_vr2_2]) 
                        surr_cov = self.surr_cov[f"{self.binning['gv']}-{self.binning['vv']}"] # select the covariance for the right binning
                        cov_tmp = np.sum(coeff_cov * surr_cov[idx1][:, idx2, :, :].reshape(len(idx1)*len(idx2), self.nkbins[self.binning['gv']], self.nkbins[self.binning['vv']]), axis=0)  # shape (nkbins, nkbins)
                    cov += [cov_tmp]

        field = np.ravel(np.array(field1)[:, None] * np.ravel(np.array(field2[0])[:, None] * np.array(field2[1])[None, :])[None, :])
        return np.sum(field[:, None, None] * cov, axis=0)  # shape (nkbins, nkbins)

    def pvvxpgv_cov(self, bv1=1, snv1=1, bfg1=0, sigmav1=0, b12=1, fnl2=0, sn2=1, bv2=1, snv2=1, bfg2=0, sigmag2=0, sigmav2=0,
                    freq1=[['90','150'], ['90','150']], field1=[[1,0], [1,0]], ell1=[1,1],
                    freq2=['90','150'], field2=[1,0], ell2=[0,1]):
        r"""Returns shape ``(nkbins, nkbins)`` cross-covariance matrix of $P_{vv}^{surr}(k) \times P_{gv}^{surr}(k)$."""

        return self.pgvxpvv_cov(b11=b12, fnl1=fnl2, sn1=sn2, bv1=bv2, snv1=snv2, bfg1=bfg2, sigmag1=sigmag2, sigmav1=sigmav2, bv2=bv1, snv2=snv1, bfg2=bfg1, sigmav2=sigmav1, freq1=freq2, field1=field2, ell1=ell2, freq2=freq1, field2=field1, ell2=ell1).T

    def pggxpvv_cov(self, b11=1, fnl1=0, sn1=1, sigmag1=0, bv2=1, snv2=1, bfg2=0, sigmav2=0,
                    ell1=[0,0], 
                    ell2=[1,1], freq2=[['90','150'], ['90','150']], field2=[[1,0], [1,0]]):
        r"""Returns shape ``(nkbins, nkbins)`` cross-covariance matrix of $P_{gg}^{surr}(k) \times P_{vv}^{surr}(k)$."""
        idx_gal1 = [self.surr_fields[f'gal-{ell1[0]}-{suff}'] for suff in self.suff_gal_list]
        idx_gal2 = [self.surr_fields[f'gal-{ell1[1]}-{suff}'] for suff in self.suff_gal_list]

        idx_vr2_1_1 = [self.surr_fields[f'{freq2[0][0]}-{ell2[0]}-{suff}'] for suff in self.suff_vel_list]
        idx_vr2_1_2 = [self.surr_fields[f'{freq2[0][1]}-{ell2[0]}-{suff}'] for suff in self.suff_vel_list]
        idx_vr2_2_1 = [self.surr_fields[f'{freq2[1][0]}-{ell2[1]}-{suff}'] for suff in self.suff_vel_list]
        idx_vr2_2_2 = [self.surr_fields[f'{freq2[1][1]}-{ell2[1]}-{suff}'] for suff in self.suff_vel_list]       

        coeffs1 = np.array([sn1, b11, self.f, fnl1*(b11 - self.p)])
        coeffs1 = np.ravel(coeffs1[:,None]*coeffs1[None,:]) # shape (6, ) ie (len(idx_gal_11)*len(idx_gal_12), )
        # Add RSD damping factor term: (Warning: only b1 / fnl terms are damped, not the shotnoise term) --> shape (6, nkbins)
        rsd1 = np.array([np.ones_like(self.k['gg']), self.D_g(self.k['gg'], sigmag1), self.D_g(self.k['gg'], sigmag1), self.D_g(self.k['gg'], sigmag1)])
        rsd1 = rsd1[:,None] * rsd1[None,:]
        rsd1 = rsd1.reshape(rsd1.shape[0]*rsd1.shape[1], rsd1.shape[2])
        coeffs1 = coeffs1[:,None] * rsd1 # shape (6, nkbins)

        coeffs2 = np.array([snv2, bv2, bfg2]) if self.sim_surr_fg else np.array([snv2, bv2])
        coeffs2 = coeffs2 = np.ravel(coeffs2[:,None]*coeffs2[None,:])
        # Add RSD damping factor term:
        rsd2 = np.array([np.ones_like(self.k['vv']), self.D_v(self.k['vv'], sigmav2), np.ones_like(self.k['vv'])]) if self.sim_surr_fg else np.array([np.ones_like(self.k['vv']), self.D_v(self.k['vv'], sigmav2)])
        rsd2 = rsd2[:,None] * rsd2[None,:]
        rsd2 = rsd2.reshape(rsd2.shape[0]*rsd2.shape[1], rsd2.shape[2])
        coeffs2 = coeffs2[:,None] * rsd2

        # shape (..., nkbins, nkbins)
        coeff_cov = (coeffs1[:,None,:,None] * coeffs2[None,:,None,:]).reshape(coeffs1.shape[0]*coeffs2.shape[0], coeffs1.shape[1], coeffs2.shape[1])

        cov = []
        # for speed up reason, we do not want to compute covariance matrices that are not needed (for instance if fields2=[1,0]) 
        for i, idx_vr1 in enumerate([idx_vr2_1_1, idx_vr2_1_2]):
            for j, idx_vr2 in enumerate([idx_vr2_2_1, idx_vr2_2_2]):
                if field2[0][i] == 0 or field2[1][j] == 0:
                    cov_tmp = np.zeros((self.nkbins[self.binning['gg']], self.nkbins[self.binning['vv']]))  # shape (nkbins, nkbins)
                else:
                    idx1 = np.array([ii*self.nsurr_fields + jj for ii in idx_gal1 for jj in idx_gal2]) 
                    idx2 = np.array([ii*self.nsurr_fields + jj for ii in idx_vr1 for jj in idx_vr2]) 
                    surr_cov = self.surr_cov[f"{self.binning['gg']}-{self.binning['vv']}"] # select the covariance for the right binning
                    cov_tmp = np.sum(coeff_cov * surr_cov[idx1][:, idx2, :, :].reshape(len(idx1)*len(idx2), self.nkbins[self.binning['gg']], self.nkbins[self.binning['vv']]), axis=0)  # shape (nkbins, nkbins)
                cov += [cov_tmp]

        field = np.ravel(np.array(field2[0])[:, None] * np.array(field2[1])[None, :])
        return np.sum(field[:, None, None] * cov, axis=0)  # shape (nkbins, nkbins)

    def pvvxpgg_cov(self, bv1=1, snv1=1, bfg1=0, sigmav1=0, b12=1, fnl2=0, sn2=1, sigmag2=0, 
                    ell1=[1,1], freq1=[['90','150'], ['90','150']], field1=[[1,0], [1,0]],
                    ell2=[0,0]):
        r"""Returns shape ``(nkbins, nkbins)`` cross-covariance matrix of $P_{vv}^{surr}(k) \times P_{gg}^{surr}(k)$."""
        return self.pggxpvv_cov(b11=b12, fnl1=fnl2, sn1=sn2, sigmag1=sigmag2, bv2=bv1, snv2=snv1, bfg2=bfg1, sigmav2=sigmav1, ell1=ell2, ell2=ell1, freq2=freq1, field2=field1).T

    def _pgg_rms(self, b1=1, fnl=0, sn=1, sigmag=0, ell=[0, 0]):
        r"""For plotting purpose, returns shape ``(nkbins,)`` array, containing sqrt(Var($P_{gg}^{surr}(k)$))."""
        assert self.nsurr >= 2
        return np.sqrt(np.var(self._pgg_surr(b1=b1, fnl=fnl, sn=sn, sigmag=sigmag, ell=ell), axis=0))

    def _pgv_rms(self, b1=1, fnl=0, sn=1, bv=1, snv=1, bfg=0, sigmag=0, sigmav=0, freq=['90','150'], field=[1, 0], ell=[0, 1]):
        r"""For plotting purpose, returns shape ``(nkbins,)`` array containing sqrt(Var($P_{gv}^{surr}(k)$))."""
        assert self.nsurr >= 2
        return np.sqrt(np.var(self._pgv_surr(b1=b1, fnl=fnl, sn=sn, bv=bv, snv=snv, bfg=bfg, sigmag=sigmag, sigmav=sigmav, freq=freq, field=field, ell=ell), axis=0))

    def _pvv_rms(self, bv=1, snv=1, bfg=0, sigmav=0, freq=[['90','150'], ['90','150']], field=[[1,0], [1,0]], ell=[1,1]):
        r"""For plotting purpose, returns shape ``(nkbins,)`` array containing sqrt(var($P_{vv}^{data}(k)$))"""
        assert self.nsurr >= 2
        return np.sqrt(np.var(self._pvv_surr(bv=bv, snv=snv, bfg=bfg, sigmav=sigmav, freq=freq, field=field, ell=ell), axis=0))

    def _pgg_surr(self, b1=1, fnl=0, sn=1, sigmag=0, ell=[0, 0]):
        """For plotting purpose, returns shape (nsurr, nkbins) array, containing P_{gg} for each surrogate"""
        params_gal = np.array([sn, b1, self.f, fnl*(b1 - self.p)])
        params_gal = params_gal[:,None] * [np.ones_like(self.k['gg']), self.D_g(self.k['gg'], sigmag), self.D_g(self.k['gg'], sigmag), self.D_g(self.k['gg'], sigmag)]  # shape (4, nkbins)

        idx_gal_1 = [self.surr_fields[f'gal-{ell[0]}-{suff}'] for suff in self.suff_gal_list]
        idx_gal_2 = [self.surr_fields[f'gal-{ell[1]}-{suff}'] for suff in self.suff_gal_list]

        pgg =  self.pk_surr[self.binning['gg']][:, idx_gal_1, :, :]       # shape (nsurr, 2, #fields, nkbins)
        pgg = np.sum(params_gal[None, :, None, :] * pgg, axis=1)  # shape (nsurr, #fields, nkbins)
        return np.sum(params_gal[None, :, :] * pgg[:, idx_gal_2, :], axis=1)      # shape (nsurr, nkbins)

    def _pgv_surr(self, b1=1, fnl=0, sn=1, bv=1, snv=1, bfg=0, sigmag=0, sigmav=0, freq=['90','150'], field=[1,0], ell=[0, 1]):
        """For plotting purpose, returns shape (nsurr, nkbins) array, containing P_{gv} for each surrogate"""
        field = self._check_field(field)
        assert ell[0] in self.spin_gal
        assert ell[1] in self.spin_vr

        params_gal = np.array([sn, b1, self.f, fnl*(b1 - self.p)])
        params_gal = params_gal[:,None] * [np.ones_like(self.k['gv']), self.D_g(self.k['gv'], sigmag), self.D_g(self.k['gv'], sigmag), self.D_g(self.k['gv'], sigmag)] # shape (4, nkbins)
        idx_gal = [self.surr_fields[f'gal-{ell[0]}-{suff}'] for suff in self.suff_gal_list]

        params_vel = np.array([snv, bv, bfg]) if self.sim_surr_fg else np.array([snv, bv])
        params_vel = params_vel[:, None] * self.D_v(self.k['gv'], sigmav)[None, :]  # shape (2 or 3, nkbins)
        idx_vr_1 = [self.surr_fields[f'{freq[0]}-{ell[1]}-{suff}'] for suff in self.suff_vel_list]
        idx_vr_2 = [self.surr_fields[f'{freq[1]}-{ell[1]}-{suff}'] for suff in self.suff_vel_list]

        pgv = self.pk_surr[self.binning['gv']][:, idx_gal,:,:] # shape (nsurr, 2, #fields, nkbins)
        # this is the g term:
        pgv = np.sum(params_gal[None, :, None, :] * pgv, axis=1) # shape (nsurr, #fields, nkbins)
        # combine or not the different frequency fields:
        pgv = field[0] * pgv[:, idx_vr_1,:] + field[1] * pgv[:, idx_vr_2,:]      # shape (nsurr, 2 or 3, nkbins)
        return np.sum(params_vel[None, :, :] * pgv, axis=1) # shape (nsurr, nkbins)

    def _pvv_surr(self, bv=1, snv=1, bfg=0, sigmav=0, freq=[['90','150'], ['90','150']], field=[[1,0], [1,0]], ell=[1,1]):
        """For plotting purpose, returns shape (nsurr, nkbins) array, containing P_{vv} for each surrogate"""
        assert ell[0] in self.spin_vr
        assert ell[1] in self.spin_vr

        params_vel = np.array([snv, bv, bfg]) if self.sim_surr_fg else np.array([snv, bv])
        params_vel = params_vel[:, None] * self.D_v(self.k['vv'], sigmav)[None, :]  # shape (3, nkbins)

        idx_vr_11 = [self.surr_fields[f'{freq[0][0]}-{ell[0]}-{suff}'] for suff in self.suff_vel_list]
        idx_vr_12 = [self.surr_fields[f'{freq[0][1]}-{ell[0]}-{suff}'] for suff in self.suff_vel_list]
        idx_vr_21 = [self.surr_fields[f'{freq[1][0]}-{ell[1]}-{suff}'] for suff in self.suff_vel_list]
        idx_vr_22 = [self.surr_fields[f'{freq[1][1]}-{ell[1]}-{suff}'] for suff in self.suff_vel_list]

        pvv = field[0][0] * self.pk_surr[self.binning['vv']][:,idx_vr_11,:,:] + field[0][1] * self.pk_surr[self.binning['vv']][:,idx_vr_12,:,:] # shape (nsurr, #params_vel, #fields, nkbins)
        pvv = np.sum(params_vel[None, :, None, :] * pvv, axis=1) # shape (nsurr, #fields, nkbins)
        pvv = field[1][0] * pvv[:, idx_vr_21,:] + field[1][1] * pvv[:, idx_vr_22,:]      # shape (nsurr, #params_vel, nkbins)
        return np.sum(params_vel[None, :, :] * pvv, axis=1)  # shape (nsurr, nkbins)


class CombineKszPipeOutdir(KszPipeOutdir):
    """Class to combine two KszPipeOutdir objects, with a weight w1 for the first one and (1-w1) for the second one."""
    def __init__(self, out1, out2, w1=0.5):

        for key in ['binning', 'k', 'nkbins', 'kmin', 'kmax', 'kstep', 'cmb_fields', 'spin_gal', 'spin_vr', 'data_fields', 'surr_fields', 'sim_surr_fg', 'suff_gal_list', 'suff_vel_list', 'p', 'f', 'nsurr', 'nsurr_fields']:
            # because they do not have all the type, ect I will assume that they are the same between out1 and out2
            # assert getattr(out1, key) == getattr(out2, key), f"The attribute '{key}' must be the same for both out1 and out2"
            setattr(self, key, getattr(out1, key))


        w_ini = w1 if isinstance(w1, float) else 0.5
        w1_data = w_ini * np.ones((len(out1.data_fields), len(out1.data_fields)))
        w1_surr = w_ini * np.ones((len(out1.surr_fields), len(out1.surr_fields)))
        if not isinstance(w1, float):
            for key, val in w1.items():
                kk1, kk2 = key.split('-')
                # data:
                for i in range(len(out1.data_fields)):
                    for j in range(len(out1.data_fields)):
                        totest1 = list(out1.data_fields.keys())[i].split('-')[0]
                        totest2 = list(out1.data_fields.keys())[j].split('-')[0]
                        if (kk1 == totest1 and kk2 == totest2) or (kk2 == totest1 and kk1 == totest2):
                            w1_data[i, j] = val
                # surr:
                for i in range(len(out1.surr_fields)):
                    for j in range(len(out1.surr_fields)):
                        totest1 = list(out1.surr_fields.keys())[i].split('-')[0]
                        totest2 = list(out1.surr_fields.keys())[j].split('-')[0]
                        if (kk1 == totest1 and kk2 == totest2) or (kk2 == totest1 and kk1 == totest2):
                            w1_surr[i, j] = val
                            
        # print(w1_data, w1_surr)
        print(w1_data)
        print(w1_surr)

        self.pk_data = [w1_data[:,:,None]*out1.pk_data[i] + (1 - w1_data)[:,:,None]*out2.pk_data[i] for i in range(len(out1.pk_data))]
        self.pk_surr = [w1_surr[None,:,:,None]*out1.pk_surr[i] + (1 - w1_surr)[None,:,:,None]*out2.pk_surr[i] for i in range(len(out1.pk_surr))]

        # Compute the mean and the covariances for the combination !
        self._precompute_mean_and_cov()
        print('CombineKszPipeOutdir initialized.')