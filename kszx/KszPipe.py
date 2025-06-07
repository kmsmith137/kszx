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

            self.kmax = params['kmax']
            self.nkbins = params['nkbins']            
            self.kbin_edges = np.linspace(0, self.kmax, self.nkbins+1)
            self.kbin_centers = (self.kbin_edges[1:] + self.kbin_edges[:-1]) / 2.

        self.box = io_utils.read_pickle(f'{input_dir}/bounding_box.pkl')
        self.kernel = 'cubic'   # hardcoded for now
        self.deltac = 1.68      # hardcoded for now

        self.pk_data_filename = f'{output_dir}/pk_data.npy'
        self.pk_surr_filename = f'{output_dir}/pk_surrogates.npy'
        self.pk_single_surr_filenames = [f'{output_dir}/tmp/pk_surr_{i}.npy' for i in range(self.nsurr)]

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
    
    def get_pk_data(self, run=False, force=False):
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

        # Rescale window function (by roughly a factor Ngal/Nrand in each footprint).
        w = np.array(w)
        wf = self.window_function[idx,:][:,idx] * w[:, None] * w[None, :]

        # Estimate power spectra. and normalize by dividing by window function.
        pk = core.estimate_power_spectrum(self.box, fourier_space_maps, self.kbin_edges)
        pk /= wf[:, :, None]

        # Save 'pk_data.npy' to disk. Note that the file format is specified here:
        # https://kszx.readthedocs.io/en/latest/kszpipe.html#kszpipe-details
        io_utils.write_npy(self.pk_data_filename, pk)
        
        return pk

    def get_pk_surrogate(self, isurr, run=False, force=False):
        r"""Returns a shape (A,A,nkbins) array, and saves it in ``pipeline_outdir/tmp/pk_surr_{isurr}.npy``,
        where A = #spin_gal * 2 + #spin_vr * #term * #cmb_fields and # term = 2 if no forgeround and 3 if foregrounds are simulated.
        
        The returned array contains auto and cross power spectra of the following fields, for a single surrogate:
          - surrogate galaxy field $S_g$ with $f_{NL}=0$ with spin in self.spin_gal.
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
        
        if (not force) and os.path.exists(fname):
            return io_utils.read_npy(fname)

        if not (run or force):
            raise RuntimeError(f'KszPipe.get_pk_surrogate(): run=False was specified, and file {fname} not found')

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
        # First we compute the coefficient arrays.
        # For more info, see the overleaf, or the sphinx docs: https://kszx.readthedocs.io/en/latest/kszpipe.html#kszpipe-details
        Sg = (ngal/nrand) * rweights * (self.surr_bg * self.surrogate_factory.delta + eta)
        dSg_dfnl = (ngal/nrand) * rweights * (2 * self.deltac) * (self.surr_bg-1) * self.surrogate_factory.phi
        Sv_noise = {f'{freq}': vweights * self.surrogate_factory.M * self.rcat.get_column(f'tcmb_{freq}') for freq in self.cmb_fields}
        Sv_signal = {f'{freq}': (ngal/nrand) * vweights * self.rcat.get_column(f'bv_{freq}') * self.surrogate_factory.vr for freq in self.cmb_fields}
        if self.sim_surr_fg:
            Sv_fg = {f'{freq}': (ngal/nrand) * vweights * self.rcat.get_column(f'bv_{freq}') * self.surrogate_factory.delta for freq in self.cmb_fields}

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
            for freq in self.cmb_fields:
                Sv_noise[freq] = utils.subtract_binned_means(Sv_noise[freq], zobs, self.nzbins_vr)
                Sv_signal[freq] = utils.subtract_binned_means(Sv_signal[freq], zobs, self.nzbins_vr)
                if self.sim_surr_fg:
                    Sv_fg[freq] = utils.subtract_binned_means(Sv_fg[freq], zobs, self.nzbins_vr)

        # (Coefficient arrays) -> (Fourier-space fields).
        fourier_space_maps, idx = [], []
        for spin in self.spin_gal:
            for term in [Sg, dSg_dfnl]:
                fourier_space_maps += [core.grid_points(self.box, self.rcat_xyz_obs, term, kernel=self.kernel, fft=True, spin=spin, compensate=True)]
                idx += [0]
        for spin in self.spin_vr:
            for i, freq in enumerate(self.cmb_fields): 
                terms = [Sv_noise, Sv_signal, Sv_fg] if self.sim_surr_fg else [Sv_noise, Sv_signal]
                for term in terms:
                    fourier_space_maps += [core.grid_points(self.box, self.rcat_xyz_obs, term[freq], kernel=self.kernel, fft=True, spin=spin, compensate=True)]
                    idx += [i + 1]

        # Rescale window function, by a factor (ngal/nrand) in each footprint.
        wf = (ngal/nrand)**2 * self.window_function
        # Expand window function from shape (3,3) to shape (14,14).
        wf = wf[idx,:][:,idx]

        # Estimate power spectra. and normalize by dividing by window function.
        pk = core.estimate_power_spectrum(self.box, fourier_space_maps, self.kbin_edges)
        pk /= wf[:,:,None]

        io_utils.write_npy(fname, pk)
        return pk

    def get_pk_surrogates(self):
        """Returns the concatenation of all surrogates generated with get_pk_surrogate(), and saves it in ``pipeline_outdir/pk_surrogates.npy``.

        This function only reads files from disk -- it does not run the pipeline.
        To run the pipeline, use :meth:`~kszx.KszPipe.run()`.
        """
        
        if os.path.exists(self.pk_surr_filename):
            return io_utils.read_npy(self.pk_surr_filename)

        if not all(os.path.exists(f) for f in self.pk_single_surr_filenames):
            raise RuntimeError(f'KszPipe.read_pk_surrogates(): necessary files do not exist; you need to call KszPipe.run()')

        pk = np.array([io_utils.read_npy(f) for f in self.pk_single_surr_filenames])

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

        data_fields = params['data_fields']
        pk_data = io_utils.read_npy(f'{dirname}/pk_data.npy')
        if pk_data.shape != (len(data_fields), len(data_fields), nkbins):
            raise RuntimeError(f'Got {pk_data.shape=}, expected ({len(data_fields)},{len(data_fields)},nkbins) where {nkbins=}')

        surr_fields = params['surrogate_fields']
        if nsurr is None:
            pk_surr = io_utils.read_npy(f'{dirname}/pk_surrogates.npy')
            if (pk_surr.ndim != 4) or (pk_surr.shape[1:] != (len(surr_fields),len(surr_fields),nkbins)):
                raise RuntimeError(f'Got {pk_surr.shape=}, expected (nsurr,{len(surr_fields)},{len(surr_fields)},nkbins) where {nkbins=}')
        else:
            pk_surr = [ ]
            for i in range(nsurr):
                pk_surrogate = io_utils.read_npy(f'{dirname}/tmp/pk_surr_{i}.npy')
                if pk_surrogate.shape != (len(surr_fields), len(surr_fields), nkbins):
                    raise RuntimeError(f'Got {pk_surrogate.shape=}, expected ({len(surr_fields)},{len(surr_fields)},nkbins) where {nkbins=}')
                pk_surr.append(pk_surrogate)
            pk_surr = np.array(pk_surr)
            pk_surr = np.reshape(pk_surr, (nsurr, len(surr_fields), len(surr_fields), nkbins))   # needed if nsurr==0

        self.k = kbin_centers
        self.nkbins = nkbins
        self.kmax = kmax
        self.dk = kmax / nkbins

        self.cmb_fields = params['cmb_fields']
        self.spin_gal = params['estimate_spin_gal']
        self.spin_vr = params['estimate_spin_vr']

        self.data_fields = data_fields
        self.pk_data = pk_data
        self.surr_fields = surr_fields
        self.surr_bg = params['surr_bg']
        self.sim_surr_fg = params['simulate_surrogate_foregrounds']

        self.pk_surr = np.array(pk_surr)
        self.nsurr = self.pk_surr.shape[0]

        # Precompute for each surrogate field dedicated mean and covariance for speed up purpose.
        mean = np.mean(self.pk_surr, axis=0)
        self.surr_mean = np.ascontiguousarray(mean) # make contiguous

        nsurr_fields = len(surr_fields)
        self.nsurr_fields = nsurr_fields
        cov = np.cov(self.pk_surr.reshape(self.nsurr, nsurr_fields*nsurr_fields*nkbins), rowvar=False)
        cov = cov.reshape((nsurr_fields*nsurr_fields, nkbins, nsurr_fields*nsurr_fields, nkbins))
        cov = cov.transpose(0, 2, 1, 3)  # reorder axes to have cov matrix in the last two indices for each field.
        self.surr_cov = np.ascontiguousarray(cov) # make contiguous

    def _check_field(self, field):
        """Checks that 'field' is a 1-d array of length 2."""
        field = np.array(field, dtype=float)
        if field.shape != (2,):
            raise RuntimeError(f"Expected 'field' argument to be a 1-d array of length 2, got {field.shape=}")
        return field

    def pgg_data(self, ell=[0, 0]):
        r"""Returns shape ``(nkbins,)`` array containing $P_{gg}^{data}(k)$."""
        assert ell[0] in self.spin_gal
        assert ell[1] in self.spin_gal
        idx1 = self.data_fields[f'gal-{ell[0]}']
        idx2 = self.data_fields[f'gal-{ell[1]}']
        return self.pk_data[idx1, idx2, :]
        
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

        return field[0]*self.pk_data[idx_gal, idx_vr_1] + field[1]*self.pk_data[idx_gal, idx_vr_2]

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

        t = field[0][0]*self.pk_data[idx_vr_11,:] + field[0][1]*self.pk_data[idx_vr_12,:]
        return field[1][0]*t[idx_vr_21] + field[1][1]*t[idx_vr_22]


    def pgg_mean(self, fnl=0, ell=[0,0]):
        r"""Returns shape ``(nkbins,)`` array, containing $\langle P_{gg}^{surr}(k) \rangle$."""
        assert ell[0] in self.spin_gal
        assert ell[1] in self.spin_gal

        params_gal = np.array([1, fnl])
        suff_gal_list = ['null', 'fnl']
        idx_gal_1 = [self.surr_fields[f'gal-{ell[0]}-{suff}'] for suff in suff_gal_list]
        idx_gal_2 = [self.surr_fields[f'gal-{ell[1]}-{suff}'] for suff in suff_gal_list]

        pgg = self.surr_mean[idx_gal_1, :, :]       # shape (2, #fields, nkbins)
        pgg = np.sum(params_gal[:, None, None] * pgg, axis=0)  # shape (nsurr,  #fields, nkbins)
        return np.sum(params_gal[:, None] * pgg[idx_gal_2, :], axis=0)       # shape (nsurr, nkbins)

    def pgv_mean(self, fnl=0, bv=1, bfg=0, freq=['90','150'], field=[1,0], ell=[0, 1]):
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

        params_gal = np.array([1, fnl])
        suff_gal_list = ['null', 'fnl']
        idx_gal = [self.surr_fields[f'gal-{ell[0]}-{suff}'] for suff in suff_gal_list]

        params_vel = np.array([1, bv, bfg]) if self.sim_surr_fg else np.array([1, bv])
        suff_vel_list = ['null', 'bv', 'bfg'] if self.sim_surr_fg else ['null', 'bv']
        idx_vr_1 = [self.surr_fields[f'{freq[0]}-{ell[1]}-{suff}'] for suff in suff_vel_list]
        idx_vr_2 = [self.surr_fields[f'{freq[1]}-{ell[1]}-{suff}'] for suff in suff_vel_list]

        pgv = self.surr_mean[idx_gal,:,:] # shape (2, #fields, nkbins)
        # this is the g term:
        pgv = np.sum(params_gal[:, None, None] * pgv, axis=0) # shape (#fields, nkbins)
        # combine or not the different frequency fields:
        pgv = field[0] * pgv[idx_vr_1,:] + field[1] * pgv[idx_vr_2,:]      # shape (2 or 3, nkbins)
        return np.sum(params_vel[:, None] * pgv, axis=0)   # shape (nkbins)

    def pvv_mean(self, bv=1, bfg=0, freq=[['90','150'], ['90','150']], field=[[1,0], [1,0]], ell=[1,1]):
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

        params_vel = np.array([1, bv, bfg]) if self.sim_surr_fg else np.array([1, bv])
        suff_vel_list = ['null', 'bv', 'bfg'] if self.sim_surr_fg else ['null', 'bv']
        idx_vr_11 = [self.surr_fields[f'{freq[0][0]}-{ell[0]}-{suff}'] for suff in suff_vel_list]
        idx_vr_12 = [self.surr_fields[f'{freq[0][1]}-{ell[0]}-{suff}'] for suff in suff_vel_list]
        idx_vr_21 = [self.surr_fields[f'{freq[1][0]}-{ell[1]}-{suff}'] for suff in suff_vel_list]
        idx_vr_22 = [self.surr_fields[f'{freq[1][1]}-{ell[1]}-{suff}'] for suff in suff_vel_list]

        pvv = field[0][0] * self.surr_mean[idx_vr_11,:,:] + field[0][1] * self.surr_mean[idx_vr_12,:,:] # shape (#params_vel, #fields, nkbins)
        pvv = np.sum(params_vel[:, None, None] * pvv, axis=0) # shape (#fields, nkbins)
        pvv = field[1][0] * pvv[idx_vr_21,:] + field[1][1] * pvv[idx_vr_22,:]      # shape (#params_vel, nkbins)
        return np.sum(params_vel[:, None] * pvv, axis=0)   # shape (nkbins)

    def pggxpgg_cov(self, fnl=0, ell=[0, 0]):
        r"""Returns shape ``(nkbins, nkbins)`` covariance matrix of $P_{gg}^{surr}(k) x P_{gg}^{surr}(k)$."""
        suff_gal_list = ['null', 'fnl']
        idx_gal_1 = [self.surr_fields[f'gal-{ell[0]}-{suff}'] for suff in suff_gal_list]
        idx_gal_2 = [self.surr_fields[f'gal-{ell[1]}-{suff}'] for suff in suff_gal_list]

        coeffs = np.array([1, fnl, fnl, fnl**2]) # shape (4, ) ie (len(idx_gal_1)*len(idx_gal_2), )
        coeff_cov = np.ravel(coeffs[:,None] * coeffs[None,:]) # shape (len(idx_gal_1)*len(idx_gal_2)**2, )
        # keep only the indices that are needed for the covariance matrix:
        idx = np.array([i*self.nsurr_fields + j for i in idx_gal_1 for j in idx_gal_2])  # shape (len(idx_gal_1)*len(idx_gal_2), ) 
        cov = coeff_cov[:, None, None] * self.surr_cov[idx][:, idx, :, :].reshape(len(idx)**2, self.nkbins, self.nkbins)  

        return np.sum(cov, axis=0)  # shape (nkbins, nkbins)

    def pgvxpgv_cov(self, fnl=0, bv=1, bfg=0, freq1=['90','150'], field1=[1,0], ell1=[0, 1], freq2=['90','150'], field2=[1,0], ell2=[0, 1]):
        r"""Returns shape ``(nkbins, nkbins)`` covariance matrix of $P_{gv}^{surr}(k) x P_{gv}^{surr}(k)$.
        The ``field`` argument is a length-2 vector, selecting a linear combination
        of the 90+150 GHz velocity reconstructions. For example:
           - ``field=[1,0]`` for 90 GHz reconstruction
           - ``field=[0,1]`` for 150 GHz reconstruction
           - ``field=[0.5,0.5]`` for mean (90+150) GHz reconstruction **not optimal**.
           - ``field=[1,-1]`` for null (90-150) GHz reconstruction.
        """
        field1, field2 = self._check_field(field1), self._check_field(field2)

        suff_gal_list = ['null', 'fnl']
        idx_gal1 = [self.surr_fields[f'gal-{ell1[0]}-{suff}'] for suff in suff_gal_list]
        idx_gal2 = [self.surr_fields[f'gal-{ell2[0]}-{suff}'] for suff in suff_gal_list]

        suff_vel_list = ['null', 'bv', 'bfg'] if self.sim_surr_fg else ['null', 'bv']
        idx_vr1_1 = [self.surr_fields[f'{freq1[0]}-{ell1[1]}-{suff}'] for suff in suff_vel_list]
        idx_vr1_2 = [self.surr_fields[f'{freq1[1]}-{ell1[1]}-{suff}'] for suff in suff_vel_list]
        idx_vr2_1 = [self.surr_fields[f'{freq2[0]}-{ell2[1]}-{suff}'] for suff in suff_vel_list]
        idx_vr2_2 = [self.surr_fields[f'{freq2[1]}-{ell2[1]}-{suff}'] for suff in suff_vel_list]       

        coeffs = np.array([1, bv, bfg, fnl, fnl*bv, fnl*bfg]) if self.sim_surr_fg else np.array([1, bv, fnl, fnl*bv])
        coeff_cov = np.ravel(coeffs[:,None] * coeffs[None,:])  # shape (len(idx_gal)*len(idx_vr), )

        cov = []
        # for speed up reason, we do not want to compute covariance matrices that are not needed (for instance if fields=[1,0]) 
        for i, idx_vr1 in enumerate([idx_vr1_1, idx_vr1_2]):
            for j, idx_vr2 in enumerate([idx_vr2_1, idx_vr2_2]):
                if field1[i] == 0 or field2[j] == 0:
                    cov_tmp = np.zeros((self.nkbins, self.nkbins))  # shape (nkbins, nkbins)
                else:
                    idx1 = np.array([ii*self.nsurr_fields + jj for ii in idx_gal1 for jj in idx_vr1])  # shape (len(idx_gal_1)*len(idx_vr1), ) 
                    idx2 = np.array([ii*self.nsurr_fields + jj for ii in idx_gal2 for jj in idx_vr2])  # shape (len(idx_gal_2)*len(idx_vr2), )
                    cov_tmp = np.sum(coeff_cov[:, None, None] * self.surr_cov[idx1][:, idx2, :, :].reshape(len(idx1)*len(idx2), self.nkbins, self.nkbins), axis=0)  # shape (nkbins, nkbins)
                cov += [cov_tmp]

        field = np.ravel(field1[:, None] * field2[None, :])
        return np.sum(field[:, None, None] * cov, axis=0)  # shape (nkbins, nkbins)

    def pvvxpvv_cov(self, bv=1, bfg=0, freq1=[['90','150'], ['90','150']], field1=[[1,0], [1,0]], ell1=[1,1], freq2=[['90','150'], ['90','150']], field2=[[1,0], [1,0]], ell2=[1,1]):
        r"""Returns shape ``(nkbins, nkbins)`` covariance matrix of $P_{vv}^{surr}(k)$."""

        suff_vel_list = ['null', 'bv', 'bfg'] if self.sim_surr_fg else ['null', 'bv']

        idx_vr1_1_1 = [self.surr_fields[f'{freq1[0][0]}-{ell1[0]}-{suff}'] for suff in suff_vel_list]
        idx_vr1_1_2 = [self.surr_fields[f'{freq1[0][1]}-{ell1[0]}-{suff}'] for suff in suff_vel_list]
        idx_vr1_2_1 = [self.surr_fields[f'{freq1[1][0]}-{ell1[1]}-{suff}'] for suff in suff_vel_list]
        idx_vr1_2_2 = [self.surr_fields[f'{freq1[1][1]}-{ell1[1]}-{suff}'] for suff in suff_vel_list]    

        idx_vr2_1_1 = [self.surr_fields[f'{freq2[0][0]}-{ell2[0]}-{suff}'] for suff in suff_vel_list]
        idx_vr2_1_2 = [self.surr_fields[f'{freq2[0][1]}-{ell2[0]}-{suff}'] for suff in suff_vel_list]
        idx_vr2_2_1 = [self.surr_fields[f'{freq2[1][0]}-{ell2[1]}-{suff}'] for suff in suff_vel_list]
        idx_vr2_2_2 = [self.surr_fields[f'{freq2[1][1]}-{ell2[1]}-{suff}'] for suff in suff_vel_list]       

        coeffs = np.array([1, bv, bfg, bv, bv**2, bv*bfg, bfg, bfg*bv, bfg**2]) if self.sim_surr_fg else np.array([1, bv, bv, bv**2])
        coeff_cov = np.ravel(coeffs[:,None] * coeffs[None,:])  # shape (len(idx_gal)*len(idx_vr), )

        cov = []  # it will be a 16 x nkbins x nkbins array to cover any possible combination of fields1 / fields2.
        # for speed up reason, we do not want to compute covariance matrices that are not needed (for instance if fields=[1,0]) 
        for i, idx_vr1_1 in enumerate([idx_vr1_1_1, idx_vr1_1_2]):
            for j, idx_vr1_2 in enumerate([idx_vr1_2_1, idx_vr1_2_2]):
                for k, idx_vr2_1 in enumerate([idx_vr2_1_1, idx_vr2_1_2]):
                    for l, idx_vr2_2 in enumerate([idx_vr2_2_1, idx_vr2_2_2]):
                        if field1[0][i] == 0 or field1[1][j] == 0 or field2[0][k] == 0 or field2[1][l] == 0:
                            cov_tmp = np.zeros((self.nkbins, self.nkbins))  # shape (nkbins, nkbins)
                        else:
                            idx1 = np.array([ii*self.nsurr_fields + jj for ii in idx_vr1_1 for jj in idx_vr2_1])  # shape (len(idx_gal_1)*len(idx_vr1), ) 
                            idx2 = np.array([ii*self.nsurr_fields + jj for ii in idx_vr1_2 for jj in idx_vr2_2])  # shape (len(idx_gal_2)*len(idx_vr2), )
                            cov_tmp = np.sum(coeff_cov[:, None, None] * self.surr_cov[idx1][:, idx2, :, :].reshape(len(idx1)*len(idx2), self.nkbins, self.nkbins), axis=0)  # shape (nkbins, nkbins)
                        cov += [cov_tmp]

        field = np.ravel(np.ravel(np.array(field1[0])[:, None] * np.array(field1[1])[None, :])[:, None] * np.ravel(np.array(field2[0])[:, None] * np.array(field2[1])[None, :])[None, :])
        return np.sum(field[:, None, None] * cov, axis=0)  # shape (nkbins, nkbins)

    def pggxpgv_cov(self, fnl=0, bv=1, bfg=0, freq=['90','150'], field=[1,0], ell1=[0, 0], ell2=[0, 1]):
        r"""Returns shape ``(nkbins, nkbins)`` cross-covariance matrix of $P_{gg}^{surr}(k) \times P_{gv}^{surr}(k)$."""
        
        field = self._check_field(field)

        suff_gal_list = ['null', 'fnl']
        idx_gal1 = [self.surr_fields[f'gal-{ell1[0]}-{suff}'] for suff in suff_gal_list]
        idx_gal2 = [self.surr_fields[f'gal-{ell1[1]}-{suff}'] for suff in suff_gal_list]
        idx_gal3 = [self.surr_fields[f'gal-{ell2[0]}-{suff}'] for suff in suff_gal_list]

        suff_vel_list = ['null', 'bv', 'bfg'] if self.sim_surr_fg else ['null', 'bv']
        idx_vr1_1 = [self.surr_fields[f'{freq[0]}-{ell2[1]}-{suff}'] for suff in suff_vel_list]
        idx_vr1_2 = [self.surr_fields[f'{freq[1]}-{ell2[1]}-{suff}'] for suff in suff_vel_list]

        coeffs1 = np.array([1, fnl, fnl, fnl**2])
        coeffs2 = np.array([1, bv, bfg, fnl, fnl*bv, fnl*bfg]) if self.sim_surr_fg else np.array([1, bv, fnl, fnl*bv])
        coeff_cov = np.ravel(coeffs1[:,None] * coeffs2[None,:])  # shape (len(idx_gal)*len(idx_vr), )

        cov = []
        # for speed up reason, we do not want to compute covariance matrices that are not needed (for instance if fields=[1,0]) 
        for i, idx_vr in enumerate([idx_vr1_1, idx_vr1_2]):
            if field[i] == 0:
                cov_tmp = np.zeros((self.nkbins, self.nkbins))  # shape (nkbins, nkbins)
            else:
                idx1 = np.array([ii*self.nsurr_fields + jj for ii in idx_gal1 for jj in idx_gal2]) 
                idx2 = np.array([ii*self.nsurr_fields + jj for ii in idx_gal3 for jj in idx_vr]) 
                cov_tmp = np.sum(coeff_cov[:, None, None] * self.surr_cov[idx1][:, idx2, :, :].reshape(len(idx1)*len(idx2), self.nkbins, self.nkbins), axis=0)  # shape (nkbins, nkbins)
            cov += [cov_tmp]

        return np.sum(field[:, None, None] * cov, axis=0)  # shape (nkbins, nkbins)

    def pgvxpvv_cov(self, fnl=0, bv=1, bfg=0, ell1=[0,1], ell2=[1,1], freq1=['90','150'], field1=[1,0], freq2=[['90','150'], ['90','150']], field2=[[1,0], [1,0]]):
        r"""Returns shape ``(nkbins, nkbins)`` cross-covariance matrix of $P_{gv}^{surr}(k) \times P_{vv}^{surr}(k)$."""
        suff_gal_list = ['null', 'fnl']
        idx_gal1 = [self.surr_fields[f'gal-{ell1[0]}-{suff}'] for suff in suff_gal_list]

        suff_vel_list = ['null', 'bv', 'bfg'] if self.sim_surr_fg else ['null', 'bv']
        idx_vr1_1 = [self.surr_fields[f'{freq1[0]}-{ell1[1]}-{suff}'] for suff in suff_vel_list]
        idx_vr1_2 = [self.surr_fields[f'{freq1[1]}-{ell1[1]}-{suff}'] for suff in suff_vel_list]
        idx_vr2_1_1 = [self.surr_fields[f'{freq2[0][0]}-{ell2[0]}-{suff}'] for suff in suff_vel_list]
        idx_vr2_1_2 = [self.surr_fields[f'{freq2[0][1]}-{ell2[0]}-{suff}'] for suff in suff_vel_list]
        idx_vr2_2_1 = [self.surr_fields[f'{freq2[1][0]}-{ell2[1]}-{suff}'] for suff in suff_vel_list]
        idx_vr2_2_2 = [self.surr_fields[f'{freq2[1][1]}-{ell2[1]}-{suff}'] for suff in suff_vel_list]       

        coeffs1 = np.array([1, bv, bfg, fnl, fnl*bv, fnl*bfg]) if self.sim_surr_fg else np.array([1, bv, fnl, fnl*bv])
        coeffs2 = np.array([1, bv, bfg, bv, bv**2, bv*bfg, bfg, bfg*bv, bfg**2]) if self.sim_surr_fg else np.array([1, bv, bv, bv**2])
        coeff_cov = np.ravel(coeffs1[:,None] * coeffs2[None,:])  # shape (len(idx_gal)*len(idx_vr), )

        cov = []
        # for speed up reason, we do not want to compute covariance matrices that are not needed (for instance if fields=[1,0]) 
        for i, idx_vr1 in enumerate([idx_vr1_1, idx_vr1_2]):
            for j, idx_vr2_1 in enumerate([idx_vr2_1_1, idx_vr2_1_2]):
                for k, idx_vr2_2 in enumerate([idx_vr2_2_1, idx_vr2_2_2]):
                    if field1[i] == 0 or field2[0][j] == 0 or field2[1][k] == 0:
                        cov_tmp = np.zeros((self.nkbins, self.nkbins))  # shape (nkbins, nkbins)
                    else:
                        idx1 = np.array([ii*self.nsurr_fields + jj for ii in idx_gal1 for jj in idx_vr1]) 
                        idx2 = np.array([ii*self.nsurr_fields + jj for ii in idx_vr2_1 for jj in idx_vr2_2]) 
                        cov_tmp = np.sum(coeff_cov[:, None, None] * self.surr_cov[idx1][:, idx2, :, :].reshape(len(idx1)*len(idx2), self.nkbins, self.nkbins), axis=0)  # shape (nkbins, nkbins)
                    cov += [cov_tmp]

        field = np.ravel(np.array(field1)[:, None] * np.ravel(np.array(field2[0])[:, None] * np.array(field2[1])[None, :])[None, :])
        return np.sum(field[:, None, None] * cov, axis=0)  # shape (nkbins, nkbins)

    def pggxpvv_cov(self, fnl=0, bv=1, bfg=0, ell1=[0,0], ell2=[1,1], freq=[['90','150'], ['90','150']], field=[[1,0], [1,0]]):
        r"""Returns shape ``(nkbins, nkbins)`` cross-covariance matrix of $P_{gg}^{surr}(k) \times P_{vv}^{surr}(k)$."""
        suff_gal_list = ['null', 'fnl']
        idx_gal1 = [self.surr_fields[f'gal-{ell1[0]}-{suff}'] for suff in suff_gal_list]
        idx_gal2 = [self.surr_fields[f'gal-{ell1[1]}-{suff}'] for suff in suff_gal_list]

        suff_vel_list = ['null', 'bv', 'bfg'] if self.sim_surr_fg else ['null', 'bv']
        idx_vr2_1_1 = [self.surr_fields[f'{freq[0][0]}-{ell2[0]}-{suff}'] for suff in suff_vel_list]
        idx_vr2_1_2 = [self.surr_fields[f'{freq[0][1]}-{ell2[0]}-{suff}'] for suff in suff_vel_list]
        idx_vr2_2_1 = [self.surr_fields[f'{freq[1][0]}-{ell2[1]}-{suff}'] for suff in suff_vel_list]
        idx_vr2_2_2 = [self.surr_fields[f'{freq[1][1]}-{ell2[1]}-{suff}'] for suff in suff_vel_list]       

        coeffs1 = np.array([1, fnl, fnl, fnl**2])
        coeffs2 = np.array([1, bv, bfg, bv, bv**2, bv*bfg, bfg, bfg*bv, bfg**2]) if self.sim_surr_fg else np.array([1, bv, bv, bv**2])
        coeff_cov = np.ravel(coeffs1[:,None] * coeffs2[None,:])  # shape (len(idx_gal)*len(idx_vr), )

        cov = []
        # for speed up reason, we do not want to compute covariance matrices that are not needed (for instance if fields=[1,0]) 
        for i, idx_vr1 in enumerate([idx_vr2_1_1, idx_vr2_1_2]):
            for j, idx_vr2 in enumerate([idx_vr2_2_1, idx_vr2_2_2]):
                if field[0][i] == 0 or field[1][j] == 0:
                    cov_tmp = np.zeros((self.nkbins, self.nkbins))  # shape (nkbins, nkbins)
                else:
                    idx1 = np.array([ii*self.nsurr_fields + jj for ii in idx_gal1 for jj in idx_gal2]) 
                    idx2 = np.array([ii*self.nsurr_fields + jj for ii in idx_vr1 for jj in idx_vr2]) 
                    cov_tmp = np.sum(coeff_cov[:, None, None] * self.surr_cov[idx1][:, idx2, :, :].reshape(len(idx1)*len(idx2), self.nkbins, self.nkbins), axis=0)  # shape (nkbins, nkbins)
                cov += [cov_tmp]

        field = np.ravel(np.array(field[0])[:, None] * np.array(field[1])[None, :])
        return np.sum(field[:, None, None] * cov, axis=0)  # shape (nkbins, nkbins)

    def _pgg_rms(self, fnl=0, ell=[0, 0]):
        r"""For plotting purpose, returns shape ``(nkbins,)`` array, containing sqrt(Var($P_{gg}^{surr}(k)$))."""
        assert self.nsurr >= 2
        return np.sqrt(np.var(self._pgg_surr(fnl=0, ell=[0, 0]), axis=0))

    def _pgv_rms(self, fnl=0, bv=1, bfg=0, freq=['90','150'], field=[1, 0], ell=[0, 1]):
        r"""For plotting purpose, returns shape ``(nkbins,)`` array containing sqrt(Var($P_{gv}^{surr}(k)$))."""
        assert self.nsurr >= 2
        return np.sqrt(np.var(self._pgv_surr(fnl=0, bv=1, bfg=0, freq=['90','150'], field=[1,0], ell=[0, 1]), axis=0))

    def _pvv_rms(self, bv=1, bfg=0, freq=[['90','150'], ['90','150']], field=[[1,0], [1,0]], ell=[1,1]):
        r"""For plotting purpose, returns shape ``(nkbins,)`` array containing sqrt(var($P_{vv}^{data}(k)$))"""
        assert self.nsurr >= 2
        return np.sqrt(np.var(self._pvv_surr(bv=bv, bfg=bfg, freq=freq, field=field, ell=ell), axis=0))

    def _pgg_surr(self, fnl=0, ell=[0, 0]):
        """For plotting purpose, returns shape (nsurr, nkbins) array, containing P_{gg} for each surrogate"""
        params_gal = np.array([1, fnl])
        suff_gal_list = ['null', 'fnl'] 
        idx_gal_1 = [self.surr_fields[f'gal-{ell[0]}-{suff}'] for suff in suff_gal_list]
        idx_gal_2 = [self.surr_fields[f'gal-{ell[1]}-{suff}'] for suff in suff_gal_list]

        pgg =  self.pk_surr[:, idx_gal_1, :, :]       # shape (nsurr, 2, #fields, nkbins)
        pgg = np.sum(params_gal[None, :, None, None] * pgg, axis=1)  # shape (nsurr, #fields, nkbins)
        return np.sum(params_gal[None, :, None] * pgg[:, idx_gal_2, :], axis=1)       # shape (nsurr, nkbins)

    def _pgv_surr(self, fnl=0, bv=1, bfg=0, freq=['90','150'], field=[1,0], ell=[0, 1]):
        """For plotting purpose, returns shape (nsurr, nkbins) array, containing P_{gv} for each surrogate"""
        field = self._check_field(field)
        assert ell[0] in self.spin_gal
        assert ell[1] in self.spin_vr

        params_gal = np.array([1, fnl])
        suff_gal_list = ['null', 'fnl'] 
        idx_gal = [self.surr_fields[f'gal-{ell[0]}-{suff}'] for suff in suff_gal_list]

        params_vel = np.array([1, bv, bfg]) if self.sim_surr_fg else np.array([1, bv])
        suff_vel_list = ['null', 'bv', 'bfg'] if self.sim_surr_fg else ['null', 'bv']
        idx_vr_1 = [self.surr_fields[f'{freq[0]}-{ell[1]}-{suff}'] for suff in suff_vel_list]
        idx_vr_2 = [self.surr_fields[f'{freq[1]}-{ell[1]}-{suff}'] for suff in suff_vel_list]

        pgv = self.pk_surr[:, idx_gal,:,:] # shape (nsurr, 2, #fields, nkbins)
        # this is the g term:
        pgv = np.sum(params_gal[None, :, None, None] * pgv, axis=1) # shape (nsurr, #fields, nkbins)
        # combine or not the different frequency fields:
        pgv = field[0] * pgv[:, idx_vr_1,:] + field[1] * pgv[:, idx_vr_2,:]      # shape (nsurr, 2 or 3, nkbins)
        return np.sum(params_vel[None, :, None] * pgv, axis=1)   # shape (nsurr, nkbins)

    def _pvv_surr(self, bv=1, bfg=0, freq=[['90','150'], ['90','150']], field=[[1,0], [1,0]], ell=[1,1]):
        """For plotting purpose, returns shape (nsurr, nkbins) array, containing P_{vv} for each surrogate"""
        assert ell[0] in self.spin_vr
        assert ell[1] in self.spin_vr

        params_vel = np.array([1, bv, bfg]) if self.sim_surr_fg else np.array([1, bv])
        suff_vel_list = ['null', 'bv', 'bfg'] if self.sim_surr_fg else ['null', 'bv']
        idx_vr_11 = [self.surr_fields[f'{freq[0][0]}-{ell[0]}-{suff}'] for suff in suff_vel_list]
        idx_vr_12 = [self.surr_fields[f'{freq[0][1]}-{ell[0]}-{suff}'] for suff in suff_vel_list]
        idx_vr_21 = [self.surr_fields[f'{freq[1][0]}-{ell[1]}-{suff}'] for suff in suff_vel_list]
        idx_vr_22 = [self.surr_fields[f'{freq[1][1]}-{ell[1]}-{suff}'] for suff in suff_vel_list]

        pvv = field[0][0] * self.pk_surr[:,idx_vr_11,:,:] + field[0][1] * self.pk_surr[:,idx_vr_12,:,:] # shape (nsurr, #params_vel, #fields, nkbins)
        pvv = np.sum(params_vel[None, :, None, None] * pvv, axis=1) # shape (nsurr, #fields, nkbins)
        pvv = field[1][0] * pvv[:, idx_vr_21,:] + field[1][1] * pvv[:, idx_vr_22,:]      # shape (nsurr, #params_vel, nkbins)
        return np.sum(params_vel[None, :, None] * pvv, axis=1)   # shape (nsurr, nkbins)