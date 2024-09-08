import time
import camb
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt

from . import utils   # is_sorted(), spline1d(), spline2d()


####################################################################################################


class CosmologicalParams:
    def __init__(self, name=None):
        """The following values of 'name' are currently supported:

             name='planck18+bao'    https://arxiv.org/abs/1807.06209 (last column of Table 2)
             name='hmvec'           match defaults in Mat Madhavacheril's hmvec code

        If no name is specified, then caller must set {ombh2,omch2,...} after calling constructor.
        """
        
        self.ombh2 = None
        self.omch2 = None
        self.scalar_amp = None
        self.tau = None
        self.ns = None
        self.h = None

        self.kpivot = 0.05
        self.nnu = 3.046
        self.mnu = 0.06
        self.zmax = 10.0
        self.kmax = 100.0
        self.dz = 0.1
        self.lmax = 5000

        # Accuracy settings
        # See sec 5 of https://arxiv.org/abs/2103.05582 for comments on high-l CMB
        
        self.accuracy_boost = 1.0
        self.l_sample_boost = 1.0
        self.l_accuracy_boost = 1.0
        self.do_late_rad_truncation = True
        self.lens_potential_accuracy = 8   # This parameter doesn't affect running time much
        self.lens_margin = 2050            # This parameter doesn't affect running time much
        
        if name == 'planck18+bao':
            # Reference: https://arxiv.org/abs/1807.06209 (last column of Table 2)
            self.ombh2 = 0.02242  # \pm 0.00014
            self.omch2 = 0.11933  # \pm 0.00091
            self.scalar_amp = 2.105e-9  # \pm 0.03
            self.tau = 0.0561     # \pm 0.0071
            self.ns = 0.9665      # \pm 0.0038
            self.h = 0.6766       # \pm 0.42
        elif name == 'hmvec':
            # Match defaults in Mat's hmvec code.
            # Reference: https://github.com/simonsobs/hmvec/blob/master/hmvec/params.py
            self.ombh2 = 0.02225
            self.omch2 = 0.1198
            self.scalar_amp = 2.2e-9
            self.tau = 0.06
            self.ns = 0.9645
            self.h = 0.673
            self.mnu = 0.0   # note zero neutrino mass
        elif name is not None:
            raise RuntimeError(f"CosmologicalParams: name '{name}' not recognized")

        
    def validate(self):
        for k in [ 'ombh2', 'omch2', 'scalar_amp', 'tau', 'ns', 'h' ]:
            if getattr(self, k) is None:
                raise RuntimeError(f"Must either initialize CosmologicalParams.{k}, or call e.g. CosmologicalParams('planck18+bao')")

        # Catch accidental use of 100*h instead of h.
        assert self.h < 2.0

        # FIXME could add more checks here, e.g.
        #  assert self.ombh2 > 0
        #  assert self.zmax > 0


####################################################################################################

        
class Cosmology:
    def __init__(self, params):
        """The 'params' arg can either be a CosmologicalParams, or a name, e.g. 'planck18+bao' or 'hmvec'.

        NOTE: no h-units! All distances are Mpc (not h^{-1} Mpc), and all wavenumbers are Mpc^{-1} (not h Mpc^{-1}).

        This is mostly a thin wrapper around CAMB. I like the wrapper for a few reasons:

           - The Cosmology object is pickleable (unlike the camb 'results' object).
           - I find the syntax a little more intuitive.
           - It's a convenient place to add new methods (e.g. frsd(), alpha()).

        Note: methods require caller to specify keywords, e.g. caller must call Cosmology.Plin(k=xx, z=xx) 
        instead of Plin(xx,xx). This is intentional, to reduce the chances that I'll create a bug by swapping
        arguments or using the wrong time coordinate (e.g. a instead of z).
        """
        
        if isinstance(params, str):
            print(f"Initializing '{params}' cosmology")
            params = CosmologicalParams(params)   # 'params' arg is a string, e.g. 'planck18+bao' or 'hmvec'.
        elif not isinstance(params, CosmologicalParams):
            raise RuntimeError("Cosmology constructor: argument must be either a CosmologicalParams, or a string e.g. 'planck18+bao'")

        params.validate()
        
        self.params = params
        self.zmax = params.zmax
        self.kmax = params.kmax
        self.lmax = params.lmax
        self.h = params.h

        # Comoving matter density (z-independent), units Msol Mpc^{-3}
        self.rhom_comoving = 2.7754e11 * (params.ombh2 + params.omch2)

        # CAMB code below loosely follows:
        # https://camb.readthedocs.io/en/latest/CAMBdemo.html
        
        camb_params = camb.CAMBparams()
        
        camb_params.set_cosmology(
            H0 = 100 * params.h,
            ombh2 = params.ombh2,
            omch2 = params.omch2,
            nnu = params.nnu,
            mnu = params.mnu,
            tau = params.tau
        )
        
        camb_params.InitPower.set_params(
            As = params.scalar_amp,
            ns = params.ns
        )

        camb_params.set_for_lmax(
            params.lmax,
            lens_potential_accuracy = params.lens_potential_accuracy,
            lens_margin = params.lens_margin
        )

        camb_params.set_accuracy(
            AccuracyBoost = params.accuracy_boost,
            lSampleBoost = params.l_sample_boost,
            lAccuracyBoost = params.l_accuracy_boost,
            DoLateRadTruncation = params.do_late_rad_truncation
        )

        # Options here are: (NonLinear_none, NonLinear_pk, NonLinear_lens, NonLinear_both)
        camb_params.NonLinear = camb.model.NonLinear_both
        
        nz_pk = int(round(params.zmax / params.dz)) + 1
        nz_pk = max(nz_pk, 2)
        z_pk = np.linspace(params.zmax, 0, nz_pk)
        
        camb_params.set_matter_power(redshifts=z_pk, kmax=params.kmax)

        print(f'Running CAMB')
        
        camb_timer = time.time()
        camb_results = camb.get_results(camb_params)
        camb_timer = time.time() - camb_timer

        # Currently, these arrays are only used for testing.
        #   self._sigma8_z = 1-d array of redshifts (ordered from highest to lowest)
        #   self._sigma8 = sigma8 value at each redshift

        self._sigma8_z = z_pk
        self._sigma8 = camb_results.get_sigma8()

        # CMB power spectra
        
        lmax = self.lmax
        powers = camb_results.get_cmb_power_spectra(camb_params, CMB_unit='muK', raw_cl=True)

        self.cltt_unl = powers['unlensed_total'][:(lmax+1),0].copy()
        self.cltt_len = powers['total'][:(lmax+1),0].copy()
        self.clphi = powers['lens_potential'][:(lmax+1),0].copy()

        # Large-scale structure power spectra
        
        k, z, pzk = camb_results.get_linear_matter_power_spectrum(var1=None, var2=None, hubble_units=False, k_hunit=False, nonlinear=False)
        print(f'Done running CAMB [{camb_timer} seconds]')

        assert utils.is_sorted(k)
        assert utils.is_sorted(z)
        assert k[0] > 0.0
        assert k[-1] >= self.params.kmax
        assert z[0] == 0.0
        assert np.all(pzk > 0.0)
        assert pzk.shape == (len(z), len(k))

        # Q(k,z) = log(P(k,z)/k). Note that we transpose the (k,z) axes relative to pzk.
        Q = np.log(pzk.T / k.reshape((-1,1)))
        
        self.pk_kmin = k[0]
        self.pk0_interp = utils.spline1d(np.log(k), np.copy(Q[:,0]))  # interpolate log(k) -> Q
        self.pkz_interp = utils.spline2d(np.log(k), z, Q)             # interpolate (log(k),z) -> Q

        # Growth function.
        self.Dfit_z0 = self.Dfit(z=0, z0norm=False)
        self.Dfit_zhi = self.Dfit(z=z[-1], z0norm=False)
        ik = np.searchsorted(k, 0.1)
        # print(f'Inferring growth function D(z) from P(k,z) at k={k[ik]}')
        logD = 0.5 * np.log(pzk[:,ik])
        logD -= logD[0]
        self.logD_interp = utils.spline1d(z, logD)           # interpolate z -> log(D(z)), normalized so that D(0)=1.
        self.logD_shift = np.log(self.Dfit_zhi) - logD[-1]   # log(D) shift to normalize so that D(z) -> 1/(1+z) at high z.

        # Background expansion
        
        nz = 1000
        zvec = np.linspace(0.0, self.zmax, nz)
        Hvec = camb_results.h_of_z(zvec)
        chivec = camb_results.comoving_radial_distance(zvec)

        # Define R = chi/z
        Rvec = np.zeros(nz)
        Rvec[1:] = chivec[1:] / zvec[1:]
        Rvec[0] = 1.0 / Hvec[0]

        self.chimax = chivec[-1]
        self.Hz_interp = utils.spline1d(zvec, Hvec)           # interpolate zvec -> Hvec
        self.Rz_interp = utils.spline1d(zvec, Rvec)           # interpolate z -> (chi/z)
        self.Rchi_interp = utils.spline1d(chivec, 1.0/Rvec)   # interpolate chi -> (z/chi)
        

    def H(self, *, z, check=True):
        """Returns Hubble expansion rate H(z)."""
        
        if check:
            assert np.all(z >= 0.0)
            assert np.all(z <= self.zmax * (1.0+1.0e-8))
        
        return self.Hz_interp(z)   # Hz_interp interpolates z -> H(z)


    def chi(self, *, z, check=True):
        """Returns comoving distance chi(z)."""

        z = np.asarray(z)
        
        if check:
            assert np.all(z >= 0.0)
            assert np.all(z <= self.zmax * (1.0+1.0e-8))
        
        return z * self.Rz_interp(z)   # Rz_interp interpolates z -> (chi/z)

    
    def z(self, *, chi, check=True):
        """Returns redshift z corresponding to specified comoving distance chi."""

        chi = np.asarray(chi)
        
        if check:
            assert np.all(chi >= 0.0)
            assert np.all(chi <= self.chimax * (1.0+1.0e-8))
        
        return chi * self.Rchi_interp(chi)   # Rchi_interp interpolates chi -> (z/chi)


    def Plin_z0(self, *, k, check=True):
        """Returns linear power spectrum P(k,z=0). 

        Slightly faster than calling Plin(k,z) with z=0, and in most 
        situations, calling Plin(k,z) is not necessary, since:

           Plin(k,z) ~ Plin(k,z=0) * D(z,normz0=True)

        This approximation slightly breaks down (at the ~0.5% level!)
        on large scales k <~ 10^(-3) and small scales k >~ 0.1.
        """
        

        k = np.asarray(k)
        
        if check:
            assert np.all(k >= 0.0)
            assert np.all(k <= self.kmax * (1.0+1.0e-8))

        kk = np.maximum(k, self.pk_kmin)
        Q = self.pk0_interp(np.log(kk))   # Q = log(P(k)/k)
        return k * np.exp(Q)


    def Plin(self, *, k, z, kzgrid=False, check=True):
        """
        Returns linear power spectrum P(k,z).

        The 'kzgrid' argument has the following meaning:

          kzgrid=False  means "broadcast k,z to get a set of points"
          kzgrid=True   means "take the Cartesian product of k,z to get a kzgrid"

        (If kzgrid=True, then the returned array has shape (nk,nz) -- note that this
        convention is transposed relative to CAMB or hmvec.)
        """

        k, z = np.asarray(k), np.asarray(z)
        
        if check:
            assert np.all(k >= 0.0)
            assert np.all(k <= self.kmax * (1.0+1.0e-8))
            assert np.all(z >= 0.0)
            assert np.all(z <= self.zmax * (1.0+1.0e-8))

        # Note: the 'grid' argument to scipy.interpolate.RectBivariateSpline.__call__()
        # has the semantics as our 'kzgrid'.
        
        kk = np.maximum(k, self.pk_kmin)
        Q = self.pkz_interp(np.log(kk), z, grid=kzgrid)   # Q = log(P(k)/k)

        if kzgrid:
            k = k.reshape(k.shape + ((-1,)*z.ndim))
        
        return k * np.exp(Q)
    

    def D(self, *, z, z0norm, check=True):
        """Return the growth function D(z).
        If z0norm=True, normalize so that D(0)=1. 
        If z0norm=False, normalize so that D -> 1/(1+z) at high z."""

        z = np.asarray(z)
        
        if check:
            assert np.all(z >= 0.0)
            assert np.all(z <= self.zmax * (1.0+1.0e-8))

        logD = self.logD_interp(z)
        if not z0norm:
            logD += self.logD_shift
        
        return np.exp(logD)
    
    def Dfit(self, *, z, z0norm, check=True):
        """Return the growth function D(z), computed using a popular fitting function.
        You probably want Cosmology.D(), not Cosmology.Dfit()!"""

        z = np.asarray(z)
        
        if check:
            assert np.all(z >= 0.0)
            assert np.all(z <= self.zmax * (1.0+1.0e-8))
            
        omm0 = (self.params.ombh2 + self.params.omch2) / self.h**2
        oml0 = 1.0 - omm0
        omlz = oml0 / (oml0 + omm0 * (1+z)**3)
        ommz = 1.0 - omlz
        D = 2.5 * ommz / (ommz**(4.0/7.0) - omlz + (1.0+ommz/2.0)*(1.0+omlz/70.0)) / (1.0+z)
        return (D / self.Dfit_z0) if z0norm else D

    
    def frsd(self, *, z, check=True):
        """Return RSD function f(z) = d(log D)/d(log a). Uses a fitting function for now!"""

        z = np.asarray(z)
        
        if check:
            assert np.all(z >= 0.0)
            assert np.all(z <= self.zmax * (1.0+1.0e-8))
        
        # FIXME using fitting function for now
        omm0 = (self.params.ombh2 + self.params.omch2) / self.h**2
        oml0 = 1.0 - omm0
        ommz = omm0 / (omm0 + oml0 * (1+z)**(-3.))
        return ommz**(5./9.)


    def alpha(self, *, k, z, kzgrid=False, check=True):
        """Defined by delta_m(k,z) = alpha(k,z) (3/5) zeta(k).
        The 'kzgrid' argument has the same meaning as in Plin() -- see Plin docstring.
        
        The function alpha(k,z) arises in non-Gaussian halo bias as:

          b(k,z) = b_g + 2 b_{ng} fNL / alpha(k,z)

        where b_{ng} = d(log n)/d(log sigma8) ~ delta_c (b_g - 1).

        Note that for fixed k, alpha(k,z) is proportional to D(z), to an excellent
        approximation.
        """

        kpiv = self.params.kpivot
        Delta2 = self.params.scalar_amp
        ns = self.params.ns

        # Easiest way to compute alpha(k,z):
        #
        #  alpha(k,z) = (5/3) (Pm(k,z) / Pzeta(k))**0.5
        #
        #  Pzeta(k) = (2pi^2 / k^3) * Delta^2 * (k/kpiv)**(ns-1)
        #  1/Pzeta(k) = (kpiv^3 / (2 pi^2 Delta^2)) * (k/kpiv)**(4-ns)
        
        k, z = np.asarray(k), np.asarray(z)
        Pm = self.Plin(k=k, z=z, kzgrid=kzgrid)
        Pz_rec = (kpiv**3 / (2 * np.pi**2 * Delta2)) * (k/kpiv)**(4-ns)
        
        if kzgrid:
            Pz_rec = Pz_rec.reshape(Pz_rec.shape + (-1,)*z.ndim)

        return (5/3.) * np.sqrt(Pm * Pz_rec)