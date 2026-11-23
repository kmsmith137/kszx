import numpy as np

from . import core

from .Box import Box
from .Catalog import Catalog
from .Cosmology import Cosmology


class SurrogateFactory:
    def __init__(self, box, cosmo, randcat, ngal_mean, ngal_rms, ztrue_col='z', kernel='cubic'):
        r"""Helper class for simulating surrogate fields defined on a random catalog.

        Constructor arguments:

          - ``box`` (:class:`~kszx.Box`): defines pixel size, bounding box size, and location of observer.

          - ``cosmo`` (:class:`~kszx.Cosmology`). Used to simulate linear densities/velocities.

          - ``randcat`` (:class:`~kszx.Catalog`): random catalog, defines survey footprint and redshift
            distribution. The randcat must contain columns ``ra_deg`` and ``dec_deg``.

          - ``surr_ngal_mean`` and ``surr_ngal_rms`` (float): In the surrogate sims, I decided to allow $N_{\rm gal}$
            to vary from one surrogate sim to the next. (The idea is to make the surrogate sims more similar to
            mocks, where $N_{\rm gal}$ varies between mocks. Indeed, I find that allowing $N_{\rm gal}$ to vary
            in the surrogate sims does improve overall agreement with mocks.)

            In each surrogate sim, $N_{\rm gal}$ is a Gaussian random variable with mean/rms given by
            the ``surr_ngal_mean`` and ``surr_ngal_rms`` constructor args. If mocks are available, then 
            one way to get sensible values for these arguments is to use the mean/variance in the mocks.
            As a simple placeholder, you could also take ``surr_ngal_rms=0`` (to disable varying $N_{\rm gal}$
            entirely) or ``surr_ngal_rms = sqrt(surr_ngal_mean)`` (Poisson statistics).

          - ``ztrue_col`` (string): name of the randcat column containing redshifts. (If the randcat
            is photometric, then these should be true redshifts, not observed redshifts.)

          - ``kernel`` (string): either ``'cic'`` or ``'cubic'`` (more options will be defined later).

        Most of the interesting members are computed in ``simulate_surrogate()``, but the constructor does
        computes a few useful members:

          - ``self.D`` (1-d array of length nrand): growth function $D(z)$ evaluated on random
            catalog, normalized to $D=1$ at $z=0$.

          - ``self.faH`` (1-d array of length nrand): parameter combination $f_{rsd}(z) H(z) / (1+z)$,
            evaluated on random catalog.

          - ``self.sigma2`` (scalar): variance of linear density field at $z=0$.
        """
        
        assert isinstance(box, Box)
        assert isinstance(cosmo, Cosmology)
        assert isinstance(randcat, Catalog)
        
        assert 0 <= ngal_rms < 0.2 * ngal_mean
        assert ngal_mean + 3*ngal_rms < randcat.size
        
        self.box = box
        self.cosmo = cosmo
        self.kernel = kernel

        self.ngal_mean = ngal_mean
        self.ngal_rms = ngal_rms
        self.ngal_min = ngal_mean - 3 * ngal_rms
        self.ngal_max = ngal_mean + 3 * ngal_rms
        self.nrand = randcat.size

        ztrue = getattr(randcat, ztrue_col)
        self.xyz_true = randcat.get_xyz(cosmo, ztrue_col)
        
        self.D = cosmo.D(z=ztrue, z0norm=True)
        self.f = cosmo.frsd(z=ztrue) 
        self.faH = cosmo.frsd(z=ztrue) * cosmo.H(z=ztrue) / (1+ztrue)
        self.sigma2 = self._integrate_kgrid(box, cosmo.Plin_z0(box.get_k()))

    
    def simulate_surrogate(self):
        r"""Simulates linear density/velocity fields on the random catalog.

        Initializes the following members:

          - ``self.ngal`` (integer): includes random scatter, see class docstring.
        
          - ``self.delta`` (1-d array of length nrand): linear density field
            $\delta(x)$ evaluated on random catalog.
        
          - ``self.phi`` (1-d array of length nrand): field
            $\phi(x) = \delta(x) / \alpha(x)$ evaluated on random catalog.
        
          - ``self.vr`` (1-d array of length nrand): radial velocity field
            $v_r(x)$  evaluated on random catalog.
        
          - ``self.M`` (1-d array of length nrand): random array with
            $(N_{rand} - N_{gal})$ zeroes and $N_{gal}$ ones.

        Reminder: non-Gaussian galaxy bias takes the form
        $\delta_g(x) = b_g \delta(x) + 2 f_{NL} \delta_c (b_g-1) \phi(x)$.
        """

        ngal = self.ngal_mean + (self.ngal_rms * np.random.normal())
        ngal = np.clip(ngal, self.ngal_min, self.ngal_max)
        ngal = int(ngal+0.5)  # round
        assert 0 < ngal <= self.nrand

        # delta is in fourier space:
        delta = core.simulate_gaussian_field(self.box, self.cosmo.Plin_z0)
        core.apply_kernel_compensation(self.box, delta, self.kernel)

        # Let's create the different fields in real space, so that we can interpolate them to the random catalog.
        # Note that phi = delta/alpha is independent of z (in linear theory)
        phi = core.multiply_kfunc(self.box, delta, lambda k: 1.0/self.cosmo.alpha_z0(k=k), dc=0)
        phi = core.interpolate_points(self.box, phi, self.xyz_true, self.kernel, fft=True, spin=0)
        
        # vr = (faHD/k) delta, evaluated at xyz_true.
        vr = core.multiply_kfunc(self.box, delta, lambda k: 1.0/k, dc=0)
        vr = core.interpolate_points(self.box, vr, self.xyz_true, self.kernel, fft=True, spin=1)
        vr *= self.faH * self.D
        
        # Add Kaiser term (L_2(mu) = 1/2(3*mu**2 -1)).
        rsd0 = core.interpolate_points(self.box, delta, self.xyz_true, self.kernel, fft=True, spin=0)
        rsd2 = core.interpolate_points(self.box, delta, self.xyz_true, self.kernel, fft=True, spin=2)
        rsd = 2 / 3 * self.D * rsd2 + 1 / 3 * self.D * rsd0  # This term will be multiply by self.f 

        # delta = core.interpolate_points(self.box, delta, self.xyz_true, self.kernel, fft=True)  # avoid running one more FFT.
        delta = rsd0
        delta *= self.D
            
        # M = random vector with (nrand-ngal) 0s and (ngal) 1s.
        # (This way of generating M is a little slow, but I don't think there's a faster way to do
        # it in numpy, and it's not currently a bottleneck.)
        M = np.zeros(self.nrand)
        M[:ngal] = 1.0
        M = np.random.permutation(M)

        self.ngal = ngal
        self.delta = delta
        self.rsd = rsd
        self.phi = phi
        self.vr = vr
        self.M = M

    
    @staticmethod
    def _integrate_kgrid(box, kgrid):
        """Helper method for constructor. Could be moved somewhere more general (e.g. kszx.core)."""

        assert kgrid.shape == box.fourier_space_shape
    
        w = np.full(box.fourier_space_shape[-1], 2.0)
        w[0] = 1.0
        w[-1] = 2.0 if (box.real_space_shape[-1] % 2) else 1.0
        w = np.reshape(w, (1,)*(box.ndim-1) + (-1,))

        # FIXME could be optimized
        ret = np.sum(w * kgrid) / box.box_volume
        return np.real(ret)
