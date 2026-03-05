from . import utils
from . import pixell_utils
from . import plot

from .Cosmology import Cosmology

import pixell
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt


class CmbClFitter:
    def __init__(self, cosmo, cmb_map, weight_map, bl, lmin, lmax, ivar=None, iterative=False, fit_cmb_amplitude=False, fit_alpha=True, fit_lred=True, beamed_red_noise=True, uniform_lweight=False, alpha_fid=-3, lred_fid=1500, lpad=1000):
        r"""Given a CMB map, fit a power spectrum of the form $C_l=$ (lensed Cls) + (power-law red noise) + (white noise).
        
        Based on utils.get_single_frequency_alms() in Alex + Mat's kSZquest software:
            https://github.com/alexlague/kSZquest/blob/main/utils.py

        An example notebook where CmbClFitter is used:
            https://github.com/kmsmith137/kszx_notebooks/blob/main/05_sdss_pipeline/03_prepare_cmb_weighting.ipynb

        With default constructor arguments, the CmbClFitter fits the power spectrum of the
        specified ``cmb_map`` (with pixel weighting specified by ``weight_map``, and for 
        $l_{\rm min} \le l \le l_{\rm max}$) to a model of the form:

        $$C_l &= b_l^2 C_l^{\rm CAMB} + A_{\rm red} b_l^2 \max\left( \frac{l}{l_{\rm red}}, 1 \right)^\alpha + A_{\rm white}$$
        with parameters $(A_{\rm red}, A_{\rm white}, \alpha, l_{\rm red})$. Here, $C_l^{CAMB}$ is the 
        **lensed** CMB power spectrum, and does not include kSZ or foregrounds.

        I experimented with a lot of flags that turned out not to really matter. The default
        constructor arguments above (with ``lmin=1500`` and ``lmax=8000``) are what I've been
        using for ACT. 

        If you want to reproduce Alex + Mat's fits from kSZquest, use the following arguments 
        (only args which differ from default values are shown):

           - lmin = 1000
           - lmax = 8000
           - fit_cmb_amplitude = True
           - fit_alpha = False
           - fit_lred = False
           - beamed_red_noise = False
           - uniform_lweight = True

        **Note:** CmbClFitter currently uses pixell maps (for ``cmb_map``, ``weight_map``, ``ivar``), but could
        easily be modified to allow healpix maps -- let me know if this would be useful.
            
        The output of the fitting procedure is contained in the following class members, which are initialized
        by the constructor::

            # Scalar params
            self.alpha           # spectral index of red noise (see above)
            self.red_ampl        # coefficient A_red of red noise contribution to C_l (see above)
            self.write_ampl      # coefficient A_white of white noise contribution to C_l (see above)
            self.cmb_ampl        # coefficient A_cmb of CMB contribution to C_l (see 'fit_cmb_amplitude' below)
            self.l_knee          # value of l where white/red noise are equal
            self.uK_arcmin       # equivalent to A_white, just reparameterized as noise RMS
            self.ivar_uK_arcmin  # equivalent white noise level of ivar map (if ivar is specified)

            # Model power spectra
            self.cl_beamed_cmb    # beam-convolved, noise-free
            self.cl_red_noise     # power-law in l (possibly beam-convolved, see 'beamed_red_noise' below)
            self.cl_white_noise   # constant in l
            self.cl_tot           # sum of previous 3 contributions

            # Weighted or "pseudo" power spectra. These are used by CmbClFitter.plot(), but are probably
            # not useful otherwise. For now, I'm converting model Cls to pseudo Cls by running on simulation.
            # Some day I'll improve this, by implementing the pseudo-Cl transfer matrix.
            
            self.pcl_beamed_cmb   # beam-convolved, noise-free, from one sim
            self.pcl_red_noise    # power spectrum of red noise, from one sim
            self.pcl_white_noise  # constant in l
            self.pcl_tot          # sum of previous 3 contributions
            self.pcl_data         # empirical spectrum of cmb map

        Constructor arguments:

          - ``cosmo`` (:class:`~kszx.Cosmology`): only needed for CAMB lensed $C_l$s.

          - ``cmb_map`` (pixell.enmap.ndmap): CMB temperature map with units $\mu$K
            (e.g. from calling :func:`~kszx.act.read_cmb()`).

          - ``weight_map`` (pixell.enmap.ndmap): applied to all maps before computing
            alms or cls. Recommend including a foreground mask, multiplied by some 
            downweighting of noisy regions (e.g. hard cutoff, ivar weighting, or FKP
            weighting).

          - ``bl`` (1-d numpy array): beam transfer function (e.g. from calling 
            :func:`~kszx.act.read_ivar()`).

          - ``lmin``, ``lmax`` (integers): range of $l$-values over which fit is performed.
            For ACT I've been using ``lmin=1500`` and ``lmax=8000`` (note that the CmbClFitter
            is intended to fit high $l$-values).

          - ``ivar`` (pixell.enmap.ndmap, optional): inverse variance map (e.g. from calling
            :func:`~kszx.act.read_ivar()`). This is not used in the fitting procedure! If ivar
            is specified, then the constructor will initialize ``self.ivar_uK_arcmin``, which
            can be compared to ``self.uK_arcmin``, to get a sense for how well the fitted white
            noise level $A_{\rm white}$ compares to the ivar map.

          - ``iterative`` (boolean, default False): If False, then we assume that pseudo-Cls and
            true Cls are related by a factor $W_2 = \int d^2\theta W(\theta)^2/4\pi$, where $W(\theta)$
            is the ``weight_map``. If True, then we try to improve this approximation using a Monte
            Carlo based, iterative approach. This turned out to make almost no difference, so I don't
            recommend that you use ``iterative=True``!

          - ``fit_cmb_amplitude`` (boolean, default False): If True, then the amplitude of the
            CMB power spectrum is a free (fitted) parameter, i.e. the CMB term is 
            $(A_{\rm cmb} b_l^2 C_l^{\rm CAMB})$ rather than $(b_l^2 C_l^{\rm CAMB})$.

          - ``fit_alpha`` (boolean, default True): If True, then the spectral index $\alpha$
            of the red noise is a free (fitted) parameter. If False, then $\alpha$ is fixed
            to the value ``alpha_fid`` (another constructor argument, see below).

          - ``fit_lred`` (boolean, default True): The $l_{\rm red}$ parameter regulates the
            red noise at low $l$ (see equation near the beginning of this docstring).
            If ``fit_red=True``, then $l_{\rm red}$ is a free (fitted) parameter. If ``fit_red=False``,
            then $l_{\rm red}$ is fixed to the value ``lred_fid`` (another constructor argument,
            see below).

          - ``beamed_red_noise`` (boolean, default True): If True, then the red noise is a 
            beam-convolved power law $C_l = (A_{\rm red} b_l^2 \max(l/l_{\rm red},1)^\alpha)$.
            If False, then the red noise is a power law $C_l = (A_{\rm red} \max(l/l_{\rm red},1)^\alpha)$.

          - ``uniform_lweight`` (boolean, default False): determines how $\chi^2$ is defined 
            (the fit is implemented by minimizing $chi^2$).

               - if False, then $\chi^2 = \sum_{l=l_{min}}^{l_{max}} (\Delta C_l)^2$.

               - if True, then $\chi^2 = \sum_{l=l_{min}}^{l_{max}} (\Delta C_l)^2/(l C_l^2)$.

          - ``alpha_fid`` (float, default -3): Fiducial value of red noise spectral index $\alpha$.
            (If ``fit_alpha=True``, then the value of ``alpha_fid`` shouldn't matter -- we only use it
            to set initial parameter values for the $\chi^2$ minimization.)

          - ``lred`` (float, default 1500): Fiducial value of the $l_{\rm red}$ parameter, which
            regulates the red noise at low $l$ (see equation near the beginning of this docstring).
            (if ``fit_lred=True``, then the value of ``lred`` shouldn't matter -- we only use it to
            to set initial parameter values for the $\chi^2$ minimization.)
        
          - ``lpad`` (integer, default 1000): After doing the fit, we validate the fit by comparing
            the empirical pseudo-$C_l$s of the data to a simulation of the model. We simulate the
            CMB to a multipole $(l_{\rm max} + l_{\rm pad})$ which is larger than the maximum
            multipole $l_{\rm max}$ of the power spectrum estimation/fitting.
        """

        # Argument checking starts here.
        
        bl = utils.asarray(bl, 'CmbClFitter', 'bl', dtype=float)
        
        assert isinstance(cosmo, Cosmology)
        assert isinstance(cmb_map, pixell.enmap.ndmap)
        assert isinstance(weight_map, pixell.enmap.ndmap)
        assert cmb_map.ndim == 2
        assert weight_map.ndim == 2
        assert 2 <= lmin < lmax
        assert lmin <= lred_fid < lmax
        assert alpha_fid < 0
        assert bl.ndim == 1
        assert lpad > 0
        
        beam_lmax = len(bl)-1
        assert beam_lmax >= (lmax + lpad)
        
        if ivar is not None:
            assert isinstance(ivar, pixell.enmap.ndmap)
            assert ivar.ndim == 2
            
        if cosmo.lmax < (lmax + lpad):
            raise RuntimeError(f"CmbClFitter: {cosmo.lmax=} must be >= {(lmax+lpad)=}. To increase cosmo.lmax,"
                               + f" call the Cosmology constructor with the 'lmax' argument specified")
        
        # Argument checking ends here.
        
        self.lmin = lmin
        self.lmax = lmax
        self.lpad = lpad
        self.lred = lred_fid
        self.alpha = alpha_fid
        self.fit_lred = fit_lred
        self.fit_alpha = fit_alpha
        self.fit_cmb_amplitude = fit_cmb_amplitude
        self.beamed_red_noise = beamed_red_noise
        self.uniform_lweight = uniform_lweight
        self.iterative = iterative
        self.bl = bl[:(lmax+lpad+1)]
        
        # self._init_white_noise() initializes the following members:
        #
        #  self.w2               # approximate normalization used to compute pseudo-Cls
        #  self.ivar_uK_arcmin   # equivalent noise level of ivar map
        
        self._init_white_noise(weight_map, ivar)

        # _pcl_from_weighted_map() must come after _init_white_noise()
        self.pcl_data = self._pcl_from_weighted_map(cmb_map * weight_map)
        
        # self._do_fit() initializes or updates the following members:
        #
        #  self.alpha           # current value of alpha
        #  self.red_ampl        # current value of A in C_l = A (l/lmin)^alpha
        #  self.white_ampl      # current value of B in C_l = B
        #  self.cmb_ampl        # multiplier of C_l^{lensed}
        #  self.red_template    # pseudo-cls corresponding to (l/lmin)^alpha, from one sim
        #  self.cmb_template    # pseudo-cls corresponding to C_l^{lensed}, from one sim
        #  self.l_knee          # value of l where white/red noise are equal
        #  self.uK_arcmin       # reparameterization of self.white_ampl

        niter = 2 if iterative else 1
        for _ in range(niter):
            self._do_fit(cosmo, weight_map)
        
        # Done -- just need to initialize remaining members.
        
        self.cl_beamed_cmb = self.cmb_ampl * cosmo.cltt_len[:(lmax+1)] * bl[:(lmax+1)]**2
        self.cl_red_noise = self.red_ampl * self._cl_red_noise()
        self.cl_white_noise = np.full(lmax+1, self.white_ampl)
        self.cl_tot = self.cl_beamed_cmb + self.cl_red_noise + self.cl_white_noise

        self.pcl_beamed_cmb = self.cmb_ampl * self.cmb_template
        self.pcl_red_noise = self.red_ampl * self.red_template
        self.pcl_white_noise = np.full(lmax+1, self.white_ampl)
        self.pcl_tot = self.pcl_beamed_cmb + self.pcl_red_noise + self.pcl_white_noise
        # Reminder: self.pcl_data was initialized above.
        
         
    def make_plots(self, filename_prefix=None, suffix='pdf', s=30):
        """Make three diagnostic plots, either on-screen (filename_prefix=None) or saved to disk.
        
          - Plot 1: pseudo-Cls, data vs model (filename ``{filename_prefix}_pcl.{suffix}``).
          - Plot 2: pseudo-Cls, data/model ratio (filename ``{filename_prefix}_pcl_ratio.{suffix}``).
          - Plot 3: Model Cls (filename ``{filename_prefix}_cl.{suffix}``).
        """

        filename = (filename_prefix + '_pcl' + suffix) if (filename_prefix is not None) else None
        self._plot_smoothed(self.pcl_data, s, label='data', l2=True)
        self._plot_smoothed(self.pcl_tot, s, label='model (total)', l2=True)
        self._plot_smoothed(self.pcl_beamed_cmb, s, label='model (cmb)', l2=True)
        self._plot_smoothed(self.pcl_red_noise, s, label='model (red)', l2=True)
        self._plot_smoothed(self.pcl_white_noise, s, label='model (white)', l2=True)
        plt.title(r'Weighted (pseudo) $C_l$s: data vs model')
        plt.legend(loc = 'lower right')
        plt.xlabel(r'$l$')
        plt.ylabel(r'$l(l+1)C_l/(2\pi)$, arbitrary normalization')
        plt.yscale('log')
        plot.savefig(filename)

        filename = (filename_prefix + '_pcl_ratio' + suffix) if (filename_prefix is not None) else None
        self._plot_smoothed(self.pcl_data/self.pcl_tot, s, label='data/model')
        plt.title(r'(Data pseudo $C_l$) / (Model pseudo $C_l$)')
        plt.legend(loc='lower right')
        plt.xlabel(r'$l$')
        plot.savefig(filename)

        filename = (filename_prefix + '_cl' + suffix) if (filename_prefix is not None) else None
        self._plot_smoothed(self.cl_tot, s, label='model (total)', l2=True)
        self._plot_smoothed(self.cl_beamed_cmb, s, label='model (cmb)', l2=True)
        self._plot_smoothed(self.cl_red_noise, s, label='model (red)', l2=True)
        self._plot_smoothed(self.cl_white_noise, s, label='model (white)', l2=True)
        plt.title(r'Contributions to model $C_l$s (normalized to $\mu$K$^2$)')
        plt.legend(loc = 'lower right')
        plt.xlabel(r'$l$')
        plt.ylabel(r'$l(l+1)C_l/(2\pi)$ ($\mu$K$^2$)')
        plt.yscale('log')
        plot.savefig(filename)      


    def _do_fit(self, cosmo, weight_map):
        """Fits model parameters to self.pcl_data.

        Initializes or updates:

           self.alpha           # current value of alpha
           self.red_ampl        # current value of A in C_l = A (l/lmin)^alpha
           self.white_ampl      # current value of B in C_l = B
           self.cmb_ampl        # multiplier of C_l^{lensed}
           self.red_template    # pseudo-cls corresponding to (l/lmin)^alpha, from one sim
           self.cmb_template    # pseudo-cls corresponding to C_l^{lensed}, from one sim
           self.l_knee          # value of l where white/red noise are equal
           self.uK_arcmin       # reparameterization of self.white_ampl

        Assumes caller has initialized self.alpha. If we're doing iterative fitting, and this
        is not the first call to _do_fit(), then (self.red_template, self.cmb_template) will
        also be used.
        """

        lmin, lmax, lpad = self.lmin, self.lmax, self.lpad

        cl_red = self._cl_red_noise()
        cl_cmb = cosmo.cltt_len[:(lmax+lpad+1)] * self.bl**2   # beamed
        red_template = getattr(self, 'red_template', cl_red)
        cmb_template = getattr(self, 'cmb_template', cl_cmb[:(lmax+1)])
        
        # ninv = inverse variance of C_l, used to determine weighting in fit.
        ninv = np.zeros(lmax+1)

        if self.uniform_lweight:
            ninv[lmin:] = 1.0
        else:
            ninv[lmin:] = 1.0 / (np.arange(lmin,lmax+1) * self.pcl_data[lmin:]**2)

        if False:
            # d = data vector (pcl_data - cmb_template)
            d = self.pcl_data - cmb_template
        
            # M = template matrix (ntemplates by (lmax+1))
            m = [ red_template, white_template ]
            m += [ cmb_template ] if self.fit_cmb_amplitude else []
            m = np.array(m)
            
            a = np.dot(m*ninv, m.T)
            v = np.dot(m*ninv, d)
            coeffs = np.dot(np.linalg.inv(a), v)

            self.red_ampl = coeffs[0]
            self.white_ampl = coeffs[1]
            self.cmb_ampl = (coeffs[2] + 1.0) if self.fit_cmb_amplitude else 1.0

        # Parameter ordering is (red, white, cmb-1, alpha, lred)
        x0 = [0,0]
        x0 += ([0] if self.fit_cmb_amplitude else [])
        x0 += ([self.alpha] if self.fit_alpha else [])
        x0 += ([self.lred] if self.fit_lred else [])
        
        def residual_cl(x):
            alpha = self.alpha
            lred = self.lred
            
            if self.fit_lred:
                lred = x[-1]
                x = x[:-1]

            if self.fit_alpha:
                alpha = x[-1]
                x = x[:-1]

            rcl = self.pcl_data - cmb_template

            # Note: calling _cl_red_noise() at current (alpha,lred), not (self.alpha,self.lred).
            red_cl_ratio = self._cl_red_noise(alpha=alpha,lred=lred) / cl_red
            rcl -= x[0] * red_cl_ratio * red_template
            rcl -= x[1]   # white noise template is C_l = 1

            if self.fit_cmb_amplitude:
                rcl -= x[2] * cmb_template

            return rcl * np.sqrt(ninv)

        lsq = scipy.optimize.least_squares(residual_cl, x0)
        x1 = list(np.copy(lsq['x']))

        self.red_ampl = x1.pop(0)
        self.white_ampl = x1.pop(0)
        self.cmb_ampl = (x1.pop(0)+1) if self.fit_cmb_amplitude else 1.0
        self.alpha = x1.pop(0) if self.fit_alpha else self.alpha
        self.lred = x1.pop(0) if self.fit_lred else self.lred
        self.l_knee = lmin * (self.white_ampl / self.red_ampl)**(1/self.alpha)
        self.uK_arcmin = np.sqrt(self.white_ampl) * (60*180/np.pi)

        # Recompute 'cl_red', in case (alpha, lred) have changed.
        cl_red = self._cl_red_noise(lmax = lmax+lpad)
            
        if not hasattr(self, 'cmb_template'):
            self.cmb_template = self._pcl_from_cl(cl_cmb, weight_map)
            
        if not hasattr(self, 'red_template') or self.fit_alpha:
            self.red_template = self._pcl_from_cl(cl_red, weight_map)

        print('CmbClFitter._do_fit()')
        print(f'    alpha = {self.alpha}')
        print(f'    lred = {self.lred}')
        print(f'    cmb_ampl = {self.cmb_ampl}')
        print(f'    white_ampl = {self.white_ampl}  (uK_arcmin = {self.uK_arcmin})')
        print(f'    red_ampl = {self.red_ampl}    (l_knee = {self.l_knee})')


    def _init_white_noise(self, weight_map, ivar=None):
        """Helper method called by constructor.
        
        self.w2 = sum(weight_map^2 * pixsize) / (4pi)
                = approximate normalization used to compute pseudo-Cls

        self.w2_ivar = sum(weight_map^2 * pixsize^2 / ivar) / (4pi)

        self.ivar_uK_arcmin = sqrt(w2_ivar / w2) * (60*180/pi)
                            = equivalent noise level of ivar map
        """

        wp = weight_map * weight_map.pixsizemap()
        self.w2 = np.vdot(weight_map, wp) / (4*np.pi)
        
        if ivar is None:
            self.w2_ivar = None
            return
        
        invalid = (ivar <= 0)
        
        if np.any(weight_map * invalid):
            raise RuntimeError("CmbClFitter: weight map is nonzero in pixel with ivar <= 0")
        
        wpi = wp / np.where(invalid, 1.0, ivar)   # weight_map * pixsize / ivar
        w2_ivar = np.vdot(wp, wpi) / (4*np.pi)
        self.ivar_uK_arcmin = np.sqrt(w2_ivar/self.w2) * 60*180/np.pi
        print(f'CmbClFitter: equivalent noise level of ivar map = {self.ivar_uK_arcmin} uK-arcmin')
        

    def _cl_red_noise(self, *, alpha=None, lred=None, lmax=None):
        if alpha is None:
            alpha = self.alpha
        if lred is None:
            lred = self.lred
        if lmax is None:
            lmax = self.lmax

        cl = np.maximum(np.arange(lmax+1)/lred, 1.0)**(alpha)
        
        if self.beamed_red_noise:
            cl *= self.bl[:(lmax+1)]**2

        return cl


    def _pcl_from_weighted_map(self, wm):
        alm = pixell_utils.map2alm(wm, self.lmax)
        cl = pixell.curvedsky.alm2cl(alm)
        return cl / self.w2


    def _pcl_from_cl(self, cl_true, weight_map):
        """Simulate map with power spectrum 'cl_true', apply weight map, return (pseudo) Cls.
        
        Some day, I'll implement the pseudo-Cl transfer matrix, and this simulation-based
        approach won't be needed."""

        assert cl_true.shape == ((self.lmax + self.lpad + 1),)
        alm = pixell.curvedsky.rand_alm(cl_true)
        m = pixell_utils.alm2map(alm, weight_map.shape, weight_map.wcs)
        m *= weight_map
        return self._pcl_from_weighted_map(m)

        
    def _plot_smoothed(self, y, s, label=None, l2=False):
        """Helper method for plot()."""
        
        assert y.shape == (self.lmax+1,)
        l = np.arange(self.lmax+1)
        y = (y*l*(l+1)/(2*np.pi)) if l2 else y
        l_smoothed = utils.boxcar_sum(l[self.lmin:], s, normalize=True)
        y_smoothed = utils.boxcar_sum(y[self.lmin:], s, normalize=True)
        plt.plot(l_smoothed, y_smoothed, label=label)
