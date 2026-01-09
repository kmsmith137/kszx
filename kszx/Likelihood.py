import sys

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

from . import utils

from .KszPipe import KszPipeOutdir


class BaseLikelihood:
    def __init__(self):
        r""" Base class for likelihoods. 
        Todo: Add all the self.attributes that are required ! """
        raise NotImplementedError("This is an abstract class. Please use a subclass of BaseLikelihood.")

    def mean_and_cov(self, force_compute_cov=False, return_cov=True, return_grad=False, **params):
        r""" Computes the mean and covariance matrix for the given parameters. 
            If return_grad is True, it also returns the gradients of the mean and covariance with respect to the parameters.
            If return_cov is False, only the mean is returned.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    @staticmethod
    def compute_derived_params(derived_params, params):
        """ Compute derived parameters from a dictionary of formulas and a dictionary of parameters."""
        derived = {}
        for key, formula in derived_params.items():
            # params dict must contain all referenced variables
            derived[key] = eval(formula, {"np": np}, params)
        return derived

    def uniform_log_prior(self, **params):
        for key in self.params:
            low, high = self.params[key]['prior']
            if not (low < params[key] < high):
                return -np.inf
        return 0.0

    def log_likelihood(self, *params):
        r""" """
        params = {key: params[i] for i, key in enumerate(self.params)}

        return_cov = not (self.cov_fix_params or self.cov_interp)
        return_grad = self.jeffreys_prior

        to_unpack = self.mean_and_cov(**params, return_grad=return_grad, return_cov=return_cov)
        if return_cov: 
            if return_grad:
                mean, cov, grad_mean, grad_cov = to_unpack
            else:
                mean, cov = to_unpack
        else: 
            mean = to_unpack

        # Cholesky decompotision:
        x = self.data - mean

        if self.cov_fix_params:
            cov_cholesky = self.cov_cholesky
            logdet_cov = self.logdet_cov
        elif self.cov_interp:
            cov_cholesky = self.cov_cholesky_interp([params[name] for name in self.params])
            logdet_cov = self.logdet_interp([params[name] for name in self.params])
            if np.isnan(cov_cholesky).any(): 
                # We can compute the cholesky interpolation outside the range since the prior is acting below (at self.uniform_log_prior step)
                # It is not a problem if the range of the interpolation is as wide as the prior range, the likelihood will be -np.inf due to the prior as well.
                cov_cholesky, logdet_cov = np.eye(cov_cholesky.shape[0]), 0
        else: 
            cov_cholesky = np.linalg.cholesky(cov)
            logdet_cov = np.linalg.slogdet(cov)[1]

        # Compute the logLikelihood:
        linv_x = scipy.linalg.solve_triangular(cov_cholesky, x, lower=True)
        logL = -(0.5 * np.dot(linv_x, linv_x) + logdet_cov)  # + x.size*np.log(2*np.pi))

        # Add uniform prior: 
        logL += self.uniform_log_prior(**params)

        # if self.jeffreys_prior:
        #     B, D = (self.B, self.D)
        #     linv_dmu = scipy.linalg.solve_triangular(l, grad_mean.T, lower=True)  # shape (D,2)
        #     f = np.dot(linv_dmu.T, linv_dmu)   # first term in 2-by-2 Fisher matrix

        #     # Second term in 2-by-2 Fisher matrix
        #     # F_{ij} = (1/2) Tr(C^{-1} dC_i C^{-1} dC_j)
        #     #        = (1/2) Tr(S_i S_j)  where S_i = L^{-1} dC_i L^{-T}
            
        #     t = grad_cov.reshape(((B+1)*D, D))
        #     u = scipy.linalg.solve_triangular(l, t.T, lower=True)  # shape (D, (B+1)*D)
        #     u = u.reshape((D*(B+1), D))
        #     v = scipy.linalg.solve_triangular(l, u.T, lower=True)  # shape (D,D*(B+1))
        #     v = v.reshape((D*D, B+1))
        #     f += 0.5 * np.dot(v.T, v)

        #     # Jeffreys prior is equivalent to including sqrt(det(F)) in the likelihood.
        #     sign, logabsdet_F = np.linalg.slogdet(f)
        #     logL += 0.5 * logabsdet_F
        #     assert sign == 1

        return logL

    def interpolate_cholesky(self, method='linear'):
        """ Interpolate choleksy and slogdet values with RegularGridInterpolator. method='cubic' is much more accurate but slow down the evaluation by x100. """
        import tqdm
        from scipy.interpolate import RegularGridInterpolator

        params_interp = [np.linspace(self.params[name]['interp'][0], self.params[name]['interp'][1], self.params[name]['interp'][2]) for name in self.params]
        nbins = [self.params[name]['interp'][2] for name in self.params]
        slogdet_values, chol_values = [], []

        mesh = [np.ravel(mm) for mm in np.meshgrid(*params_interp, indexing='ij')]

        for i in tqdm.tqdm(range(len(mesh[0]))):
            params_to_eval = {name: mesh[n][i] for n, name in enumerate(self.params)}
            _, cov = self.mean_and_cov(**params_to_eval)
            chol_values += [utils.flatten_cholesky(np.linalg.cholesky(cov))]
            slogdet_values += [np.linalg.slogdet(cov)[1]]

        chol_values = np.array(chol_values).reshape(nbins + [sum([i+1 for i in range(self.k.size)])])
        slogdet_values = np.array(slogdet_values).reshape(nbins)

        chol_interp = RegularGridInterpolator(params_interp, chol_values, method=method, bounds_error=False)
        slogdet_interp = RegularGridInterpolator(params_interp, slogdet_values, method=method, bounds_error=False)

        return slogdet_interp, chol_interp

    def validate_cholesky_interpolation(self, ntest=2, seed=32, nticks=4, vmin=-0.1, vmax=0.1, fn_fig=None):
        """ Test the cholesky interpolation that can be used for the inference and the profiling. """
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        np.random.seed(seed)
        for i in range(ntest): 
            params_test = {name: np.random.uniform(low=self.params[name]['prior'][0], high=self.params[name]['prior'][1], size=1)[0] for name in self.params}
            print(params_test)
            _, cov = self.mean_and_cov(**params_test)
            chol = np.linalg.cholesky(cov)
            chol_test = self.cov_cholesky_interp([params_test[name] for name in self.params])

            plt.figure(figsize=(2.5*len(self.fields) + 1, 2.5*len(self.fields) + 2))
            ax = plt.gca()
            im = ax.imshow((chol - chol_test) / chol * 100, vmin=vmin, vmax=vmax)  # (chol - chol_test) / np.diag(chol) * 100

            idx = np.arange(0, len(self.k), len(self.k) // (nticks*len(self.fields)))
            ax.set_xticks(idx, [f'{self.k[i]:2.2f}' for i in idx])
            ax.set_yticks(idx, [f'{self.k[i]:2.2f}' for i in idx])
            ax.set_xlabel('$k$ [Mpc$^{-1}$]')
            ax.set_ylabel('$k$ [Mpc$^{-1}$]')

            # create an axes on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.1 inch.
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im, cax=cax, label=r'relative errors [$\%$]')

            plt.tight_layout()
            if fn_fig is not None: plt.savefig(fn_fig)
            plt.show()

    def run_profiling(self, nprofiles=5, fn_profile=None, maxiter=None, verbose=True):
        r"""Returns bestfit value for self.params after nprofiles different profiling with scipy.optimize.minimize.  """

        x0 = np.zeros((nprofiles, len(self.params)))
        for i, key in enumerate(self.params):
            x0[:, i] = np.random.uniform(self.params[key]['prior'][0], self.params[key]['prior'][1], size=nprofiles)

        f = lambda x: - self.log_likelihood(*x)  # note minus sign

        results = []
        for i in range(nprofiles):
            results += [scipy.optimize.minimize(f, x0[i,:], method='Nelder-Mead', options={'maxiter': maxiter})]
        
        # select the bestfit with the lowest log_likelihood:
        sel_min = np.argmin([result.fun for result in results])
        self.result = results[sel_min]
        if verbose: print(self.result)
        self.bestfit = {key: self.result.x[i] for i, key in enumerate(self.params)}
        if fn_profile is not None: np.save(self.result, fn_profile)

        return self.bestfit      

    def run_mcmc(self, ncpu=1, nwalkers=8, nsamples=10000, discard=1000, thin=5, progress='notebook', extend_chain=False, fn_chain=None):
        r""" 
        Run mcmc with emcee. 

        Parameters:
        - nwalkers: number of walkers
        - nsamples: number of samples to draw
        - discard: number of samples to discard at the beginning of the chain (burnin phase)
        - thin: thinning factor (only keep every `thin`-th sample)
        - progress: 'notebook' or 'text' to show the progress bar of emcee (requires tqdm), False to disable it.
        - extend_chain: if True, use the previous chain as a starting point for the new chain. If False, start from a random uniform distribution.
        - fn_chain: if not None, save the chain to this file.
        
        """
        import emcee
        print(f'MCMC start: {nwalkers=}, {nsamples=}, {discard=}, {thin=}')

        if extend_chain:
            x0 = None
        else:
            x0 = np.zeros((nwalkers, len(self.params)))
            for i, key in enumerate(self.params):
                x0[:, i] = np.random.uniform(self.params[key]['prior'][0], self.params[key]['prior'][1], size=nwalkers)

        logL = lambda x: self.log_likelihood(*x)

        # It is super slow ... in the notebook at NERSC for some reason...:
        # First need to write a wrapper function to use the logL method (OUTSIDE the class):
        # def logL(self, x):
        #     print(x)
        #     return self.log_likelihood(*x)
        #from multiprocessing import Pool
        #import kszx.utils as utils
        #with utils.Pool(ncpu) as pool:
        #from multiprocessing import Pool
        #with Pool(ncpu) as pool:
        #    sampler = emcee.EnsembleSampler(nwalkers, len(self.params), self.logL, pool=pool)
        #    sampler.run_mcmc(x0, nsamples, progress=progress)

        if not extend_chain: self.sampler = emcee.EnsembleSampler(nwalkers, len(self.params), logL)
        sampler = self.sampler
        sampler.run_mcmc(x0, nsamples, progress=progress)

        self.samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
        print('MCMC done. To see the results, call the show_mcmc() method.')

        # Create getdist samples --> save for later use
        from getdist import MCSamples
        self.gdsamples = MCSamples(samples=self.samples, names=[name for name in self.params], labels=[self.params[name]['latex'] for name in self.params]) 

        if fn_chain is not None: 
            print(f'Chain is saved in {fn_chain}')
            np.save(self.samples, fn_chain)
    
    def read_mcmc(self, fn_chain):
        """ Read mcmc already run """
        self.samples = np.load(fn_chain)

    def show_mcmc(self, params=None, add_expected_value=True, add_bestfit=True, params_fid=None, add_marker=None,
                  add_chains=None, legend_label=None, colors=None, fig_width_inch=5, title_limit=None, return_fig=False, fn_fig=None):
        r"""Makes a corner plot from MCMC results. Intended to be called from jupyter."""

        if not hasattr(self, 'gdsamples'):
            raise RuntimeError('Must call Likelihood.run_mcmc() before Likelihood.show_mcmc().')
        
        from getdist import plots

        # Triangle plot
        g = plots.get_subplot_plotter()
        g.settings.fig_width_inch = fig_width_inch
        g.settings.fontsize = 14
        g.settings.axes_labelsize = 14
        g.settings.axes_fontsize = 14
        g.settings.alpha_filled_add = 0.8
        g.settings.legend_loc = 'upper right'
        g.settings.legend_fontsize = 12
        g.settings.figure_legend_frame = False
        g.settings.legend_colored_text = False

        to_display = [self.gdsamples] if add_chains is None else [self.gdsamples] + add_chains
        contour_colors = colors
        line_args = None if colors is None else [{'color': cc} for cc in colors]

        g.triangle_plot(to_display, params, filled=True, legend_labels=legend_label, contour_colors=contour_colors, line_args=line_args, title_limit=title_limit, show=False)

        if add_expected_value:
            params = self.params if params is None else params
            # add expected value:
            params_fid = [self.params[name]['ref'] for name in params] if params_fid is None else params_fid
            bestfit = [self.bestfit[name] for name in params] if (hasattr(self, 'bestfit') and add_bestfit) else None
            marker = [add_marker[name] for name in params] if (add_marker is not None) else None
            for i in range(len(params)):
                for irow in range(i, len(params)):
                    if params_fid[i] is not None: g.subplots[irow, i].axvline(params_fid[i], color='grey', ls=(0, (2, 3)), lw=1)
                    if irow > i :
                        if params_fid[irow] is not None: g.subplots[irow, i].axhline(params_fid[irow], color='grey', ls=(0, (2, 3)), lw=1)
                        if (bestfit is not None): g.subplots[irow, i].scatter(bestfit[i], bestfit[irow], marker='*', color='red')
                        if (marker is not None): g.subplots[irow, i].scatter(marker[i], marker[irow], marker='*', color='black')
        if fn_fig is not None: plt.savefig(fn_fig)
        if return_fig: 
            return g
        else:
            plt.show()

    def show_table(self, sigfig=2, add_bestfit=False, fn_tab=None):
        from tabulate import tabulate

        if add_bestfit:
            tab = [["Parameter", "BestFit", "Mean", "Std", "Interval:1sigma"]]
            for i, name in enumerate(self.params):
                mean, std = np.mean(self.samples[:,i]), np.std(self.samples[:,i])
                lower, median, upper = np.percentile(self.samples[:,i], [15.87, 50, 84.13]) 
                tab += [[name, utils.std_notation(self.bestfit[name], sigfig), utils.std_notation(mean, sigfig), utils.std_notation(std, sigfig), 
                utils.std_notation(lower-mean, sigfig) + "/" + utils.std_notation(upper-mean, sigfig)]] 
        else:
            tab = [["Parameter", "Mean", "Std", "Interval:1sigma"]]
            for i, name in enumerate(self.params):
                mean, std = np.mean(self.samples[:,i]), np.std(self.samples[:,i])
                lower, median, upper = np.percentile(self.samples[:,i], [15.87, 50, 84.13]) 
                tab += [[name, utils.std_notation(mean, sigfig), utils.std_notation(std, sigfig), 
                utils.std_notation(lower-mean, sigfig) + "/" + utils.std_notation(upper-mean, sigfig)]] 

        print(tabulate(tab, headers="firstrow", tablefmt='rounded_outline'))

        if fn_tab is not None: 
            with open(fn_tab, "w") as f:
                f.write('Chi2 info: ' + ', '.join([f'{key}={value:2.3f}' for key, value in self.analyze_chi2().items()]) + '\n')
                f.write('\n')
                f.write(tabulate(tab, headers="firstrow", tablefmt='latex_booktabs'))


    def show_quantile(self):
        qlevels = [0.025, 0.16, 0.5, 0.84, 0.975]
        for i, key in enumerate(self.params):
            quantiles = np.quantile(self.samples[:,i], qlevels)
            print(f'\n{key}: ')
            for q,val in zip(qlevels, quantiles):
                s = f'  ({(val-quantiles[2]):+.03f})' if (q != 0.5) else ''
                print(f'{(100*q):.03f}% quantile: {val=:.03f}{s}')

    def analyze_chi2(self, params=None, ddof=None, force_compute_cov=False):
        r"""Computes a $\chi^2$ statistic, which compares data to model with given set of parameters.

        Returns ($\chi^2$, $N_{dof}$, $\chi^2$ / N_{dof}$, $p$-value).

        If params is None use the self.bestfit as default. params can be either a list of value or a dictionary.

        The ``ddof`` argument is used to compute the number of degrees of freedom, as
        ``ndof = nkbins - ddof``. If ``ddof=None``, then it will be equal to the
        number of nonzero (fnl, bias) params. (This is usually the correct choice.)

        force_compute_cov: If True, it evals the covariance matrix at either the besfit params or the provided params. If False, uses the covariance matrix used during the inference ie. at the fiducial parameters.

        Warning: Do not include percival factor in the covariance.
        """

        if params is None: params = self.bestfit
        if isinstance(params, list): params = {key: params[i] for i, key in enumerate(self.params)}
        
        mean, cov = self.mean_and_cov(**params, force_compute_cov=force_compute_cov)
        if hasattr(self, 'percival_factor'): cov = cov / self.percival_factor  # remove percival factor applied in mean_and_cov.
        if ddof is None: ddof = len(params)
        
        x = self.data - mean
        chi2 = np.dot(x, np.linalg.solve(cov,x))
        ndof = self.data.size - ddof
        pte = scipy.stats.chi2.sf(chi2, ndof)
        snr = np.sqrt(np.dot(mean, np.linalg.solve(cov, mean)))
        
        return {'chi2': float(chi2), 'ndof': int(ndof), 'red_chi2': float(chi2 / ndof), 'snr': float(snr), 'pte': float(pte)}


class Likelihood(BaseLikelihood):
    """ Likelihood for power spectrum inference."""
    def __init__(self, pout, 
                 params={'fnl': {'ref': 0, 'prior': [-100, 100], 'latex': r'$f_{\rm NL}^{\rm loc}$'}},
                 fields={'gv': {'freq': ['90', '150'], 'field': [1, 0], 'ell': [0, 1], 'name_params': {'fnl':'fnl', 'bv':'bv', 'bfg':'bfg'}}}, 
                 first_kbin={'gv': None}, last_kbin={'gv': None}, k_rebin={'gv': 1}, jeffreys_prior=False,
                 cov_fix_params=False, params_cov=None, rescale_cross_cov=None, cov_correction='hartlap-percival', 
                 cov_interp=False, interp_method='linear'):
        r""" 
        pout: KszPipeOutdir object. 
        params: dictionary of parameters to vary in the inference. Each parameter should have a 'ref' value, a 'prior' range and a 'latex' label for plotting. 
                Optionally, an 'interp' range can be provided for interpolation of the covariance matrix.
        fields: dictionary of fields to include in the likelihood ('gg', 'gv', 'vv'). For each field you need to provide a dictionary with the entries to call  
                pout.pgg_mean() / pout.pgv_mean() ... ('gv': 'freq', 'field', 'ell') and all field have 'name_params'. name_params should be a dictionary with the mapping between the parameter names used in the KszPipeOutdir methods and the parameter names used in the inference (i.e. in params). If you want to fix some parameters to a given value, you can provide a 'fix_params' entry with a dictionary of the parameters to fix and their values. For example, 'fix_params': {'bv': 1.0} will fix bv to 1.0 in the likelihood evaluation.
        first_kbin, last_kbin: dictionary with the same key entries than fields.
        jeffreys_prior: if True, include the Jeffreys prior in the likelihood. Not ready yet.

        cov_fix_params: if True, compute the covariance matrix only once at the fiducial value of the parameters (given in params_cov) and use it for all  
                        likelihood evaluations. This speeds up the inference but ignores the parameter dependence of the covariance matrix.
        params_cov: dictionary of parameter values at which to compute the covariance matrix if cov_fix_params is True.
        rescale_cross_cov: dictionary to use different parameters value for the cross-covariance terms between different fields.
                           example: {'gv_1_90-gv_1_150': {'snv1': snv90x150, 'snv2': snv90x150}, 'gv}

        cov_correction can be 'hartlap', 'percival', 'hartlap-percival' or None. WARNING: percival correction should not be applied when the chi2 is computed.
        """
        assert isinstance(pout, KszPipeOutdir)

        # check if we have first_kbin and last_kbin for each field:
        for name in fields: 
            assert (name in first_kbin) and (name in last_kbin)

        # check if parameters used for each field is defined in params:
        for name in fields: 
            for nn in fields[name]['name_params']:
                assert fields[name]['name_params'][nn] in params

        self.pout = pout
        self.first_kbin = first_kbin.copy()
        self.last_kbin = last_kbin.copy()
        self.k_rebin = k_rebin.copy()
        self.k_norebin = np.concatenate([pout.k[name[:2]][first_kbin[name]:last_kbin[name]] for name in fields])
        self.nk_norebin = [pout.k[name[:2]][first_kbin[name]:last_kbin[name]].size for name in fields]
        
        self.fields = fields.copy()

        self.k = [self._rebin_vector(pout.k[name[:2]][first_kbin[name]:last_kbin[name]], pout.k[name[:2]][first_kbin[name]:last_kbin[name]], k_rebin[name]) for name in fields]
        self.nk = [len(kk) for kk in self.k]
        self.k = np.concatenate(self.k)

        self.params = params.copy()
        for field in fields:
            for name in fields[field]['name_params'].values(): 
                assert name in params

        self.jeffreys_prior = jeffreys_prior
        if jeffreys_prior:
            print('ERROR: jeffreys_prior is not ready yet')
            sys.exit(8)

        data = []
        for i, field in enumerate(self.fields): 
            field_split = field.split('_')
            ff = self.fields[field].copy()
            _ = ff.pop('name_params')
            if 'fix_params' in ff: _ = ff.pop('fix_params')
            if 'derived_params' in ff: _ = ff.pop('derived_params')
            data_tmp = None
            if field_split[0] == 'gg':
                data_tmp = self.pout.pgg_data(**ff)[self.first_kbin[field]:self.last_kbin[field]]
            elif field_split[0] == 'gv':
                data_tmp = self.pout.pgv_data(**ff)[self.first_kbin[field]:self.last_kbin[field]]
            elif field_split[0] == 'vv':
                data_tmp = self.pout.pvv_data(**ff)[self.first_kbin[field]:self.last_kbin[field]]
            else:
                print(f"{field_split} in not in gg, gv, vv")
                sys.exit(2)
            kk = self.k_norebin[int(np.sum(self.nk_norebin[:i])):int(np.sum(self.nk_norebin[:i+1]))]
            data += [self._rebin_vector(data_tmp, kk, k_rebin=self.k_rebin[field])]
        self.data = np.concatenate(data)

        self.hartlap_factor, self.percival_factor = 1.0, 1.0
        if 'hartlap' in cov_correction:
            # See: https://arxiv.org/abs/astro-ph/0608064
            self.hartlap_factor = ((self.pout.nsurr - self.data.size - 2) / (self.pout.nsurr - 1))**(-1)
            print(f'Using Hartlap correction for covariance: {self.hartlap_factor=}')
        if 'percival' in cov_correction:
            # See: https://arxiv.org/pdf/1312.4841.
            A = 2 / ((self.pout.nsurr - self.data.size - 1)*(self.pout.nsurr - self.data.size - 4))
            B = (self.pout.nsurr - self.data.size - 2) / ((self.pout.nsurr - self.data.size - 1)*(self.pout.nsurr - self.data.size - 4))
            self.percival_factor = (1 + B*(self.data.size - len(self.params))) / (1 + A + B*(len(self.params) - 1))
            print(f'Using Percival correction for covariance: {self.percival_factor=}')
        # WARNING: percival_factor should NOT be applied when computing the SNR/chi2/...
        self.factor_cov_correction = self.hartlap_factor * self.percival_factor
        print(f'Total factor applied to the covariance matrix: {self.factor_cov_correction=}')

        self.cov_fix_params = cov_fix_params  # Speed up the mcmc by using covariance at fiducial value of the parameters.
        self.cov_interp = cov_interp
        self.interp_method = interp_method
        self.rescale_cross_cov = rescale_cross_cov

        if cov_fix_params:
            print(f'Precompute the covariance matrix with {params_cov=}')
            self.params_cov = params_cov.copy()
            _, cov = self.mean_and_cov(force_compute_cov=True, **params_cov)
            self.cov = cov
            self.cov_inv = np.linalg.inv(cov)
            self.logdet_cov = np.linalg.slogdet(cov)[1]
            # cholesky decomposition is faster (for large matrix) than linalg.inv !
            self.cov_cholesky = np.linalg.cholesky(cov)
        
        if not cov_fix_params and cov_interp: 
            print(f'Interpolate the Cholesky decomposition of the covariance matrix with RegularGridInterpolator and method={interp_method}')
            # Check if the interp range is as wide as the prior range:
            for name in self.params:
                prior, interp = params[name]['prior'], params[name]['interp']
                assert prior[0] >= interp[0], f'params[{name}]: {params[name]}'
                assert prior[1] <= interp[1], f'params[{name}]: {params[name]}'

            logdet_interp, cov_choleskk_interp = self.interpolate_cholesky(method=interp_method)
            self.logdet_interp = logdet_interp
            self.cov_cholesky_interp = lambda pp: utils.unflatten_cholesky(cov_choleskk_interp(pp), dim=self.k.size)

    def _rebin_vector(self, vect, k, k_rebin=1):
        " We take the average of every two points above k_rebin. k are assumed to be increasing ... "

        non_rebin_vect = vect[k < k_rebin]
        rebin_vect = vect[k >= k_rebin]
        
        if rebin_vect.size % 2 != 0:
            # we drop the last element, bye bye:
            rebin_vect = rebin_vect[:-1]
        
        rebin_vect = rebin_vect.reshape(-1, 2).mean(axis=1)

        return np.concatenate([non_rebin_vect, rebin_vect])

    def _rebin_cov(self, cov, k1, k2, k_rebin1=1, k_rebin2=1):
        """ We rebin the covariance matrix.."""
        A = cov[np.ix_(k1 < k_rebin1, k2 < k_rebin2)]
        B = cov[np.ix_(k1 < k_rebin1, k2 >= k_rebin2)]
        C = cov[np.ix_(k1 >= k_rebin1, k2 < k_rebin2)]
        D = cov[np.ix_(k1 >= k_rebin1, k2 >= k_rebin2)]
        
        if np.sum(k1>=k_rebin1) % 2 != 0:
            C = C[:-1, :]
            D = D[:-1, :]
        
        if np.sum(k2 >= k_rebin2) % 2 != 0:
            B = B[:, :-1]
            D = D[:, :-1]

        D = 1/4 * D.reshape(D.shape[0]//2, 2, D.shape[1]//2, 2).sum(axis=(1,3))
        B = 1/4 * B.reshape(B.shape[0], B.shape[1]//2, 2).sum(axis=2)
        C = 1/4 * C.reshape(C.shape[0]//2, 2, C.shape[1]).sum(axis=1)

        return np.block([[A, B], [C, D]]) 

    def mean_and_cov(self, force_compute_cov=False, return_cov=True, return_grad=False, **params):
        r""" Computes the model prediction and covariance matrix for the given parameters from the surrogates."""

        mu = []
        for i, field in enumerate(self.fields):
            field_split = field.split('_')
            ff = self.fields[field].copy()

            # use the value of the params to evaluate the theory:
            name_params = ff.pop('name_params')
            ff.update({nn: params[name_params[nn]] for nn in name_params})
            # add chosen default value for some parameters
            if 'fix_params' in ff:
                fix_params = ff.pop('fix_params')
                ff.update(fix_params)
            # add params that are derived from others:
            if 'derived_params' in ff:
                derived_params = ff.pop('derived_params')
                derived = self.compute_derived_params(derived_params, params)
                ff.update(derived)

            mu_tmp = None
            if field_split[0] == 'gg':
                mu_tmp = self.pout.pgg_mean(**ff)[self.first_kbin[field]:self.last_kbin[field]]
            elif field_split[0] == 'gv':
                mu_tmp = self.pout.pgv_mean(**ff)[self.first_kbin[field]:self.last_kbin[field]]
            elif field_split[0] == 'vv':
                mu_tmp = self.pout.pvv_mean(**ff)[self.first_kbin[field]:self.last_kbin[field]]
            kk = self.k_norebin[int(np.sum(self.nk_norebin[:i])):int(np.sum(self.nk_norebin[:i+1]))]
            mu += [self._rebin_vector(mu_tmp, kk, k_rebin=self.k_rebin[field])]

        mu = np.concatenate(mu)

        if not return_cov:
            return mu
        else:
            if self.cov_fix_params and not force_compute_cov:
                cov = self.cov
            else:
                cov = []
                for i, field1 in enumerate(self.fields): 
                    field1_split = field1.split('_')
                    for j, field2 in enumerate(self.fields): 
                        field2_split = field2.split('_')

                        ff = {f"{key}1": value for key, value in self.fields[field1].items()}
                        ff.update({f"{key}2": value for key, value in self.fields[field2].items()})

                        # use the value of the params to evaluate the covariance:
                        name_params1 = ff.pop('name_params1')
                        ff.update({nn+'1': params[name_params1[nn]] for nn in name_params1})
                        name_params2 = ff.pop('name_params2')
                        ff.update({nn+'2': params[name_params2[nn]] for nn in name_params2})
                        # add chosen default value for some parameters
                        if 'fix_params1' in ff:
                            fix_params = ff.pop('fix_params1')
                            ff.update({nn+'1': fix_params[nn]for nn in fix_params})
                        if 'fix_params2' in ff:
                            fix_params = ff.pop('fix_params2')
                            ff.update({nn+'2': fix_params[nn] for nn in fix_params})
                        if 'derived_params1' in ff:
                            derived_params = ff.pop('derived_params1')
                            derived = self.compute_derived_params(derived_params, params)
                            ff.update({nn+'1': derived[nn]for nn in derived_params})
                        if 'derived_params2' in ff:
                            derived_params = ff.pop('derived_params2')
                            derived = self.compute_derived_params(derived_params, params)
                            ff.update({nn+'2': derived[nn]for nn in derived_params})

                        if self.rescale_cross_cov is not None:
                            update_params = self.rescale_cross_cov.get(f"{field1}-{field2}", None)
                            if update_params is not None: ff.update(update_params)

                        # not super elegant... 
                        cov_tmp = None
                        if field1_split[0] == 'gg':
                            if field2_split[0] == 'gg':
                                cov_tmp = self.pout.pggxpgg_cov(**ff)[self.first_kbin[field1]:self.last_kbin[field1], self.first_kbin[field2]:self.last_kbin[field2]]
                            elif field2_split[0] == 'gv':
                                cov_tmp = self.pout.pggxpgv_cov(**ff)[self.first_kbin[field1]:self.last_kbin[field1], self.first_kbin[field2]:self.last_kbin[field2]]
                            elif field2_split[0] == 'vv':
                                cov_tmp = self.pout.pggxpvv_cov(**ff)[self.first_kbin[field1]:self.last_kbin[field1], self.first_kbin[field2]:self.last_kbin[field2]]
                        elif field1_split[0] == 'gv':
                            if field2_split[0] == 'gg':
                                cov_tmp = self.pout.pgvxpgg_cov(**ff)[self.first_kbin[field1]:self.last_kbin[field1], self.first_kbin[field2]:self.last_kbin[field2]]
                            elif field2_split[0] == 'gv':
                                cov_tmp = self.pout.pgvxpgv_cov(**ff)[self.first_kbin[field1]:self.last_kbin[field1], self.first_kbin[field2]:self.last_kbin[field2]]
                            elif field2_split[0] == 'vv':
                                cov_tmp = self.pout.pgvxpvv_cov(**ff)[self.first_kbin[field1]:self.last_kbin[field1], self.first_kbin[field2]:self.last_kbin[field2]]
                        elif field1_split[0] == 'vv':
                            if field2_split[0] == 'gg':
                                cov_tmp = self.pout.pvvxpgg_cov(**ff)[self.first_kbin[field1]:self.last_kbin[field1], self.first_kbin[field2]:self.last_kbin[field2]]
                            elif field2_split[0] == 'gv':
                                cov_tmp = self.pout.pvvxpgv_cov(**ff)[self.first_kbin[field1]:self.last_kbin[field1], self.first_kbin[field2]:self.last_kbin[field2]]
                            elif field2_split[0] == 'vv':
                                cov_tmp = self.pout.pvvxpvv_cov(**ff)[self.first_kbin[field1]:self.last_kbin[field1], self.first_kbin[field2]:self.last_kbin[field2]]
    
                        kk1 = self.k_norebin[int(np.sum(self.nk_norebin[:i])):int(np.sum(self.nk_norebin[:i+1]))]
                        kk2 = self.k_norebin[int(np.sum(self.nk_norebin[:j])):int(np.sum(self.nk_norebin[:j+1]))]
                        cov += [self._rebin_cov(cov_tmp, kk1, kk2, k_rebin1=self.k_rebin[field1], k_rebin2=self.k_rebin[field2])]
                        
                cov = np.block([[cov[i*len(self.fields) + j] for j in range(len(self.fields))] for i in range(len(self.fields))])
            
            cov = self.factor_cov_correction * cov  # apply the correction factor to the covariance matrix

            if not return_grad:
                return mu, cov

            else:
                print('NOT READY FOR NOW --> need to add dmu and dcov in KszPipeOutDir !')
                sys.exit(3)
                mu_grad = None
                cov_grad = None
                return mu, cov, mu_grad, cov_grad

    def snr_wo_noise(self, remove_foregrounds=False):
        """ Compute the SNR without noise. Useful to check if the non-noise signal is detected or not."""
        # Zero all parameters except those related to shotnoise:
        params_shotnoise = self.bestfit.copy()
        for key in params_shotnoise:
            if remove_foregrounds:
                if not (('sn' in key) or ('snv' in key) or ('bfg' in key)):    
                    params_shotnoise[key] = 0.0
            else:
                if not (('sn' in key) or ('snv' in key)):    
                    params_shotnoise[key] = 0.0
        # remove shotnoise / foregrounds: 
        shotnoise = self.mean_and_cov(**params_shotnoise)[0]
        mean, cov = self.mean_and_cov(**self.bestfit, force_compute_cov=True)
        mean -= shotnoise

        if hasattr(self, 'percival_factor'): cov = cov / self.percival_factor  # remove percival factor applied in mean_and_cov (percival needs to be applied only for the mcmc)

        return float(np.sqrt(np.dot(mean, np.linalg.solve(cov, mean))))

    def plot_data(self, params=None, add_bestfit=True, remove_shotnoise=False, remove_foregrounds=False, add_residuals=False, nsigma=3, fn_fig=None, return_fig=False):
            """ """
            data = self.data

            # Compute theory predicition:    
            bestfit = self.mean_and_cov(**self.bestfit)[0] if add_bestfit else None
            theory = self.mean_and_cov(**params)[0] if params is not None else None

            # Compute the errors:
            try: 
                cov = self.mean_and_cov(force_compute_cov=True, **self.bestfit)[1]
                err = np.sqrt(np.diag(cov))
            except:
                if params is not None: 
                    print(f'Use {params=}')
                    cov = self.mean_and_cov(force_compute_cov=True, **params)[1]
                    err = np.sqrt(np.diag(cov))
                else: 
                    print('no error display')
                    err = np.zeros_like(data)
            
            if remove_shotnoise or remove_foregrounds:
                # Zero all parameters except those related to shotnoise:
                params_to_remove = params.copy() if params is not None else self.bestfit.copy()
                for key in params_to_remove:
                    if remove_foregrounds:
                        if remove_shotnoise and not (('sn' in key) or ('snv' in key) or ('bfg' in key)):
                            params_to_remove[key] = 0.0
                        elif not remove_shotnoise and not ('bfg' in key):
                            params_to_remove[key] = 0.0
                    else:
                        if not (('sn' in key) or ('snv' in key)) :    
                            params_to_remove[key] = 0.0

                # remove shotnoise: 
                shotnoise_foreground = self.mean_and_cov(**params_to_remove)[0]
                data = data - shotnoise_foreground
                if bestfit is not None: bestfit = bestfit - shotnoise_foreground
                if theory is not None: theory = theory - shotnoise_foreground

            if add_residuals:
                fig, axs = plt.subplots(2, len(self.fields), figsize=(2.7*len(self.fields) + 1, 3.7), gridspec_kw={'height_ratios': [3, 1]}, sharex='col')
            else: 
                fig, axs = plt.subplots(1, len(self.fields), figsize=(2.7*len(self.fields) + 1, 2.7))

            for i, key in enumerate(self.fields):
                if len(self.fields) == 1:
                    ax = axs[0] if add_residuals else axs
                else:
                    ax = axs[0,i] if add_residuals else axs[i]

                start, end = np.sum(self.nk[:i], dtype='int'), np.sum(self.nk[:i+1], dtype='int')
                if 'gg' in key:
                    k_power = 0
                elif 'gv' in key:
                    k_power = 1
                else:
                    k_power = 1 if remove_shotnoise else 2

                # Should be remove once the estimator in kSZPIPE is corrected:
                correct_factor = (2*self.fields[key]['ell'][0] + 1) * (2*self.fields[key]['ell'][1] + 1)

                ax.errorbar(self.k[start:end], correct_factor*self.k[start:end]**k_power*data[start:end], correct_factor*self.k[start:end]**k_power*err[start:end], ls='', marker='.', label='data', zorder=10)
                if add_bestfit: ax.plot(self.k[start:end], correct_factor*self.k[start:end]**k_power*bestfit[start:end], label='bestfit', ls='--', c='r', zorder=0)
                if params is not None: ax.plot(self.k[start:end], correct_factor*self.k[start:end]**k_power*theory[start:end], ls=':', c='orange', zorder=1)

                if 'gg' in key:
                    ax.set_ylabel(r'$P^{gg}_{\ell=0}$ [Mpc$^3$]')
                    ax.set_xscale('log')
                    ax.set_yscale('log')
                elif 'gv' in key:
                    if self.fields[key]['ell'][1] == 0:
                        ax.set_ylabel(r'$k P^{gv}_{\ell=0}$ [Mpc$^2$]')
                    else:
                        ax.set_ylabel(r'$k P^{gv}_{\ell=1}$ [Mpc$^2$]')
                else:
                    if remove_shotnoise:
                        ax.set_ylabel(r'$k P^{vv}_{\ell=' + str(self.fields[key]['ell'][0]) +r',\ell=' + str(self.fields[key]['ell'][1]) + r'}$ [Mpc$^2$]')
                    else:
                        ax.set_ylabel(r'$k^2P^{vv}_{\ell=' + str(self.fields[key]['ell'][0]) +r',\ell=' + str(self.fields[key]['ell'][1]) + r'}$ [Mpc]')
                        ax.set_yscale('log')

                ax.yaxis.set_label_coords(-0.17, 0.5)
                if not add_residuals: ax.set_xlabel('$k$ [Mpc$^{-1}$]')
                if add_residuals: ax.set_xticks([])
                ax.legend()

                if add_residuals:
                    if len(self.fields) == 1:
                        ax = axs[1]
                    else:
                        ax = axs[1,i]

                    if add_bestfit: ax.plot(self.k[start:end], (data[start:end] - bestfit[start:end]) / err[start:end], ls=':', marker='.', zorder=10)
                    if params is not None: ax.plot(self.k[start:end], (data[start:end] - theory[start:end]) / err[start:end], ls=':', marker='.', zorder=10)

                    if 'gg' in key:
                        ax.set_ylabel(r'$\Delta P^{gg}_{\ell=0} / \sigma$')
                        ax.set_xscale('log')
                    elif 'gv' in key:
                        if self.fields[key]['ell'][1] == 0:
                            ax.set_ylabel(r'$\Delta P^{gv}_{\ell=0} / \sigma$')
                            ax.set_xscale('linear')
                        else:
                            ax.set_ylabel(r'$\Delta P^{gv}_{\ell=1} / \sigma$')
                            ax.set_xscale('linear')
                    else:
                        ax.set_ylabel(r'$\Delta P^{vv}_{\ell=' + str(self.fields[key]['ell'][0]) +r',\ell=' + str(self.fields[key]['ell'][1]) + r'} / \sigma$')
                        if remove_shotnoise: 
                            ax.set_xscale('linear')
                        else:
                            ax.set_xscale('log')

                    ax.set_xlabel('$k$ [Mpc$^{-1}$]')
                    ax.axhline(2, color='lightgrey', ls='--', lw=1, zorder=-1)
                    ax.axhline(0, color='k', ls='-', lw=1, zorder=-1)
                    ax.axhline(-2, color='lightgrey', ls='--', lw=1, zorder=-1)
                    ax.set_ylim(-nsigma, nsigma)
                    ax.axhspan(-1, 1, alpha=0.3, color='lightgray', zorder=0)

            #ax.set_ylabel('Label', loc='center')
            ax.yaxis.set_label_coords(-0.17, 0.5)

            plt.tight_layout(h_pad=0.5)
            if fn_fig is not None: plt.savefig(fn_fig)
            if return_fig: 
                return fig
            else:
                plt.show()

    def plot_residuals(self, params=None, nsigma=2.4, fn_fig=None, return_fig=False):
        """ Plot only the residuals. """
        # Compute theory predicition:
        params = self.bestfit if params is None else params
        theory = self.mean_and_cov(**params)[0]

        cov = self.mean_and_cov(force_compute_cov=True, **params)[1]
        err = np.sqrt(np.diag(cov))

        plt.figure(figsize=(2.7*len(self.fields) + 1, 2.7))
        for i, key in enumerate(self.fields):
            plt.subplot(1, len(self.fields), 1+i)

            start, end = np.sum(self.nk[:i], dtype='int'), np.sum(self.nk[:i+1], dtype='int')

            plt.plot(self.k[start:end], (self.data[start:end] - theory[start:end]) / err[start:end], ls=':', marker='.', zorder=10)

            if 'gg' in key:
                plt.ylabel(r'$\Delta P^{gg}_{\ell=0} / \sigma$')
                plt.xscale('log')
            elif 'gv' in key:
                if self.fields[key]['ell'][1] == 0:
                    plt.ylabel(r'$\Delta P^{gv}_{\ell=0} / \sigma$')
                    plt.xscale('log')
                else:
                    plt.ylabel(r'$\Delta P^{gv}_{\ell=1} / \sigma$')
            else:
                plt.ylabel(r'$\Delta P^{vv}_{\ell=' + str(self.fields[key]['ell'][0]) +r',\ell=' + str(self.fields[key]['ell'][1]) + r'} / \sigma$')
                #plt.xscale('log')

            plt.axhline(2, color='grey', ls='--', lw=1, zorder=0)
            plt.axhline(0, color='k', ls='-', lw=1, zorder=0)
            plt.axhline(-2, color='grey', ls='--', lw=1, zorder=0)

            plt.xlabel('$k$ [Mpc$^{-1}$]')
            plt.ylim(-nsigma, nsigma)

        plt.tight_layout()
        if fn_fig is not None: plt.savefig(fn_fig)
        if return_fig: 
            return plt.gcf()
        else:
            plt.show()

    def plot_cov(self, params=None, nticks=5, fn_fig=None, return_fig=False):
        """ """
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        if params is not None: 
            cov = self.mean_and_cov(force_compute_cov=True, **params)[1]
        else: 
            cov = self.mean_and_cov(force_compute_cov=True, **self.bestfit)[1]

        plt.figure(figsize=(2.2*len(self.fields) + 1, 2.2*len(self.fields) + 2))
        ax = plt.gca()
        im = ax.imshow(np.log10(np.abs(cov)))

        tmp = np.concatenate([np.array([0]), np.cumsum(self.nk)])
        idx = np.concatenate([np.arange(tmp[i], tmp[i+1], nticks) for i in range(len(self.fields))])

        ax.set_xticks(idx, [f'{self.k[i]:2.2f}' for i in idx])
        ax.set_yticks(idx, [f'{self.k[i]:2.2f}' for i in idx])

        ax.set_xlabel('$k$ [Mpc$^{-1}$]')
        ax.set_ylabel('$k$ [Mpc$^{-1}$]')

        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.1 inch.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)

        plt.colorbar(im, cax=cax, label=r'$\log(\vert cov \vert)$')
        plt.tight_layout()
        if fn_fig is not None: plt.savefig(fn_fig)
        if return_fig: 
            return plt.gcf()
        else:
            plt.show()

    def plot_corr(self, params=None, nticks=6, fn_fig=None, return_fig=False):
        """ """
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        if params is not None: 
            cov = self.mean_and_cov(force_compute_cov=True, **params)[1]
        else: 
            cov = self.mean_and_cov(force_compute_cov=True, **self.bestfit)[1]

        corr = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))

        plt.figure(figsize=(2.2*len(self.fields) + 1, 2.2*len(self.fields) + 2))
        ax = plt.gca()

        # Generate a custom diverging colormap: difficult to be nice :(
        import seaborn as sns
        cmap = sns.diverging_palette(220, 25, as_cmap=True)

        im = ax.imshow(corr, vmin=-1, vmax=1, cmap=cmap)

        tmp = np.concatenate([np.array([0]), np.cumsum(self.nk)])
        idx = np.concatenate([np.arange(tmp[i], tmp[i+1], nticks) for i in range(len(self.fields))])

        ax.set_xticks(idx, [f'{self.k[i]:2.3f}' for i in idx])
        ax.set_yticks(idx, [f'{self.k[i]:2.3f}' for i in idx])

        ax.set_xlabel('$k$ [Mpc$^{-1}$]')
        ax.set_ylabel('$k$ [Mpc$^{-1}$]')

        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.1 inch.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)

        plt.colorbar(im, cax=cax, label=r'R')
        plt.tight_layout()
        if fn_fig is not None: plt.savefig(fn_fig)
        if return_fig: 
            return plt.gcf()
        else:
            plt.show()

class CombineTracerLikelihood(Likelihood):
    def __init__(self, *likelihoods):
        """ Sum the different likelihoods, neglecting the cross-correlation between them ! """

        self.likelihoods = likelihoods

        self.fields = {}
        # I add a prefix to the field names to avoid duplicates and be able to plot all the data together:
        # This is a hacky trick only for plotting purpose, self.fields is not used in the computations.
        for i, lik in enumerate(self.likelihoods): 
            new_fields = {f'{i}_'+key: value for key, value in lik.fields.items()}
            self.fields.update(new_fields)

        self.params = {}
        for lik in self.likelihoods: self.params.update(lik.params)

        self.nk = np.concatenate([lik.nk for lik in self.likelihoods])
        self.k = np.concatenate([lik.k for lik in self.likelihoods])
        
        self.data = np.concatenate([lik.data for lik in self.likelihoods])

        self.cov_fix_params = all([lik.cov_fix_params for lik in self.likelihoods])
        self.jeffreys_prior = all([lik.jeffreys_prior for lik in self.likelihoods])

        if self.cov_fix_params:
            self.cov = scipy.linalg.block_diag(*[lik.cov for lik in self.likelihoods])
            #self.cov_inv = np.linalg.inv(self.cov)
            self.logdet_cov = np.linalg.slogdet(self.cov)[1]
            self.cov_cholesky = np.linalg.cholesky(self.cov)
 
    def mean_and_cov(self, force_compute_cov=False, return_cov=True, return_grad=False, **params): 
        mean, cov, grad = [], [], []
        for lik in self.likelihoods:
            to_unpack = lik.mean_and_cov(force_compute_cov=force_compute_cov, return_cov=return_cov, return_grad=return_grad, **params)
            if not return_cov and not return_grad: 
                mean.append(to_unpack)
            else:
                mean.append(to_unpack[0])
            if return_cov:  cov.append(to_unpack[1])
            if return_grad: grad.append(to_unpack[2])

        if not return_cov and not return_grad:
            return np.concatenate(mean)
        elif return_cov and not return_grad:
            return np.concatenate(mean), scipy.linalg.block_diag(*cov)
        else:
            import sys 
            print('NOT IMPLEMENTED YET')
            sys.exit(1)


""" DEVELOPMENT ..."""

# class FieldLevelLikelihood(BaseLikelihood):

# class NewLikelihood(Likelihood):
#     def __init__(self, lik1, lik2):
#         print('Not ready yet')
#     def mean_and_cov(self, force_compute_cov=False, return_cov=True, return_grad=False, **params): 
#            return mean, cov