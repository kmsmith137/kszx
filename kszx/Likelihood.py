import sys

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

from . import utils

from .KszPipe import KszPipeOutdir


class Likelihood:
    def __init__(self, pout, 
                 params={'fnl': {'ref': 0, 'prior': [-100, 100], 'latex': r'$f_{\rm NL}^{\rm loc}$'}},
                 fields={'gv': {'freq': ['90', '150'], 'field': [1, 0], 'ell': [0, 1], 'name_params': {'fnl':'fnl', 'bv':'bv', 'bfg':'bfg'}}}, 
                 first_kbin={'gv': None}, last_kbin={'gv': None}, jeffreys_prior=False,
                 cov_fix_params=False, params_cov=None, cov_correction='hartlap-percival', 
                 cov_interp=False, interp_method='linear'):
        r""" TODO. 
        
        name_params should be in params !
        
        first_kbin and last_kbin should have the same key entries than fields. --> data[first_kbin, last_kbin] if None, None, use all the data vector !
        
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
        self.first_kbin = first_kbin
        self.last_kbin = last_kbin
        self.k = np.concatenate([pout.k[name[:2]][first_kbin[name]:last_kbin[name]] for name in fields])
        self.nk = [pout.k[name[:2]][first_kbin[name]:last_kbin[name]].size for name in fields]
        self.fields = fields

        self.params = params
        for field in fields:
            for name in fields[field]['name_params'].values(): 
                assert name in params

        self.jeffreys_prior = jeffreys_prior
        if jeffreys_prior:
            print('ERROR: jeffreys_prior is not ready yet')
            sys.exit(8)

        data = []
        for field in self.fields: 
            field_split = field.split('_')
            ff = self.fields[field].copy()
            _ = ff.pop('name_params')
            if 'fix_params' in ff: _ = ff.pop('fix_params')
            if field_split[0] == 'gg':
                data += [self.pout.pgg_data(**ff)[self.first_kbin[field]:self.last_kbin[field]]]
            elif field_split[0] == 'gv':
                data += [self.pout.pgv_data(**ff)[self.first_kbin[field]:self.last_kbin[field]]]
            elif field_split[0] == 'vv':
                data += [self.pout.pvv_data(**ff)[self.first_kbin[field]:self.last_kbin[field]]]
            else:
                print(f"{field_split} in not in gg, gv, vv")
                sys.exit(2)
        self.data = np.concatenate(data)

        self.factor_cov_correction = 1.0
        if 'hartlap' in cov_correction:
            # See: https://arxiv.org/abs/astro-ph/0608064
            hartlap_factor = ((self.pout.nsurr - self.data.size - 2) / (self.pout.nsurr - 1))**(-1)
            print(f'Using Hartlap correction for covariance: {hartlap_factor=}')
            self.factor_cov_correction *= hartlap_factor
        if 'percival' in cov_correction:
            # See: https://arxiv.org/pdf/1312.4841
            A = 2 / ((self.pout.nsurr - self.data.size - 1)*(self.pout.nsurr - self.data.size - 4))
            B = (self.pout.nsurr - self.data.size - 2) / ((self.pout.nsurr - self.data.size - 1)*(self.pout.nsurr - self.data.size - 4))  
            percival_factor = (1 + B*(self.data.size - len(self.params))) / (1 + A + B*(len(self.params) - 1))
            print(f'Using Percival correction for covariance: {percival_factor=}')
            self.factor_cov_correction *= percival_factor

        self.cov_fix_params = cov_fix_params  # Speed up the mcmc by using covariance at fiducial value of the parameters.
        self.cov_interp = cov_interp

        if cov_fix_params:
            print(f'Precompute the covariance matrix with {params_cov=}')
            self.params_cov = params_cov
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

    def mean_and_cov(self, force_compute_cov=False, return_cov=True, return_grad=False, **params):
        r""" TODO. """

        mu = []
        for field in self.fields: 
            field_split = field.split('_')
            ff = self.fields[field].copy()

            # use the value of the params to evaluate the theory:
            name_params = ff.pop('name_params')
            ff.update({nn: params[name_params[nn]] for nn in name_params})
            # add chosen default value for some parameters
            if 'fix_params' in ff:
                fix_params = ff.pop('fix_params')
                ff.update(fix_params)

            if field_split[0] == 'gg':
                mu += [self.pout.pgg_mean(**ff)[self.first_kbin[field]:self.last_kbin[field]]]
            elif field_split[0] == 'gv':
                mu += [self.pout.pgv_mean(**ff)[self.first_kbin[field]:self.last_kbin[field]]]
            elif field_split[0] == 'vv':
                mu += [self.pout.pvv_mean(**ff)[self.first_kbin[field]:self.last_kbin[field]]]
        mu = np.concatenate(mu)

        if not return_cov:
            return mu
        else:
            if self.cov_fix_params and not force_compute_cov:
                cov = self.cov
            else:
                cov = []
                for field1 in self.fields: 
                    field1_split = field1.split('_')
                    for field2 in self.fields: 
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

                        # not super elegant... 
                        if field1_split[0] == 'gg':
                            if field2_split[0] == 'gg':
                                cov = [self.pout.pggxpgg_cov(**ff)[self.first_kbin[field1]:self.last_kbin[field1], self.first_kbin[field2]:self.last_kbin[field2]]]
                            elif field2_split[0] == 'gv':
                                cov += [self.pout.pggxpgv_cov(**ff)[self.first_kbin[field1]:self.last_kbin[field1], self.first_kbin[field2]:self.last_kbin[field2]]]
                            elif field2_split[0] == 'vv':
                                cov += [self.pout.pggxpvv_cov(**ff)[self.first_kbin[field1]:self.last_kbin[field1], self.first_kbin[field2]:self.last_kbin[field2]]]
                        elif field1_split[0] == 'gv':
                            if field2_split[0] == 'gg':
                                cov += [self.pout.pgvxpgg_cov(**ff)[self.first_kbin[field1]:self.last_kbin[field1], self.first_kbin[field2]:self.last_kbin[field2]]]
                            elif field2_split[0] == 'gv':
                                cov += [self.pout.pgvxpgv_cov(**ff)[self.first_kbin[field1]:self.last_kbin[field1], self.first_kbin[field2]:self.last_kbin[field2]]]
                            elif field2_split[0] == 'vv':
                                cov += [self.pout.pgvxpvv_cov(**ff)[self.first_kbin[field1]:self.last_kbin[field1], self.first_kbin[field2]:self.last_kbin[field2]]]
                        elif field1_split[0] == 'vv':
                            if field2_split[0] == 'gg':
                                cov += [self.pout.pvvxpgg_cov(**ff)[self.first_kbin[field1]:self.last_kbin[field1], self.first_kbin[field2]:self.last_kbin[field2]]]
                            elif field2_split[0] == 'gv':
                                cov += [self.pout.pvvxpgv_cov(**ff)[self.first_kbin[field1]:self.last_kbin[field1], self.first_kbin[field2]:self.last_kbin[field2]]]
                            elif field2_split[0] == 'vv':
                                cov += [self.pout.pvvxpvv_cov(**ff)[self.first_kbin[field1]:self.last_kbin[field1], self.first_kbin[field2]:self.last_kbin[field2]]]
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


    def run_profiling(self, nprofiles=5, fn_profile=None, verbose=True):
        r"""Returns bestfit value for self.params after nprofiles different profiling with scipy.optimize.minimize.  """

        x0 = np.zeros((nprofiles, len(self.params)))
        for i, key in enumerate(self.params):
            x0[:, i] = np.random.uniform(self.params[key]['prior'][0], self.params[key]['prior'][1], size=nprofiles)

        f = lambda x: - self.log_likelihood(*x)  # note minus sign

        results = []
        for i in range(nprofiles):
            results += [scipy.optimize.minimize(f, x0[i,:], method='Nelder-Mead')]
        
        # select the bestfit with the lowest log_likelihood:
        sel_min = np.argmin([result.fun for result in results])
        self.result = results[sel_min]
        if verbose: print(self.result)
        self.bestfit = {key: self.result.x[i] for i, key in enumerate(self.params)}
        if fn_profile is not None: np.save(self.result, fn_profile)

        return self.bestfit      

    def run_mcmc(self, nwalkers=8, nsamples=10000, discard=1000, thin=5, fn_chain=None):
        r"""Initializes ``self.samples`` to an array of shape (N,len(self.params)) where N is large."""
        import emcee
        print(f'MCMC start: {nwalkers=}, {nsamples=}, {discard=}, {thin=}')

        x0 = np.zeros((nwalkers, len(self.params)))
        for i, key in enumerate(self.params):
            x0[:, i] = np.random.uniform(self.params[key]['prior'][0], self.params[key]['prior'][1], size=nwalkers)

        logL = lambda x: self.log_likelihood(*x)

        sampler = emcee.EnsembleSampler(nwalkers, len(self.params), logL)
        sampler.run_mcmc(x0, nsamples)
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

    def show_mcmc(self, add_expected_value=True, legend_label=None, fn_fig=None):
        r"""Makes a corner plot from MCMC results. Intended to be called from jupyter."""

        if not hasattr(self, 'gdsamples'):
            raise RuntimeError('Must call Likelihood.run_mcmc() before Likelihood.show_mcmc().')
        
        from getdist import plots

        # Triangle plot
        g = plots.get_subplot_plotter()
        g.settings.fig_width_inch = 5
        g.settings.fontsize = 14
        g.settings.legend_fontsize = 12
        g.settings.figure_legend_frame = False
        g.settings.axes_labelsize = 14
        g.settings.axes_fontsize = 14
        g.settings.alpha_filled_add = 0.8
        g.settings.legend_loc = 'upper right'
        g.settings.legend_colored_text = True

        g.triangle_plot(self.gdsamples, filled=True, legend_labels=legend_label, show=False)

        if add_expected_value:
            # add expected value:
            params_fid = [self.params[name]['ref'] for name in self.params]
            for i in range(len(self.params)):
                for irow in range(i, len(self.params)):
                    if params_fid[i] is not None: g.subplots[irow, i].axvline(params_fid[i], color='grey', ls=(0, (2, 3)), lw=1)
                    if irow > i :
                        if params_fid[irow] is not None: g.subplots[irow, i].axhline(params_fid[irow], color='grey', ls=(0, (2, 3)), lw=1)

        if fn_fig is not None: plt.savefig(fn_fig)
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
        if fn_tab is not None: np.savetxt(tab, fn_tab)

    def show_quantile(self):
        qlevels = [0.025, 0.16, 0.5, 0.84, 0.975]
        for i, key in enumerate(self.params):
            quantiles = np.quantile(self.samples[:,i], qlevels)
            print(f'\n{key}: ')
            for q,val in zip(qlevels, quantiles):
                s = f'  ({(val-quantiles[2]):+.03f})' if (q != 0.5) else ''
                print(f'{(100*q):.03f}% quantile: {val=:.03f}{s}')

            print(f'\nSNR: {self.compute_snr():.03f}')

    def compute_snr(self, params=None):
        r"""Returns total SNR at the bestfit parameters. 
            # This is not the definition used previoulsy, do we want to change ? """
        
        if params is None: params = self.bestfit
        if isinstance(params, list): params = {key: params[i] for i, key in enumerate(self.params)}
        mean, cov = self.mean_and_cov(**params)

        return np.sqrt(np.dot(mean, np.linalg.solve(cov, mean)))

    def analyze_chi2(self, params=None, ddof=None):
        r"""Computes a $\chi^2$ statistic, which compares data to model with given set of parameters.

        Returns ($\chi^2$, $N_{dof}$, $\chi^2$ / N_{dof}$, $p$-value).

        If params is None use the self.bestfit as default. params can be either a list of value or a dictionary.

        The ``ddof`` argument is used to compute the number of degrees of freedom, as
        ``ndof = nkbins - ddof``. If ``ddof=None``, then it will be equal to the
        number of nonzero (fnl, bias) params. (This is usually the correct choice.)
        """

        if params is None: params = self.bestfit
        if isinstance(params, list): params = {key: params[i] for i, key in enumerate(self.params)}
        
        mean, cov = self.mean_and_cov(**params)
        if ddof is None: ddof = len(params)
        
        x = self.data - mean
        chi2 = np.dot(x, np.linalg.solve(cov,x))
        ndof = self.data.size - ddof
        pte = scipy.stats.chi2.sf(chi2, ndof)
        snr = self.compute_snr(params)
        
        return chi2, ndof, chi2 / ndof, snr, pte

    def plot_data(self, params=None, add_bestfit=True, fn_fig=None):
        """ """
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
                err = np.zeros(self.data)
        
        plt.figure(figsize=(3.5*len(self.fields) + 1, 3))
        for i, key in enumerate(self.fields):
            plt.subplot(1, len(self.fields), 1+i)

            start, end = np.sum(self.nk[:i], dtype='int'), np.sum(self.nk[:i+1], dtype='int')
            if 'gg' in key:
                k_power = 0
            elif 'gv' in key:
                k_power = 0 if self.fields[key]['ell'][1] == 0 else 1
            else:
                k_power = 2

            plt.errorbar(self.k[start:end], self.k[start:end]**k_power*self.data[start:end], self.k[start:end]**k_power*err[start:end], ls='', marker='.', label='data', zorder=10)
            if add_bestfit: plt.plot(self.k[start:end], self.k[start:end]**k_power*bestfit[start:end], label='bestfit', ls='--', c='r', zorder=0)
            if params is not None: plt.plot(self.k[start:end], self.k[start:end]**k_power*theory[start:end], ls=':', c='orange', zorder=1)

            if 'gg' in key:
                plt.ylabel('$P^{gg}_{\ell=0}$ [Mpc$^3$]')
                plt.xscale('log')
                plt.yscale('log')
            elif 'gv' in key:
                if self.fields[key]['ell'][1] == 0:
                    plt.ylabel('$P^{gv}_{\ell=0}$ [Mpc$^3$]')
                    plt.xscale('log')
                    plt.yscale('log')
                else:
                    plt.ylabel('$k P^{gv}_{\ell=1}$ [Mpc$^2$]')
            else:
                plt.ylabel('$k^2P^{vv}_{\ell=' + str(self.fields[key]['ell'][0]) +',\ell=' + str(self.fields[key]['ell'][1]) + '}$ $[Mpc^3]$')
                plt.xscale('log')
                plt.yscale('log')
            plt.xlabel('$k$ [Mpc$^{-1}$]')
            plt.legend()
        plt.tight_layout()
        if fn_fig is not None: plt.savefig(fn_fig)
        plt.show()

    def plot_cov(self, params=None, nticks=5, fn_fig=None):
        """ """
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        if params is not None: 
            cov = self.mean_and_cov(force_compute_cov=True, **params)[1]
        else: 
            cov = self.mean_and_cov(force_compute_cov=True, **self.bestfit)[1]

        plt.figure(figsize=(2.5*len(self.fields) + 1, 2.5*len(self.fields) + 2))
        ax = plt.gca()
        im = ax.imshow(np.log10(np.abs(cov)))

        idx = np.arange(0, len(self.k), len(self.k) // (nticks*len(self.fields)))
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
        plt.show()

    def plot_corr(self, params=None, nticks=5, fn_fig=None):
        """ """
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        if params is not None: 
            cov = self.mean_and_cov(force_compute_cov=True, **params)[1]
        else: 
            cov = self.mean_and_cov(force_compute_cov=True, **self.bestfit)[1]

        corr = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))

        plt.figure(figsize=(2.5*len(self.fields) + 1, 2.5*len(self.fields) + 2))
        ax = plt.gca()

        # Generate a custom diverging colormap: difficult to be nice :(
        import seaborn as sns
        cmap = sns.diverging_palette(220, 25, as_cmap=True)

        im = ax.imshow(corr, vmin=-1, vmax=1, cmap=cmap)

        idx = np.arange(0, len(self.k), len(self.k) // (nticks*len(self.fields)))
        ax.set_xticks(idx, [f'{self.k[i]:2.2f}' for i in idx])
        ax.set_yticks(idx, [f'{self.k[i]:2.2f}' for i in idx])

        ax.set_xlabel('$k$ [Mpc$^{-1}$]')
        ax.set_ylabel('$k$ [Mpc$^{-1}$]')

        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.1 inch.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)

        plt.colorbar(im, cax=cax, label=r'R')
        plt.tight_layout()
        if fn_fig is not None: plt.savefig(fn_fig)
        plt.show()


class CombineRegionLikelihood(Likelihood):
    def __init__(self, lik1, lik2, f1=None, params_for_f=None):

        assert np.all(lik1.k == lik2.k), "k bins must be the same for both likelihoods"
        self.k = lik1.k
        self.nk = lik1.nk
        assert lik1.fields == lik2.fields, "Fields must be the same for both likelihoods"
        self.fields = lik1.fields
        assert lik1.params == lik2.params, "Parameters must be the same for both likelihoods"
        self.params = lik1.params
        assert lik1.jeffreys_prior == lik2.jeffreys_prior, "Jeffreys priors must be the same for both likelihoods"
        self.jeffreys_prior = lik1.jeffreys_prior

        self.lik1 = lik1
        self.lik2 = lik2

        if f1 is None:
            if params_for_f is None:
                print('Warning: f1 and params_for_f are not provided, using default values of 0.5 for both likelihoods and for each fields. Otherwise, provide params_for_f..')
                f1, f2 = np.array([0.5]*np.sum(self.nk)), np.array([0.5]*np.sum(self.nk))
            else:
                print('Compute f1 for each fields as 1 / (1 + np.mean((np.diag(cov1) / np.diag(cov2))))')
                _, cov1 = lik1.mean_and_cov(**params_for_f)
                _, cov2 = lik2.mean_and_cov(**params_for_f)
                f1 = [1 / (1 + np.mean(np.diag(cov1)[sum(self.nk[:i]):sum(self.nk[:i+1])] / np.diag(cov2)[sum(self.nk[:i]):sum(self.nk[:i+1])])) for i in range(len(self.fields))]
                print(f"{f1=}")
                f1 = np.array(np.concatenate([[f]*self.nk[i] for i, f in enumerate(f1)]))
                f2 = 1 - f1
        else: 
            assert type(f1) is not float, 'f1 should be a list of lenght equal to the number of fields, for automatic computation pass f1=None' 
            f1 = np.array(np.concatenate([[f]*self.nk[i] for i, f in enumerate(f1)]))
            f2 = 1 - f1

        self.f1, self.f2 = f1, f2
        self.f1_cov, self.f2_cov = self.f1[:,None]*self.f1[None,:], self.f2[:,None]*self.f2[None,:]

        assert lik1.cov_fix_params == lik2.cov_fix_params, "cov_fix_params must be the same for both likelihoods"
        self.cov_fix_params = lik1.cov_fix_params 
        if self.cov_fix_params: 
            assert np.all(lik1.params_cov == lik2.params_cov), "params_cov must be the same for both likelihoods"
            self.cov = self.f1_cov*lik1.cov + self.f2_cov*lik2.cov
            self.cov_inv = np.linalg.inv(self.cov)
            self.logdet_cov = np.linalg.slogdet(self.cov)[1]
            # cholesky decomposition is faster (for large matrix) than linalg.inv !
            self.cov_cholesky = np.linalg.cholesky(self.cov)

        assert lik1.cov_interp == lik2.cov_interp, "cov_interp must be the same for both likelihoods"
        assert lik1.interp_method == lik2.interp_method, "interp_method must be the same for both likelihoods"
        self.cov_interp, self.interp_method = lik1.cov_interp, lik1.interp_method
        if not self.cov_fix_params and self.cov_interp: 
            logdet_interp, cov_choleskk_interp = self.interpolate_cholesky(method=interp_method)
            self.logdet_interp = logdet_interp
            self.cov_cholesky_interp = lambda pp: utils.unflatten_cholesky(cov_choleskk_interp(pp), dim=self.k.size)

        assert lik1.factor_cov_correction == lik2.factor_cov_correction, "factor_cov_correction must be the same for both likelihoods (hints: use same number of surrogates for both likelihoods)"
        #In this case, factor_cov_correction is the same for the combine likelihoods than for the two likelihoods, so we can just leave as it is.
        #self.factor_cov_correction = lik1.factor_cov_correction

        self.data = self.f1*lik1.data + self.f2*lik2.data

    def mean_and_cov(self, force_compute_cov=False, return_cov=True, return_grad=False, **params): 
        if return_grad:
            print('Not ready yet')
            sys.exit(31)

        to_unpack1 = self.lik1.mean_and_cov(force_compute_cov=force_compute_cov, return_cov=return_cov, return_grad=return_grad, **params)
        to_unpack1 = self.lik2.mean_and_cov(force_compute_cov=force_compute_cov, return_cov=return_cov, return_grad=return_grad, **params)

        if return_cov:
            mean1, cov1 = to_unpack1
            mean2, cov2 = to_unpack2
            mean = self.f1*mean1 + self.f2*mean2
            cov = self.f1_cov*cov1 + self.f2_cov*cov2
            return mean, cov

        else:
            mean1, mean2 = to_unpack1, to_unpack2
            mean = self.f1*mean1 + self.f2*mean2
            return mean


# class NewLikelihood(Likelihood):
#     def __init__(self, lik1, lik2):

#         self.something = something

        # if fix_cov: self.cov ... (pour le loglikelihood? non le reecrire ca sera plus facile ?!)

#     def mean_and_cov(self, force_compute_cov=False, return_cov=True, return_grad=False, **params): 
#            return mean, cov