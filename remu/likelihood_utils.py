from __future__ import division
import numpy as np
from matplotlib import pyplot as plt

def plot_bin_efficiencies(likelihood_machine, filename, plot_limits=False, bins_per_plot=20):
    """Plot bin by bin efficiencies.

    Uses Matplotlibs ``boxplot``, showing the median (line), quartiles
    (box), 5% and 95% percentile (error bars), and mean (point) of the
    efficiencies over all matrices.

    Parameters
    ----------

    filename : string
        Where to save the plot.
    plot_limits : bool, optional
        Also plot the truth limits for each bin on a second axis.
    bins_per_plot : int, optional
        How many bins are combined into a single plot.

    Returns
    -------

    fig : Figure
    ax : list of Axis

    """

    eff = likelihood_machine.response_matrix.sum(axis=-2)
    eff.shape = (np.prod(eff.shape[:-1], dtype=int), eff.shape[-1])
    if eff.shape[0] == 1:
        # Trick boxplot into working even if there is only one efficiency per bin
        eff = np.broadcast_to(eff, (2, eff.shape[1]))

    nplots = int(np.ceil(eff.shape[-1] / bins_per_plot))
    fig, ax= plt.subplots(nplots, squeeze=False, figsize=(8,nplots*3), sharey=True)
    ax = ax[:,0]
    for i in range(nplots):
        x = np.arange(i*bins_per_plot, min((i+1)*bins_per_plot, eff.shape[-1]), dtype=int)
        ax[i].set_ylabel("Efficiency")
        ax[i].boxplot(eff[:,i*bins_per_plot:(i+1)*bins_per_plot], whis=[5., 95.], sym='|', showmeans=True, whiskerprops={'linestyle': 'solid'}, positions=x)
        # TODO: Weighted plot for weighted matrices?
        if plot_limits:
            ax2 = ax[i].twinx()
            ax2.plot(x, likelihood_machine.truth_limits[i*bins_per_plot:(i+1)*bins_per_plot], drawstyle='steps-mid', color='green')
            ax2.set_ylabel("Truth limits")
    ax[-1].set_xlabel("Truth bin #")
    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename)

    return fig, ax

    def mcmc(self, composite_hypothesis, prior_only=False):
        """Return a Marcov Chain Monte Carlo object for the hypothesis.

        The hypothesis must define priors for its parameters.

        Parameters
        ----------

        composite_hypothesis : CompositeHypothesis
        prior_only : bool, optional
            Use only the prior infomation. Ignore the data. Useful for testing
            purposes.

        Notes
        -----

        See documentation of `PyMC` for a description of the `MCMC` class.

        See also
        --------

        plr
        likelihood_p_value
        max_likelihood_p_value
        max_likelihood_ratio_p_value

        """

        # Load pymc on demand
        global pymc
        if pymc is None:
            import pymc as _pymc
            pymc = _pymc

        priors = composite_hypothesis.parameter_priors

        names = composite_hypothesis.parameter_names
        if names is None:
            names = [ 'par_%d'%(i,) for i in range(len(priors)) ]

        # Toy index as additional stochastic
        n_toys = np.prod(self.response_matrix.shape[:-2])
        if self.log_matrix_weights is None:
            toy_index = pymc.DiscreteUniform('toy_index', lower=0, upper=(n_toys-1))
        else:
            p = np.exp(self.log_matrix_weights)
            p /= np.sum(p)
            toy_index = pymc.Categorical('toy_index', p=p)

        # The parameter pymc stochastics
        parameters = []
        names_priors = list(zip(names, priors))
        for n,p in names_priors:
            # Get default value of prior
            if isinstance(p, JeffreysPrior):
                # Jeffreys prior?
                default = p.default_values
                parents = {'toy_index': toy_index}
            else:
                # Regular function
                default = inspect.getargspec(p).defaults[0]
                parents = {}

            parameters.append(pymc.Stochastic(
                logp = p,
                doc = '',
                name = n,
                parents = parents,
                value = default,
                dtype=float))

        # The data likelihood
        if prior_only:
            def logp(value=self.data_vector, parameters=parameters, toy_index=toy_index):
                """Do not consider the data likelihood."""
                return 0
        else:
            def logp(value=self.data_vector, parameters=parameters, toy_index=toy_index):
                """The reconstructed data."""
                return self.log_likelihood(composite_hypothesis.translate(parameters), systematics=(toy_index,))
        data = pymc.Stochastic(
            logp = logp,
            doc = '',
            name = 'data',
            parents = {'parameters': parameters, 'toy_index': toy_index},
            value = self.data_vector,
            dtype = int,
            observed = True)

        M = pymc.MCMC({'data': data, 'parameters': parameters, 'toy_index': toy_index})
        M.use_step_method(pymc.DiscreteMetropolis, toy_index, proposal_distribution='Prior')

        return M

    def marginal_log_likelihood(self, composite_hypothesis, parameters, toy_indices):
        """Calculate the marginal likelihood.

        Parameters
        ----------

        composite_hypothesis : CompositeHypothesis
            The composite hypotheses for which the likelihood will be calculated.

        parameters : array like
            Array of parameter vectors, drawn from the prior or posterior distribution
            of the hypothesis, e.g. with the MCMC objects::

                parameters = [
                    [1.0, 2.0, 3.0],
                    [1.1, 1.9, 2.8],
                    ...
                    ]

        toy_indices, : array_like
            The corresponding systematic toy indices, in an array of equal
            dimensionality. That means, even if the toy index is just a single
            integer, it must be provided as arrays of length 1::

                toy_indices0 = [
                    [0],
                    [3],
                    ...
                    ]

        Returns
        -------

        L : float
            The marginal log-likelihood.

        Notes
        -----

        The marginal likelihood is used in the construction of bayes factors,
        when comparing the evidence in the data for two hypotheses::

            bayes_factor = marginal_likelihood0 / marginal_likelihood1

        or in the case of log-likelihoods::

            log_bayes_factor = marginal_log_likelihood0 - marginal_log_likelihood1

        """

        L = self.log_likelihood(composite_hypothesis.translate(parameters),
            systematics=toy_indices)
        return np.logaddexp.reduce(L) - np.log(len(L))

    def plr(self, H0, parameters0, toy_indices0, H1, parameters1, toy_indices1):
        """Calculate the Posterior distribution of the log Likelihood Ratio.

        Parameters
        ----------

        H0, H1 : CompositeHypothesis
            Composite Hypotheses to be compared.

        parameters0, parameters1 : array like
            Arrays of parameter vectors, drawn from the posterior distribution
            of the hypotheses, e.g. with the MCMC objects::

                parameters0 = [
                    [1.0, 2.0, 3.0],
                    [1.1, 1.9, 2.8],
                    ...
                    ]

        toy_indices0, toy_indices1 : array_like
            The corresponding systematic toy indices, in an array of equal
            dimensionality. That means, even if the toy index is just a single
            integer, it must be provided as arrays of length 1::

                toy_indices0 = [
                    [0],
                    [3],
                    ...
                    ]

        Returns
        -------

        PLR : ndarray
            A sample from the PLR as calculated from the parameter sets.
        model_preference : float
            The resulting model preference.

        Notes
        -----

        The model preference is calculated as the fraction of likelihood ratios
        in the PLR that prefer H1 over H0::

            model_preference = N(PLR > 0) / N(PLR)

        It can be interpreted as the posterior probability for the data
        prefering H1 over H0.

        The PLR is symmetric::

            PLR(H0, H1) = -PLR(H1, H0)
            preference(H0, H1) = 1. - preference(H1, H0) # modulo the cases of PLR = 0.

        See also
        --------

        mcmc

        """

        L0 = self.log_likelihood(H0.translate(parameters0), systematics=toy_indices0)
        L1 = self.log_likelihood(H1.translate(parameters1), systematics=toy_indices1)
        # Build all possible combinations
        # Assumes posteriors are independent, I guess
        PLR = np.array(np.meshgrid(L1, -L0)).T.sum(axis=-1).flatten()
        preference = float(np.count_nonzero(PLR > 0)) / PLR.size
        return PLR, preference

def plot_truth_bin_traces(likelihood_machine, filename, trace, plot_limits=False, bins_per_plot=20):
    """Plot bin by bin MCMC truth traces.

    Uses Matplotlibs ``boxplot``, showing the traces' median (line),
    quartiles (box), 5% and 95% percentile (error bars), and mean (point).

    Parameters
    ----------

    filename : string
        Where to save the plot.
    trace : array like
        The posterior trace of the truth bin values of an MCMC.
    plot_limits : bool or 'relative', optional
        Also plot the truth limits.
        If 'relative', the values are divided by the limits before plotting.
    bins_per_plot : int, optional
        How many bins are combined into a single plot.

    Returns
    -------

    fig : Figure
    ax : list of Axis

    See also
    --------

    mcmc
    plot_reco_bin_traces

    """

    trace = trace.reshape( (np.prod(trace.shape[:-1], dtype=int), trace.shape[-1]) )
    if trace.shape[0] == 1:
        # Trick boxplot into working even if there is only one trace entry per bin
        trace = np.broadcast_to(trace, (2, trace.shape[1]))

    if plot_limits == 'relative':
        trace = trace / np.where(likelihood_machine.truth_limits > 0, likelihood_machine.truth_limits, 1.)

    nplots = int(np.ceil(trace.shape[-1] / bins_per_plot))
    fig, ax= plt.subplots(nplots, squeeze=False, figsize=(8,nplots*3), sharey=True)
    ax = ax[:,0]
    for i in range(nplots):
        x = np.arange(i*bins_per_plot, min((i+1)*bins_per_plot, trace.shape[-1]), dtype=int)
        ax[i].boxplot(trace[:,i*bins_per_plot:(i+1)*bins_per_plot], whis=[5., 95.], sym='|', showmeans=True, whiskerprops={'linestyle': 'solid'}, positions=x)
        if plot_limits == 'relative':
            ax[i].set_ylabel("Value / Truth limit")
        else:
            ax[i].set_ylabel("Value")
            if plot_limits:
                ax[i].plot(x, likelihood_machine.truth_limits[i*bins_per_plot:(i+1)*bins_per_plot], drawstyle='steps-mid', color='green', label="Truth limit")
                ax[i].legend(loc='best')
    ax[-1].set_xlabel("Truth bin #")
    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename)

    return fig, ax

def plot_reco_bin_traces(likelihood_machine, filename, trace, toy_index=None, plot_data=False, bins_per_plot=20):
    """Plot bin by bin MCMC reco traces.

    Uses Matplotlibs ``boxplot``, showing the traces' median (line),
    quartiles (box), 5% and 95% percentile (error bars), and mean (point).

    Parameters
    ----------

    filename : string
        Where to save the plot.
    trace : array like
        The posterior trace of the *truth* bin values of an MCMC.
    toy_index : array like, optional
        The posterior trace of the chosen toy matrices of an MCMC.
    plot_data : bool or 'relative', optional
        Also plot the actual data content of the reco bins.
        If 'relative', the values are divided by the data before plotting.
    bins_per_plot : int, optional
        How many bins are combined into a single plot.

    Returns
    -------

    fig : Figure
    ax : list of Axis

    See also
    --------

    mcmc
    plot_truth_bin_traces

    """

    resp = likelihood_machine._reduced_response_matrix
    if toy_index is not None:
        resp = resp[toy_index,...]

    trace = likelihood_machine._reduce_truth_vector(trace)[...,np.newaxis,:]
    trace = np.einsum('...i,...i->...', resp, trace)

    # Reshape for boxplotting
    trace = trace.reshape( (np.prod(trace.shape[:-1], dtype=int), trace.shape[-1]) )

    # Trick boxplot into working even if there is only one trace entry per bin
    if trace.shape[0] == 1:
        trace = np.broadcast_to(trace, (2,) + trace.shape[1:])

    if plot_data == 'relative':
        trace = trace / np.where(likelihood_machine.data_vector > 0, likelihood_machine.data_vector, 1.)

    nplots = int(np.ceil(trace.shape[-1] / bins_per_plot))
    fig, ax= plt.subplots(nplots, squeeze=False, figsize=(8,nplots*3), sharey=True)
    ax = ax[:,0]
    for i in range(nplots):
        x = np.arange(i*bins_per_plot, min((i+1)*bins_per_plot, trace.shape[-1]), dtype=int)
        ax[i].boxplot(trace[:,i*bins_per_plot:(i+1)*bins_per_plot], whis=[5., 95.], sym='|', showmeans=True, whiskerprops={'linestyle': 'solid'}, positions=x)
        if plot_data == 'relative':
            ax[i].set_ylabel("Value / Data")
        else:
            ax[i].set_ylabel("Value")
            if plot_data:
                ax[i].plot(x, likelihood_machine.data_vector[i*bins_per_plot:(i+1)*bins_per_plot], drawstyle='steps-mid', color='green', label="Data")
                ax[i].legend(loc='best')
    ax[-1].set_xlabel("Reco bin #")
    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename)

    return fig, ax

def pseudo_chi2(self, truth_vector):
    """Calculate the pseudo chi2 goodness of fit of a parameter set.

    This calculates the likelihood ratio between the given data and the
    best possible data fit to the truth vector::

        -2 * ln(L(data) / L(best possible data))

    It should *not* be confused with the ratio of likelihoods when
    maximising over a composite hypothesis parameter space!

    See also
    --------

    max_likelihood_p_valuey
    max_likelihood_ratio_p_valuey
    wilks_max_likelihood_ratio_p_value

    """

    best_LL = self.log_likelihood.data_model.max

    return 2. * (self.best_possible_log_likelihood(truth_vector) - self.log_likelihood(truth_vector))
