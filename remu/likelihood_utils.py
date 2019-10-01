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
