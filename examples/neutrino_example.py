"""Example analysis using somewhat realistic data.

This example analysis shows how to do a Bayesian analysis of (somewhat)
realistic neutrino interaction data. The data consists of the true and
reconstructed muon momentum and angle in a given selection and the true
values of all simulated events.
"""

from __future__ import print_function
from __future__ import division
from six.moves import map, zip

import numpy as np
from matplotlib import pyplot as plt
from pymc.Matplot import plot as mcplot
from pymc.Matplot import summary_plot

# Set working directory and sys.path
import sys, os
pathname = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), '..'))
sys.path.append(pathname)
os.chdir(os.path.dirname(sys.argv[0]))

import argparse

import binning
import migration
import likelihood

# Parallelization
from multiprocessing import Pool

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", help="do a quick test-run", action='store_true')
    parser.add_argument("--quicktest", help="do a quicker test-run", action='store_true')
    args = parser.parse_args()

    # We split the MC into parts for training and testing
    if args.quicktest:
        N_toy = 2
    elif args.test:
        N_toy = 10
    else:
        N_toy = 100
    toy_div = 100

    print("Loading the truth and reco binnings...")

    with open("neutrino_data/truth-binning.yml", 'r') as f:
        truth_binning = binning.yaml.load(f)

    with open("neutrino_data/reco-binning.yml", 'r') as f:
        reco_binning = binning.yaml.load(f)

    print("Creating a response matrix object...")
    response = migration.ResponseMatrix(reco_binning, truth_binning)

    print("Filling the response matrix...")
    response.fill_from_csv_file("neutrino_data/selectedEvents.csv")
    #response.fill_up_truth_from_csv_file("neutrino_data/trueEvents.csv")
    response.fill_up_truth_from_csv_file("neutrino_data/trueEvents_forward.csv")

    print("Getting normalised response matrix...")
    true_resp = response.get_response_matrix_as_ndarray()
    # Fake a 10% efficiency uncertainty
    resp = np.array([true_resp * x for x in np.linspace(0.9, 1.1, 10)])
    eff = resp.sum(axis=1)
    i_signal = [4, 5, 7, 8, 10, 11]

    print("Creating test data...")
    truth = response.get_truth_values_as_ndarray() / toy_div
    signal_truth = truth[i_signal]
    data = likelihood.LikelihoodMachine.generate_random_data_sample(true_resp, truth, N_toy)

    print("Creating likelihood machines...")
    lm = [ likelihood.LikelihoodMachine(x, resp) for x in data ]

    print("Calculating absolute maximum likelihood...")
    def lm_maxlog(x):
        return x.absolute_max_log_likelihood(kwargs={'niter':20}) # 'niter' should be increased for higher chance of finding true maximum
    pool = Pool()
    ret = pool.map(lm_maxlog, lm)
    del pool

    print("Calculating posterior probabilities...")
    ntruth = len(truth)

    # Super hypothesis
    def trans(par):
        """Free floating truth bins vs given shape.

        Just return the parameters 1-to-1 as truth bin values"""
        return par
    prior = likelihood.JeffreysPrior(resp, trans, [(0,None)]*len(truth), [100.]*len(truth), total_truth_limit=6000)
    H0 = likelihood.CompositeHypothesis(trans, parameter_priors=[prior])

    # Alternative hypothesis
    def alt_trans(par):
        """Fixed shape with free scale parameter."""
        par = np.asarray(par)
        shape = np.concatenate((par.shape[:-1], np.asarray(truth).shape))
        ret = np.broadcast_to(truth, shape)
        ret = (par[...,0] * ret.T).T
        return ret
    alt_prior = likelihood.JeffreysPrior(resp, alt_trans, [(0,None)], [1.], total_truth_limit=6000)
    H1 = likelihood.CompositeHypothesis(alt_trans, parameter_priors=[alt_prior])

    # Yet another alternative hypothesis
    def alt_trans2(par):
        """Fixed shape with free scale parameter."""
        par = np.asarray(par)
        shape = tuple(list(par.shape[:-1]) + list(np.asarray(truth).shape))
        ret = np.broadcast_to(truth, shape)
        ret = (par[...,0] * ret.T).T
        ret[...,7] *= 0.5
        ret[...,8] *= 0.5
        return ret
    alt_prior2 = likelihood.JeffreysPrior(resp, alt_trans2, [(0,None)], [1.], total_truth_limit=6000)
    H2 = likelihood.CompositeHypothesis(alt_trans2, parameter_priors=[alt_prior2])

    M = [ lm[i].mcmc(H0) for i in range(N_toy) ] # Super hypothesis
    M += [ lm[i].mcmc(H1) for i in range(N_toy) ] # Alternative hypothesis
    M += [ lm[i].mcmc(H2) for i in range(N_toy) ] # Alternative hypothesis
    M.insert(0, lm[0].mcmc(H0, prior_only=True)) # Add MCMC to sample prior
    def f_MCMC(i):
        # Do the actual MCMC sampling
        if args.quicktest:
            M[i].sample(200*100, burn=100*100, thin=100, tune_interval=1000, tune_throughout=True, progress_bar=True)
        elif args.test:
            M[i].sample(400*100, burn=100*100, thin=100, tune_interval=1000, tune_throughout=True, progress_bar=True)
        else:
            M[i].sample(1000*100, burn=100*100, thin=100, tune_interval=1000, tune_throughout=True, progress_bar=False)
        # Get all traces and save them in an array
        # Must be done here, because MCMC object are not pickleable
        ret = {}
        for par in M[i].stochastics:
            ret[str(par)] = M[i].trace(par)[:]
        # Parameters are combined in 'par_0'
        arr = np.array( ret['par_0'].T )
        toy_trace = np.array(ret['toy_index'])
        return arr, toy_trace
    pool = Pool()
    all_trace, all_toy_index = zip(*pool.map(f_MCMC, range(3*N_toy+1)))
    del pool
    #trace, toy_index = zip(*map(f_MCMC, range(N_toy)))
    prior_trace = np.array(all_trace[0])
    trace = np.array(all_trace[1:N_toy+1])
    alt_trace = np.array(all_trace[N_toy+1:2*N_toy+1])
    alt_trace2 = np.array(all_trace[2*N_toy+1:])
    prior_toy_index = np.array(all_toy_index[0])
    toy_index = np.array(all_toy_index[1:N_toy+1])
    alt_toy_index = np.array(all_toy_index[N_toy+1:2*N_toy+1])
    alt_toy_index2 = np.array(all_toy_index[2*N_toy+1:])
    median = np.median(trace, axis=-1)
    mean = np.mean(trace, axis=-1)
    total = np.sum(trace, axis=-2)
    signal_total = np.sum(trace[:,i_signal,:], axis=-2)

    print("Calculating Posterior distribution of Likelihood Ratio...")
    def f_PLR(i):
        return lm[i].plr(H0, trace[i].T, toy_index[i,...,np.newaxis], H1, alt_trace[i].T, alt_toy_index[i,...,np.newaxis])
    def f_PLR2(i):
        return lm[i].plr(H0, trace[i].T, toy_index[i,...,np.newaxis], H2, alt_trace2[i].T, alt_toy_index2[i,...,np.newaxis])
    pool = Pool()
    PLR, pref = zip(*pool.map(f_PLR, range(N_toy)))
    PLR2, pref2 = zip(*pool.map(f_PLR2, range(N_toy)))
    del pool
    PLR = np.array(PLR)
    PLR2 = np.array(PLR2)

    print("Plotting results...")
    print("- response")
    response.plot_values('response.png', variables=(None, None))

    print("- migration")
    response.plot_values('migration.png', variables=(['muMomRec','muCosThetaRec'], ['muMomTrue','muCosThetaTrue']))

    print("- efficiencies")
    lm[0].plot_bin_efficiencies('efficiencies.png')

    print("- parameter traces")
    # Debug plots to check convergence of MCMC
    for i, t in enumerate(prior_trace):
        mcplot(t, 'prior_par_%d'%(i,), last=False)
        fig = plt.gcf()
        ax = fig.axes[-1]
        ax.axvline(truth[i], linewidth=2, linestyle='dashed', color='r')
        fig.savefig('prior_par_%d.png'%(i,))
    mcplot(prior_toy_index, 'prior_toy_index')
    for i, t in enumerate(trace[0]):
        mcplot(t, 'par_%d'%(i,), last=False)
        fig = plt.gcf()
        ax = fig.axes[-1]
        ax.axvline(truth[i], linewidth=2, linestyle='dashed', color='r')
        fig.savefig('par_%d.png'%(i,))
    mcplot(toy_index[0], 'toy_index')
    for i, t in enumerate(alt_trace[0]):
        mcplot(t, 'alt_par_%d'%(i,), last=False)
        fig = plt.gcf()
        ax = fig.axes[-1]
        fig.savefig('alt_par_%d.png'%(i,))
    mcplot(alt_toy_index[0], 'alt_toy_index')
    for i, t in enumerate(alt_trace2[0]):
        mcplot(t, 'alt_par2_%d'%(i,), last=False)
        fig = plt.gcf()
        ax = fig.axes[-1]
        fig.savefig('alt_par2_%d.png'%(i,))
    mcplot(alt_toy_index2[0], 'alt_toy_index2')

    percentiles = [0.5, 2.5, 16., 50., 84., 97.5, 99.5]
    col = ['red', 'orange', 'green', 'black', 'green', 'orange', 'red']

    print("- posterior median")
    figax = None
    figax = truth_binning.plot_ndarray('posterior-median-truth.png', truth, figax=figax,
                kwargs1d={'linestyle': 'dashed', 'color': 'c', 'linewidth': 2., 'label': "Truth", 'zorder': 1})
    median_percentiles = np.percentile(H0.translate(median), percentiles, axis=0)
    for p, m, c in zip(percentiles, median_percentiles, col):
        figax = truth_binning.plot_ndarray('posterior-median-truth.png', m, figax=figax,
                    kwargs1d={'linestyle': 'solid', 'linewidth': 2.0, 'color': c, 'label': "$P_\mathrm{post}$ median %.1f%%ile"%(p,), 'zorder': -abs(p-50.)})

    print("- posterior mean")
    figax = None
    figax = truth_binning.plot_ndarray('posterior-mean-truth.png', truth, figax=figax,
                kwargs1d={'linestyle': 'dashed', 'color': 'c', 'linewidth': 2., 'label': "Truth", 'zorder': 1})
    mean_percentiles = np.percentile(H0.translate(median), percentiles, axis=0)
    for p, m, c in zip(percentiles, mean_percentiles, col):
        figax = truth_binning.plot_ndarray('posterior-mean-truth.png', m, figax=figax,
                    kwargs1d={'linestyle': 'solid', 'linewidth': 2.0, 'color': c, 'label': "$P_\mathrm{post}$ mean %.1f%%ile"%(p,), 'zorder': -abs(p-50.)})

    print("- total")
    fig, ax = plt.subplots()
    ax.set_xlabel("Fake data throw")
    ax.set_ylabel("Total true events")
    #ax.hist2d(np.indices(total.shape)[0].flatten(), total.flatten(), [len(total), 20])
    ax.boxplot(total.T, whis=[5., 95.], sym='|', showmeans=True, whiskerprops={'linestyle': 'solid'})
    ax.axhline(np.sum(truth), color='orange', linewidth=2., linestyle='dashed', label="Truth")
    ax.legend(loc='best', framealpha=0.5)
    fig.tight_layout()
    fig.savefig('total.png')

    print("- signal total")
    fig, ax = plt.subplots()
    ax.set_xlabel("Fake data throw")
    ax.set_ylabel("Total true signal events")
    #ax.hist2d(np.indices(signal_total.shape)[0].flatten(), signal_total.flatten(), [len(signal_total), 20])
    ax.boxplot(signal_total.T, whis=[5., 95.], sym='|', showmeans=True, whiskerprops={'linestyle': 'solid'})
    ax.axhline(np.sum(signal_truth), color='orange', linewidth=2., linestyle='dashed', label="Truth")
    ax.legend(loc='best', framealpha=0.5)
    fig.tight_layout()
    fig.savefig('signal_total.png')

    print("- Posterior distribution of Likelihood Ratios")
    fig, ax = plt.subplots(1)
    ax.set_xlabel("Fake data throw")
    ax.set_ylabel("PLR")
    ax.boxplot(PLR.T, whis=[5., 95.], sym='|', showmeans=True, whiskerprops={'linestyle': 'solid'})
    fig.tight_layout()
    fig.savefig('PLR.png')
    fig, ax = plt.subplots(1)
    ax.set_xlabel("Fake data throw")
    ax.set_ylabel("PLR")
    ax.boxplot(PLR2.T, whis=[5., 95.], sym='|', showmeans=True, whiskerprops={'linestyle': 'solid'})
    fig.tight_layout()
    fig.savefig('PLR2.png')

    print("- model preferences")
    fig, ax = plt.subplots(1)
    ax.set_xlabel("model preference")
    ax.hist(pref, 20, (0.0, 1.0), histtype='step', hatch='/', label='Scaled truth')
    ax.hist(pref2, 20, (0.0, 1.0), histtype='step', hatch='\\', label='Scaled other')
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig('model_preference.png')

    print("- max likelihood")
    figax = None
    figax = truth_binning.plot_ndarray('maxL-truth.png', truth, figax=figax,
                kwargs1d={'linestyle': 'dashed', 'color': 'c', 'linewidth': 2., 'label': "Truth", 'zorder': 1})
    maxlik_percentiles = np.percentile([H0.translate(r.x) for r in ret], percentiles, axis=0)
    for p, m, c in zip(percentiles, maxlik_percentiles, col):
        figax = truth_binning.plot_ndarray('maxL-truth.png', m, figax=figax,
                    kwargs1d={'linestyle': 'solid', 'linewidth': 2.0, 'color': c, 'label': "$L_\mathrm{max}$ %.1f%%ile"%(p,), 'zorder': -abs(p-50.)})
