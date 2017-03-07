"""Example analysis using somewhat realistic data.

This exampla analysis shows how to do a Bayesian analysis of (somewhat)
realistic neutrino interaction data. The data consists of the true and
reconstructed muon momentum and angle in a given selection and the true
values of all simulated events.
"""

from __future__ import print_function

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
        N_toy = 20
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
    # HACK: Re-fill the truth binning to simulate the detector efficiency
    response._truth_binning.reset()
    response._truth_binning.fill_from_csv_file("neutrino_data/trueEvents.csv")

    print("Getting normalised response matrix...")
    true_resp = response.get_response_matrix_as_ndarray()
    # Fake a 10% efficiency uncertainty
    resp = np.array([true_resp * x for x in np.linspace(0.9, 1.1, 10)])
    eff = resp.sum(axis=1)
    min_eff = eff.min(axis=0)
    max_eff = eff.max(axis=0)
    i_eff = (min_eff > 0.2)

    print("Creating test data...")
    truth = response.get_truth_values_as_ndarray() / toy_div
    eff_truth = truth[i_eff]
    data = likelihood.LikelihoodMachine.generate_random_data_sample(true_resp, truth, N_toy)

    print("Creating likelihood machines...")
    lm = [ likelihood.LikelihoodMachine(x, resp) for x in data ]

    print("Calculating absolute maximum likelihood...")
    def lm_maxlog(x):
        return x.absolute_max_log_likelihood(kwargs={'niter':20})
    pool = Pool()
    ret = pool.map(lm_maxlog, lm)
    del pool

    print("Calculating posterior probabilities...")
    def prior(value=1.):
        return -np.inf if value < 0. else -0.5*np.log(value) # P = 1/sqrt(lambda), Jeffreys prior for Poisson expectation values
    # Free floating hypothesis
    prior = likelihood.JeffreysPrior(resp, lambda x: x, [(0,None)]*len(truth), [100.]*len(truth))
    H = likelihood.CompositeHypothesis(lambda x: x, parameter_priors=[prior])
    M = [ lm[i].MCMC(H) for i in range(N_toy) ]
    def f_MCMC(i):
        # Do the actual MCMC sampling
        if args.quicktest or args.test:
            M[i].sample(100*1000, burn=10*1000, thin=1000, tune_interval=1000, tune_throughout=True, progress_bar=True)
        else:
            M[i].sample(1000*1000, burn=100*1000, thin=10*1000, tune_interval=1000, tune_throughout=True, progress_bar=False)
        # Get all traces and save them in an array
        # Must be done here, because MCMC object are not pickleable
        ret = {}
        for par in M[i].stochastics:
            ret[str(par)] = M[i].trace(par)[:]
        # Parameters are namedd 'par_i'
        arr = np.array( ret['par_0'].T )
        toy_trace = np.array(ret['toy_index'])
        return arr, toy_trace
    pool = Pool()
    trace, toy_index = zip(*pool.map(f_MCMC, range(N_toy)))
    del pool
    #trace, toy_index = zip(*map(f_MCMC, range(N_toy)))
    trace = np.array(trace)
    toy_index = np.array(toy_index)
    median = np.median(trace, axis=-1)
    mean = np.mean(trace, axis=-1)
    total = np.sum(trace, axis=-2)
    eff_total = np.sum(trace[:,i_eff,:], axis=-2)

    print("Plotting results...")
    print("- response")
    response.plot_values('response.png', variables=(None, None))

    print("- migration")
    response.plot_values('migration.png', variables=(['muMomRec','muCosThetaRec'], ['muMomTrue','muCosThetaTrue']))

    print("- efficiencies")
    lm[0].plot_bin_efficiencies('efficiencies.png')

    print("- parameter example traces")
    # Debug plots to check convergence of MCMC
    for i, t in enumerate(trace[0]):
        mcplot(t, 'par_%d'%(i,), last=False)
        fig = plt.gcf()
        ax = fig.axes[-1]
        ax.axvline(truth[i], linewidth=2, linestyle='dashed', color='r')
        fig.savefig('par_%d.png'%(i,))
    mcplot(toy_index[0], 'toy_index')

    percentiles = [0.5, 2.5, 16., 50., 84., 97.5, 99.5]
    col = ['red', 'orange', 'green', 'black', 'green', 'orange', 'red']

    print("- posterior median")
    figax = None
    figax = truth_binning.plot_ndarray('posterior-median-truth.png', truth, figax=figax,
                kwargs1d={'linestyle': 'dashed', 'color': 'c', 'linewidth': 2., 'label': "Truth", 'zorder': 1})
    median_percentiles = np.percentile(H.translate(median), percentiles, axis=0)
    for p, m, c in zip(percentiles, median_percentiles, col):
        figax = truth_binning.plot_ndarray('posterior-median-truth.png', m, figax=figax,
                    kwargs1d={'linestyle': 'solid', 'linewidth': 2.0, 'color': c, 'label': "$P_\mathrm{post}$ median %.1f%%ile"%(p,), 'zorder': -abs(p-50.)})

    print("- posterior mean")
    figax = None
    figax = truth_binning.plot_ndarray('posterior-mean-truth.png', truth, figax=figax,
                kwargs1d={'linestyle': 'dashed', 'color': 'c', 'linewidth': 2., 'label': "Truth", 'zorder': 1})
    mean_percentiles = np.percentile(H.translate(median), percentiles, axis=0)
    for p, m, c in zip(percentiles, mean_percentiles, col):
        figax = truth_binning.plot_ndarray('posterior-mean-truth.png', m, figax=figax,
                    kwargs1d={'linestyle': 'solid', 'linewidth': 2.0, 'color': c, 'label': "$P_\mathrm{post}$ mean %.1f%%ile"%(p,), 'zorder': -abs(p-50.)})

    print("- total")
    fig, ax = plt.subplots()
    ax.set_xlabel("Fake data throw")
    ax.set_ylabel("Total true events")
    #ax.hist2d(np.indices(total.shape)[0].flatten(), total.flatten(), [len(total), 20])
    ax.boxplot(total.T, whis=[2.5, 97.5], showmeans=True, whiskerprops={'linestyle': 'solid'})
    ax.axhline(np.sum(truth), color='c', linewidth=2., linestyle='dashed', label="Truth")
    ax.legend(loc='best', framealpha=0.5)
    fig.savefig('total.png')

    print("- eff total")
    fig, ax = plt.subplots()
    ax.set_xlabel("Fake data throw")
    ax.set_ylabel("Total true efficient events")
    #ax.hist2d(np.indices(eff_total.shape)[0].flatten(), eff_total.flatten(), [len(eff_total), 20])
    ax.boxplot(eff_total.T, whis=[2.5, 97.5], showmeans=True, whiskerprops={'linestyle': 'solid'})
    ax.axhline(np.sum(eff_truth), color='c', linewidth=2., linestyle='dashed', label="Truth")
    ax.legend(loc='best', framealpha=0.5)
    fig.savefig('eff_total.png')

    print("- max likelihood")
    figax = None
    figax = truth_binning.plot_ndarray('maxL-truth.png', truth, figax=figax,
                kwargs1d={'linestyle': 'dashed', 'color': 'c', 'linewidth': 2., 'label': "Truth", 'zorder': 1})
    maxlik_percentiles = np.percentile([H.translate(r.x) for r in ret], percentiles, axis=0)
    for p, m, c in zip(percentiles, maxlik_percentiles, col):
        figax = truth_binning.plot_ndarray('maxL-truth.png', m, figax=figax,
                    kwargs1d={'linestyle': 'solid', 'linewidth': 2.0, 'color': c, 'label': "$L_\mathrm{max}$ %.1f%%ile"%(p,), 'zorder': -abs(p-50.)})
