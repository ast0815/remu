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

# Set working directory and sys.path
import sys, os
pathname = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), '..'))
sys.path.append(pathname)
os.chdir(os.path.dirname(sys.argv[0]))

import binning
import migration
import likelihood

# Parallelization
from multiprocessing import Pool

# We split the MC into parts for training and testing
N_toy = 4
toy_div = 100

if __name__ == '__main__':

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
    resp = response.get_response_matrix_as_ndarray()

    print("Creating test data...")
    truth = response.get_truth_values_as_ndarray() / toy_div
    data = likelihood.LikelihoodMachine.generate_random_data_sample(resp, truth, N_toy)

    print("Creating likelihood machines...")
    lm = [ likelihood.LikelihoodMachine(x, resp) for x in data ]

    print("Calculating absolute maximum likelihood...")
    def lm_maxlog(x):
        return x.absolute_max_log_likelihood(kwargs={'niter':100})
    pool = Pool()
    ret = pool.map(lm_maxlog, lm)
    del pool

    print("Calculating posterior probabilities...")
    def prior(value=1.):
        return -np.inf if value < 0. else -0.5*np.log(value) # P = 1/sqrt(lambda), Jeffreys prior for Poisson expectation values
    # Free floating hypothesis
    H = [ likelihood.CompositeHypothesis(lambda x: x, parameter_priors=[prior]*len(truth)) for i in range(N_toy) ]
    M = [ lm[i].MCMC(H[i]) for i in range(N_toy) ]
    def f_MCMC(i):
        M[i].sample(10000, burn=5000, thin=10, tune_interval=100, progress_bar=False)
        # Debug plots to check convergence of MCMC
        mcplot(M[i])
        # Get all traces and save them in an array
        # Must be done here, because MCMC object are not pickleable
        ret = {}
        for par in M[i].stochastics:
            ret[str(par)] = M[i].trace(par)[:]
        # Parameters are namedd 'par_i'
        ret = np.array([ ret['par_%d'%(j,)] for j in range(len(truth)) ])
        return ret
    pool = Pool()
    trace = np.array(pool.map(f_MCMC, range(N_toy)))
    del pool
    median = np.median(trace, axis=-1)
    total = np.sum(trace, axis=-2)

    print("Plotting results...")
    print("- response")
    response.plot_values('response.png', variables=(None, None))

    print("- migration")
    response.plot_values('migration.png', variables=(['muMomRec','muCosThetaRec'], ['muMomTrue','muCosThetaTrue']))

    print("- max likelihood")
    col = ['r', 'g', 'b', 'm']
    figax = None
    figax = truth_binning.plot_ndarray('maxL-truth.png', truth, figax=figax,
                kwargs1d={'linestyle': 'solid', 'color': 'k', 'label': "Truth"})
    for i in range(N_toy):
        figax = truth_binning.plot_ndarray('maxL-truth.png', H[i].translate(ret[i].x), figax=figax,
                    kwargs1d={'linestyle': 'dashed', 'linewidth': 2.0, 'color': col[i], 'label': "$L_\mathrm{max,%02d}$"%(i,)})
    figax = None
    for i in range(N_toy):
        figax = reco_binning.plot_ndarray('maxL-data.png', data[i], figax=figax,
                    kwargs1d={'linestyle': 'solid', 'color': col[i], 'label': "Data$_{%02d}$"%(i,)})
        figax = reco_binning.plot_ndarray('maxL-data.png', resp.dot(H[i].translate(ret[i].x)), figax=figax,
                    kwargs1d={'linestyle': 'dashed', 'linewidth': 2.0, 'color': col[i], 'label': "$L_\mathrm{max,%02d}$"%(i,)})

    print("- median posterior")
    figax = None
    figax = truth_binning.plot_ndarray('posterior-truth.png', truth, figax=figax,
                kwargs1d={'linestyle': 'solid', 'color': 'k', 'label': "Truth"})
    for i in range(N_toy):
        figax = truth_binning.plot_ndarray('posterior-truth.png', H[i].translate(median[i]), figax=figax,
                    kwargs1d={'linestyle': 'dashed', 'linewidth': 2.0, 'color': col[i], 'label': "median $P_\mathrm{post,%02d}$"%(i,)})
    figax = None
    for i in range(N_toy):
        figax = reco_binning.plot_ndarray('posterior-data.png', data[i], figax=figax,
                    kwargs1d={'linestyle': 'solid', 'color': col[i], 'label': "Data$_{%02d}$"%(i,)})
        figax = reco_binning.plot_ndarray('posterior-data.png', resp.dot(H[i].translate(median[i])), figax=figax,
                    kwargs1d={'linestyle': 'dashed', 'linewidth': 2.0, 'color': col[i], 'label': "median $P_\mathrm{post,%02d}$"%(i,)})


    fig, ax = plt.subplots()
    ax.set_xlabel("total true events")
    ax.hist(total.T, histtype='step', color=col, label=["$P_\mathrm{post,%02d}$"%(i,) for i in range(N_toy)])
    ax.axvline(np.sum(truth), color='k', linewidth=2., linestyle='dotted', label="Truth")
    ax.legend(loc='best', framealpha=0.5)
    fig.savefig('total.png')
