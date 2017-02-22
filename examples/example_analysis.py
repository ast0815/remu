from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plt

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

# Performance profiling
import timeit
from bprofile import BProfile
profile = BProfile('call_profile.png')

if __name__ == '__main__':

    print("Loading the truth and reco binnings...")
    with open("toydata/toy-truth-binning.yml", 'r') as f:
        truth_binning = binning.yaml.load(f)
    with open("toydata/toy-reco-binning.yml", 'r') as f:
        reco_binning = binning.yaml.load(f)

    print("Creating a response matrix object...")
    response = migration.ResponseMatrix(reco_binning, truth_binning)

    print("Filling the response matrix...")
    response.fill_from_csv_file("toydata/model1_detector1.csv")

    print("Getting normalised response matrix...")
    resp = response.get_response_matrix_as_ndarray()

    print("Loading test data...")
    reco_binning.reset()
    reco_binning.fill_from_csv_file("toydata/reco1_detector1.csv")
    data1 = reco_binning.get_entries_as_ndarray()
    data1_x0 = reco_binning.project(['x_0_reco']).get_entries_as_ndarray()
    data1_x1 = reco_binning.project(['x_1_reco']).get_entries_as_ndarray()
    reco_binning.reset()
    reco_binning.fill_from_csv_file("toydata/reco2_detector1.csv")
    data2 = reco_binning.get_entries_as_ndarray()
    data2_x0 = reco_binning.project(['x_0_reco']).get_entries_as_ndarray()
    data2_x1 = reco_binning.project(['x_1_reco']).get_entries_as_ndarray()

    print("Loading test models...")
    truth_binning.reset()
    truth_binning.fill_from_csv_file("toydata/truth1.csv")
    truth1 = truth_binning.get_entries_as_ndarray()
    truth1_x0 = truth_binning.project(['x_0_truth']).get_entries_as_ndarray()
    truth1_x1 = truth_binning.project(['x_1_truth']).get_entries_as_ndarray()
    truth_binning.reset()
    truth_binning.fill_from_csv_file("toydata/truth2.csv")
    truth2 = truth_binning.get_entries_as_ndarray()
    truth2_x0 = truth_binning.project(['x_0_truth']).get_entries_as_ndarray()
    truth2_x1 = truth_binning.project(['x_1_truth']).get_entries_as_ndarray()

    print("Creating likelihood machines...")
    lm1 = likelihood.LikelihoodMachine(data1, resp)
    lm2 = likelihood.LikelihoodMachine(data2, resp)

    print("Calculating absolute maximum likelihood...")
    start_time = timeit.default_timer()
    def lm_maxlog(lm):
        return lm.absolute_max_log_likelihood(kwargs={'niter':10})
    pool = Pool()
    ret1, ret2 = pool.map(lm_maxlog, [lm1, lm2])
    del pool
    elapsed = timeit.default_timer() - start_time
    print("Time: %.1f"%(elapsed,))

    print("Calculating p-values...")
    start_time = timeit.default_timer()
    print("Data1: N=%.1f, ll=%.1f, p=%.3f"%(np.sum(ret1.x), ret1.L, lm1.likelihood_p_value(ret1.x)))
    print("Data2: N=%.1f, ll=%.1f, p=%.3f"%(np.sum(ret2.x), ret2.L, lm2.likelihood_p_value(ret2.x)))
    elapsed = timeit.default_timer() - start_time
    print("Time: %.1f"%(elapsed,))

    print("Calculating event number limits...")
    def shape_to_truth(shape, total):
        # Shape integral
        s = np.sum(shape)
        # Calculate missing bin from integral of shape
        x = np.exp(-s)
        truth = np.append(shape, x)
        truth *= total / np.sum(truth)

        return truth

    eff = lm1._eff
    n_eff = lm1._n_eff

    class TransFun(object):
        """Helper class to translate shape parameters to truth vectors."""

        def __init__(self, total):
            self.total = total

        def __call__(self, x):
            truth = np.zeros_like(truth1, dtype=float)
            truth[eff] = shape_to_truth(x, self.total)
            return truth

    def hyp_factory(total):
        return likelihood.CompositeHypothesis(TransFun(total), [(0.,None)]*(n_eff-1))

    test_total = np.linspace(950, 1050, 11)
    test_hyp = map(hyp_factory, test_total)

    start_time = timeit.default_timer()
    def pair_maxlog(pair):
        return pair[0].max_log_likelihood(pair[1], kwargs={'niter':20})
    pool = Pool()
    test_ret = pool.map(pair_maxlog, [(lm1,h) for h in test_hyp])
    del pool
    elapsed = timeit.default_timer() - start_time
    print("Time: %.1f"%(elapsed,))

    print("Plotting results...")
    print("- 'response.png'")
    response.plot_values('response.png', variables=(None, None))

    print("- 'reco1.png'")
    figax = reco_binning.plot_ndarray('reco1.png', data1,
                kwargs1d={'linestyle': 'solid', 'label': "Data"})
    figax = reco_binning.plot_ndarray('reco1.png', resp.dot(ret1.x), figax=figax,
                kwargs1d={'linestyle': 'dashed', 'linewidth': 2.0, 'color': 'r', 'label': "$L_\mathrm{max}$"})

    print("- 'truth1.png'")
    figax = truth_binning.plot_ndarray('truth1.png', truth1,
                kwargs1d={'linestyle': 'solid', 'label': "Truth"})
    figax = truth_binning.plot_ndarray('truth1.png', ret1.x, figax=figax,
                kwargs1d={'linestyle': 'dashed', 'linewidth': 2.0, 'color': 'r', 'label': "$L_\mathrm{max}$"})

    print("- 'reco2.png'")
    figax = reco_binning.plot_ndarray('reco2.png', data2,
                kwargs1d={'linestyle': 'solid', 'label': "Data"})
    figax = reco_binning.plot_ndarray('reco2.png', resp.dot(ret2.x), figax=figax,
                kwargs1d={'linestyle': 'dashed', 'linewidth': 2.0, 'color': 'r', 'label': "$L_\mathrm{max}$"})

    print("- 'truth2.png'")
    figax = truth_binning.plot_ndarray('truth2.png', truth2,
                kwargs1d={'linestyle': 'solid', 'label': "Truth"})
    figax = truth_binning.plot_ndarray('truth2.png', ret2.x, figax=figax,
                kwargs1d={'linestyle': 'dashed', 'linewidth': 2.0, 'color': 'r', 'label': "$L_\mathrm{max}$"})

    for tot, hyp, ret in zip(test_total, test_hyp, test_ret):
        fname = 'N%d.png'%(tot,)
        print("- '%s'"%(fname,))
        figax = truth_binning.plot_ndarray(fname, truth1,
                    kwargs1d={'linestyle': 'solid', 'label': "Truth"})
        tr = hyp.translate(ret.x)
        figax = truth_binning.plot_ndarray('N%d.png'%(tot,), tr, figax=figax,
                    kwargs1d={'linestyle': 'dashed', 'linewidth': 2.0, 'color': 'r', 'label': "$L^*_\mathrm{max}$"})

    print("- 'Nall.png'")
    fig, ax = plt.subplots()
    test_L = [-2.*r.L for r in test_ret]
    L_min = min(test_L)
    ax.plot(test_total, test_L)
    ax.axhline(L_min, color='r', linestyle='dashed')
    ax.axhline(L_min+1, color='r', linestyle='dashed')
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"$-2\log(L)$")
    fig.savefig('Nall.png')
