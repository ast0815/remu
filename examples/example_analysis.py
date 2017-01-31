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
    with profile:
        start_time = timeit.default_timer()
        ret1 = lm1.absolute_max_log_likelihood()
        print("Data1: N=%.1f, ll=%.1f, p=%.3f"%(np.sum(ret1.x), ret1.L, lm1.likelihood_p_value(ret1.x)))
        ret2 = lm2.absolute_max_log_likelihood()
        print("Data2: N=%.1f, ll=%.1f, p=%.3f"%(np.sum(ret2.x), ret2.L, lm2.likelihood_p_value(ret2.x)))
        elapsed = timeit.default_timer() - start_time
        print("Time: %.1f"%(elapsed,))


    print("Plotting results...")
    print("- 'reco1.png'")
    fig, ax = plt.subplots(2)
    reco_binning.set_values_from_ndarray(resp.dot(ret1.x))
    ax[0].set_xlabel(r"$x_{0,reco}$")
    edg = reco_binning.binedges['x_0_reco']
    N = data1_x0
    ax[0].plot(edg[:-1], N, drawstyle='steps-post', label="Data")
    N = reco_binning.project(['x_0_reco']).get_values_as_ndarray()
    ax[0].plot(edg[:-1], N, '--r', linewidth=2, drawstyle='steps-post', label="$L_{max}$")
    ax[0].legend(loc='best')
    ax[1].set_xlabel(r"$x_{1,reco}$")
    edg = reco_binning.binedges['x_1_reco']
    N = data1_x1
    ax[1].plot(edg[:-1], N, drawstyle='steps-post', label="Data")
    N = reco_binning.project(['x_1_reco']).get_values_as_ndarray()
    ax[1].plot(edg[:-1], N, '--r', linewidth=2, drawstyle='steps-post', label="$L_{max}$")
    ax[1].legend(loc='best')
    fig.savefig('reco1.png')

    print("- 'truth1.png'")
    fig, ax = plt.subplots(2)
    truth_binning.set_values_from_ndarray(ret1.x)
    ax[0].set_xlabel(r"$x_{0,truth}$")
    edg = truth_binning.binedges['x_0_truth']
    N = truth1_x0
    ax[0].plot(edg[:-1], N, drawstyle='steps-post', label="Data")
    N = truth_binning.project(['x_0_truth']).get_values_as_ndarray()
    ax[0].plot(edg[:-1], N, '--r', linewidth=2, drawstyle='steps-post', label="$L_{max}$")
    ax[0].legend(loc='best')
    ax[1].set_xlabel(r"$x_{1,truth}$")
    edg = truth_binning.binedges['x_0_truth']
    N = truth1_x1
    ax[1].plot(edg[:-1], N, drawstyle='steps-post', label="Data")
    N = truth_binning.project(['x_1_truth']).get_values_as_ndarray()
    ax[1].plot(edg[:-1], N, '--r', linewidth=2, drawstyle='steps-post', label="$L_{max}$")
    ax[1].legend(loc='best')
    fig.savefig('truth1.png')

    print("- 'reco2.png'")
    fig, ax = plt.subplots(2)
    reco_binning.set_values_from_ndarray(resp.dot(ret2.x))
    ax[0].set_xlabel(r"$x_{0,reco}$")
    edg = reco_binning.binedges['x_0_reco']
    N = data2_x0
    ax[0].plot(edg[:-1], N, drawstyle='steps-post', label="Data")
    N = reco_binning.project(['x_0_reco']).get_values_as_ndarray()
    ax[0].plot(edg[:-1], N, '--r', linewidth=2, drawstyle='steps-post', label="$L_{max}$")
    ax[0].legend(loc='best')
    ax[1].set_xlabel(r"$x_{1,reco}$")
    edg = reco_binning.binedges['x_1_reco']
    N = data2_x1
    ax[1].plot(edg[:-1], N, drawstyle='steps-post', label="Data")
    N = reco_binning.project(['x_1_reco']).get_values_as_ndarray()
    ax[1].plot(edg[:-1], N, '--r', linewidth=2, drawstyle='steps-post', label="$L_{max}$")
    ax[1].legend(loc='best')
    fig.savefig('reco2.png')

    print("- 'truth2.png'")
    fig, ax = plt.subplots(2)
    truth_binning.set_values_from_ndarray(ret2.x)
    ax[0].set_xlabel(r"$x_{0,truth}$")
    edg = truth_binning.binedges['x_0_truth']
    N = truth2_x0
    ax[0].plot(edg[:-1], N, drawstyle='steps-post', label="Data")
    N = truth_binning.project(['x_0_truth']).get_values_as_ndarray()
    ax[0].plot(edg[:-1], N, '--r', linewidth=2, drawstyle='steps-post', label="$L_{max}$")
    ax[0].legend(loc='best')
    ax[1].set_xlabel(r"$x_{1,truth}$")
    edg = truth_binning.binedges['x_0_truth']
    N = truth2_x1
    ax[1].plot(edg[:-1], N, drawstyle='steps-post', label="Data")
    N = truth_binning.project(['x_1_truth']).get_values_as_ndarray()
    ax[1].plot(edg[:-1], N, '--r', linewidth=2, drawstyle='steps-post', label="$L_{max}$")
    ax[1].legend(loc='best')
    fig.savefig('truth2.png')
