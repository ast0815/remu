from six import print_
import numpy as np
from matplotlib import pyplot as plt
from remu import binning
from remu import plotting
from remu import likelihood
from remu import likelihood_utils
import emcee

with open("../01/reco-binning.yml", "rt") as f:
    reco_binning = binning.yaml.full_load(f)
with open("../01/optimised-truth-binning.yml", "rt") as f:
    truth_binning = binning.yaml.full_load(f)

reco_binning.fill_from_csv_file("../00/real_data.txt")
data = reco_binning.get_entries_as_ndarray()
data_model = likelihood.PoissonData(data)

response_matrix = "../03/response_matrix.npz"
matrix_predictor = likelihood.ResponseMatrixPredictor(response_matrix)
calc = likelihood.LikelihoodCalculator(data_model, matrix_predictor)

truth_binning.fill_from_csv_file("../00/modelA_truth.txt")
modelA = truth_binning.get_values_as_ndarray()
modelA /= np.sum(modelA)

modelA_shape = likelihood.TemplatePredictor([modelA])
calcA = calc.compose(modelA_shape)

samplerA = likelihood_utils.emcee_sampler(calcA)
guessA = likelihood_utils.emcee_initial_guess(calcA)

state = samplerA.run_mcmc(guessA, 100)
chain = samplerA.get_chain(flat=True)
with open("chain_shape.txt", "w") as f:
    print_(chain.shape, file=f)

fig, ax = plt.subplots()
ax.hist(chain[:, 0])
ax.set_xlabel("model A weight")
fig.savefig("burn_short.png")

with open("burn_short_tau.txt", "w") as f:
    try:
        tau = samplerA.get_autocorr_time()
        print_(tau, file=f)
    except emcee.autocorr.AutocorrError as e:
        print_(e, file=f)

samplerA.reset()
state = samplerA.run_mcmc(guessA, 200 * 50)
chain = samplerA.get_chain(flat=True)

with open("burn_long_tau.txt", "w") as f:
    try:
        tau = samplerA.get_autocorr_time()
        print_(tau, file=f)
    except emcee.autocorr.AutocorrError as e:
        print_(e, file=f)

fig, ax = plt.subplots()
ax.hist(chain[:, 0])
ax.set_xlabel("model A weight")
fig.savefig("burn_long.png")

samplerA.reset()
state = samplerA.run_mcmc(state, 100 * 50)
chain = samplerA.get_chain(flat=True)

with open("tauA.txt", "w") as f:
    try:
        tau = samplerA.get_autocorr_time()
        print_(tau, file=f)
    except emcee.autocorr.AutocorrError as e:
        print_(e, file=f)

fig, ax = plt.subplots()
ax.hist(chain[:, 0])
ax.set_xlabel("model A weight")
fig.savefig("weightA.png")

truth, _ = modelA_shape(chain)
truth.shape = (np.prod(truth.shape[:-1]), truth.shape[-1])
pltr = plotting.get_plotter(truth_binning)
pltr.plot_array(truth, stack_function=np.median, label="Post. median", hatch=None)
pltr.plot_array(truth, stack_function=0.68, label="Post. 68%", scatter=0)
pltr.legend()
pltr.savefig("truthA.png")

reco, _ = calcA.predictor(chain)
reco.shape = (np.prod(reco.shape[:-1]), reco.shape[-1])
pltr = plotting.get_plotter(reco_binning)
pltr.plot_array(reco, stack_function=np.median, label="Post. median", hatch=None)
pltr.plot_array(reco, stack_function=0.68, label="Post. 68%")
pltr.plot_array(data, label="Data", hatch=None, linewidth=2)
pltr.legend()
pltr.savefig("recoA.png")

truth_binning.reset()
truth_binning.fill_from_csv_file("../00/modelB_truth.txt")
modelB = truth_binning.get_values_as_ndarray()
modelB /= np.sum(modelB)

combined = likelihood.TemplatePredictor([modelA, modelB])
calcC = calc.compose(combined)

samplerC = likelihood_utils.emcee_sampler(calcC)
guessC = likelihood_utils.emcee_initial_guess(calcC)

state = samplerC.run_mcmc(guessC, 200 * 50)
chain = samplerC.get_chain(flat=True)
with open("combined_chain_shape.txt", "w") as f:
    print_(chain.shape, file=f)

with open("burn_combined_tau.txt", "w") as f:
    try:
        tau = samplerC.get_autocorr_time()
        print_(tau, file=f)
    except emcee.autocorr.AutocorrError as e:
        print_(e, file=f)

samplerC.reset()
state = samplerC.run_mcmc(state, 100 * 50)
chain = samplerC.get_chain(flat=True)
with open("combined_tau.txt", "w") as f:
    try:
        tau = samplerC.get_autocorr_time()
        print_(tau, file=f)
    except emcee.autocorr.AutocorrError as e:
        print_(e, file=f)

fig, ax = plt.subplots()
ax.hist2d(chain[:, 0], chain[:, 1])
ax.set_xlabel("model A weight")
ax.set_ylabel("model B weight")
fig.savefig("combined.png")

fig, ax = plt.subplots()
ax.hist(np.sum(chain, axis=-1))
ax.set_xlabel("model A weight + model B weight")
fig.savefig("total.png")
