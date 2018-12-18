from six import print_
import numpy as np
from matplotlib import pyplot as plt
from remu import binning
from remu import likelihood
import pymc

with open("../01/reco-binning.yml", 'rt') as f:
    reco_binning = binning.yaml.load(f)
with open("../01/optimised-truth-binning.yml", 'rt') as f:
    truth_binning = binning.yaml.load(f)

reco_binning.fill_from_csv_file("../00/real_data.txt")
data = reco_binning.get_entries_as_ndarray()

response_matrix = np.load("../03/response_matrix.npy")
generator_truth = np.load("../03/generator_truth.npy")
response_matrix.shape = (np.prod(response_matrix.shape[:-2]),) + response_matrix.shape[-2:]
lm = likelihood.LikelihoodMachine(data, response_matrix, truth_limits=generator_truth, limit_method='prohibit')

truth_binning.fill_from_csv_file("../00/modelA_truth.txt")
modelA = truth_binning.get_values_as_ndarray()
modelA /= np.sum(modelA)
truth_binning.reset()
truth_binning.fill_from_csv_file("../00/modelB_truth.txt")
modelB = truth_binning.get_values_as_ndarray()
modelB /= np.sum(modelB)

def flat_prior(value=100):
    if value >= 0 and value <= 2000:
        return 0
    else:
        return -np.inf

modelA_shape = likelihood.TemplateHypothesis([modelA], parameter_priors=[flat_prior], parameter_names=["template_weight"])
modelB_shape = likelihood.TemplateHypothesis([modelB], parameter_priors=[flat_prior], parameter_names=["template_weight"])

mcmcA = lm.mcmc(modelA_shape)

mcmcA.sample(iter=1000)
pymc.Matplot.plot(mcmcA, suffix='_noburn')

mcmcA.sample(iter=1000, burn=100)
pymc.Matplot.plot(mcmcA, suffix='_burn')

mcmcA.sample(iter=1000*10, burn=100, thin=10)
pymc.Matplot.plot(mcmcA, suffix='_A')

mcmcB = lm.mcmc(modelB_shape)

mcmcB.sample(iter=1000*10, burn=100, thin=10)
pymc.Matplot.plot(mcmcB, suffix='_B')

with open("stats.txt", 'wt') as f:
    print_(mcmcA.stats(), file=f)
    print_(mcmcB.stats(), file=f)

traceA = mcmcA.trace('template_weight')[:]
traceB = mcmcB.trace('template_weight')[:]
with open("percentiles.txt", 'wt') as f:
    print_(np.percentile(traceA, [2.5, 16., 50, 84, 97.5]), file=f)
    print_(np.percentile(traceB, [2.5, 16., 50, 84, 97.5]), file=f)

toysA = mcmcA.trace('toy_index')[:]
toysB = mcmcB.trace('toy_index')[:]
tA = traceA[:,np.newaxis]
iA = toysA[:,np.newaxis]
tB = traceB[:,np.newaxis]
iB = toysB[:,np.newaxis]
ratios, preference = lm.plr(modelA_shape, tA, iA, modelB_shape, tB, iB)
with open("plr.txt", 'wt') as f:
    print_(preference, file=f)

mixed_model = likelihood.TemplateHypothesis([modelA, modelB])

prior = likelihood.JeffreysPrior(
    response_matrix = response_matrix,
    translation_function = mixed_model.translate,
    parameter_limits = [(0,None), (0,None)],
    default_values = [10., 10.],
    total_truth_limit = 2000)

mixed_model.parameter_priors = [prior]
mixed_model.parameter_names = ["weights"]

mcmc = lm.mcmc(mixed_model, prior_only=True)
mcmc.sample(iter=25000, burn=1000, tune_throughout=True, thin=25)
pymc.Matplot.plot(mcmc, suffix='_prior')

mcmc = lm.mcmc(mixed_model)
mcmc.sample(iter=250000, burn=10000, tune_throughout=True, thin=250)
pymc.Matplot.plot(mcmc, suffix='_mixed')

weights = mcmc.trace('weights')[:]
fig, ax = plt.subplots()
ax.hist2d(weights[:,0],weights[:,1], bins=20)
ax.set_xlabel("model A weight")
ax.set_ylabel("model B weight")
fig.savefig("posterior.png")

fig, ax = plt.subplots()
ax.hist(weights.sum(axis=1), bins=20)
ax.set_xlabel("sum of template weights")
fig.savefig("sum_posterior.png")

truth_trace = mixed_model.translate(weights)
toys = mcmc.trace('toy_index')[:]
lm.plot_truth_bin_traces("truth_traces.png", truth_trace, plot_limits='relative')

lm.plot_reco_bin_traces("reco_traces.png", truth_trace, toy_index=toys, plot_data=True)
