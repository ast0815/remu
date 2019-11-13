from six import print_
import numpy as np
from remu import binning
from remu import plotting
from remu import likelihood
from multiprocess import Pool
pool = Pool(8)
likelihood.mapper = pool.map

response_matrix = "../01/response_matrix.npz"

with open("../01/reco-binning.yml", 'rt') as f:
    reco_binning = binning.yaml.full_load(f)
with open("../01/optimised-truth-binning.yml", 'rt') as f:
    truth_binning = binning.yaml.full_load(f)

reco_binning.fill_from_csv_file("../00/real_data.txt")
data = reco_binning.get_entries_as_ndarray()
data_model = likelihood.PoissonData(data)
matrix_predictor = likelihood.ResponseMatrixPredictor(response_matrix)

calc = likelihood.LikelihoodCalculator(data_model, matrix_predictor)
test = likelihood.HypothesisTester(calc)

truth_binning.fill_from_csv_file("../00/modelA_truth.txt")
modelA = truth_binning.get_values_as_ndarray()
modelA /= np.sum(modelA)

truth_binning.reset()
truth_binning.fill_from_csv_file("../00/modelB_truth.txt")
modelB = truth_binning.get_values_as_ndarray()
modelB /= np.sum(modelB)

with open("simple_hypotheses.txt", 'w') as f:
    print_(calc(modelA*1000), file=f)
    print_(test.likelihood_p_value(modelA*1000), file=f)
    print_(calc(modelB*1000), file=f)
    print_(test.likelihood_p_value(modelB*1000), file=f)

modelA_shape = likelihood.TemplatePredictor([modelA])
modelA_reco_shape = matrix_predictor.compose(modelA_shape)
calcA = likelihood.LikelihoodCalculator(data_model, modelA_reco_shape)

maxi = likelihood.BasinHoppingMaximizer()
retA = maxi(calcA)
with open("modelA_fit.txt", 'w') as f:
    print_(retA, file=f)

modelB_shape = likelihood.TemplatePredictor([modelB])
calcB = calc.compose(modelB_shape)
retB = maxi(calcB)
with open("modelB_fit.txt", 'w') as f:
    print_(retB, file=f)

testA = likelihood.HypothesisTester(calcA, maximizer=maxi)
testB = likelihood.HypothesisTester(calcB, maximizer=maxi)

with open("fit_p-values.txt", 'w') as f:
    print_(testA.max_likelihood_p_value(), file=f)
    print_(testB.max_likelihood_p_value(), file=f)

pltr = plotting.get_plotter(reco_binning)
pltr.plot_entries(label='data', hatch=None)
modelA_reco, modelA_weights = modelA_reco_shape(retA.x)
modelA_logL = calcA(retA.x)
modelA_p = testA.likelihood_p_value(retA.x)
modelB_reco, modelB_weights = calcB.predictor(retB.x)
modelB_logL = calcB(retB.x)
modelB_p = testB.likelihood_p_value(retB.x)
pltr.plot_array(modelA_reco, label='model A: $\log L=%.1f$, $p=%.3f$'%(modelA_logL, modelA_p), hatch=None, linestyle='dashed')
pltr.plot_array(modelB_reco, label='model B: $\log L=%.1f$, $p=%.3f$'%(modelB_logL, modelB_p), hatch=None, linestyle='dotted')
pltr.legend(loc='lower center')
pltr.savefig("reco-comparison.png")

mix_model = likelihood.TemplatePredictor([modelA, modelB])
calc_mix = calc.compose(mix_model)
ret = maxi.maximize_log_likelihood(calc_mix)
with open("mix_model_fit.txt", 'w') as f:
    print_(ret, file=f)

test = likelihood.HypothesisTester(calc_mix)
with open("mix_model_p_value.txt", 'w') as f:
    print_(test.max_likelihood_p_value(), file=f)

p_values = []
A_values = np.linspace(0, 1000, 11)
for A in A_values:
    p = test.max_likelihood_ratio_p_value((A,None))
    print_(A, p)
    p_values.append(p)

wilks_p_values = []
fine_A_values = np.linspace(0, 1000, 100)
for A in fine_A_values:
    p = test.wilks_max_likelihood_ratio_p_value((A,None))
    print_(A, p)
    wilks_p_values.append(p)

from matplotlib import pyplot as plt
fig, ax = plt.subplots()
ax.set_xlabel("Model A weight")
ax.set_ylabel("p-value")
ax.plot(A_values, p_values, label="Profile plug-in")
ax.plot(fine_A_values, wilks_p_values, label="Wilks")
ax.axvline(ret.x[0], color='k', linestyle='solid')
ax.axhline(0.32, color='k', linestyle='dashed')
ax.axhline(0.05, color='k', linestyle='dashed')
ax.legend(loc='best')
fig.savefig("p-values.png")
