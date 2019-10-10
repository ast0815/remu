from six import print_
import numpy as np
from matplotlib import pyplot as plt
from remu import binning
from remu import likelihood

response_matrix = "../01/response_matrix.npz"

with open("../01/reco-binning.yml", 'rt') as f:
    reco_binning = binning.yaml.full_load(f)
with open("../01/optimised-truth-binning.yml", 'rt') as f:
    truth_binning = binning.yaml.full_load(f)

reco_binning.fill_from_csv_file("../00/real_data.txt")
data = reco_binning.get_entries_as_ndarray()

lm = likelihood.LikelihoodMachine(data, response_matrix, limit_method='prohibit')

truth_binning.fill_from_csv_file("../00/modelA_truth.txt")
modelA = truth_binning.get_values_as_ndarray()
modelA /= np.sum(modelA)
truth_binning.reset()
truth_binning.fill_from_csv_file("../00/modelB_truth.txt")
modelB = truth_binning.get_values_as_ndarray()
modelB /= np.sum(modelB)

with open("simple_hypotheses.txt", 'w') as f:
    print_(lm.log_likelihood(modelA*1000), file=f)
    print_(lm.likelihood_p_value(modelA*1000), file=f)
    print_(lm.log_likelihood(modelB*1000), file=f)
    print_(lm.likelihood_p_value(modelB*1000), file=f)

with open("modelA_fit.txt", 'w') as f:
    modelA_shape = likelihood.TemplateHypothesis([modelA])
    retA = lm.max_log_likelihood(modelA_shape)
    print_(retA, file=f)

with open("modelB_fit.txt", 'w') as f:
    modelB_shape = likelihood.TemplateHypothesis([modelB])
    retB = lm.max_log_likelihood(modelB_shape)
    print_(retB, file=f)

with open("fit_p-values.txt", 'w') as f:
    print_(lm.max_likelihood_p_value(modelA_shape, nproc=4), file=f)
    print_(lm.max_likelihood_p_value(modelB_shape, nproc=4), file=f)

figax = reco_binning.plot_values(None, kwargs1d={'label': 'data', 'color': 'k'})
modelA_reco = lm.fold(modelA_shape.translate(retA.x))
modelA_chi2 = lm.pseudo_chi2(modelA_shape.translate(retA.x))
modelB_reco = lm.fold(modelB_shape.translate(retB.x))
modelB_chi2 = lm.pseudo_chi2(modelB_shape.translate(retB.x))
reco_binning.plot_ndarray(None, modelA_reco, kwargs1d={'label': 'model A: $\chi^2=%.1f$'%(modelA_chi2), 'color': 'b'}, sqrt_errors=True, error_xoffset=-0.1, figax=figax)
reco_binning.plot_ndarray("reco-comparison.png", modelB_reco, kwargs1d={'label': 'model B: $\chi^2=%.1f$'%(modelB_chi2), 'color': 'r'}, sqrt_errors=True, error_xoffset=+0.1, figax=figax)

with open("mix_model_fit.txt", 'w') as f:
    mix_model = likelihood.TemplateHypothesis([modelA, modelB])
    ret = lm.max_log_likelihood(mix_model)
    print_(ret, file=f)

with open("mix_model_p_value.txt", 'w') as f:
    print_(lm.max_likelihood_p_value(mix_model, nproc=4), file=f)

p_values = []
A_values = np.linspace(0, 1000, 11)
for A in A_values:
    fixed_model = mix_model.fix_parameters((A, None))
    p = lm.max_likelihood_ratio_p_value(fixed_model, mix_model, nproc=4)
    print_(A, p)
    p_values.append(p)

wilks_p_values = []
fine_A_values = np.linspace(0, 1000, 100)
for A in fine_A_values:
    fixed_model = mix_model.fix_parameters((A, None))
    p = lm.wilks_max_likelihood_ratio_p_value(fixed_model, mix_model)
    print_(A, p)
    wilks_p_values.append(p)

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
