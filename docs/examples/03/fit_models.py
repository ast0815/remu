from six import print_
import numpy as np
from matplotlib import pyplot as plt
from remu import binning
from remu import likelihood

with open("../01/reco-binning.yml", 'rt') as f:
    reco_binning = binning.yaml.load(f)
with open("../01/optimised-truth-binning.yml", 'rt') as f:
    truth_binning = binning.yaml.load(f)

reco_binning.fill_from_csv_file("../00/real_data.txt")
data = reco_binning.get_entries_as_ndarray()

# No systematics LikelihoodMachine
response_matrix = "../01/response_matrix.npz"
lm = likelihood.LikelihoodMachine(data, response_matrix, limit_method='prohibit')

# Systematics LikelihoodMachine
response_matrix_syst = "response_matrix.npz"
lm_syst = likelihood.LikelihoodMachine(data, response_matrix_syst, limit_method='prohibit')

truth_binning.fill_from_csv_file("../00/modelA_truth.txt")
modelA = truth_binning.get_values_as_ndarray()
modelA /= np.sum(modelA)
truth_binning.reset()
truth_binning.fill_from_csv_file("../00/modelB_truth.txt")
modelB = truth_binning.get_values_as_ndarray()
modelB /= np.sum(modelB)

modelA_shape = likelihood.TemplateHypothesis([modelA])
with open("modelA_fit.txt", 'w') as f:
    retA = lm.max_log_likelihood(modelA_shape)
    print_(retA, file=f)
with open("modelA_fit_syst.txt", 'w') as f:
    retA_syst = lm_syst.max_log_likelihood(modelA_shape)
    print_(retA_syst, file=f)

modelB_shape = likelihood.TemplateHypothesis([modelB])
with open("modelB_fit.txt", 'w') as f:
    retB = lm.max_log_likelihood(modelB_shape)
    print_(retB, file=f)
with open("modelB_fit_syst.txt", 'w') as f:
    retB_syst = lm_syst.max_log_likelihood(modelB_shape)
    print_(retB_syst, file=f)

figax = reco_binning.plot_values(None, kwargs1d={'color': 'k', 'label': 'data'}, sqrt_errors=True)
modelA_reco = lm.fold(modelA_shape.translate(retA.x))
modelB_reco = lm.fold(modelB_shape.translate(retB.x))
modelA_reco_syst = lm_syst.fold(modelA_shape.translate(retA_syst.x))
modelB_reco_syst = lm_syst.fold(modelB_shape.translate(retB_syst.x))
reco_binning.plot_ndarray(None, modelA_reco, kwargs1d={'color': 'b', 'label': 'model A'}, figax=figax)
reco_binning.plot_ndarray(None, modelA_reco_syst, kwargs1d={'color': 'b', 'label': 'model A syst', 'hatch': '\\\\', 'facecolor': 'none'}, error_band='step', figax=figax)
reco_binning.plot_ndarray(None, modelB_reco, kwargs1d={'color': 'r', 'label': 'model B'}, figax=figax)
reco_binning.plot_ndarray("reco-comparison.png", modelB_reco_syst, kwargs1d={'color': 'r', 'label': 'model B syst', 'hatch': '//', 'facecolor': 'none'}, error_band='step', figax=figax)

with open("fit_p-values.txt", 'w') as f:
    print_(lm.max_likelihood_p_value(modelA_shape, nproc=4), file=f)
    print_(lm.max_likelihood_p_value(modelB_shape, nproc=4), file=f)
with open("fit_p-values_syst.txt", 'w') as f:
    print_(lm_syst.max_likelihood_p_value(modelA_shape, nproc=4), file=f)
    print_(lm_syst.max_likelihood_p_value(modelB_shape, nproc=4), file=f)

p_values_A = []
p_values_B = []
p_values_A_syst = []
p_values_B_syst = []
values = np.linspace(600, 1600, 21)
for v in values:
    fixed_model_A = modelA_shape.fix_parameters((v,))
    fixed_model_B = modelB_shape.fix_parameters((v,))
    A = lm.max_likelihood_p_value(fixed_model_A, nproc=4)
    A_syst = lm_syst.max_likelihood_p_value(fixed_model_A, nproc=4)
    B = lm.max_likelihood_p_value(fixed_model_B, nproc=4)
    B_syst = lm_syst.max_likelihood_p_value(fixed_model_B, nproc=4)
    print_(v, A, A_syst, B, B_syst)
    p_values_A.append(A)
    p_values_B.append(B)
    p_values_A_syst.append(A_syst)
    p_values_B_syst.append(B_syst)

fig, ax = plt.subplots()
ax.set_xlabel("Model weight")
ax.set_ylabel("p-value")
ax.plot(values, p_values_A, label="Model A", color='b', linestyle='dotted')
ax.plot(values, p_values_A_syst, label="Model A syst", color='b', linestyle='solid')
ax.plot(values, p_values_B, label="Model B", color='r', linestyle='dotted')
ax.plot(values, p_values_B_syst, label="Model B syst", color='r', linestyle='solid')
ax.axvline(retA.x[0], color='b', linestyle='dotted')
ax.axvline(retA_syst.x[0], color='b', linestyle='solid')
ax.axvline(retB.x[0], color='r', linestyle='dotted')
ax.axvline(retB_syst.x[0], color='r', linestyle='solid')
ax.axhline(0.32, color='k', linestyle='dashed')
ax.axhline(0.05, color='k', linestyle='dashed')
ax.legend(loc='best')
fig.savefig("p-values.png")

p_values_A = []
p_values_B = []
p_values_A_syst = []
p_values_B_syst = []
values = np.linspace(600, 1600, 21)
for v in values:
    fixed_model_A = modelA_shape.fix_parameters((v,))
    fixed_model_B = modelB_shape.fix_parameters((v,))
    A = lm.max_likelihood_ratio_p_value(fixed_model_A, modelA_shape, nproc=4)
    A_syst = lm_syst.max_likelihood_ratio_p_value(fixed_model_A, modelA_shape, nproc=4)
    B = lm.max_likelihood_ratio_p_value(fixed_model_B, modelB_shape, nproc=4)
    B_syst = lm_syst.max_likelihood_ratio_p_value(fixed_model_B, modelB_shape, nproc=4)
    print_(v, A, A_syst, B, B_syst)
    p_values_A.append(A)
    p_values_B.append(B)
    p_values_A_syst.append(A_syst)
    p_values_B_syst.append(B_syst)

p_values_A_wilks = []
p_values_B_wilks = []
fine_values = np.linspace(600, 1600, 100)
for v in fine_values:
    fixed_model_A = modelA_shape.fix_parameters((v,))
    fixed_model_B = modelB_shape.fix_parameters((v,))
    A = lm_syst.wilks_max_likelihood_ratio_p_value(fixed_model_A, modelA_shape)
    B = lm_syst.wilks_max_likelihood_ratio_p_value(fixed_model_B, modelB_shape)
    print_(v, A, B)
    p_values_A_wilks.append(A)
    p_values_B_wilks.append(B)

fig, ax = plt.subplots()
ax.set_xlabel("Model weight")
ax.set_ylabel("p-value")
ax.plot(values, p_values_A, label="Model A", color='b', linestyle='dotted')
ax.plot(values, p_values_A_syst, label="Model A syst", color='b', linestyle='solid')
ax.plot(fine_values, p_values_A_wilks, label="Model A Wilks", color='b', linestyle='dashed')
ax.plot(values, p_values_B, label="Model B", color='r', linestyle='dotted')
ax.plot(values, p_values_B_syst, label="Model B syst", color='r', linestyle='solid')
ax.plot(fine_values, p_values_B_wilks, label="Model B Wilks", color='r', linestyle='dashed')
ax.axvline(retA.x[0], color='b', linestyle='dotted')
ax.axvline(retA_syst.x[0], color='b', linestyle='solid')
ax.axvline(retB.x[0], color='r', linestyle='dotted')
ax.axvline(retB_syst.x[0], color='r', linestyle='solid')
ax.axhline(0.32, color='k', linestyle='dashed')
ax.axhline(0.05, color='k', linestyle='dashed')
ax.legend(loc='best')
fig.savefig("ratio-p-values.png")
