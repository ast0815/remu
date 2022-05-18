from six import print_
import numpy as np
from matplotlib import pyplot as plt
from remu import binning
from remu import plotting
from remu import likelihood
from multiprocess import Pool

pool = Pool(8)
likelihood.mapper = pool.map

with open("../01/reco-binning.yml", "rt") as f:
    reco_binning = binning.yaml.full_load(f)
with open("../01/optimised-truth-binning.yml", "rt") as f:
    truth_binning = binning.yaml.full_load(f)

reco_binning.fill_from_csv_file("../00/real_data.txt")
data = reco_binning.get_entries_as_ndarray()
data_model = likelihood.PoissonData(data)

# No systematics LikelihoodCalculator
response_matrix = "../01/response_matrix.npz"
matrix_predictor = likelihood.ResponseMatrixPredictor(response_matrix)
calc = likelihood.LikelihoodCalculator(data_model, matrix_predictor)

# Systematics LikelihoodCalculator
response_matrix_syst = "response_matrix.npz"
matrix_predictor_syst = likelihood.ResponseMatrixPredictor(response_matrix_syst)
calc_syst = likelihood.LikelihoodCalculator(data_model, matrix_predictor_syst)

truth_binning.fill_from_csv_file("../00/modelA_truth.txt")
modelA = truth_binning.get_values_as_ndarray()
modelA /= np.sum(modelA)
truth_binning.reset()
truth_binning.fill_from_csv_file("../00/modelB_truth.txt")
modelB = truth_binning.get_values_as_ndarray()
modelB /= np.sum(modelB)
maxi = likelihood.BasinHoppingMaximizer()

modelA_shape = likelihood.TemplatePredictor([modelA])
calcA = calc.compose(modelA_shape)
retA = maxi(calcA)
with open("modelA_fit.txt", "w") as f:
    print_(retA, file=f)

calcA_syst = calc_syst.compose(modelA_shape)
retA_syst = maxi(calcA_syst)
with open("modelA_fit_syst.txt", "w") as f:
    print_(retA_syst, file=f)

modelB_shape = likelihood.TemplatePredictor([modelB])
calcB = calc.compose(modelB_shape)
retB = maxi(calcB)
with open("modelB_fit.txt", "w") as f:
    print_(retB, file=f)

calcB_syst = calc_syst.compose(modelB_shape)
retB_syst = maxi(calcB_syst)
with open("modelB_fit_syst.txt", "w") as f:
    print_(retB_syst, file=f)

pltr = plotting.get_plotter(reco_binning)
pltr.plot_values(edgecolor="C0", label="data", hatch=None, linewidth=2.0)
modelA_reco, modelA_weights = calcA.predictor(retA.x)
modelB_reco, modelB_weights = calcB.predictor(retB.x)
modelA_syst_reco, modelA_syst_weights = calcA_syst.predictor(retA.x)
modelB_syst_reco, modelB_syst_weights = calcB_syst.predictor(retB.x)
pltr.plot_array(modelA_reco, label="model A", edgecolor="C1", hatch=None)
pltr.plot_array(
    modelA_syst_reco,
    label="model A syst",
    edgecolor="C1",
    hatch=r"//",
    stack_function=0.68,
)
pltr.plot_array(modelB_reco, label="model B", edgecolor="C2", hatch=None)
pltr.plot_array(
    modelB_syst_reco,
    label="model B syst",
    edgecolor="C2",
    hatch=r"\\",
    stack_function=0.68,
)
pltr.legend()
pltr.savefig("reco-comparison.png")

testA = likelihood.HypothesisTester(calcA)
testB = likelihood.HypothesisTester(calcB)
with open("fit_p-values.txt", "w") as f:
    print_(testA.max_likelihood_p_value(), file=f)
    print_(testB.max_likelihood_p_value(), file=f)

testA_syst = likelihood.HypothesisTester(calcA_syst)
testB_syst = likelihood.HypothesisTester(calcB_syst)
with open("fit_p-values_syst.txt", "w") as f:
    print_(testA_syst.max_likelihood_p_value(), file=f)
    print_(testB_syst.max_likelihood_p_value(), file=f)

p_values_A = []
p_values_B = []
p_values_A_syst = []
p_values_B_syst = []
values = np.linspace(600, 1600, 21)
for v in values:
    A = testA.max_likelihood_p_value([v])
    A_syst = testA_syst.max_likelihood_p_value([v])
    B = testB.max_likelihood_p_value([v])
    B_syst = testB_syst.max_likelihood_p_value([v])
    print_(v, A, A_syst, B, B_syst)
    p_values_A.append(A)
    p_values_B.append(B)
    p_values_A_syst.append(A_syst)
    p_values_B_syst.append(B_syst)

fig, ax = plt.subplots()
ax.set_xlabel("Model weight")
ax.set_ylabel("p-value")
ax.plot(values, p_values_A, label="Model A", color="C1", linestyle="dotted")
ax.plot(values, p_values_A_syst, label="Model A syst", color="C1", linestyle="solid")
ax.plot(values, p_values_B, label="Model B", color="C2", linestyle="dotted")
ax.plot(values, p_values_B_syst, label="Model B syst", color="C2", linestyle="solid")
ax.axvline(retA.x[0], color="C1", linestyle="dotted")
ax.axvline(retA_syst.x[0], color="C1", linestyle="solid")
ax.axvline(retB.x[0], color="C2", linestyle="dotted")
ax.axvline(retB_syst.x[0], color="C2", linestyle="solid")
ax.axhline(0.32, color="k", linestyle="dashed")
ax.axhline(0.05, color="k", linestyle="dashed")
ax.legend(loc="best")
fig.savefig("p-values.png")

p_values_A = []
p_values_B = []
p_values_A_syst = []
p_values_B_syst = []
values = np.linspace(600, 1600, 21)
for v in values:
    A = testA.max_likelihood_ratio_p_value([v])
    A_syst = testA_syst.max_likelihood_ratio_p_value([v])
    B = testB.max_likelihood_ratio_p_value([v])
    B_syst = testB_syst.max_likelihood_ratio_p_value([v])
    print_(v, A, A_syst, B, B_syst)
    p_values_A.append(A)
    p_values_B.append(B)
    p_values_A_syst.append(A_syst)
    p_values_B_syst.append(B_syst)

p_values_A_wilks = []
p_values_B_wilks = []
fine_values = np.linspace(600, 1600, 100)
for v in fine_values:
    A = testA_syst.wilks_max_likelihood_ratio_p_value([v])
    B = testB_syst.wilks_max_likelihood_ratio_p_value([v])
    print_(v, A, B)
    p_values_A_wilks.append(A)
    p_values_B_wilks.append(B)

fig, ax = plt.subplots()
ax.set_xlabel("Model weight")
ax.set_ylabel("p-value")
ax.plot(values, p_values_A, label="Model A", color="C1", linestyle="dotted")
ax.plot(values, p_values_A_syst, label="Model A syst", color="C1", linestyle="solid")
ax.plot(
    fine_values, p_values_A_wilks, label="Model A Wilks", color="C1", linestyle="dashed"
)
ax.plot(values, p_values_B, label="Model B", color="C2", linestyle="dotted")
ax.plot(values, p_values_B_syst, label="Model B syst", color="C2", linestyle="solid")
ax.plot(
    fine_values, p_values_B_wilks, label="Model B Wilks", color="C2", linestyle="dashed"
)
ax.axvline(retA.x[0], color="C1", linestyle="dotted")
ax.axvline(retA_syst.x[0], color="C1", linestyle="solid")
ax.axvline(retB.x[0], color="C2", linestyle="dotted")
ax.axvline(retB_syst.x[0], color="C2", linestyle="solid")
ax.axhline(0.32, color="k", linestyle="dashed")
ax.axhline(0.05, color="k", linestyle="dashed")
ax.legend(loc="best")
fig.savefig("ratio-p-values.png")
