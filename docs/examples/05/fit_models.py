import numpy as np
from multiprocess import Pool

from remu import binning, likelihood, plotting

pool = Pool(8)
likelihood.mapper = pool.map

with open("../01/reco-binning.yml") as f:
    reco_binning = binning.yaml.full_load(f)
with open("truth-binning.yml") as f:
    truth_binning = binning.yaml.full_load(f)

reco_binning.fill_from_csv_file("real_data.txt")
data = reco_binning.get_entries_as_ndarray()
data_model = likelihood.PoissonData(data)

response_matrix = "response_matrix.npz"
matrix_predictor = likelihood.ResponseMatrixPredictor(response_matrix)

calc = likelihood.LikelihoodCalculator(data_model, matrix_predictor)
maxi = likelihood.BasinHoppingMaximizer()

import numpy.lib.recfunctions as rfn


def set_signal(data):
    return rfn.append_fields(data, "event_type", np.full_like(data["true_x"], 1.0))


def set_bg(data):
    return rfn.append_fields(data, "event_type", np.full_like(data["true_x"], 0.0))


truth_binning.fill_from_csv_file("../00/modelA_truth.txt", cut_function=set_signal)
modelA = truth_binning.get_values_as_ndarray()
modelA /= np.sum(modelA)

truth_binning.reset()
truth_binning.fill_from_csv_file("../00/modelB_truth.txt", cut_function=set_signal)
modelB = truth_binning.get_values_as_ndarray()
modelB /= np.sum(modelB)

truth_binning.reset()
truth_binning.fill_from_csv_file("bg_truth.txt", cut_function=set_bg)
bg = truth_binning.get_values_as_ndarray()
bg /= np.sum(bg)

truth_binning.reset()
noise = truth_binning.get_values_as_ndarray()
noise[0] = 1.0

modelA_only = likelihood.TemplatePredictor([modelA])
calcA_only = calc.compose(modelA_only)

retA_only = maxi(calcA_only)
with open("modelA_only_fit.txt", "w") as f:
    print(retA_only, file=f)

testA_only = likelihood.HypothesisTester(calcA_only)
with open("modelA_only_gof.txt", "w") as f:
    print(testA_only.likelihood_p_value(retA_only.x), file=f)

with open("modelA_only_p_value.txt", "w") as f:
    print(testA_only.max_likelihood_p_value(), file=f)

modelA_bg = likelihood.TemplatePredictor([noise, bg, modelA])
calcA_bg = calc.compose(modelA_bg)

retA_bg = maxi(calcA_bg)
with open("modelA_bg_fit.txt", "w") as f:
    print(retA_bg, file=f)

testA_bg = likelihood.HypothesisTester(calcA_bg)
with open("modelA_bg_gof.txt", "w") as f:
    print(testA_bg.likelihood_p_value(retA_bg.x), file=f)

with open("modelA_bg_p_value.txt", "w") as f:
    print(testA_bg.max_likelihood_p_value(), file=f)

modelB_only = likelihood.TemplatePredictor([modelB])
calcB_only = calc.compose(modelB_only)

retB_only = maxi(calcB_only)
with open("modelB_only_fit.txt", "w") as f:
    print(retB_only, file=f)

testB_only = likelihood.HypothesisTester(calcB_only)
with open("modelB_only_gof.txt", "w") as f:
    print(testB_only.likelihood_p_value(retB_only.x), file=f)

with open("modelB_only_p_value.txt", "w") as f:
    print(testB_only.max_likelihood_p_value(), file=f)

modelB_bg = likelihood.TemplatePredictor([noise, bg, modelB])
calcB_bg = calc.compose(modelB_bg)

retB_bg = maxi(calcB_bg)
with open("modelB_bg_fit.txt", "w") as f:
    print(retB_bg, file=f)

testB_bg = likelihood.HypothesisTester(calcB_bg)
with open("modelB_bg_gof.txt", "w") as f:
    print(testB_bg.likelihood_p_value(retB_bg.x), file=f)

with open("modelB_bg_p_value.txt", "w") as f:
    print(testB_bg.max_likelihood_p_value(), file=f)

pltr = plotting.get_plotter(reco_binning)
modelA_reco, modelA_weights = calcA_only.predictor(retA_only.x)
modelB_reco, modelB_weights = calcB_only.predictor(retB_only.x)
modelA_bg_reco, modelA_bg_weights = calcA_bg.predictor(retA_bg.x)
modelB_bg_reco, modelB_bg_weights = calcB_bg.predictor(retB_bg.x)
pltr.plot_array(
    modelA_reco, label="model A only", stack_function=0.68, hatch=r"//", edgecolor="C1"
)
pltr.plot_array(
    modelA_bg_reco,
    label="model A + bg",
    stack_function=0.68,
    hatch=r"*",
    edgecolor="C1",
)
pltr.plot_array(
    modelB_reco, label="model B only", stack_function=0.68, hatch=r"\\", edgecolor="C2"
)
pltr.plot_array(
    modelB_bg_reco,
    label="model B + bg",
    stack_function=0.68,
    hatch=r"O",
    edgecolor="C2",
)
pltr.plot_entries(edgecolor="C0", label="data", hatch=None, linewidth=2.0)
pltr.legend()
pltr.savefig("reco-comparison.png")

pltr = plotting.get_plotter(truth_binning)
pltr.plot_array(
    modelA_only(retA_only.x)[0],
    label="model A only",
    hatch=r"//",
    edgecolor="C1",
    density=False,
)
pltr.plot_array(
    modelA_bg(retA_bg.x)[0],
    label="model A + bg",
    hatch=r"*",
    edgecolor="C1",
    density=False,
)
pltr.plot_array(
    modelB_only(retB_only.x)[0],
    label="model B only",
    hatch=r"\\",
    edgecolor="C2",
    density=False,
)
pltr.plot_array(
    modelB_bg(retB_bg.x)[0],
    label="model B + bg",
    hatch=r"O",
    edgecolor="C2",
    density=False,
)
pltr.legend(loc="upper left")
pltr.savefig("truth-comparison.png")
