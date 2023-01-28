#!/env/python

import numpy as np
import pandas as pd
from multiprocess import Pool

from remu import binning, likelihood, plotting

pool = Pool(8)
likelihood.mapper = pool.map

with open("../05/truth-binning.yml") as f:
    truth_binning = binning.yaml.full_load(f)

# Define flux binning
with open("flux-binning.yml") as f:
    flux_binning = binning.yaml.full_load(f)

# Fill flux with neutrinos per m^2
from numpy.random import default_rng

rng = default_rng()
E = rng.normal(loc=8.0, scale=2.0, size=1000)
df = pd.DataFrame({"E": E})
flux_binning.fill(df, weight=0.01)

pltr = plotting.get_plotter(flux_binning)
pltr.plot_values()
pltr.savefig("flux.png")

# Get truth binnings for BG and signal
bg_truth_binning = truth_binning.subbinnings[1].clone()
signal_truth_binning = truth_binning.subbinnings[2].clone()

# Create cross-section binnings
bg_flux_binning = flux_binning.clone()
bg_xsec_binning = binning.CartesianProductBinning((bg_truth_binning, bg_flux_binning))
signal_flux_binning = flux_binning.clone()
signal_xsec_binning = binning.CartesianProductBinning(
    (signal_truth_binning, signal_flux_binning)
)

# Check binning structure
n_bg_truth = bg_truth_binning.data_size
n_signal_truth = signal_truth_binning.data_size
n_flux = flux_binning.data_size
with open("check_binning.txt", "w") as f:
    print(n_bg_truth, n_signal_truth, n_flux, file=f)
    print(signal_xsec_binning.bins[0], file=f)
    print(signal_xsec_binning.bins[1], file=f)
    print(signal_xsec_binning.bins[n_flux], file=f)

# Create event number predictors
flux = flux_binning.get_values_as_ndarray()
bg_predictor = likelihood.LinearEinsumPredictor(
    "ij,...kj->...ik",
    [flux],
    reshape_parameters=(n_bg_truth, n_flux),
    bounds=[(0.0, np.inf)] * bg_xsec_binning.data_size,
)
signal_predictor = likelihood.LinearEinsumPredictor(
    "ij,...kj->...ik",
    [flux],
    reshape_parameters=(n_signal_truth, n_flux),
    bounds=[(0.0, np.inf)] * signal_xsec_binning.data_size,
)

# Test cross-section predictions
signal_xsec_binning.reset()
signal_xsec_binning.fill({"E": 8.0, "true_x": 0.0, "true_y": 0.0}, 100.0)
signal_xsec = signal_xsec_binning.get_values_as_ndarray()

signal_events, weights = signal_predictor(signal_xsec)
signal_truth_binning.set_values_from_ndarray(signal_events)
pltr = plotting.get_plotter(signal_truth_binning)
pltr.plot_values(density=False)
pltr.savefig("many_events.png")

signal_xsec_binning.reset()
signal_xsec_binning.fill({"E": 3.0, "true_x": 0.0, "true_y": 0.0}, 100.0)
signal_xsec = signal_xsec_binning.get_values_as_ndarray()

signal_events, weights = signal_predictor(signal_xsec)
signal_truth_binning.set_values_from_ndarray(signal_events)
pltr = plotting.get_plotter(signal_truth_binning)
pltr.plot_values(density=False)
pltr.savefig("few_events.png")

# Create noise predictor and combine
noise_predictor = likelihood.TemplatePredictor([[1.0]])
event_predictor = likelihood.ConcatenatedPredictor(
    [noise_predictor, bg_predictor, signal_predictor]
)

parameter_binning = truth_binning.marginalize_subbinnings()
parameter_binning = parameter_binning.insert_subbinning(1, bg_xsec_binning)
parameter_binning = parameter_binning.insert_subbinning(2, signal_xsec_binning)

# Create some theory templates
noise_template = np.zeros(parameter_binning.data_size)
noise_template[0] = 1.0

bg_template = np.zeros(parameter_binning.data_size)
bg_xsec = np.zeros(bg_xsec_binning.data_size)
bg_offset = 1
for i, b in enumerate(bg_xsec_binning.bins):
    # Get truth and flux bin from Cartesian Product
    truth_bin, flux_bin = b.get_marginal_bins()

    print(i)
    print(truth_bin)
    print(flux_bin)

    break

from scipy.stats import expon, norm, uniform


def calculate_bg_xsec(E_min, E_max, x_min, x_max, y_min, y_max):
    """Calculate the cross section for the BG process."""

    # We need to make an assumption about the E dsitribution within the E bin
    # Bin edges can be +/- np.inf
    if np.isfinite(E_min) and np.isfinite(E_max):
        # Uniform in given bounds
        E_dist = uniform(loc=E_min, scale=E_max - E_min)
    elif np.isfinite(E_min):
        # Exponential from E_min to inf
        E_dist = expon(loc=E_min)
    else:
        # Exponential from -inf to E_max
        E_dist = expon(loc=E_max, scale=-1)

    # Simple overall cross section: One unit of exposure yields 30 true events
    xsec = 30.0

    # True x is True E with a shift and scale
    # Average XSEC in bin is proportional to overlap
    E_0 = (x_min - 0.5) * 2 / np.sqrt(0.5) + 8
    E_1 = (x_max - 0.5) * 2 / np.sqrt(0.5) + 8
    lower = max(E_min, E_0)
    upper = min(E_max, E_1)
    if upper >= lower:
        xsec *= E_dist.cdf(upper) - E_dist.cdf(lower)
    else:
        xsec = 0.0

    # Differential XSEC in y is Gaussian
    # Independent of x
    y_dist = norm(loc=0.5, scale=np.sqrt(0.5))
    xsec *= y_dist.cdf(y_max) - y_dist.cdf(y_min)

    return xsec


for i, b in enumerate(bg_xsec_binning.bins):
    # Get truth and flux bin from Cartesian Product
    truth_bin, flux_bin = b.get_marginal_bins()

    E_min, E_max = flux_bin.edges[0]
    x_min, x_max = truth_bin.edges[0]
    y_min, y_max = truth_bin.edges[1]

    bg_xsec[i] = calculate_bg_xsec(E_min, E_max, x_min, x_max, y_min, y_max)
    bg_template[i + bg_offset] = bg_xsec[i]

pltr = plotting.get_plotter(bg_xsec_binning)
pltr.plot_array(bg_xsec)
pltr.savefig("bg_xsec.png")

bg_xsec_plot_binning = binning.RectilinearBinning(
    bg_truth_binning.variables + (flux_binning.variable,),
    bg_truth_binning.bin_edges + (flux_binning.bin_edges,),
)
pltr = plotting.get_plotter(bg_xsec_plot_binning)
pltr.plot_array(bg_xsec, density=[0, 1], hatch=None)
pltr.savefig("bg_xsec_pretty.png")

bg_truth_binning.reset()
bg_truth_binning.fill_from_csv_file("../05/bg_truth.txt", weight=0.1)
pltr = plotting.get_plotter(bg_truth_binning)
pltr.plot_values(scatter=500, label="generator")

bg_pred, w = bg_predictor(bg_xsec)
pltr.plot_array(bg_pred, scatter=500, label="xsec")

pltr.legend()
pltr.savefig("bg_prediction.png")

model_A_xsec = np.zeros(signal_xsec_binning.data_size)
model_A_template = np.zeros(parameter_binning.data_size)
signal_offset = 1 + bg_xsec_binning.data_size


def calculate_model_A_xsec(E_min, E_max, x_min, x_max, y_min, y_max):
    """Calculate the cross section for the model A process."""

    # We need to make an assumption about the E dsitribution within the E bin
    # Bin edges can be +/- np.inf
    if np.isfinite(E_min) and np.isfinite(E_max):
        # Uniform in given bounds
        E_dist = uniform(loc=E_min, scale=E_max - E_min)
    elif np.isfinite(E_min):
        # Exponential from E_min to inf
        E_dist = expon(loc=E_min)
    else:
        # Exponential from -inf to E_max
        E_dist = expon(loc=E_max, scale=-1)

    # Simple overall cross section: One unit of exposure yields 100 true events
    xsec = 100.0

    # True x is True E with a shift and scale
    # Average XSEC in bin is proportional to overlap
    E_0 = (x_min - 0.1) * 2.0 + 8
    E_1 = (x_max - 0.1) * 2.0 + 8
    lower = max(E_min, E_0)
    upper = min(E_max, E_1)
    if upper >= lower:
        xsec *= E_dist.cdf(upper) - E_dist.cdf(lower)
    else:
        xsec = 0.0

    # Differential XSEC in y is Gaussian
    # Independent of x
    y_dist = norm(loc=0.2, scale=1.0)
    xsec *= y_dist.cdf(y_max) - y_dist.cdf(y_min)

    return xsec


for i, b in enumerate(signal_xsec_binning.bins):
    # Get truth and flux bin from Cartesian Product
    truth_bin, flux_bin = b.get_marginal_bins()

    E_min, E_max = flux_bin.edges[0]
    x_min, x_max = truth_bin.edges[0]
    y_min, y_max = truth_bin.edges[1]

    model_A_xsec[i] = calculate_model_A_xsec(E_min, E_max, x_min, x_max, y_min, y_max)
    model_A_template[i + signal_offset] = model_A_xsec[i]

signal_xsec_plot_binning = binning.RectilinearBinning(
    signal_truth_binning.variables + (flux_binning.variable,),
    signal_truth_binning.bin_edges + (flux_binning.bin_edges,),
)
pltr = plotting.get_plotter(signal_xsec_plot_binning)
pltr.plot_array(model_A_xsec, density=[0, 1], hatch=None)
pltr.savefig("model_A_xsec.png")

signal_truth_binning.reset()
signal_truth_binning.fill_from_csv_file("../00/modelA_truth.txt", weight=0.1)
pltr = plotting.get_plotter(signal_truth_binning)
pltr.plot_values(scatter=500, label="generator")

model_A_pred, w = signal_predictor(model_A_xsec)
pltr.plot_array(model_A_pred, scatter=500, label="xsec")

pltr.legend()
pltr.savefig("model_A_prediction.png")

model_B_xsec = np.zeros(signal_xsec_binning.data_size)
model_B_template = np.zeros(parameter_binning.data_size)
signal_offset = 1 + bg_xsec_binning.data_size


def calculate_model_B_xsec(E_min, E_max, x_min, x_max, y_min, y_max):
    """Calculate the cross section for the model A process."""

    # We need to make an assumption about the E dsitribution within the E bin
    # Bin edges can be +/- np.inf
    if np.isfinite(E_min) and np.isfinite(E_max):
        # Uniform in given bounds
        E_dist = uniform(loc=E_min, scale=E_max - E_min)
    elif np.isfinite(E_min):
        # Exponential from E_min to inf
        E_dist = expon(loc=E_min)
    else:
        # Exponential from -inf to E_max
        E_dist = expon(loc=E_max, scale=-1)

    # Simple overall cross section: One unit of exposure yields 100 true events
    xsec = 100.0

    # True x is True E with a shift and scale
    # Average XSEC in bin is proportional to overlap
    E_0 = (x_min - 0.0) * 2.0 + 8
    E_1 = (x_max - 0.0) * 2.0 + 8
    lower = max(E_min, E_0)
    upper = min(E_max, E_1)
    if upper >= lower:
        xsec *= E_dist.cdf(upper) - E_dist.cdf(lower)
    else:
        xsec = 0.0

    # Differential XSEC in y is Gaussian
    # Correalted with x
    # Should integrate 2D distribution of x/E and y
    # Instead, cheat and assume median E value
    if np.isfinite(lower) and np.isfinite(upper):
        E_m = (upper + lower) / 2
    elif np.isfinite(lower):
        E_m = lower + 1
    elif np.isfinite(upper):
        E_m = upper - 1
    else:
        E_m = 0.0
    x_m = (E_m - 8.0) / 2

    y_dist = norm(loc=0.5 * x_m, scale=1.0 - 0.5**2)
    xsec *= y_dist.cdf(y_max) - y_dist.cdf(y_min)

    return xsec


for i, b in enumerate(signal_xsec_binning.bins):
    # Get truth and flux bin from Cartesian Product
    truth_bin, flux_bin = b.get_marginal_bins()

    E_min, E_max = flux_bin.edges[0]
    x_min, x_max = truth_bin.edges[0]
    y_min, y_max = truth_bin.edges[1]

    model_B_xsec[i] = calculate_model_B_xsec(E_min, E_max, x_min, x_max, y_min, y_max)
    model_B_template[i + signal_offset] = model_B_xsec[i]

pltr = plotting.get_plotter(signal_xsec_plot_binning)
pltr.plot_array(model_B_xsec, density=[0, 1], hatch=None)
pltr.savefig("model_B_xsec.png")

signal_truth_binning.reset()
signal_truth_binning.fill_from_csv_file("../00/modelB_truth.txt", weight=0.1)
pltr = plotting.get_plotter(signal_truth_binning)
pltr.plot_values(scatter=500, label="generator")

model_B_pred, w = signal_predictor(model_B_xsec)
pltr.plot_array(model_B_pred, scatter=500, label="xsec")

pltr.legend()
pltr.savefig("model_B_prediction.png")

# Template predictor for noise, bg, model A, model B
xsec_template_predictor = likelihood.TemplatePredictor(
    [noise_template, bg_template, model_A_template, model_B_template]
)

# Load data and response matrix

with open("../01/reco-binning.yml") as f:
    reco_binning = binning.yaml.full_load(f)

reco_binning.fill_from_csv_file("../05/real_data.txt")
data = reco_binning.get_entries_as_ndarray()
data_model = likelihood.PoissonData(data)

response_matrix = "../05/response_matrix.npz"
matrix_predictor = likelihood.ResponseMatrixPredictor(response_matrix)

# Combine into linear predictor
data_predictor = likelihood.ComposedMatrixPredictor(
    [matrix_predictor, event_predictor, xsec_template_predictor]
)

# Likelihood caclulator and hypothesis tester
calc = likelihood.LikelihoodCalculator(data_model, data_predictor)
maxi = likelihood.BasinHoppingMaximizer()

# Fit everything
ret = maxi.maximize_log_likelihood(calc)
print(ret)

# Calculate p-values for hypotheses overall
calc_A = calc.fix_parameters([None, None, None, 0.0])
calc_B = calc.fix_parameters([None, None, 0.0, None])
test_A = likelihood.HypothesisTester(calc_A)
test_B = likelihood.HypothesisTester(calc_B)
print(test_A.max_likelihood_p_value())
print(test_B.max_likelihood_p_value())


# Get Wilks' p-values for models
norms = np.linspace(0.5, 1.5, 50)
p_values_A = []
p_values_B = []
for n in norms:
    p_values_A.append(test_A.wilks_max_likelihood_ratio_p_value([None, None, n]))
    p_values_B.append(test_B.wilks_max_likelihood_ratio_p_value([None, None, n]))

from matplotlib import pyplot as plt

fig, ax = plt.subplots()
ax.set_xlabel("Model weight")
ax.set_ylabel("p-value")
ax.plot(norms, p_values_A, label="Model A", color="C1")
ax.plot(norms, p_values_B, label="Model B", color="C2")
ax.axhline(0.32, color="k", linestyle="dashed")
ax.axhline(0.05, color="k", linestyle="dashed")
ax.legend(loc="best")
fig.savefig("wilks-p-values.png")
