import numpy as np
from remu import binning
from remu import migration
from remu import plotting
from remu import matrix_utils

with open("reco-binning.yml", "rt") as f:
    reco_binning = binning.yaml.full_load(f)
with open("coarse-truth-binning.yml", "rt") as f:
    truth_binning = binning.yaml.full_load(f)

respA = migration.ResponseMatrix(reco_binning, truth_binning)

respA.fill_from_csv_file("../00/modelA_data.txt")
respA.fill_up_truth_from_csv_file("../00/modelA_truth.txt")

matrix_utils.plot_mean_response_matrix(respA, "response_matrix_A.png")

pltr = plotting.get_plotter(respA.truth_binning)
pltr.plot_entries()
pltr.savefig("entries_A.png")

pltr = plotting.get_plotter(respA.truth_binning)
pltr.plot_entries(density=False, hatch=None)
pltr.savefig("abs_entries_A.png")

reco_binning = reco_binning.clone()
truth_binning = truth_binning.clone()
reco_binning.reset()
truth_binning.reset()
respB = migration.ResponseMatrix(reco_binning, truth_binning)

respB.fill_from_csv_file("../00/modelB_data.txt")
respB.fill_up_truth_from_csv_file("../00/modelB_truth.txt")

matrix_utils.plot_mean_response_matrix(respB, "response_matrix_B.png")

matrix_utils.plot_mahalanobis_distance(respA, respB, "mahalanobis_distance.png")
matrix_utils.plot_compatibility(respA, respB, "compatibility.png")

with open("reco-binning.yml", "rt") as f:
    reco_binning = binning.yaml.full_load(f)
with open("fine-truth-binning.yml", "rt") as f:
    truth_binning = binning.yaml.full_load(f)

respA = migration.ResponseMatrix(reco_binning, truth_binning)
reco_binning = reco_binning.clone()
truth_binning = truth_binning.clone()
respB = migration.ResponseMatrix(reco_binning, truth_binning)

respA.fill_from_csv_file("../00/modelA_data.txt")
respA.fill_up_truth_from_csv_file("../00/modelA_truth.txt")

respB.fill_from_csv_file("../00/modelB_data.txt")
respB.fill_up_truth_from_csv_file("../00/modelB_truth.txt")

pltr = plotting.get_plotter(respB.truth_binning)
pltr.plot_entries()
pltr.savefig("fine_entries_B.png")

matrix_utils.plot_mean_response_matrix(respB, "fine_response_matrix_A.png")

matrix_utils.plot_mean_efficiency(respA, "fine_efficiency_A.png")

matrix_utils.plot_mean_efficiency(respB, "fine_efficiency_B.png")

matrix_utils.plot_mahalanobis_distance(respA, respB, "fine_mahalanobis_distance.png")
matrix_utils.plot_compatibility(respA, respB, "fine_compatibility.png")

resp = respA + respB

matrix_utils.plot_in_bin_variation(resp, "fine_inbin_var.png")
matrix_utils.plot_statistical_uncertainty(resp, "fine_stat_var.png")
matrix_utils.plot_relative_in_bin_variation(resp, "fine_rel_inbin_var.png")

pltr = plotting.get_plotter(resp.truth_binning)
pltr.plot_entries()
pltr.savefig("fine_entries.png")

entries = resp.get_truth_entries_as_ndarray()
optimised = resp
while np.min(entries) < 10:
    optimised = matrix_utils.improve_stats(optimised)
    entries = optimised.get_truth_entries_as_ndarray()

pltr = plotting.get_plotter(optimised.truth_binning)
pltr.plot_entries()
pltr.savefig("optimised_entries.png")

pltr = plotting.get_plotter(optimised.truth_binning)
pltr.plot_entries(density=False, label="min", hatch=None, margin_function=np.min)
pltr.plot_entries(density=False, label="max", hatch=None, margin_function=np.max)
pltr.plot_entries(density=False, label="median", hatch=None, margin_function=np.median)
pltr.legend()
pltr.savefig("optimised_abs_entries.png")

matrix_utils.plot_mean_efficiency(optimised, "optimised_efficiency.png")
matrix_utils.plot_relative_in_bin_variation(optimised, "optimised_rel_inbin_var.png")

reco_binning = optimised.reco_binning.clone()
reco_binning.reset()
truth_binning = optimised.truth_binning.clone()
truth_binning.reset()

respA = migration.ResponseMatrix(reco_binning, truth_binning)
reco_binning = reco_binning.clone()
truth_binning = truth_binning.clone()
respB = migration.ResponseMatrix(reco_binning, truth_binning)
respA.fill_from_csv_file("../00/modelA_data.txt")
respA.fill_up_truth_from_csv_file("../00/modelA_truth.txt")
respB.fill_from_csv_file(["../00/modelB_data.txt"])
respB.fill_up_truth_from_csv_file(["../00/modelB_truth.txt"])
matrix_utils.plot_mahalanobis_distance(
    respA, respB, "optimised_mahalanobis_distance.png"
)
matrix_utils.plot_compatibility(respA, respB, "optimised_compatibility.png")

with open("optimised-truth-binning.yml", "w") as f:
    binning.yaml.dump(optimised.truth_binning, f)

optimised.export("response_matrix.npz")
