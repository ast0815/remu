import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from remu import binning
from remu import migration
from copy import deepcopy

with open("reco-binning.yml", 'rt') as f:
    reco_binning = binning.yaml.load(f)
with open("coarse-truth-binning.yml", 'rt') as f:
    truth_binning = binning.yaml.load(f)

respA = migration.ResponseMatrix(reco_binning, truth_binning)
reco_binning = deepcopy(reco_binning)
truth_binning = deepcopy(truth_binning)
respB = migration.ResponseMatrix(reco_binning, truth_binning)

respA.fill_from_csv_file("../00/modelA_data.txt")
respA.fill_up_truth_from_csv_file("../00/modelA_truth.txt")
respA.plot_values("response_matrixA.png", variables=(None, None))
respA.plot_expected_efficiency("efficiencyA.png")
respA.plot_in_bin_variation("inbin_var_A.png", variables=(None, None))

respB.fill_from_csv_file(["../00/modelB_data.txt"])
respB.fill_up_truth_from_csv_file(["../00/modelB_truth.txt"])
respB.plot_values("response_matrixB.png", variables=(None, None))
respB.plot_expected_efficiency("efficiencyB.png")
respB.plot_in_bin_variation("inbin_var_B.png", variables=(None, None))

respA.plot_distance("mahalanobis_distance.png", respB, expectation=True, variables=(None, None))
respA.plot_compatibility("compatibility.png", respB)

with open("reco-binning.yml", 'rt') as f:
    reco_binning = binning.yaml.load(f)
with open("fine-truth-binning.yml", 'rt') as f:
    truth_binning = binning.yaml.load(f)

respA = migration.ResponseMatrix(reco_binning, truth_binning)
reco_binning = deepcopy(reco_binning)
truth_binning = deepcopy(truth_binning)
respB = migration.ResponseMatrix(reco_binning, truth_binning)

respA.fill_from_csv_file("../00/modelA_data.txt")
respA.fill_up_truth_from_csv_file("../00/modelA_truth.txt")
respA.plot_values("fine_response_matrixA.png", variables=(None, None))
respA.plot_statistical_variance("fine_stat_varA.png", variables=(None, None))
respA.plot_expected_efficiency("fine_efficiencyA.png")
respA.plot_in_bin_variation("fine_inbin_varA.png", variables=(None, None))

respB.fill_from_csv_file(["../00/modelB_data.txt"])
respB.fill_up_truth_from_csv_file(["../00/modelB_truth.txt"])
respB.plot_values("fine_response_matrixB.png", variables=(None, None))
respB.plot_statistical_variance("fine_stat_varB.png", variables=(None, None))
respB.plot_expected_efficiency("fine_efficiencyB.png")
respB.plot_in_bin_variation("fine_inbin_varB.png", variables=(None, None))

respA.plot_distance("fine_mahalanobis_distance.png", respB, expectation=True, variables=(None, None))
respA.plot_compatibility("fine_compatibility.png", respB)

resp = respA + respB
resp.plot_values("fine_response_matrix.png", variables=(None, None))
resp.plot_statistical_variance("fine_stat_var.png", variables=(None, None))
resp.plot_in_bin_variation("fine_inbin_var.png", variables=(None, None))

entries = resp.get_response_entries_as_ndarray()
optimised = resp.maximize_stats_by_rebinning()
entries = optimised.get_response_entries_as_ndarray()
optimised.plot_values("optimised_response_matrix.png", variables=(None, None))
optimised.plot_statistical_variance("optimised_stat_var.png", variables=(None, None))
optimised.plot_expected_efficiency("optimised_efficiency.png", variables=(None, None))
optimised.plot_in_bin_variation("optimised_inbin_var.png", variables=(None, None))

with open("optimised-truth-binning.yml", 'w') as f:
    binning.yaml.dump(optimised.truth_binning, f)

M = optimised.get_mean_response_matrix_as_ndarray()
np.save("response_matrix.npy", M)

entries = optimised.get_truth_entries_as_ndarray()
np.save("generator_truth.npy", entries)

with open("reco-binning.yml", 'rt') as f:
    reco_binning = binning.yaml.load(f)
with open("optimised-truth-binning.yml", 'rt') as f:
    truth_binning = binning.yaml.load(f)

respA = migration.ResponseMatrix(reco_binning, truth_binning)
reco_binning = deepcopy(reco_binning)
truth_binning = deepcopy(truth_binning)
respB = migration.ResponseMatrix(reco_binning, truth_binning)
respA.fill_from_csv_file("../00/modelA_data.txt")
respA.fill_up_truth_from_csv_file("../00/modelA_truth.txt")
respB.fill_from_csv_file(["../00/modelB_data.txt"])
respB.fill_up_truth_from_csv_file(["../00/modelB_truth.txt"])
respA.plot_distance("optimised_mahalanobis_distance.png", respB, expectation=True, variables=(None, None))
respA.plot_compatibility("optimised_compatibility.png", respB)
