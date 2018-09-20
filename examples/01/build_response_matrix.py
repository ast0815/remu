import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from remu import binning
from remu import migration
from copy import deepcopy

with open("../00/reco-binning.yml", 'rt') as f:
    reco_binning = binning.yaml.load(f)
with open("../00/truth-binning.yml", 'rt') as f:
    truth_binning = binning.yaml.load(f)

respA = migration.ResponseMatrix(reco_binning, truth_binning)
reco_binning = deepcopy(reco_binning)
truth_binning = deepcopy(truth_binning)
respB = migration.ResponseMatrix(reco_binning, truth_binning)

respA.fill_from_csv_file("../00/modelA_data.txt")
respA.fill_up_truth_from_csv_file("../00/modelA_truth.txt")
respA.plot_values("response_matrixA.png", variables=(None, None))
respA.plot_expected_efficiency("efficiencyA.png")
respA.plot_in_bin_variation("inbin_var_A.png")

respB.fill_from_csv_file(["../00/modelB_data.txt"])
respB.fill_up_truth_from_csv_file(["../00/modelB_truth.txt"])
respB.plot_values("response_matrixB.png", variables=(None, None))
respB.plot_expected_efficiency("efficiencyB.png")
respB.plot_in_bin_variation("inbin_var_B.png")

respA.plot_distance("mahalanobis_distance.png", respB, expectation=True, variables=(None, None))
respA.plot_compatibility("compatibility.png", respB)

with open("reco-binning.yml", 'rt') as f:
    reco_binning = binning.yaml.load(f)
with open("truth-binning.yml", 'rt') as f:
    truth_binning = binning.yaml.load(f)

respA = migration.ResponseMatrix(reco_binning, truth_binning)
reco_binning = deepcopy(reco_binning)
truth_binning = deepcopy(truth_binning)
respB = migration.ResponseMatrix(reco_binning, truth_binning)

respA.fill_from_csv_file("../00/modelA_data.txt")
respA.fill_up_truth_from_csv_file("../00/modelA_truth.txt")
respA.plot_values("optimised_response_matrixA.png", variables=(None, None))
respA.plot_expected_efficiency("optimised_efficiencyA.png")
respA.plot_in_bin_variation("optimised_inbin_var_A.png")

respB.fill_from_csv_file(["../00/modelB_data.txt"])
respB.fill_up_truth_from_csv_file(["../00/modelB_truth.txt"])
respB.plot_values("optimised_response_matrixB.png", variables=(None, None))
respB.plot_expected_efficiency("optimised_efficiencyB.png")
respB.plot_in_bin_variation("optimised_inbin_var_B.png")

respA.plot_distance("optimised_mahalanobis_distance.png", respB, expectation=True, variables=(None, None))
respA.plot_compatibility("optimised_compatibility.png", respB)
