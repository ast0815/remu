import numpy as np
from remu import binning
from remu import migration
from remu import matrix_utils
from remu import plotting

builder = migration.ResponseMatrixArrayBuilder(1)

with open("../01/reco-binning.yml", "rt") as f:
    reco_binning = binning.yaml.full_load(f)
with open("../01/optimised-truth-binning.yml", "rt") as f:
    signal_binning = binning.yaml.full_load(f)

bg_binning = signal_binning.clone()
resp = migration.ResponseMatrix(reco_binning, bg_binning)
i = 0
resp.fill_from_csv_file(
    "bg_data.txt",
    weightfield="weight_%i" % (i,),
    rename={"reco_x_%i" % (i,): "reco_x"},
    buffer_csv_files=True,
)
resp.fill_up_truth_from_csv_file("bg_truth.txt", buffer_csv_files=True)
entries = resp.get_truth_entries_as_ndarray()
while np.min(entries) < 10:
    resp = matrix_utils.improve_stats(resp)
    entries = resp.get_truth_entries_as_ndarray()
bg_binning = resp.truth_binning
bg_binning.reset()
reco_binning.reset()

truth_binning = binning.LinearBinning(
    variable="event_type",
    bin_edges=[-1.5, -0.5, 0.5, 1.5],
    subbinnings={
        1: bg_binning,
        2: signal_binning,
    },
)

with open("truth-binning.yml", "wt") as f:
    binning.yaml.dump(truth_binning, f)

resp = migration.ResponseMatrix(reco_binning, truth_binning, nuisance_indices=[0])

import numpy.lib.recfunctions as rfn


def set_signal(data):
    return rfn.append_fields(data, "event_type", np.full_like(data["true_x"], 1.0))


def set_bg(data):
    return rfn.append_fields(data, "event_type", np.full_like(data["true_x"], 0.0))


def set_noise(data):
    return rfn.append_fields(data, "event_type", np.full_like(data["reco_x"], -1.0))


n_toys = 100
for i in range(n_toys):
    resp.reset()
    resp.fill_from_csv_file(
        ["../03/modelA_data.txt", "../03/modelB_data.txt"],
        weightfield="weight_%i" % (i,),
        rename={"reco_x_%i" % (i,): "reco_x"},
        cut_function=set_signal,
        buffer_csv_files=True,
    )
    resp.fill_up_truth_from_csv_file(
        ["../00/modelA_truth.txt", "../00/modelB_truth.txt"],
        cut_function=set_signal,
        buffer_csv_files=True,
    )
    resp.fill_from_csv_file(
        "bg_data.txt",
        weightfield="weight_%i" % (i,),
        rename={"reco_x_%i" % (i,): "reco_x"},
        cut_function=set_bg,
        buffer_csv_files=True,
    )
    resp.fill_up_truth_from_csv_file(
        "bg_truth.txt", cut_function=set_bg, buffer_csv_files=True
    )
    # Calling `fill_up_truth_from_csv` twice only works because
    # the files fill completely different bins
    resp.fill_from_csv_file(
        "noise_data.txt", cut_function=set_noise, buffer_csv_files=True
    )
    builder.add_matrix(resp)

builder.export("response_matrix.npz")

pltr = plotting.get_plotter(truth_binning)
pltr.plot_values(density=False)
pltr.savefig("truth.png")

pltr = plotting.BinningPlotter(truth_binning)
pltr.plot_values(density=False)
pltr.savefig("all_truth.png")

pltr = plotting.get_plotter(signal_binning)
pltr.plot_values()
pltr.savefig("signal_truth.png")

pltr = plotting.get_plotter(bg_binning)
pltr.plot_values()
pltr.savefig("bg_truth.png")

matrix_utils.plot_mean_efficiency(resp, "efficiency.png")
