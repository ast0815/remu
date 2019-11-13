from remu import binning
from remu import plotting
import numpy as np

with open("../01/reco-binning.yml", 'r') as f:
    reco_binning = binning.yaml.full_load(f)

# Get real data
reco_binning.fill_from_csv_file("../00/real_data.txt")
data = reco_binning.get_values_as_ndarray()

# Get toys
n_toys = 100
toyA = []
toyB = []
for i in range(n_toys):
    reco_binning.reset()
    reco_binning.fill_from_csv_file("modelA_data.txt",
        weightfield='weight_%i'%(i,), rename={'reco_x_%i'%(i,): 'reco_x'},
        buffer_csv_files=True)
    toyA.append(reco_binning.get_values_as_ndarray())

    reco_binning.reset()
    reco_binning.fill_from_csv_file("modelB_data.txt",
        weightfield='weight_%i'%(i,), rename={'reco_x_%i'%(i,): 'reco_x'},
        buffer_csv_files=True)
    toyB.append(reco_binning.get_values_as_ndarray())

# Plot
pltr = plotting.get_plotter(reco_binning)
for A, B in zip(toyA, toyB):
    pltr.plot_array(A/10., alpha=0.05, edgecolor='C1', hatch=None)
    pltr.plot_array(B/10., alpha=0.05, edgecolor='C2', hatch=None)

A = np.mean(toyA, axis=0)
B = np.mean(toyB, axis=0)
pltr.plot_array(data, label="Data", hatch=None, edgecolor='C0', linewidth=2)
pltr.plot_array(A/10., label="Model A", hatch=None, edgecolor='C1')
pltr.plot_array(B/10., label="Model B", hatch=None, edgecolor='C2')
pltr.legend()
pltr.savefig("data.png")
