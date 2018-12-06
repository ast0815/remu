from remu import binning
import numpy as np

with open("../01/reco-binning.yml", 'r') as f:
    reco_binning = binning.yaml.load(f)

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
figax = None
for A, B in zip(toyA, toyB):
    figax = reco_binning.plot_ndarray(None, A/10.,
        kwargs1d={'alpha':0.05, 'color':'b'}, figax=figax)
    figax = reco_binning.plot_ndarray(None, B/10.,
        kwargs1d={'alpha':0.05, 'color':'r'}, figax=figax)

A = np.mean(toyA, axis=0)
B = np.mean(toyB, axis=0)
figax = reco_binning.plot_ndarray(None, A/10.,
    kwargs1d={'color':'b', 'label':"Model A"}, figax=figax)
figax = reco_binning.plot_ndarray(None, B/10.,
    kwargs1d={'color':'r', 'label':"Model B"}, figax=figax)
figax = reco_binning.plot_ndarray("data.png", data,
    kwargs1d={'color':'k', 'label':"Data"}, sqrt_errors=True, figax=figax)
