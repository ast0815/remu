import numpy as np
import pandas as pd

from remu import binning, plotting

pd.set_option("display.max_rows", 10)

px = np.random.randn(1000) * 20
py = np.random.randn(1000) * 20
pz = np.random.randn(1000) * 20
df = pd.DataFrame({"px": px, "py": py, "pz": pz})
with open("df.txt", "w") as f:
    print(df, file=f)

with open("muon-binning.yml") as f:
    muon_binning = binning.yaml.full_load(f)

muon_binning.fill(df)

pltr = plotting.get_plotter(muon_binning, ["py", "pz"], ["px"])
pltr.plot_values()
pltr.savefig("pandas.png")

import uproot

flat_tree = uproot.open("Zmumu.root")["events"]
with open("flat_keys.txt", "w") as f:
    print(flat_tree.keys(), file=f)

df = flat_tree.pandas.df()
with open("flat_df.txt", "w") as f:
    print(df, file=f)

muon_binning.reset()
muon_binning.fill(df, rename={"px1": "px", "py1": "py", "pz1": "pz"})

pltr = plotting.get_plotter(muon_binning, ["py", "pz"], ["px"])
pltr.plot_values()
pltr.savefig("flat_muons.png")

structured_tree = uproot.open("HZZ.root")["events"]
with open("structured_keys.txt", "w") as f:
    print(structured_tree.keys(), file=f)

df = structured_tree.pandas.df(flatten=False)
with open("structured_df.txt", "w") as f:
    print(df, file=f)

df = structured_tree.pandas.df(["NMuon", "Muon_Px", "Muon_Py", "Muon_Pz"])
with open("flattened_df.txt", "w") as f:
    print(df, file=f)

df = df.loc[(slice(None), 0), :]
with open("sliced_df.txt", "w") as f:
    print(df, file=f)

muon_binning.reset()
muon_binning.fill(df, rename={"Muon_Px": "px", "Muon_Py": "py", "Muon_Pz": "pz"})

pltr = plotting.get_plotter(muon_binning, ["py", "pz"], ["px"])
pltr.plot_values()
pltr.savefig("sliced_muons.png")
