#!/usr/bin/env python

"""Script to generate toy detector variations.

This modifies the output data of a MC simulation.
"""

import argparse
import experiment
import numpy as np
import numpy.lib.recfunctions as rfn
import csv

parser = argparse.ArgumentParser(
    description="Modify the reconstructed events of a simulation."
)
parser.add_argument("inputfilename", help="where to get the data")
parser.add_argument("datafilename", help="where to store the data")
args = parser.parse_args()

# Nominal detector
nominal_detector = experiment.Detector()
# Toy parameters
np.random.seed(1337)  # Make sure the variations are always the same
n_toys = 100
eff_slopes = np.abs(1.0 + 0.1 * np.random.randn(n_toys))
eff_offsets = 0.1 * np.random.randn(n_toys)
smear_sigmas = np.abs(1.0 + 0.1 * np.random.randn(n_toys))
max_effs = 1.0 - np.abs(0.1 + 0.03 * np.random.randn(n_toys))

# Toy detectors
toy_detectors = []
for slope, offset, sigma, max_eff in zip(
    eff_slopes, eff_offsets, smear_sigmas, max_effs
):
    toy_detectors.append(
        experiment.Detector(
            eff_slope=slope, eff_offset=offset, smear_sigma=sigma, max_eff=max_eff
        )
    )

events = np.genfromtxt(args.inputfilename, names=True, delimiter=",")

# Calculate weights as efficiency ratio of nominal and toy detectors
weights = []
nominal = nominal_detector.efficiency(events)
for i, toy in enumerate(toy_detectors):
    weights.append(
        np.array(toy.efficiency(events) / nominal, dtype=[("weight_%i" % (i,), float)])
    )
weights = rfn.merge_arrays(weights, flatten=True, usemask=False)

# Modify x smearing by sigma ratio
reco_x = []
nominal = 1.0
for i, toy in enumerate(smear_sigmas):
    tru = events["true_x"]
    dif = events["reco_x"] - tru
    new = toy / nominal * dif + tru
    reco_x.append(np.array(new, dtype=[("reco_x_%i" % (i,), float)]))
reco_x = rfn.merge_arrays(reco_x, flatten=True, usemask=False)

events = rfn.drop_fields(events, ["reco_x"])
events = rfn.merge_arrays([events, weights, reco_x], flatten=True, usemask=False)

csvfields = events.dtype.names
with open(args.datafilename, "wt") as f:
    writer = csv.DictWriter(f, csvfields, delimiter=",")
    writer.writerow(dict((fn, fn) for fn in csvfields))  # Write the field names
    for event in events:
        writer.writerow({k: event[k] for k in event.dtype.names})
