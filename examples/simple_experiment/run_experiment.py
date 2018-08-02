#!/usr/bin/env python

"""Script to run the toy experiment.

This creates some output data like a real experiment.
"""

import argparse
import experiment
import numpy as np
import csv

parser = argparse.ArgumentParser(description="Run the experiment and get reconstructed events.")
parser.add_argument('years', type=float, help="how many years worth of data should be geneated")
parser.add_argument('filename', help="where to store the data")
args = parser.parse_args()

# Get the "true" models
bg = experiment.BackgroundGenerator(cross_section=110, mean_momentum=52, boost_factor=0.0)
sig = experiment.ModelBGenerator(cross_section=90, mean_momentum=110, boost_factor=0.5)

# Generate the events
bg_events = bg.generate_exposed(args.years)
sig_events = sig.generate_exposed(args.years)

# Randomly mix events
true_events = np.concatenate((bg_events, sig_events))
np.random.shuffle(true_events)

# Reconstruct events
detector = experiment.Detector(momentum_threshold=49., momentum_turnon=8., cap_efficiency=0.48, barrel_efficiency=0.91, gap_costheta=0.68, gap_width=0.029, gap_turnon=0.015, momentum_resolution=0.011, angular_resolution=0.009)
reco_events = detector.reconstruct(true_events, keep_truth=False)

csvfields = reco_events.dtype.names
with open(args.filename, 'wt') as f:
    writer = csv.DictWriter(f, csvfields, delimiter=',')
    writer.writerow(dict((fn,fn) for fn in csvfields)) # Write the field names
    for event in reco_events:
        writer.writerow({k: event[k] for k in event.dtype.names})
