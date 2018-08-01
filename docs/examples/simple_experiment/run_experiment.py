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
sig = experiment.ModelBGenerator()

# Generate the events
sig_events = sig.generate_exposed(args.years)

# Randomly mix events
#true_events = np.concatenate((bg_events, sig_events))
#np.random.shuffle(true_events)
true_events = sig_events

# Reconstruct events
detector = experiment.Detector()
reco_events = detector.reconstruct(true_events, keep_truth=False)

csvfields = reco_events.dtype.names
with open(args.filename, 'wt') as f:
    writer = csv.DictWriter(f, csvfields, delimiter=',')
    writer.writerow(dict((fn,fn) for fn in csvfields)) # Write the field names
    for event in reco_events:
        writer.writerow({k: event[k] for k in event.dtype.names})
