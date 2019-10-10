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
parser.add_argument('--enable-background', help="enable background events", action='store_true')
args = parser.parse_args()

# Get the "true" models
sig = experiment.ModelBGenerator(cross_section=100)
bg = experiment.BGGenerator(cross_section=30)
noise = experiment.NoiseGenerator(cross_section=20)

# Generate the events
sig_events = sig.generate_exposed(args.years)
bg_events = bg.generate_exposed(args.years)
noise_events = noise.generate_exposed(args.years)

# Reconstruct events
detector = experiment.Detector()
reco_sig_events = detector.reconstruct(sig_events, keep_truth=False)
detector = experiment.Detector(smear_sigma=2.) # BG smears differently
reco_bg_events = detector.reconstruct(bg_events, keep_truth=False)
reco_noise_events = noise_events # re already reco

if args.enable_background:
    reco_events = np.concatenate((reco_bg_events, reco_sig_events, noise_events))
    # Randomly mix events
    np.random.shuffle(reco_events)
else:
    reco_events = reco_sig_events

csvfields = reco_events.dtype.names
with open(args.filename, 'wt') as f:
    writer = csv.DictWriter(f, csvfields, delimiter=',')
    writer.writerow(dict((fn,fn) for fn in csvfields)) # Write the field names
    for event in reco_events:
        writer.writerow({k: event[k] for k in event.dtype.names})
