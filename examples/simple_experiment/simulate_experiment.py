#!/usr/bin/env python

"""Script to simulate the toy experiment.

This creates some output data like a MC simulation.
"""

import argparse
import experiment
import numpy as np
import csv

parser = argparse.ArgumentParser(description="Simulate the experiment and get reconstructed and true events.")
parser.add_argument('years', type=float, help="how many years worth of data should be geneated")
parser.add_argument('model', help="What model to simulate: background, modelA, modelB")
parser.add_argument('datafilename', help="where to store the data")
parser.add_argument('truthfilename', help="where to store the truth")
args = parser.parse_args()

# Get the model
if args.model == 'background':
    gen = experiment.BackgroundGenerator(cross_section=100, mean_momentum=50, boost_factor=0.0)
elif args.model == 'modelA':
    gen = experiment.ModelAGenerator(cross_section=100, mean_momentum=100, boost_factor=0.5)
elif args.model == 'modelB':
    gen = experiment.ModelBGenerator(cross_section=100, mean_momentum=100, boost_factor=0.5)

# Generate the events
true_events = gen.generate_exposed(args.years)

csvfields = true_events.dtype.names
with open(args.truthfilename, 'wt') as f:
    writer = csv.DictWriter(f, csvfields, delimiter=',')
    writer.writerow(dict((fn,fn) for fn in csvfields)) # Write the field names
    for event in true_events:
        writer.writerow({k: event[k] for k in event.dtype.names})

# Reconstruct events
detector = experiment.Detector(momentum_threshold=50., momentum_turnon=10., cap_efficiency=0.50, barrel_efficiency=0.90, gap_costheta=0.7, gap_width=0.03, gap_turnon=0.02, momentum_resolution=0.001, angular_resolution=0.01)
reco_events = detector.reconstruct(true_events, keep_truth=True)

csvfields = reco_events.dtype.names
with open(args.datafilename, 'wt') as f:
    writer = csv.DictWriter(f, csvfields, delimiter=',')
    writer.writerow(dict((fn,fn) for fn in csvfields)) # Write the field names
    for event in reco_events:
        writer.writerow({k: event[k] for k in event.dtype.names})
