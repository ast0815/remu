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
parser.add_argument('model', help="what model to simulate", choices=['modelA', 'modelB', 'background', 'noise'])
parser.add_argument('datafilename', help="where to store the data")
parser.add_argument('truthfilename', help="where to store the truth")
args = parser.parse_args()

# Get the model
if args.model == 'modelA':
    gen = experiment.ModelAGenerator(cross_section=100)
elif args.model == 'modelB':
    gen = experiment.ModelBGenerator(cross_section=100)
elif args.model == 'background':
    gen = experiment.BGGenerator(cross_section=30)
elif args.model == 'noise':
    gen = experiment.NoiseGenerator(cross_section=20)

# Generate the events
true_events = gen.generate_exposed(args.years)

csvfields = true_events.dtype.names
with open(args.truthfilename, 'wt') as f:
    writer = csv.DictWriter(f, csvfields, delimiter=',')
    writer.writerow(dict((fn,fn) for fn in csvfields)) # Write the field names
    for event in true_events:
        writer.writerow({k: event[k] for k in event.dtype.names})

# Reconstruct events
if args.model == 'background':
    # Detector reacts differently to BG events
    detector = experiment.Detector(smear_sigma=2.)
    reco_events = detector.reconstruct(true_events, keep_truth=True)
elif args.model == 'noise':
    # Noise is already reco
    reco_events = true_events
else:
    detector = experiment.Detector()
    reco_events = detector.reconstruct(true_events, keep_truth=True)

csvfields = reco_events.dtype.names
with open(args.datafilename, 'wt') as f:
    writer = csv.DictWriter(f, csvfields, delimiter=',')
    writer.writerow(dict((fn,fn) for fn in csvfields)) # Write the field names
    for event in reco_events:
        writer.writerow({k: event[k] for k in event.dtype.names})
