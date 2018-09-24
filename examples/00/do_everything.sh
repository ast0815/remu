#!/bin/bash

# Create "real" data
../simple_experiment/run_experiment.py 10 real_data.txt

# Create simulations
../simple_experiment/simulate_experiment.py 100 background background_data.txt background_truth.txt
../simple_experiment/simulate_experiment.py 100 modelA modelA_data.txt modelA_truth.txt
../simple_experiment/simulate_experiment.py 100 modelB modelB_data.txt modelB_truth.txt

# Plot data
python plot_data.py
