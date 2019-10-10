#!/bin/bash

# Create "real" data
../simple_experiment/run_experiment.py 10 real_data.txt --enable-background

# Create BG simulation
../simple_experiment/simulate_experiment.py 100 background nominal_bg_data.txt bg_truth.txt

# Create Noise simulation
../simple_experiment/simulate_experiment.py 100 noise noise_data.txt noise_truth.txt

# Create Variations
../simple_experiment/vary_detector.py nominal_bg_data.txt bg_data.txt

# Create the new response matrix
python build_response_matrix.py

# Plot data
python fit_models.py
