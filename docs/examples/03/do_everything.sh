#!/bin/bash


# Vary detector simulations
../simple_experiment/vary_detector.py ../00/modelA_data.txt modelA_data.txt
../simple_experiment/vary_detector.py ../00/modelB_data.txt modelB_data.txt

# Plot varied data
python plot_data.py

# Build the uncertain response matrix
python build_response_matrix.py

# Fit stuff
python fit_models.py
