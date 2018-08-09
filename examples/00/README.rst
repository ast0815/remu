==========
Example 00
==========

-----------------------
Simple plotting of data
-----------------------

Aims
====

*   Create "real" and simulated data of the mock experiemnt
*   Load data into histograms and plot it

Instructions
============

The folder '../simple_experiment/' contains two scripts to create "real" and
simulated data. The script 'simulate_experiment.py' simulates the mock
experiment and creates two files: one file with the truth information of all
simulated events, and another file with the truth and reconstructed information
of all reconstructed events. The command line parameters determine the
properties of the simulation, e.g. whether to simulated background or signal
and what signal model to use.

The script 'run_experiment.py' creates a single file with only reconstructed
information. Of course, this file is also  the result of simulations, but since
it is supposed to represent the real results of a real experiment, no truth
information is saved.

Create "real" data corresponding to one year of running the experiment::

    $ ../simple_experiment/run_experiment.py 1 real_data.txt

Create simulated data corresponding to 10 times the real data::

    $ ../simple_experiment/simulate_experiment.py 10 background background_data.txt background_truth.txt
    $ ../simple_experiment/simulate_experiment.py 10 modelA modelA_data.txt modelA_truth.txt
    $ ../simple_experiment/simulate_experiment.py 10 modelB modelB_data.txt modelB_truth.txt
