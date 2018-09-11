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
properties of the simulation, e.g. whether to simulate background or signal
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

The file 'reco-binning.yml' contains a RectangularBinning object for the reconstructed
information::

    !RecBinning
    binedges:
    - - reco_costheta
      - [-1.0,
        -0.5,
        0.0,
        0.5,
        1.0001]
    - - reco_momentum
      - [0.0,
        100.,
        200.,
        300.,
        400.,
        500.,
        600.,
        700.,
        800.,
        900.,
        1000.,
        .inf]
    include_upper: false

A RectangularBinning object defines bin edges in multiple variables. These
variables are orthogonal to each other. The total number of bins is thus the
product of the number of bins per variable.

Let's create a binning object, load the data into it, and plot the
distributions::

    >>> from remu import binning
    >>>
    >>> with open("reco-binning.yml", 'r') as f:
    >>>     reco_binning = binning.yaml.load(f)
    >>>
    >>> reco_binning.fill_from_csv_file("real_data.txt")
    >>> reco_binning.plot_values("real_data.png", variables=(None, None))
    >>>
    >>> reco_binning.reset()
    >>> reco_binning.fill_from_csv_file("modelA_data.txt")
    >>> reco_binning.plot_values("modelA_data.png", variables=(None, None))
    >>>
    >>> reco_binning.reset()
    >>> reco_binning.fill_from_csv_file("modelB_data.txt")
    >>> reco_binning.plot_values("modelB_data.png", variables=(None, None))
    >>>
    >>> reco_binning.reset()
    >>> reco_binning.fill_from_csv_file("background_data.txt")
    >>> reco_binning.plot_values("background_data.png", variables=(None, None))

We can do the same with the true information and its respective binning in
'truth-binning.yml'::

    >>> with open("truth-binning.yml", 'r') as f:
    >>>     truth_binning = binning.yaml.load(f)
    >>>
    >>> truth_binning.fill_from_csv_file("modelA_truth.txt")
    >>> truth_binning.plot_values("modelA_truth.png", variables=(None, None))
    >>>
    >>> truth_binning.reset()
    >>> truth_binning.fill_from_csv_file("modelB_truth.txt")
    >>> truth_binning.plot_values("modelB_truth.png", variables=(None, None))
    >>>
    >>> truth_binning.reset()
    >>> truth_binning.fill_from_csv_file("background_truth.txt")
    >>> truth_binning.plot_values("background_truth.png", variables=(None, None))
