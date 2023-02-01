.. _example06:

=================================================
Example 06 -- Cross sections & flux uncertainties
=================================================

Aims
====

*   Turn cross sections as input parameters into event numbers using fluxes
*   Implement flux uncertainties

Instructions
============

So far we have concentrated on raw event numbers or simple template scaling
parameters to parameterize the number of events we expect to see in our
detector. But these raw numbers or templates themselves depend on other more
fundamental parameters.

E.g, if you are recording products of a radioactive decay, they will depend on
the number of radioactive nuclei inside the detector, the decay rates into the
different states you are distinguishing, and the total amount of time the
experiment was collecting data::

    n_true_events_in_state_j = N_nuclei * time * decay_rate_to_state_j

Of these three parameters, the decay rate is usually the truly interesting one.
It is the physics one wants to examine with the experimental setup. The number
of nuclei and the time that the experiment ran on the other hand are just
properties of the experimental setup and the data that was recorded. So just
like the detector response, it would be good if we could absorb them into a
:class:`.Predictor`, so that we can do statistical tests with the interesting
physics parameters directly.

A slightly more complicated example is the measurement of interactions of a
particle beam with a target material, e.g. a neutrino beam with a water target.
Here the number of true events depends and the neutrino flux as a function of
the neutrino energy, the number of target molecules in the detector, and the
neutrino interaction cross sections for the different final states
distinguished by the detector::

    n_true_events_j = T * sum_k(sigma_jk * F_k)

Here `T` is the number of targets, `F_k` is the integrated recorded neutrino
flux in neutrino energy bin `k` (unit: neutrinos/m^2), and `sigma_jk` is the
cross section for neutrinos in energy bin `k` to cause a reaction to the final
state `j` that can be recorded by the detector.

In general, a lot of experimental setups can be described with a matrix
multiplication in the form::

    n_true_events_j = sum_k(physics_parameter_matrix_jk * exposure_k)

Depending on the details of the experiment, the parameter matrix and exposure
vector have slightly different meanings. In the case of the radioactive decay
measurement, the physics parameters would be the decay rates, and the exposure
would be the product of number of nuclei and the recording time. In the
neutrino case the parameters would be the cross sections, and the exposure
would be the product of integrated neutrino fluxes and target mass. In the
context of a collider experiment, the exposure would probably be called an
integrated luminosity.

We will extend the previous example and treat it like a neutrino beam experiment.
First we will need to get the binning in the true event properties::

    import numpy as np
    import pandas as pd

    from remu import binning

    with open("../05/truth-binning.yml") as f:
        truth_binning = binning.yaml.full_load(f)

    # Get truth binnings for BG and signal
    bg_truth_binning = truth_binning.subbinnings[1].clone()
    signal_truth_binning = truth_binning.subbinnings[2].clone()

As a binning for the flux, we will use a simple linear binning in the neutrino
energy:

.. include:: flux-binning.yml
    :literal:

::

    # Define flux binning
    with open("flux-binning.yml") as f:
        flux_binning = binning.yaml.full_load(f)

The binning of the cross section is done as a :class:`.CartesianProductBinning`.
Every possible combination of true kinematic bin and true neutrino energy bin
gets its own cross-section values::

    # Create cross-section binnings
    bg_flux_binning = flux_binning.clone()
    bg_xsec_binning = binning.CartesianProductBinning((bg_truth_binning, bg_flux_binning))
    signal_flux_binning = flux_binning.clone()
    signal_xsec_binning = binning.CartesianProductBinning(
        (signal_truth_binning, signal_flux_binning)
    )

We can have a look at the structure of the resulting binning::

    # Check binning structure
    n_bg_truth = bg_truth_binning.data_size
    n_signal_truth = signal_truth_binning.data_size
    n_flux = flux_binning.data_size
    n_bg_xsec = bg_xsec_binning.data_size
    n_signal_xsec = signal_xsec_binning.data_size
    print(n_bg_truth, n_signal_truth, n_flux)
    print(n_bg_xsec, n_signal_xsec)
    print(signal_xsec_binning.bins[0].data_indices)
    print(signal_xsec_binning.bins[1].data_indices)
    print(signal_xsec_binning.bins[n_flux].data_indices)

.. include:: check_binning.txt
    :literal:

An increase of the bin index by one means that we increase the corresponding
data index in the flux binning by one. Once every flux bin has been stepped
through, the data index in the truth binning increases by one.

To create a :class:`.Predictor` that can turn cross sections into event
numbers, we need an actual neutrino flux. We can easily create one by filling
the flux binning::

    # Fill flux with exposure units (proportional to neutrinos per m^2)
    from remu import plotting
    from numpy.random import default_rng

    rng = default_rng()
    E = rng.normal(loc=8.0, scale=2.0, size=1000)
    df = pd.DataFrame({"E": E})
    flux_binning.fill(df, weight=0.01)

We fill the binning with a weight of 0.01 so that the total amount of exposure
in the binning is 10, corresponding with the assumed 10 years of data taking
used for the event generation in the previous examples. So one "unit" of
exposure corresponds to one year of data taking.

The "simulated" flux looks like this::

    pltr = plotting.get_plotter(flux_binning)
    pltr.plot_values()
    pltr.savefig("flux.png")

.. image:: flux.png

Of course the exact flux is never known (especially with neutrino experiments).
To simulate a flux uncertainty, we can just create lots of throws::

    # Create fluctuated flux predictions
    E_throws = rng.normal(loc=8.0, scale=2.0, size=(100,1000))
    flux = []
    for E in E_throws:
        df = pd.DataFrame({"E": E})
        flux_binning.reset()
        flux_binning.fill(df, weight=0.01)
        flux.append(flux_binning.get_values_as_ndarray())

    flux = np.asfarray(flux)
    print(flux.shape)

.. include:: flux_shape.txt
    :literal:

Now we can use those flux predictions to create a
:class:`.LinearEinsumPredictor` that will do the correct matrix multiplication
to combine the cross sections and the flux into event numbers::

    # Create event number predictors
    from multiprocess import Pool
    from remu import likelihood
    pool = Pool(8)
    likelihood.mapper = pool.map

    bg_predictor = likelihood.LinearEinsumPredictor(
        "ij,...kj->...ik",
        flux,
        reshape_parameters=(n_bg_truth, n_flux),
        bounds=[(0.0, np.inf)] * bg_xsec_binning.data_size,
    )
    signal_predictor = likelihood.LinearEinsumPredictor(
        "ij,...kj->...ik",
        flux,
        reshape_parameters=(n_signal_truth, n_flux),
        bounds=[(0.0, np.inf)] * signal_xsec_binning.data_size,
    )

Let us do a quick test of the predictions::

    # Test cross-section predictions
    signal_xsec_binning.reset()
    signal_xsec_binning.fill({"E": 8.0, "true_x": 0.0, "true_y": 0.0}, 100.0)
    signal_xsec = signal_xsec_binning.get_values_as_ndarray()

    signal_events, weights = signal_predictor(signal_xsec)
    signal_truth_binning.set_values_from_ndarray(signal_events)
    pltr = plotting.get_plotter(signal_truth_binning)
    pltr.plot_values(density=False)
    pltr.savefig("many_events.png")

.. image:: many_events.png

::

    signal_xsec_binning.reset()
    signal_xsec_binning.fill({"E": 3.0, "true_x": 0.0, "true_y": 0.0}, 100.0)
    signal_xsec = signal_xsec_binning.get_values_as_ndarray()

    signal_events, weights = signal_predictor(signal_xsec)
    signal_truth_binning.set_values_from_ndarray(signal_events)
    pltr = plotting.get_plotter(signal_truth_binning)
    pltr.plot_values(density=False)
    pltr.savefig("few_events.png")

.. image:: few_events.png

Despite filling the same cross section to the same true kinematics int the two
cases, the resulting number of events was different. This is of course because
the flux at 3 is smaller than the flux at 8. So even if the cross sections at 3
and 8 are the same, the different flux will lead to a different number of
predicted events.

Finally we need a predictor for the noise events. These are events that do not
correspond to any interesting physics and are just weighted up or down with a
single parameter as input to the response matrix, so we will just pass through
that single parameter. We will use the :class:`.TemplatePredictor` for this,
since it set the parameter limits to ``(0, np.inf)`` by default, which is
convenient here. In order or the systematics to line up, we will have to create
100 identical "variations"::

    # Create noise predictor
    noise_predictor = likelihood.TemplatePredictor([[[1.0]]] * 100)

Now that we have the three separate predictors for the noise, background, and
signal events, we need to combine them into a single
:class:`.ConcatenatedPredictor`::

    # Combine into single predictor
    event_predictor = likelihood.ConcatenatedPredictor(
        [noise_predictor, bg_predictor, signal_predictor],
        combine_systematics="same"
    )

The input of the combined predictor will be a concatenation of the separate
inputs. I.e. first the single noise scaling parameter, then the background
cross-section parameters, and finally the signal cross-section parameters. The
output likewise will be a concatenation of the single output: first the
unmodified noise scaling parameter, then the background events numbers, and
finally the signal event numbers. If we have done everything right, this
corresponds exactly to the meaning of the data of the original `truth_binning`,
so it can be used as input for the response matrix.

At this point we could combine the event predictor with a response matrix and
likelihood calculator to do statistical tests with the cross-section
parameters. With hundreds or thousands of parameters, this is a computationally
intensive task though. So for this tutorial we will again define some models as
templates and investigate them by varying their weights.

In order to to simplify that task, we will first create a binning that
encapsulates all input parameters for the `event_predictor`::

    parameter_binning = truth_binning.marginalize_subbinnings()
    parameter_binning = parameter_binning.insert_subbinning(1, bg_xsec_binning)
    parameter_binning = parameter_binning.insert_subbinning(2, signal_xsec_binning)

We started by marginalizing out all subbinnings in the original
`truth_binning`, which yield a binning that only distinguishes the three event
types in order: noise, background, signal. Then we inserted the cross-section
binnings as subbinnings into their respective top level bins. This leaves us
with a binning where the desired data structure: first a single bin for the
noise, then the background cross-section bins as defined by the
`bg_xsec_binning`, and finally the signal cross-section bins as defined by
`signal_xsec_binning`.

The template for the noise events is simple. Again, we just want to scale the
noise parameter directly::

    noise_template = np.zeros(parameter_binning.data_size)
    noise_template[0] = 1.0
