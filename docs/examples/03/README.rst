.. _example03:

====================================
Example 03 -- Detector uncertainties
====================================

Aims
====

*   Generate random detector response variations
*   Include detector uncertainties in the response matrix by using a
    :class:`.ResponseMatrixArrayBuilder`
*   Include statistical uncertainties by generating randomly varied matrices
*   Use array of response matrices in :class:`.LikelihoodMachine`

Instructions
============

The properties of experimental setups are usually only known to a finite
precision. The remaining uncertainty on these properties, and thus the
uncertainty on the detector response to true events, must be reflected in the
response matrix.

We can create an event-by-event variation of the detector response with the
provided script::

    $ ../simple_experiment/vary_detector.py ../00/modelA_data.txt modelA_data.txt
    $ ../simple_experiment/vary_detector.py ../00/modelB_data.txt modelB_data.txt

It creates 100 randomly generated variations of the simulated detector and
varies the simulation that was created in example 00 accordingly. Each
variation describes one possible real detector and is often called a "toy
variation" or a "universe".

Events are varied in two ways: The values of reconstructed variables can be
different for each toy, and each event can get assigned a weight. The former
can be used to cover uncertainties in the detector precision and accuracy,
while the latter is a simple way to deal with uncertainties in the overall
probability of reconstructing an event.

In this case, the values of the reconstructed ``reco_x`` are varied and saved
as ``reco_x_0`` to ``reco_x_99``. The detection efficiency is also varied and
the ratio of the events nominal efficiency and the efficiency assuming the toy
detector stored as ``weight_0`` up to ``weight_99``. Let us plot the different
reconstructed distributions we get from the toy variations:

.. include:: plot_data.py
    :literal:

.. image:: data.png

Here we plot the data with ``sqrt(N)`` "error bars" just to give an indication
of the expected statistical fluctuations. Also we scaled the model predictions
by a factor 10, simply because that makes them correspond roughly to the data.
In this example, the systematic uncertainties seem to be of a similar size to
the expected statistical variations.

We are using a few specific arguments to the ``fill_from_csv_file`` method
here:

``weightfield``
    Tells the object which field to use as the weight of the events.
    Does not need to follow any naming conventions.
    Here we tell it to use the field corresponding to each toy.

``rename``
    The binning of the response matrix expects a variable called ``reco_x``.
    This variable is not present in the varied data sets. Here we can specify a
    dictionary with columns in the data that will be renamed before filling the
    matrix.

``buffer_csv_files``
    Reading a CSV file from disk and turning it into an array of floating point
    values takes quite a bit of time. We can speed up the process by buffering
    the intermediate array on disk. This means the time consuming parsing of a
    text file only has to happen once per CSV file.

ReMU handles the detector uncertainties by building a response matrix for each
toy detector separately. These matrices are then used in parallel to calculate
likelihoods. In the end the marginal (i.e. average) likelihood is used as
the final answer. Some advanced features of ReMU (like sparse matrices, or
nuisance bins) require quite a bit of book-keeping to make the toy matrices
behave consistently and as expected. This is handled by the
:class:`.ResponseMatrixArrayBuilder`::

    import matplotlib.pyplot as plt
    from scipy import stats
    from remu import migration
    from copy import deepcopy

    builder = migration.ResponseMatrixArrayBuilder(1)

Its only argument is the number of randomly drawn statistical variations per
toy response matrix. The number of simulated events that are used to build the
response matrix influences the statistical uncertainty of the response matrix
elements. This can be seen as an additional "systematic" uncertainty of the
detector response. By generating randomly fluctuated response matrices from
the toy matrices, it can be handled organically with the other systematics.
In this case we generate 1 randomly varied matrix per toy matrix, yielding a
total of 100 matrices in the end.

The toy matrices themselves are built like the nominal matrix in the previous
examples and then added to the builder::

    with open("../01/reco-binning.yml", 'rt') as f:
        reco_binning = binning.yaml.load(f)
    with open("../01/optimised-truth-binning.yml", 'rt') as f:
        truth_binning = binning.yaml.load(f)

    resp = migration.ResponseMatrix(reco_binning, truth_binning)

    n_toys = 100
    for i in range(n_toys):
        resp.reset()
        resp.fill_from_csv_file(["modelA_data.txt", "modelB_data.txt"],
            weightfield='weight_%i'%(i,), rename={'reco_x_%i'%(i,): 'reco_x'},
            buffer_csv_files=True)
        resp.fill_up_truth_from_csv_file(["../00/modelA_truth.txt",
            "../00/modelB_truth.txt"], buffer_csv_files=True)
        builder.add_matrix(resp)

The builder generates the actual array of floating point numbers as soon as the
:class:`.ResponseMatrix` object is added. It retains no connection to the object
itself.

Now we just need to save the set of response matrices for later use in the
likelihood fits::

    builder.export("response_matrix.npz")

The :class:`.LikelihoodMachine` is created just like in the previous example::

    from six import print_
    from matplotlib import pyplot as plt
    from remu import likelihood

    with open("../01/reco-binning.yml", 'rt') as f:
        reco_binning = binning.yaml.load(f)
    with open("../01/optimised-truth-binning.yml", 'rt') as f:
        truth_binning = binning.yaml.load(f)

    reco_binning.fill_from_csv_file("../00/real_data.txt")
    data = reco_binning.get_entries_as_ndarray()

    # No systematics LikelihoodMachine
    response_matrix = "../01/response_matrix.npz"
    lm = likelihood.LikelihoodMachine(data, response_matrix, limit_method='prohibit')

    # Systematics LikelihoodMachine
    response_matrix_syst = "response_matrix.npz"
    lm_syst = likelihood.LikelihoodMachine(data, response_matrix_syst, limit_method='prohibit')

To show the influence of the systematic uncertainties, we create two
:class:`.LikelihoodMachine` objects here. One with the average detector
response and one with the set of systematically varied responses. The set of
response matrices is saved as an ``A x B x X x Y`` shaped NumPy array, with the
number of systematic toy variations ``A``, the number of randomly drawn
matrices per toy ``B``, the number of reco bins ``X``, and the number of truth
bins ``Y``. To make handling the matrices easier, we reshape the matrices to an
``N x X x Y`` shaped array here.

Now we can test some models against the data, just like in the previous example::

    truth_binning.fill_from_csv_file("../00/modelA_truth.txt")
    modelA = truth_binning.get_values_as_ndarray()
    modelA /= np.sum(modelA)
    truth_binning.reset()
    truth_binning.fill_from_csv_file("../00/modelB_truth.txt")
    modelB = truth_binning.get_values_as_ndarray()
    modelB /= np.sum(modelB)

    modelA_shape = likelihood.TemplateHypothesis([modelA])
    retA = lm.max_log_likelihood(modelA_shape)
    print_(retA)

.. include:: modelA_fit.txt
    :literal:

::

    retA_syst = lm_syst.max_log_likelihood(modelA_shape)
    print_(retA_syst)

.. include:: modelA_fit_syst.txt
    :literal:

::

    modelB_shape = likelihood.TemplateHypothesis([modelB])
    retB = lm.max_log_likelihood(modelB_shape)
    print_(retB)

.. include:: modelB_fit.txt
    :literal:

::

    retB_syst = lm_syst.max_log_likelihood(modelB_shape)
    print_(retB_syst)

.. include:: modelB_fit_syst.txt
    :literal:

Let us take another look at how the fitted templates and the data compare in
reco space. This time we are going to use `ReMU`'s built in function to plot a
set of bin contents, i.e. the set of model predictions varied by the detector
systematics. It will plot the mean and average value of projections of
binnings, when given a set of bin contents rather than a single one::

    figax = reco_binning.plot_values(None, kwargs1d={'color': 'k', 'label': 'data'}, sqrt_errors=True)
    modelA_reco = lm.fold(modelA_shape.translate(retA.x))
    modelB_reco = lm.fold(modelB_shape.translate(retB.x))
    modelA_reco_syst = lm_syst.fold(modelA_shape.translate(retA_syst.x))
    modelB_reco_syst = lm_syst.fold(modelB_shape.translate(retB_syst.x))
    reco_binning.plot_ndarray(None, modelA_reco,
        kwargs1d={'color': 'b', 'label': 'model A'}, figax=figax)
    reco_binning.plot_ndarray(None, modelA_reco_syst,
        kwargs1d={'color': 'b', 'label': 'model A syst', 'hatch': '\\\\', 'facecolor': 'none'},
        error_band='step', figax=figax)
    reco_binning.plot_ndarray(None, modelB_reco,
        kwargs1d={'color': 'r', 'label': 'model B'}, figax=figax)
    reco_binning.plot_ndarray("reco-comparison.png", modelB_reco_syst,
        kwargs1d={'color': 'r', 'label': 'model B syst', 'hatch': '//', 'facecolor': 'none'},
        error_band='step', figax=figax)

.. image:: reco-comparison.png

And of course we can compare the p-values of the two fitted models assuming the
nominal detector response::

    print_(lm.max_likelihood_p_value(modelA_shape, nproc=4))
    print_(lm.max_likelihood_p_value(modelB_shape, nproc=4))

.. include:: fit_p-values.txt
    :literal:

With the p-values yielded when taking the systematic uncertainties into account::

    print_(lm_syst.max_likelihood_p_value(modelA_shape, nproc=4))
    print_(lm_syst.max_likelihood_p_value(modelB_shape, nproc=4))

.. include:: fit_p-values_syst.txt
    :literal:

We can construct confidence intervals on the template weights of the models
using the maximum likelihood p-value::

    p_values_A = []
    p_values_B = []
    p_values_A_syst = []
    p_values_B_syst = []
    values = np.linspace(600, 1600, 21)
    for v in values:
        fixed_model_A = modelA_shape.fix_parameters((v,))
        fixed_model_B = modelB_shape.fix_parameters((v,))
        A = lm.max_likelihood_p_value(fixed_model_A, nproc=4)
        A_syst = lm_syst.max_likelihood_p_value(fixed_model_A, nproc=4)
        B = lm.max_likelihood_p_value(fixed_model_B, nproc=4)
        B_syst = lm_syst.max_likelihood_p_value(fixed_model_B, nproc=4)
        print_(v, A, A_syst, B, B_syst)
        p_values_A.append(A)
        p_values_B.append(B)
        p_values_A_syst.append(A_syst)
        p_values_B_syst.append(B_syst)

    fig, ax = plt.subplots()
    ax.set_xlabel("Model weight")
    ax.set_ylabel("p-value")
    ax.plot(values, p_values_A, label="Model A", color='b', linestyle='dotted')
    ax.plot(values, p_values_A_syst, label="Model A syst", color='b',
        linestyle='solid')
    ax.plot(values, p_values_B, label="Model B", color='r', linestyle='dotted')
    ax.plot(values, p_values_B_syst, label="Model B syst", color='r',
        linestyle='solid')
    ax.axvline(retA.x[0], color='b', linestyle='dotted')
    ax.axvline(retA_syst.x[0], color='b', linestyle='solid')
    ax.axvline(retB.x[0], color='r', linestyle='dotted')
    ax.axvline(retB_syst.x[0], color='r', linestyle='solid')
    ax.axhline(0.32, color='k', linestyle='dashed')
    ax.axhline(0.05, color='k', linestyle='dashed')
    ax.legend(loc='best')
    fig.savefig("p-values.png")

.. image:: p-values.png

The "fixed" models have no free parameters here, making them degenerate
:class:`.CompositeHypothesis`, equivalent to simple hypotheses. This means
using the :meth:`.max_likelihood_p_value` is equivalent to testing the
corresponding simple hypotheses using the :meth:`.likelihood_p_value`. No
actual maximisation is taking place. Note that the p-values for model A are
generally smaller than those for model B. This is consistent with previous
results showing that the data is better described by model B.

Finally, let us construct confidence intervals of the template weights,
*assuming that the corresponding model is correct*::

    p_values_A = []
    p_values_B = []
    p_values_A_syst = []
    p_values_B_syst = []
    values = np.linspace(600, 1600, 21)
    for v in values:
        fixed_model_A = modelA_shape.fix_parameters((v,))
        fixed_model_B = modelB_shape.fix_parameters((v,))
        A = lm.max_likelihood_ratio_p_value(fixed_model_A, modelA_shape,
            nproc=4)
        A_syst = lm_syst.max_likelihood_ratio_p_value(fixed_model_A,
            modelA_shape, nproc=4)
        B = lm.max_likelihood_ratio_p_value(fixed_model_B, modelB_shape,
            nproc=4)
        B_syst = lm_syst.max_likelihood_ratio_p_value(fixed_model_B,
            modelB_shape, nproc=4)
        print_(v, A, A_syst, B, B_syst)
        p_values_A.append(A)
        p_values_B.append(B)
        p_values_A_syst.append(A_syst)
        p_values_B_syst.append(B_syst)

    p_values_A_wilks = []
    p_values_B_wilks = []
    fine_values = np.linspace(600, 1600, 100)
    for v in fine_values:
        fixed_model_A = modelA_shape.fix_parameters((v,))
        fixed_model_B = modelB_shape.fix_parameters((v,))
        A = lm_syst.wilks_max_likelihood_ratio_p_value(fixed_model_A, modelA_shape)
        B = lm_syst.wilks_max_likelihood_ratio_p_value(fixed_model_B, modelB_shape)
        print_(v, A, B)
        p_values_A_wilks.append(A)
        p_values_B_wilks.append(B)

    fig, ax = plt.subplots()
    ax.set_xlabel("Model weight")
    ax.set_ylabel("p-value")
    ax.plot(values, p_values_A, label="Model A", color='b', linestyle='dotted')
    ax.plot(values, p_values_A_syst, label="Model A syst", color='b',
        linestyle='solid')
    ax.plot(fine_values, p_values_A_wilks, label="Model A Wilks", color='b',
        linestyle='dashed')
    ax.plot(values, p_values_B, label="Model B", color='r', linestyle='dotted')
    ax.plot(values, p_values_B_syst, label="Model B syst", color='r',
        linestyle='solid')
    ax.plot(fine_values, p_values_B_wilks, label="Model B Wilks", color='r',
        linestyle='dashed')
    ax.axvline(retA.x[0], color='b', linestyle='dotted')
    ax.axvline(retA_syst.x[0], color='b', linestyle='solid')
    ax.axvline(retB.x[0], color='r', linestyle='dotted')
    ax.axvline(retB_syst.x[0], color='r', linestyle='solid')
    ax.axhline(0.32, color='k', linestyle='dashed')
    ax.axhline(0.05, color='k', linestyle='dashed')
    ax.legend(loc='best')
    fig.savefig("ratio-p-values.png")

.. image:: ratio-p-values.png

Note that the p-values for both models go up to 1.0 here (within the
granularity of the parameter scan) and that the constructed confidence
intervals are very different from the ones before. This is because we are
asking a different question now. Before we asked the question "What is the
probability of getting a worse likelihood assuming that the tested
model-parameter is true?". Now we ask the question "What is the probability of
getting a worse best-fit likelihood ratio, assuming the tested model-parameter
is true?". Since the likelihood ratio is 1.0 at the best fit point and the
likelihood ratio of nested hypotheses is less than or equal to 1.0 by
construction, the p-value is 100% there.

The method :meth:`.wilks_max_likelihood_ratio_p_value` calculates the same
p-value as :meth:`.max_likelihood_ratio_p_value`, but does so assuming Wilks'
theorem holds. This does not require the generation of random data and
subsequent likelihood maximisations, so it is much faster.
