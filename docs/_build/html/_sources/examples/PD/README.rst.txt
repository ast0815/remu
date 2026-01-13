.. _examplePD:

========================================================
Example PD -- Advanced data loading with pandas and ROOT
========================================================

Aims
====

*   Use pandas :class:`DataFrame` to fill a :class:`.Binning`
*   Use uproot to load ROOT files and fill them into a :class:`.Binning`

Instructions
============

Pandas is an open source, BSD-licensed library providing high-performance,
easy-to-use data structures and data analysis tools for the Python programming
language:

https://pandas.pydata.org/

It provides a :class:`DataFrame` class, which is a useful tool to organise
structured data::

    from remu import binning
    from remu import plotting
    import numpy as np
    import pandas as pd
    pd.set_option('display.max_rows', 10)

    px = np.random.randn(1000)*20
    py = np.random.randn(1000)*20
    pz = np.random.randn(1000)*20
    df = pd.DataFrame({'px': px, 'py': py, 'pz': pz})
    print(df)

.. include:: df.txt
    :literal:

ReMU supports :class:`DataFrame` objects as inputs for all
:meth:`fill<.Binning.fill>` methods::

    with open("muon-binning.yml", 'r') as f:
        muon_binning = binning.yaml.full_load(f)

    muon_binning.fill(df)

    pltr = plotting.get_plotter(muon_binning, ['py','pz'], ['px'])
    pltr.plot_values()
    pltr.savefig("pandas.png")

.. image:: pandas.png

This way, ReMU supports the same input file formats as the pandas library,
e.g. CSV, JSON, HDF5, SQL, etc..

Using the uproot library, pandas can also be used to load ROOT files:

https://github.com/scikit-hep/uproot5

The ROOT framework is the de-facto standard for data analysis in high energy
particle physics:

https://root.cern.ch/

Uproot does *not* need the actual ROOT framework to be installed to work. It
can convert a flat ROOT :class:`TTree` directly into a usable pandas
:class:`DataFrame`::

    import uproot

    flat_tree = uproot.open("Zmumu.root")['events']
    print(flat_tree.keys())

.. include:: flat_keys.txt
    :literal:

::

    df = flat_tree.arrays(library="pd")
    print(df)

.. include:: flat_df.txt
    :literal:

::

    muon_binning.reset()
    muon_binning.fill(df, rename={'px1': 'px', 'py1': 'py', 'pz1': 'pz'})

    pltr = plotting.get_plotter(muon_binning, ['py','pz'], ['px'])
    pltr.plot_values()
    pltr.savefig("flat_muons.png")

.. image:: flat_muons.png

ReMU expects exactly one row per event. If the root file is not flat, but has a
more complicated structure, it must be converted first. For example, let us
take a look at a file where each event has varying numbers of reconstructed
particles::

    structured_tree = uproot.open("HZZ.root")['events']
    print(structured_tree.keys())

.. include:: structured_keys.txt
    :literal:

::

    df = structured_tree.arrays(["NMuon", "Muon_Px", "Muon_Py", "Muon_Pz"], library='pd')
    print(df)

.. include:: structured_df.txt
    :literal:

This kind of data frame with "lists" as cell elements can be inconvenient to
handle. But we can flatten it using the power of the `awkward`::

    import awkward as ak

    arr = structured_tree.arrays(["NMuon", "Muon_Px", "Muon_Py", "Muon_Pz"])
    df = ak.to_dataframe(arr)
    print(df)

.. include:: flattened_df.txt
    :literal:

This double-index structure is still not suitable as input for ReMU, though. We
can select only the first muon in each event, to get the required "one event
per row" structure::

    idx = pd.IndexSlice
    df = df.loc[idx[:,0], :]
    print(df)

.. include:: sliced_df.txt
    :literal:

::

    muon_binning.reset()
    muon_binning.fill(df, rename={'Muon_Px': 'px', 'Muon_Py': 'py', 'Muon_Pz': 'pz'})

    pltr = plotting.get_plotter(muon_binning, ['py','pz'], ['px'])
    pltr.plot_values()
    pltr.savefig("sliced_muons.png")

.. image:: sliced_muons.png
