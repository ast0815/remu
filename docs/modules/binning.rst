=======
binning
=======

.. automodule:: remu.binning

Classes
=======

.. toctree::
    :maxdepth: 1

    binning/PhaseSpace
    binning/Bin
    binning/RectangularBin
    binning/Binning
    binning/RectangularBinning

YAML interface
==============

All classes defined in `binning` can be stored as and read from YAML files
using the ``binning.yaml`` module::

    >>> with open("filename.yml", 'w') as f:
    >>>     binning.yaml.dump(some_binning, f)
    >>>
    >>> with open("filename.yml", 'r') as f:
    >>>     some_binning = binning.yaml.load(f)
