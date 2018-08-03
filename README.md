ReMU - Response Matrix Utilities
================================

A framework for likelihood calculations and hypothesis testing using binned events and response matrices.

[![Build Status](https://travis-ci.org/ast0815/remu.svg?branch=master)](https://travis-ci.org/ast0815/remu)
[![Coverage Status](https://coveralls.io/repos/github/ast0815/remu/badge.svg?branch=master)](https://coveralls.io/github/ast0815/remu?branch=master)
[![DOI](https://zenodo.org/badge/77051099.svg)](https://zenodo.org/badge/latestdoi/77051099)

Setup
-----

### Installing development version from source

It is recommended to run this software in a virtual Python environment
(virtualenv).  This ensures that all required packages are present in the
tested version and do not interfere with other packages installed on the
system.

    $ # Create a new virtual environment
    $ virtualenv ENV
    $ # Activate the environment
    $ . ENV/bin/activate
    $ # Upgrade pip to the latest version (optional)
    $ pip install --upgrade pip
    $ # Install all required packages
    $ pip install -r requirements.txt
    $ # Install optional packages
    $ pip install -r pymc-requirements.txt
    $ pip install -r matplotlib-requirements.txt
    $ pip install -r multiprocess-requirements.txt
    $ # Install actual package
    $ pip install -e .

You might need to install additional system libraries to compile all packages.
PyMC, Matplotlib, and Multiprocess are optional dependencies. They are only
needed if one actually wants to do any Marcov Chain Monte Carlo, plotting, or
parallel computing respectively.

ReMU requires Python 2.7 or >=3.4 for best functionality. Python 2.6 is
supported, but a lot of required packages have dropped support for Python 2.6
in newer releases. The file `requirements26.txt` can be used instead of
`requirements.txt` in the instructions above to install the packages in
versions that still support Python 2.6.

### Installing official releases with `pip`

Alternatively you can install official releases directly with `pip`:

    $ # Create a new virtual environment
    $ virtualenv ENV
    $ # Activate the environment
    $ . ENV/bin/activate
    $ # Upgrade pip to the latest version (optional)
    $ pip install --upgrade pip
    $ # Install remu and its dependencies
    $ pip install remu==1.0.0

If you want to make sure the optional dependencies are also installed,
use pip's 'Extras' syntax:

    $ # install remu including all optional dependencies
    $ pip install remu[mcmc,plotting,parallel]==1.0.0

Tests
-----

Run all test cases of the framework:

    $ pip install -r test-requirements.txt
    $ ./run_tests.sh

Examples
--------

The folder `examples` contains example analyses that show how one can use this
software.

Citing
======

If you use ReMU in a publication, please cite it as follows:

    L. Koch, ReMU - Response Matrix Utilities, http://github.com/ast0815/remu, doi:10.5281/zenodo.1217572

The DOI is different from the one shown in the badge at the top of this page.
The DOI in the citation does not change and always refers to the latest release,
while the displayed DOI in the badge refers to a specific version and changes with every release.
