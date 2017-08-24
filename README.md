likelihood-machine
==================

A framework for hypothesis testing using binned events.

[![Build Status](https://travis-ci.org/ast0815/likelihood-machine.svg?branch=master)](https://travis-ci.org/ast0815/likelihood-machine)
[![Coverage Status](https://coveralls.io/repos/github/ast0815/likelihood-machine/badge.svg?branch=master)](https://coveralls.io/github/ast0815/likelihood-machine?branch=master)

Setup
-----

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
    $ pip install -r pymc-requirements.txt
    $ pip install -r multiprocess-requirements.txt

You might need to install additional system libraries to compile all packages.
PyMC needs to be installed in a separate step, because it requires numpy
already being installed to work. The `multiprocess-requirements` are optional
and only need to be installed, if one wants to use parallel computing.

Tests
-----

Run all test cases of the framework:

    $ pip install -r test-requirements.txt
    $ ./run_tests.sh

Examples
--------

The folder `examples` contains example analyses that show how one can use this
software. The examples need some additional packages to run:

    $ pip install -r example-requirements.txt

They can be run like this:

    $ python examples/example_analysis.py
