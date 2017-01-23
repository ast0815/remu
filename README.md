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

Tests
-----

Run all test cases of the framework:

    $ pip install -r test-requirements.txt
    $ ./run_tests.sh
