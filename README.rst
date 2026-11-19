================================
ReMU - Response Matrix Utilities
================================

A framework for likelihood calculations and hypothesis testing using binned events and response matrices.

|Documentation| |Coverage| |MIT-Licence| |DOI|

Setup
=====

Installing development version from source
------------------------------------------

It is recommended to run this software in an isolated virtual Python
environment. This ensures that all required packages are present in the
tested version and do not interfere with other packages installed on the
system. The environment is managed by `uv <https://docs.astral.sh/uv/>`_.
The easiest way to create it is to simply source ``setup.sh`` after checking
out the source code::

    $ . setup.sh

This will automatically do something along these lines::

    $ # Create the virtual environment and install all dependencies
    $ uv sync --all-extras
    $ # Activate the environment
    $ . .venv/bin/activate

It will only create the environment if it does not exist already (and keep it
up to date otherwise). You can select a specific Python version by passing it
on to ``uv``, e.g. ``. setup.sh -p 3.12``. You might need to install
additional system libraries to compile all packages.

ReMU requires Python >=3.11.

Installing official releases with ``pip``
-----------------------------------------

Alternatively you can install official releases directly with ``pip``::

    $ # Create a new virtual environment
    $ virtualenv ENV
    $ # Activate the environment
    $ . ENV/bin/activate
    $ # Upgrade pip to the latest version (optional)
    $ pip install --upgrade pip
    $ # Install remu and its dependencies
    $ pip install remu==1.0.0

If you want to make sure the optional dependencies are also installed,
use pip's 'Extras' syntax::

    $ # install remu including all optional dependencies
    $ pip install remu[mcmc]==1.0.0

Tests
=====

Run all test cases of the framework with nox (uses the Python versions
installed on your system)::

    $ uv run nox -s tests

Or, inside the environment created by ``setup.sh``, run them directly with
coverage::

    $ ./run_tests.sh

Online documentation
====================

Online documentation including examples can be found on the project's readthedocs page:

    `<https://remu.readthedocs.io>`_

Citing
======

If you use ReMU in a publication, please cite it as follows::

    L. Koch, ReMU - Response Matrix Utilities, http://github.com/ast0815/remu, doi:10.5281/zenodo.1217572

Or just use the DOI and let your bibliography manager handle the rest for you.
You can cite specific versions of the software too. Just follow the link
behind the DOI badge and choose the DOI specific for the release.


.. |Documentation| image:: https://readthedocs.org/projects/remu/badge/?version=latest
    :target: https://remu.readthedocs.io/en/latest/
    :alt: [Documentation]

.. |Coverage| image:: https://coveralls.io/repos/github/ast0815/remu/badge.svg?branch=master
    :target: https://coveralls.io/github/ast0815/remu?branch=master
    :alt: [Coverage]

.. |MIT-Licence| image:: https://img.shields.io/badge/license-MIT-yellow.svg
    :target: https://opensource.org/licenses/MIT
    :alt: [license: MIT]

.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1217572.svg
    :target: https://doi.org/10.5281/zenodo.1217572
    :alt: [DOI: 10.5281/zenodo.1217572]
