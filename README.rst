================================
ReMU - Response Matrix Utilities
================================

A framework for likelihood calculations and hypothesis testing using binned events and response matrices.

|Travis-CI| |Documentation| |Coverage| |MIT-Licence| |DOI|

Setup
=====

Installing development version from source
------------------------------------------

It is recommended to run this software in a virtual Python environment
(virtualenv). This ensures that all required packages are present in the
tested version and do not interfere with other packages installed on the
system::

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
in newer releases. The file ``requirements26.txt`` can be used instead of
``requirements.txt`` in the instructions above to install the packages in
versions that still support Python 2.6.

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
    $ pip install remu[mcmc,plotting,parallel]==1.0.0

Tests
=====

Run all test cases of the framework::

    $ pip install -r test-requirements.txt
    $ ./run_tests.sh

Documentation
=============

Online documentation including examples can be found on the project's readthedocs page:

    `<https://remu.readthedocs.io>`_

Citing
======

If you use ReMU in a publication, please cite it as follows::

    L. Koch, ReMU - Response Matrix Utilities, http://github.com/ast0815/remu, doi:10.5281/zenodo.1217572

Or just use the DOI and let your bibliography manager handle the rest for you.
You can cite specififc versions of the software too. Just follow the link
behind the DOI badge and choose the DOI specific for the release.


.. |Travis-CI| image:: https://travis-ci.org/ast0815/remu.svg?branch=master
    :target: https://travis-ci.org/ast0815/remu
    :alt: [Travis-CI]

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
