from setuptools import setup

# Todo: Parse this from a proper readme file in the future
description='Response Matrix Utilities'
long_description = """ReMU - Response Matrix Utilities

ReMU is a framework for response matrices in cross-section measurements and
general counting experiments. It offers functions to build response matrices,
and use them to compare truth-space expectation values with actual experimental
results in reconstructed space.

Systematic uncertainties about the detector response can be included in the
matrices.  Done correctly, this enables anyone to test arbitrary models against
data that was published together with a response matrix, including systematic
detector effects. Intimate knowledge of the detector is only needed to build
the response matrix, *not* to use it.
"""

def get_version():
    """Get the version number by parsing the package's __init__.py."""
    with open("remu/__init__.py", 'rt') as f:
        for line in f:
            if line.startswith("__version__ = "):
                return eval(line[14:])
        else:
            raise RuntimeError("Could not determine package version!")

setup(name='remu',
    version=get_version(),
    description=description,
    long_description=long_description,
    url='http://github.com/ast0815/remu',
    author='Lukas Koch',
    author_email='lukas.koch@mailbox.org',
    license='MIT',
    packages=['remu'],
    install_requires=['pyyaml>=5.1.2', 'numpy>=1.11.3', 'scipy>=0.18.1', 'six>=1.10.0', 'matplotlib>=2.2.4'],
    extras_require = {
        "mcmc": ['emcee>=3.0.0'],
    },
    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*',
    classifiers=[
      # How mature is this project? Common values are
      #   3 - Alpha
      #   4 - Beta
      #   5 - Production/Stable
      'Development Status :: 4 - Beta',

      # Indicate who your project is intended for
      'Intended Audience :: Developers',
      'Intended Audience :: Science/Research',
      'Topic :: Scientific/Engineering',

      # Pick your license as you wish (should match "license" above)
       'License :: OSI Approved :: MIT License',

      # Specify the Python versions you support here. In particular, ensure
      # that you indicate whether you support Python 2, Python 3 or both.
      'Programming Language :: Python :: 2',
      'Programming Language :: Python :: 2.7',
      'Programming Language :: Python :: 3',
      'Programming Language :: Python :: 3.4',
    ],
    zip_safe=True)
