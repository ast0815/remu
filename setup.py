from setuptools import setup

# Todo: Parse this from a proper readme file in the future
description='Response Matrix Utilities',
long_description = """ReMU - Response Matrix Utilities
"""

setup(name='remu',
      version='0.1.0',
      description=description,
      long_description=long_description,
      url='http://github.com/ast0815/remu',
      author='Lukas Koch',
      author_email='lukas.koch@mailbox.org',
      license='MIT',
      packages=['remu'],
      install_requires=['ruamel.yaml>=0.11.15', 'numpy>=1.11.3', 'scipy>=0.18.1', 'matplotlib>=1.5.3', 'six>=1.10.0', 'pymc>=2.3.6'],
      python_requires='>=2.6, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*',
      classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',

        # Pick your license as you wish (should match "license" above)
         'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
      ],
      zip_safe=False)
