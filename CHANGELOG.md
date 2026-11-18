# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Removed
- Support for Python 3.7, as it is beyond its lifetime
- Support for Python 3.8, 3.9 and 3.10, as they are beyond their lifetime

### Added
- Support for Python 3.12.
- Support for Python 3.13, 3.14 and 3.15.

### Changed
- Coverage was bumped to 7.x, as 6.5.0 cannot analyse the bytecode of the newer Python versions (coveralls went to 4.x with it, as 3.3.1 requires coverage < 7).
- The example sessions run on 3.11 and 3.14 instead of the newest Python: `dill` 0.4.1 still reads `code.co_lnotab`, which 3.15 removed, so the examples cannot run there until dill releases a fix. The test sessions still cover 3.15.

## [0.8.0] - 2023-06-05

### Added
- Possibility to use parallelisation to speed up binning filling
- Possibility to pass additional kwargs to mcmc sampler generator

### Changed
- Sped up filling from Pandas DataFrames

## [0.7.0] - 2023-02-06

### Added
- LinearizedPredictor
- SummedPredictor
- ConcatenatedPredictor
- LinearEinsumPredictor
- Support for "same" systematics in ComposedPredictors
- `iter_subbins` method to Binnings
- `get_subbins` method to Binnings
- `get_event_subbins` method to Binnings
- `get_marginal_bins` method to CartesianProductBins
- `get_marginal_subbins` method to CartesianProductBins
- Allow selective `density` parameter for CartesianProductPlotter

### Changed
- Calculate M. distance truth-bin by bin to avoid memory issues.
- Relax requirements for exact Matplotlib version.
- Renamed LinearPredictor to MatrixPredictor
- Switched to setuptools as build system.

### Removed
- Removed support for Python versions below 3.7, test up to 3.11 now.

## [0.6.0] - 2019-11-13
