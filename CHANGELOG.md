# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Possibility to use parallelisation to speed up binning filling

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
