# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- LinearizedPredictor
- SummedPredictor
- ConcatenatedPredictor

### Changed
- Calculate M. distance truth-bin by bin to avoid memory issues.
- Relax requirements for exact Matplotlib version.
- Renamed LinearPredictor to MatrixPredictor

### Removed
- Removed support for Python versions below 3.7, test up to 3.11 now.

## [0.6.0] - 2019-11-13
