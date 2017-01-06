import numpy as np

class ResponseMatrix(object):
    """Matrix that describes the detector response to true events."""

    def __init__(self, reco_binning, truth_binning):
        """Initilize the Response Matrix.

        Arguments
        ---------

        truth_binning: The Binning object describing the truth categorization.
        reco_binning: The Binning object describing the reco categorization.

        The binnings will be combined with `cartesian_product`.
        """

        self._truth_binning = truth_binning
        self._reco_binning = reco_binning
        self._response_binning = reco_binning.cartesian_product(truth_binning)

    def fill(self, event, weight=1.):
        """Fill events into the binnings."""
        self._truth_binning.fill(event, weight)
        self._reco_binning.fill(event, weight)
        self._response_binning.fill(event, weight)

    def fill_from_csv_file(self, filename, weightfield=None):
        """Fill binnings from csv file."""
        self._truth_binning.fill_from_csv_file(filename, weightfield)
        self._reco_binning.fill_from_csv_file(filename, weightfield)
        self._response_binning.fill_from_csv_file(filename, weightfield)

    def reset(self):
        """Reset all binnings."""
        self._truth_binning.reset()
        self._reco_binning.reset()
        self._response_binning.reset()
