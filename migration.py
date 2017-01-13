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

    def get_truth_values_as_ndarray(self, shape=None):
        return self._truth_binning.get_values_as_ndarray(shape)

    def get_truth_entries_as_ndarray(self, shape=None):
        return self._truth_binning.get_entries_as_ndarray(shape)

    def get_reco_values_as_ndarray(self, shape=None):
        return self._reco_binning.get_values_as_ndarray(shape)

    def get_reco_entries_as_ndarray(self, shape=None):
        return self._reco_binning.get_entries_as_ndarray(shape)

    def get_response_values_as_ndarray(self, shape=None):
        return self._response_binning.get_values_as_ndarray(shape)

    def get_response_entries_as_ndarray(self, shape=None):
        return self._response_binning.get_entries_as_ndarray(shape)

    def get_response_matrix_as_ndarray(self, shape=None):
        """Return the ResponseMatrix as a ndarray.

        If no shape is specified, it will be set to `(N_reco, N_truth)`.
        The expected response of a truth vector can then be calculated like this:

            v_reco = response_matrix.dot(v_truth)

        """

        if shape is None:
            shape = (len(self._reco_binning.bins), len(self._truth_binning.bins))

        # Get the bin response entries
        M = self._response_binning.get_values_as_ndarray(shape)

        # Normalize to number of simulated events
        N_t = self._truth_binning.get_values_as_ndarray()
        M /= N_t

        return M
