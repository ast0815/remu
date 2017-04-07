import numpy as np
from copy import copy, deepcopy

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

    def fill_up_truth_from_csv_file(self, filename, weightfield=None):
        """Re fill the truth bins with the given csv file.

        This can be used to get proper efficiencies if the true signal events
        are saved in a separate file from the reconstructed events.

        A new truth binning is created and filled with the events from the
        provided file. Each bin is compared to the corresponding bin in the
        already present truth binning. The larger value of the two is taken as
        the new truth. This way, event types that are not present in the pure
        truth data, e.g. background, are not affected by this. It can only
        *increase* the value of the truth bins, lowering their efficiency.

        For each truth bin, one of the following *must* be true for this
        operation to make sense:

        * All events in the migration matrix are also present in the truth
          file. In this case, the additional truth events lower the efficiency
          of the truth bin. This is the case, for example, if not all true signal
          events are reconstructed.

        * All events in the truth file are also present in the migration
          matrix. In this case, the events in the truth file have no influence
          on the response matrix. This is the case, for example, if only a subset
          of the reconstructed background is saved in the truth file.

        If there are events in the response matrix that are not in the truth
        tree *and* there are events in the truth tree that are not in the
        response matrix, this method will lead to a *wrong* efficiency of the
        affected truth bin.
        """

        new_truth_binning = deepcopy(self._truth_binning)
        new_truth_binning.reset()
        new_truth_binning.fill_from_csv_file(filename, weightfield=weightfield)
        new_values = new_truth_binning.get_values_as_ndarray()
        new_entries = new_truth_binning.get_entries_as_ndarray()

        old_values = self._truth_binning.get_values_as_ndarray()
        old_entries = self._truth_binning.get_entries_as_ndarray()

        where = new_values > old_values

        self._truth_binning.set_values_from_ndarray(np.where(where, new_values, old_values))
        self._truth_binning.set_entries_from_ndarray(np.where(where, new_entries, old_entries))

    def reset(self):
        """Reset all binnings."""
        self._truth_binning.reset()
        self._reco_binning.reset()
        self._response_binning.reset()

    def get_truth_values_as_ndarray(self, shape=None):
        return self._truth_binning.get_values_as_ndarray(shape)

    def get_truth_entries_as_ndarray(self, shape=None):
        return self._truth_binning.get_entries_as_ndarray(shape)

    def get_truth_sumw2_as_ndarray(self, shape=None):
        return self._truth_binning.get_sumw2_as_ndarray(shape)

    def get_reco_values_as_ndarray(self, shape=None):
        return self._reco_binning.get_values_as_ndarray(shape)

    def get_reco_entries_as_ndarray(self, shape=None):
        return self._reco_binning.get_entries_as_ndarray(shape)

    def get_reco_sumw2_as_ndarray(self, shape=None):
        return self._reco_binning.get_sumw2_as_ndarray(shape)

    def get_response_values_as_ndarray(self, shape=None):
        return self._response_binning.get_values_as_ndarray(shape)

    def get_response_entries_as_ndarray(self, shape=None):
        return self._response_binning.get_entries_as_ndarray(shape)

    def get_response_sumw2_as_ndarray(self, shape=None):
        return self._response_binning.get_sumw2_as_ndarray(shape)

    def get_response_matrix_as_ndarray(self, shape=None):
        """Return the ResponseMatrix as a ndarray.

        If no shape is specified, it will be set to `(N_reco, N_truth)`.
        The expected response of a truth vector can then be calculated like this:

            v_reco = response_matrix.dot(v_truth)

        """

        original_shape = (len(self._reco_binning.bins), len(self._truth_binning.bins))

        # Get the bin response entries
        M = self._response_binning.get_values_as_ndarray(original_shape)

        # Normalize to number of simulated events
        N_t = self._truth_binning.get_values_as_ndarray()
        M /= np.where(N_t > 0., N_t, 1.)

        if shape is not None:
            M = M.reshape(shape, order='C')

        return M

    def plot_values(self, filename, variables=None, divide=True, kwargs1d={}, kwargs2d={}, figax=None):
        """Plot the values of the response binning.

        This plots the distribution of events that have *both* a truth and reco bin.
        """

        return self._response_binning.plot_values(filename, variables, divide, kwargs1d, kwargs2d, figax)

    def plot_entries(self, filename, variables=None, divide=True, kwargs1d={}, kwargs2d={}, figax=None):
        """Plot the entries of the response binning.

        This plots the distribution of events that have *both* a truth and reco bin.
        """

        return self._response_binning.plot_entries(filename, variables, divide, kwargs1d, kwargs2d, figax)
