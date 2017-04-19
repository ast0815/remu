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
        M = self.get_response_values_as_ndarray(original_shape)

        # Normalize to number of simulated events
        N_t = self.get_truth_values_as_ndarray()
        M /= np.where(N_t > 0., N_t, 1.)

        if shape is not None:
            M = M.reshape(shape, order='C')

        return M

    def _get_stat_error_parameters(self, expected_weight=1.):
        """Return $\alpha^t_{ij}$, $\hat{m}^t_{ij}$ and $\sigma(m^t_{ij})$.

        Used for calculations of statistical variance.
        """

        N_reco = len(self._reco_binning.bins)
        N_truth = len(self._truth_binning.bins)
        orig_shape = (N_reco, N_truth)

        resp_entries = self.get_response_entries_as_ndarray(orig_shape)
        truth_entries = self.get_truth_entries_as_ndarray()
        # Add "waste bin" of not selected events
        waste_entries = truth_entries - resp_entries.sum(axis=0)
        resp_entries = np.append(resp_entries, waste_entries[np.newaxis,:], axis=0)

        # Get Dirichlet parameters when assuming flat prior
        alpha = resp_entries + 1

        # Estimate mean weight
        resp1 = self.get_response_values_as_ndarray(orig_shape)
        resp2 = self.get_response_sumw2_as_ndarray(orig_shape)
        truth1 = self.get_truth_values_as_ndarray()
        truth2 = self.get_truth_sumw2_as_ndarray()
        # Add "waste bin" of not selected events
        waste1 = truth1 - resp1.sum(axis=0)
        resp1 = np.append(resp1, waste1[np.newaxis,:], axis=0)
        waste2 = truth2 - resp2.sum(axis=0)
        resp2 = np.append(resp2, waste2[np.newaxis,:], axis=0)

        mu = np.where(resp_entries > 0, resp1/np.where(resp_entries > 0, resp_entries, 1), expected_weight)

        # Add pseudo observation for variance estimation
        resp1_p = resp1 + expected_weight
        resp2_p = resp2 + expected_weight**2
        resp_entries_p = resp_entries + 1

        sigma = np.sqrt(((resp2_p/resp_entries_p) - (resp1_p/resp_entries_p)**2) / resp_entries_p)
        # Add an epsilon so sigma is always > 0
        sigma += 1e-12

        return alpha, mu, sigma

    def get_response_matrix_variance_as_ndarray(self, shape=None, expected_weight=1.):
        """Return the statistical variance of the single ResponseMatrix bins as ndarray.

        The variance is estimated from the actual bin contents in a Bayesian
        motivated way.

        The response matrix creation is modeled as a two step process:

        1.  Distribution of truth events among the reco bins according to a
            multinomial distribution.
        2.  Correction of the multinomial transisiton probabilities according
            to the mean weights of the events in each bin.

        So the response matrix element can be written like this:

            R_ij = m_ij * p_ij

        where p_ij is the unweighted multinomial transistion probability and
        m_ij the weight correction. The variance of R_ij is estimated by
        estimating the variances of these values separately.

        The variance of p_ij is estimated by using the Bayesian conjugate prior
        for multinomial distributions: the Dirichlet distribution. We assume a
        uniform prior for all transition probabilities and update it with the
        simulated events. The variance of the posterior distribution is taken
        as the variance of the transisiton probability.

        The variances of m_ij is estimated from the errors of the average
        weights in the matrix elemets as classical "standard error of the
        mean". To avoid problems with bins with 0 or 1 entries, we add a "prior
        expectation" point to the data. This ensures that all bins have at
        least 1 entry (no divisions by zero) and that variances can be
        estimated even for bins with only one (true) entry (from the difference
        to the expected value).

        If no shape is specified, it will be set to `(N_reco, N_truth)`.
        """

        alpha, mu, sigma = self._get_stat_error_parameters(expected_weight=expected_weight)
        beta = np.sum(alpha, axis=0)
        k = alpha.shape[0]

        # Unweighted (multinomial) transistion probabilty
        # Posterior mode estimate = Nij / Nj
        pij = np.asfarray(alpha - 1) / np.where(beta-k > 0, beta-k, 1)
        # Posterior variance
        pij_var = np.asfarray(beta - alpha)
        pij_var *= alpha
        pij_var /= (beta**2 * (beta+1))

        # Weight correction
        wij = mu
        wj = np.sum(mu * pij, axis=-2)
        mij = wij / wj

        # Standard error propagation
        # Variances of input variables
        wij_var = sigma**2
        # derivatives of input variables (adds a dimension to the matrix)
        wij_diff = (np.eye(wij.shape[0])[...,np.newaxis]*wj  - pij[:,np.newaxis,:]*wij)
        wij_diff /= wj**2
        # Putting things together
        mij_var = np.sum( wij_var * wij_diff**2, axis=0)

        # Combine uncertainties
        MM = mij**2 * pij_var + pij**2 * mij_var
        # Remove "waste bin"
        MM = MM[:-1,:]

        if shape is not None:
            MM.shape = shape

        return MM

    def generate_random_response_matrices(self, size=None, shape=None, expected_weight=1.):
        """Generate random response matrices according to the estimated variance.

        This is a two step process:

        1.  Draw the multinomial transition probabilities from a Dirichlet
            distribution.
        2.  Draw weight corrections from normal distributions.

        If no shape is specified, it will be set to `(N_reco, N_truth)`.
        """

        alpha, mu, sigma = self._get_stat_error_parameters(expected_weight=expected_weight)

        # Transpose so we have an array of dirichlet parameters
        alpha = alpha.T

        # Generate truth bin by truth bin
        pij = []
        for j in range(alpha.shape[0]):
            pij.append(np.random.dirichlet(alpha[j], size=size))
        pij = np.array(pij)

        # Reorganise axes
        pij = np.moveaxis(pij, 0, -1)

        # Append original shape to requested size of data sets
        if size is not None:
            try:
                size = list(size)
            except TypeError:
                size = [size]
            size.extend(mu.shape)

        # Generate random weights
        wij = np.abs(np.random.normal(mu, sigma, size=size))
        wj = np.sum(wij * pij, axis=-2)
        mij = (wij / wj[...,np.newaxis,:])

        response = mij * pij

        # Remove "waste bin"
        response = response[...,:-1,:]

        # Adjust shape
        if shape is None:
            shape = (len(self._reco_binning.bins), len(self._truth_binning.bins))
        response = response.reshape(list(response.shape[:-2]) + list(shape))

        return response

    def get_in_bin_variation_as_ndarray(self, shape=None, truth_only=True):
        """Returns an estimate for the variation of the response within a bin.

        The in-bin variation is estimated from the difference of the maximum
        and minimum values of the surrounding bins.

        If `truth_only` is true, only the neighbouring bins along the
        truth-axes will be considered.
        """

        nbins = self._response_binning.nbins
        resp_vars = self._response_binning.variables
        truth_vars = self._truth_binning.variables
        resp = [ self.get_response_matrix_as_ndarray(shape=nbins) ]

        # Generate the shifted matrices
        i0 = np.zeros(len(nbins), dtype=int)
        im1 = np.full(len(nbins), -1, dtype=int)
        for i, var in enumerate(resp_vars):
            # Ignore non-truth variables if flag says so
            if truth_only and var not in truth_vars:
                continue

            # Roll the array
            shifted = np.roll(resp[0], 1, axis=i)
            # Set the 'rolled-in' elements to the values of their neighbours
            i1 = np.array(i0)
            i1[i] = 1
            shifted[tuple(i0)] = shifted[tuple(i1)]
            # Add to list of shifted arrays
            resp.append( shifted )

            # Same in other direction
            # Roll the array
            shifted = np.roll(resp[0], -1, axis=i)
            # Set the 'rolled-in' elements to the values of their neighbours
            im2 = np.array(im1)
            im2[i] = -2
            shifted[tuple(im1)] = shifted[tuple(im2)]
            resp.append( shifted )

        # Get the maximum and minimum of the shifted arrays
        resp = np.array(resp)
        resp = resp.max(axis=0) - resp.min(axis=0)

        # Adjust shape
        if shape is None:
            shape = (len(self._reco_binning.bins), len(self._truth_binning.bins))
        resp.shape = shape

        return resp

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
