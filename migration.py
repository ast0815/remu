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

    def fill_from_csv_file(self, filename, **kwargs):
        """Fill binnings from csv file."""
        self._truth_binning.fill_from_csv_file(filename, **kwargs)
        self._reco_binning.fill_from_csv_file(filename, **kwargs)
        self._response_binning.fill_from_csv_file(filename, **kwargs)

    def fill_up_truth_from_csv_file(self, filename, **kwargs):
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
        new_truth_binning.fill_from_csv_file(filename, **kwargs)
        new_values = new_truth_binning.get_values_as_ndarray()
        new_entries = new_truth_binning.get_entries_as_ndarray()
        new_sumw2 = new_truth_binning.get_sumw2_as_ndarray()

        old_values = self._truth_binning.get_values_as_ndarray()
        old_entries = self._truth_binning.get_entries_as_ndarray()
        old_sumw2 = self._truth_binning.get_sumw2_as_ndarray()

        where = new_values > old_values

        self._truth_binning.set_values_from_ndarray(np.where(where, new_values, old_values))
        self._truth_binning.set_entries_from_ndarray(np.where(where, new_entries, old_entries))
        self._truth_binning.set_sumw2_from_ndarray(np.where(where, new_sumw2, old_sumw2))

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

    def _get_stat_error_parameters(self, expected_weight=1., nuisance_indices=None):
        """Return $\alpha^t_{ij}$, $\hat{m}^t_{ij}$ and $\sigma(m^t_{ij})$.

        Used for calculations of statistical variance.
        """

        if nuisance_indices is None:
            nuisance_indices = np.ndarray(0, dtype=int)

        N_reco = len(self._reco_binning.bins)
        N_truth = len(self._truth_binning.bins)
        orig_shape = (N_reco, N_truth)
        epsilon = 1e-12

        resp_entries = self.get_response_entries_as_ndarray(orig_shape)
        truth_entries = self.get_truth_entries_as_ndarray()
        # Add "waste bin" of not selected events
        waste_entries = truth_entries - resp_entries.sum(axis=0)
        resp_entries = np.append(resp_entries, waste_entries[np.newaxis,:], axis=0)

        # Get Dirichlet parameters when assuming prior flat in efficiency and
        # ignorant about reconstruction. This also means that the prior
        # expects the recosntructed events to be clustered in a small number of
        # reco bins.
        alpha = np.asfarray(resp_entries)
        prior = np.full(N_reco+1, 1./N_reco, dtype=float)
        prior[-1] = 1.
        alpha += prior[:,np.newaxis]
        # Set loss probability for nuisance truth bins to (almost) 0
        alpha[-1,nuisance_indices] = epsilon

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
        sigma += epsilon

        return alpha, mu, sigma

    def get_mean_response_matrix_as_ndarray(self, shape=None, expected_weight=1., nuisance_indices=None):
        """Return the means of the posterior distributions of the response matrix elements.

        This is different from the "raw" matrix one gets from
        `get_response_matrix_as_ndarray`. The latter simply divides the sum of
        weights in the respective bins.

        If no shape is specified, it will be set to `(N_reco, N_truth)`.
        """

        alpha, mu, sigma = self._get_stat_error_parameters(expected_weight=expected_weight, nuisance_indices=nuisance_indices)
        beta = np.sum(alpha, axis=0)

        # Unweighted (multinomial) transistion probabilty
        # Posterior mean estimate = alpha / beta
        pij = np.asfarray(alpha) / beta

        # Weight correction
        wij = mu
        wj = np.sum(mu * pij, axis=-2)
        mij = wij / wj

        # Combine the two
        MM = mij*pij

        # Remove "waste bin"
        MM = MM[:-1,:]

        if shape is not None:
            MM.shape = shape

        return MM

    def get_statistical_variance_as_ndarray(self, shape=None, expected_weight=1., nuisance_indices=None):
        """Return the statistical variance of the single ResponseMatrix elements as ndarray.

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
        prior that is uniform in the reconstruction efficiency and completely
        ignorant about reconstruction accuracies. We then update it with the
        simulated events. The variance of the posterior distribution is taken
        as the variance of the transition probability.

        If a list of `nuisance_indices` is provided, the probabilities of *not*
        reconstructing events in the respective truth categories will be fixed
        to 0. This is useful for background categories where one is not
        interested in the true number of events.

        The variances of m_ij is estimated from the errors of the average
        weights in the matrix elements as classical "standard error of the
        mean". To avoid problems with bins with 0 or 1 entries, we add a "prior
        expectation" point to the data. This ensures that all bins have at
        least 1 entry (no divisions by zero) and that variances can be
        estimated even for bins with only one (true) entry (from the difference
        to the expected value).

        This is just an estimate! The true variance of the randomly generated
        response matrices can deviate from the returned numbers. Also, these
        variances ignore the correlations between matrix elements.

        If no shape is specified, it will be set to `(N_reco, N_truth)`.
        """

        alpha, mu, sigma = self._get_stat_error_parameters(expected_weight=expected_weight, nuisance_indices=nuisance_indices)
        beta = np.sum(alpha, axis=0)

        # Unweighted (multinomial) transistion probabilty
        # Posterior mean estimate = alpha / beta
        pij = np.asfarray(alpha) / beta
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

    @staticmethod
    def _dirichlet(alpha, size=None):
        """Reimplements np.random.dirichlet.

        The original implementation is not suitable for very low alphas.
        """

        params = np.asfarray(alpha)

        if size is None:
            total_size = (len(alpha))
        else:
            try:
                total_size = tuple(list(size) + [len(alpha)])
            except TypeError:
                total_size = tuple(list([size]) + [len(alpha)])

        xs = np.zeros(total_size)

        xs[...,0] = np.random.beta(params[0], np.sum(params[1:]), size=size)
        for j in range(1,len(params)-1):
            phi = np.random.beta(params[j], sum(params[j+1:]), size=size)
            xs[...,j] = (1-np.sum(xs, axis=-1)) * phi
        xs[...,-1] = (1-np.sum(xs, axis=-1))

        return xs

    def generate_random_response_matrices(self, size=None, shape=None, expected_weight=1., nuisance_indices=None):
        """Generate random response matrices according to the estimated variance.

        This is a two step process:

        1.  Draw the multinomial transition probabilities from a Dirichlet
            distribution.
        2.  Draw weight corrections from normal distributions.

        If no shape is specified, it will be set to `(N_reco, N_truth)`.
        """

        alpha, mu, sigma = self._get_stat_error_parameters(expected_weight=expected_weight, nuisance_indices=nuisance_indices)

        # Transpose so we have an array of dirichlet parameters
        alpha = alpha.T

        # Generate truth bin by truth bin
        pij = []
        for j in range(alpha.shape[0]):
            pij.append(self._dirichlet(alpha[j], size=size))
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

    def get_in_bin_variation_as_ndarray(self, shape=None, truth_only=True, ignore_variables=[]):
        """Returns an estimate for the variation of the response within a bin.

        The in-bin variation is estimated from the maximum difference to the
        surrounding bins. The differences are normalized to the estimated
        statistical errors, so values close to one indicate a statistically
        dominated variation.

        If `truth_only` is true, only the neighbouring bins along the
        truth-axes will be considered.

        Variables specified in `ignore_variables` will not be considered.  This
        is useful to exclude categotical variables, where the response is not
        expected to vary smoothly.
        """

        nbins = self._response_binning.nbins
        resp_vars = self._response_binning.variables
        truth_vars = self._truth_binning.variables
        resp = self.get_mean_response_matrix_as_ndarray(shape=nbins)
        stat = self.get_statistical_variance_as_ndarray(shape=nbins)
        ret = np.zeros_like(resp, dtype=float)

        # Generate the shifted matrices
        i0 = np.zeros(len(nbins), dtype=int)
        im1 = np.full(len(nbins), -1, dtype=int)
        for i, var in enumerate(resp_vars):
            # Ignore non-truth variables if flag says so
            if truth_only and var not in truth_vars:
                continue

            # Ignore other specified variables
            if var in ignore_variables:
                continue

            # Roll the array
            shifted_resp = np.roll(resp, 1, axis=i)
            shifted_stat = np.roll(resp, 1, axis=i)
            # Set the 'rolled-in' elements to the values of their neighbours
            i1 = np.array(i0)
            i1[i] = 1
            shifted_resp[tuple(i0)] = shifted_resp[tuple(i1)]
            shifted_stat[tuple(i0)] = shifted_stat[tuple(i1)]

            # Get maximum difference
            ret = np.maximum(ret, np.abs(resp - shifted_resp) / np.sqrt(stat + shifted_stat))

            # Same in other direction
            # Roll the array
            shifted_resp = np.roll(resp, -1, axis=i)
            shifted_stat = np.roll(stat, -1, axis=i)
            # Set the 'rolled-in' elements to the values of their neighbours
            im2 = np.array(im1)
            im2[i] = -2
            shifted_resp[tuple(im1)] = shifted_resp[tuple(im2)]
            shifted_stat[tuple(im1)] = shifted_stat[tuple(im2)]

            # Get maximum difference
            ret = np.maximum(ret, np.abs(resp - shifted_resp) / np.sqrt(stat + shifted_stat))

        # Adjust shape
        if shape is None:
            shape = (len(self._reco_binning.bins), len(self._truth_binning.bins))
        ret.shape = shape

        return ret

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

    def plot_in_bin_variation(self, filename, variables=None, kwargs1d={}, kwargs2d={}, figax=None, **kwargs):
        """Plot the maximum in-bin variation for projections on all truth variables.

        Additional `kwargs` will be passed on to `get_in_bin_variation_as_ndarray`.
        """

        truth_binning = self._truth_binning
        inbin = self.get_in_bin_variation_as_ndarray(**kwargs)
        inbin = np.max(inbin, axis=0)

        truth_binning.plot_ndarray(filename, inbin, variables=variables, kwargs1d=kwargs1d, kwargs2d=kwargs2d, figax=figax, divide=False, reduction_function=np.max)

    def plot_statistical_variation(self, filename, variables=None, kwargs1d={}, kwargs2d={}, figax=None, **kwargs):
        """Plot the maximum statistical variation for projections on all truth variables.

        Additional `kwargs` will be passed on to `get_statistical_variance_as_ndarray`.
        """

        truth_binning = self._truth_binning
        stat = self.get_statistical_variance_as_ndarray(**kwargs)
        stat = np.sqrt(np.max(stat, axis=0))

        truth_binning.plot_ndarray(filename, stat, variables=variables, kwargs1d=kwargs1d, kwargs2d=kwargs2d, figax=figax, divide=False, reduction_function=np.max)

    def plot_min_efficiency(self, filename, variables=None, kwargs1d={}, kwargs2d={}, figax=None, **kwargs):
        """Plot minimum efficiencies for projections on all truth variables.

        This does *not* consider the statistical uncertainty of the matrix
        elements.  It uses only the mean response matrix.

        Additional `kwargs` will be passed on to `get_mean_response_matrix_as_ndarray`.
        """

        truth_binning = self._truth_binning
        eff = self.get_mean_response_matrix_as_ndarray(**kwargs)
        eff = np.sum(eff, axis=0)

        truth_binning.plot_ndarray(filename, eff, variables=variables, kwargs1d=kwargs1d, kwargs2d=kwargs2d, figax=figax, divide=False, reduction_function=np.min)
