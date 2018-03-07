import numpy as np
from copy import copy, deepcopy
from warnings import warn

from .binning import Binning

class ResponseMatrix(object):
    """Matrix that describes the detector response to true events."""

    def __init__(self, reco_binning, truth_binning, nuisance_indices=[], impossible_indices=[], response_binning=None):
        """Initilize the Response Matrix.

        Arguments
        ---------

        truth_binning: The Binning object describing the truth categorization.
        reco_binning: The Binning object describing the reco categorization.
        nuisance_indices: List of indices of nuisance truth bins.
        impossible_indices: List of indices of impossible reco bins.
        response_binning: Optional. The Binning object describing the reco and
                          truth categorization. Usually this will be generated
                          from the truth and reco binning using their
                          `cartesian_product` method.

        The binnings will be combined with `cartesian_product`.

        The truth bins corresonding to the `nuisance_indices` will be treated
        like they have a total efficiency of 1.

        The reco bins corresonding to the `impossible_indices` will be treated
        like they are filled with a probability of 0.
        """

        self.truth_binning = truth_binning
        self.reco_binning = reco_binning
        if response_binning is None:
            self.response_binning = reco_binning.cartesian_product(truth_binning)
        else:
            self.response_binning = response_binning
        self.nuisance_indices=nuisance_indices
        self.impossible_indices=impossible_indices
        self._update_filled_indices()

    def rebin(self, remove_binedges):
        """Return a new ResponseMatrix with the given bin edges removed.

        Arguments
        ---------

        remove_binedges : A dictionary specifying the bin edge indeices of each
                          variable that should be removed. Binning variables that are not part of the
                          dictionary are kept as is.  E.g. if you want to remove
                          bin edge 2 in `var_A` and bin edges 3, 4 and 7 in `var_C`:

                              remove_binedges = { 'var_A': [2],
                                                  'var_B': [3, 4, 7] }

                          The values of the bins adjacent to the removed bin edges
                          will be summed up in the resulting larger bin.
                          Please note that bin values are lost if the first or last
                          binedge of a variable are removed.

        Please note that the `nuisance_indices` and `impossible_indices` of the new matrix are set to `[]`!
        """

        new_response_binning = self.response_binning.rebin(remove_binedges)
        rb = dict( (v, remove_binedges[v]) for v in remove_binedges if v in self.reco_binning.variables )
        new_reco_binning = self.reco_binning.rebin(rb)
        rb = dict( (v, remove_binedges[v]) for v in remove_binedges if v in self.truth_binning.variables )
        new_truth_binning = self.truth_binning.rebin(rb)
        new_nuisance_indices = []
        new_impossible_indices = []

        return ResponseMatrix(reco_binning=new_reco_binning, truth_binning=new_truth_binning, response_binning=new_response_binning, nuisance_indices=new_nuisance_indices, impossible_indices=new_impossible_indices)

    def _update_filled_indices(self):
        """Update the list of filled truth indices."""
        self.filled_truth_indices = np.argwhere(self.get_truth_entries_as_ndarray() > 0).flatten()

    def fill(self, event, weight=1.):
        """Fill events into the binnings."""
        self.truth_binning.fill(event, weight)
        self.reco_binning.fill(event, weight)
        self.response_binning.fill(event, weight)
        self._update_filled_indices()

    def _fix_rounding_errors(self):
        """Fix rounding errors that cause impossible matrices."""

        resp = self.get_response_values_as_ndarray()
        truth = self.get_truth_values_as_ndarray()
        resp.shape=(resp.size // truth.size, truth.size)
        resp = np.sum(resp, axis=0)
        diff = truth-resp

        if np.any(truth < 0):
            raise RuntimeError("Illegal response matrix: Negative true weight!")
        if np.any(resp < 0):
            raise RuntimeError("Illegal response matrix: Negative total reconstructed weight!")

        if np.any(diff < -1e-9): # Allow rounding errors
            raise RuntimeError("Illegal response matrix: Higher total reconstructed than true weight!")

        if np.any(diff < 0.): # But make sure truth is >= reco
            fixed_truth = np.where(diff < 0, resp, truth)
            self.truth_binning.set_values_from_ndarray(fixed_truth)

    def fill_from_csv_file(self, *args, **kwargs):
        """Fill binnings from csv file.

        See the `Binning.fill_from_csv_file` method for a description of the arguments.
        """
        Binning.fill_multiple_from_csv_file([self.truth_binning, self.reco_binning, self.response_binning], *args, **kwargs)
        self._fix_rounding_errors()
        self._update_filled_indices()

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

        new_truth_binning = deepcopy(self.truth_binning)
        new_truth_binning.reset()
        new_truth_binning.fill_from_csv_file(filename, **kwargs)
        new_values = new_truth_binning.get_values_as_ndarray()
        new_entries = new_truth_binning.get_entries_as_ndarray()
        new_sumw2 = new_truth_binning.get_sumw2_as_ndarray()

        old_values = self.truth_binning.get_values_as_ndarray()
        old_entries = self.truth_binning.get_entries_as_ndarray()
        old_sumw2 = self.truth_binning.get_sumw2_as_ndarray()

        if np.any(new_values < 0):
            i = np.argwhere(new_values < 0)
            raise RuntimeError("Filled-up values are negative in %d bins."%(i.size,), stacklevel=2)

        where = (new_values > 0)
        diff_v = new_values - old_values
        diff_e = new_entries - old_entries
        # Check for bins where the fill-up is less than the original
        if np.any(where & (diff_v < -1e-9)):
            i = np.argwhere(where & (diff_v < -1e-9))
            warn("Filled-up values are less than the original filling in %d bins. This should not happen!"%(i.size,), stacklevel=2)
        if np.any(where & (diff_e < 0)):
            i = np.argwhere(where & (diff_e < 0))
            warn("Filled-up entries are less than the original filling in %d bins. This should not happen!"%(i.size,), stacklevel=2)

        where = (where & (diff_v >= 0) & (diff_e >= 0))

        self.truth_binning.set_values_from_ndarray(np.where(where, new_values, old_values))
        self.truth_binning.set_entries_from_ndarray(np.where(where, new_entries, old_entries))
        self.truth_binning.set_sumw2_from_ndarray(np.where(where, new_sumw2, old_sumw2))

        self._fix_rounding_errors()
        self._update_filled_indices()

    def reset(self):
        """Reset all binnings."""
        self.truth_binning.reset()
        self.reco_binning.reset()
        self.response_binning.reset()
        self._update_filled_indices()

    def get_truth_values_as_ndarray(self, *args, **kwargs):
        return self.truth_binning.get_values_as_ndarray(*args, **kwargs)

    def get_truth_entries_as_ndarray(self, *args, **kwargs):
        return self.truth_binning.get_entries_as_ndarray(*args, **kwargs)

    def get_truth_sumw2_as_ndarray(self, *args, **kwargs):
        return self.truth_binning.get_sumw2_as_ndarray(*args, **kwargs)

    def get_reco_values_as_ndarray(self, *args, **kwargs):
        return self.reco_binning.get_values_as_ndarray(*args, **kwargs)

    def get_reco_entries_as_ndarray(self, *args, **kwargs):
        return self.reco_binning.get_entries_as_ndarray(*args, **kwargs)

    def get_reco_sumw2_as_ndarray(self, *args, **kwargs):
        return self.reco_binning.get_sumw2_as_ndarray(*args, **kwargs)

    def get_response_values_as_ndarray(self, *args, **kwargs):
        return self.response_binning.get_values_as_ndarray(*args, **kwargs)

    def get_response_entries_as_ndarray(self, *args, **kwargs):
        return self.response_binning.get_entries_as_ndarray(*args, **kwargs)

    def get_response_sumw2_as_ndarray(self, *args, **kwargs):
        return self.response_binning.get_sumw2_as_ndarray(*args, **kwargs)

    @staticmethod
    def _normalize_matrix(M):
        """Make sure all efficiencies are less than or equal to 1."""

        eff = np.sum(M, axis=-2)
        eff = np.where(eff < 1., 1., eff)[...,np.newaxis,:]
        return M / eff

    def get_response_matrix_as_ndarray(self, shape=None, truth_indices=None):
        """Return the ResponseMatrix as a ndarray.

        If no shape is specified, it will be set to `(N_reco, N_truth)`.
        The expected response of a truth vector can then be calculated like this:

            v_reco = response_matrix.dot(v_truth)

        If `truth_indices` are provided, a sliced matrix with only the given
        columns will be returned.
        """

        if truth_indices is None:
            truth_indices = slice(None, None, None)

        original_shape = (len(self.reco_binning.bins), len(self.truth_binning.bins))

        # Get the bin response entries
        M = self.get_response_values_as_ndarray(original_shape)[:,truth_indices]

        # Normalize to number of simulated events
        N_t = self.get_truth_values_as_ndarray(indices=truth_indices)
        M /= np.where(N_t > 0., N_t, 1.)

        # Deal with bins where N_reco > N_truth
        M = ResponseMatrix._normalize_matrix(M)

        if shape is not None:
            M = M.reshape(shape, order='C')

        return M

    def _get_stat_error_parameters(self, expected_weight=1., nuisance_indices=None, impossible_indices=None, truth_indices=None):
        r"""Return $\beta^t_1j$, $\beta^t_2j$, $\alpha^t_{ij}$, $\hat{w}^t_{ij}$ and $\sigma(w^t_{ij})$.

        Used for calculations of statistical variance.

        If `truth_indices` are provided, a sliced matrix with only the given
        columns will be returned.
        """

        if nuisance_indices is None:
            nuisance_indices = self.nuisance_indices

        if impossible_indices is None:
            impossible_indices = self.impossible_indices

        if truth_indices is None:
            truth_indices = slice(None, None, None)
        else:
            # Translate nuisance indices to sliced indices
            i = np.searchsorted(truth_indices, nuisance_indices)
            mask = i < len(truth_indices)
            i = i[mask]
            nuisance_indices = np.asarray(nuisance_indices)[mask]
            mask = (nuisance_indices == np.asarray(truth_indices)[i])
            nuisance_indices = np.array(i[mask])
            del mask
            del i

        N_reco = len(self.reco_binning.bins)
        N_truth = len(self.truth_binning.bins)
        orig_shape = (N_reco, N_truth)
        epsilon = 1e-50

        resp_entries = self.get_response_entries_as_ndarray(orig_shape)[:,truth_indices]
        truth_entries = self.get_truth_entries_as_ndarray(indices=truth_indices)

        # Get parameters of Beta distribution characterizing the efficiency.
        # Assume a prior of Beta(1,1), i.e. flat in efficiency.
        beta1 = np.sum(resp_entries, axis=0)
        # "Waste bin" of not selected events
        waste_entries = truth_entries - beta1
        if np.any(waste_entries < 0):
            raise RuntimeError("Illegal response matrix: More reconstructed than true events!")
        beta1 = np.asfarray(beta1 + 1)
        beta2 = np.asfarray(waste_entries + 1)

        # Set efficiency of nuisance bins to 1, i.e. beta2 to (almost) zero.
        beta2[nuisance_indices] = epsilon

        # Get parameters of Dirichlet distribution characterizing the distribution within the reco bins.
        # Assume a prior where we expect most of the events to be clustered in a few reco bins.
        # Most events should end up divided into about 3 bins per reco variable:
        # the correct one and the two neighbouring ones.
        # Since the binning is orthogonal, we expect the number of bins to be roughly 3**N_variables.
        # This leads to prior parameters >1 for degenerate reco binnings with < 3 bins/variable.
        # We protect against that by setting the maximum prior value to 1.
        prior = min(1., 3.**len(self.reco_binning.variables) / (N_reco - len(impossible_indices)))
        alpha = np.asfarray(resp_entries) + prior

        # Set efficiency of impossible bins to (almost) 0
        alpha[impossible_indices] = epsilon

        # Estimate mean weight
        resp1 = self.get_response_values_as_ndarray(orig_shape)[:,truth_indices]
        resp2 = self.get_response_sumw2_as_ndarray(orig_shape)[:,truth_indices]
        truth1 = self.get_truth_values_as_ndarray(indices=truth_indices)
        truth2 = self.get_truth_sumw2_as_ndarray(indices=truth_indices)
        # Add truth bin of all events
        resp1 = np.append(resp1, truth1[np.newaxis,:], axis=0)
        resp2 = np.append(resp2, truth2[np.newaxis,:], axis=0)
        resp_entries = np.append(resp_entries, truth_entries[np.newaxis,:], axis=0)

        i = ((resp_entries > 0) & (resp1 > 0))
        mu = resp1/np.where(i, resp_entries, 1)
        mu[-1] = np.where(i[-1], mu[-1], expected_weight) # Set empty truth bins to expected weight
        mu[:-1,:] = np.where(i[:-1], mu[:-1,:], mu[-1,:]) # Set empty reco bins to average truth weight

        # Add pseudo observation for variance estimation
        resp1_p = resp1 + expected_weight
        resp2_p = resp2 + expected_weight**2
        resp_entries_p = resp_entries + 1
        resp_entries_p2 = resp_entries_p**2

        # Since `w_ij` is the mean weight, the variance is just the error of the mean.
        #
        #            |---- sum of weights
        #            v                                      |---- sample variance
        #     w_ij = W_ij / N_ij  <---- number of entries   v
        #     var(w_ij) = var(W_ij) / (N_ij)**2       = var(W) / N_ij
        #               = ( (V_ij / N_ij) - (W_ij / N_ij)**2 ) / N_ij
        #                    ^
        #                    |----- sum of squared weights
        #
        var = ((resp2_p/resp_entries_p) - (resp1_p/resp_entries_p)**2) / resp_entries_p

        sigma = np.sqrt(var)
        # Add an epsilon so sigma is always > 0
        sigma += epsilon

        return beta1, beta2, alpha, mu, sigma

    def get_mean_response_matrix_as_ndarray(self, shape=None, expected_weight=1., nuisance_indices=None, impossible_indices=None, truth_indices=None):
        """Return the means of the posterior distributions of the response matrix elements.

        This is different from the "raw" matrix one gets from
        `get_response_matrix_as_ndarray`. The latter simply divides the sum of
        weights in the respective bins.

        If no shape is specified, it will be set to `(N_reco, N_truth)`.

        If `truth_indices` are provided, a sliced matrix with only the given
        columns will be returned.
        """

        beta1, beta2, alpha, mu, sigma = self._get_stat_error_parameters(expected_weight=expected_weight, nuisance_indices=nuisance_indices, impossible_indices=impossible_indices, truth_indices=truth_indices)

        # Unweighted binomial reconstructed probability (efficiency)
        # Posterior mean estimate = beta1 / (beta1 + beta2)
        beta0 = beta1 + beta2
        effj = beta1 / beta0

        # Unweighted (multinomial) transistion probabilty
        # Posterior mean estimate = alpha / alpha0
        alpha0 = np.sum(alpha, axis=0)
        pij = np.asfarray(alpha) / alpha0

        # Weight correction
        wij = mu[:-1]
        wj = mu[-1]
        mij = wij / wj

        # Combine the three
        MM = mij*pij*effj
        # Re-normalise after weight corrections
        MM = ResponseMatrix._normalize_matrix(MM)

        if shape is not None:
            MM.shape = shape

        return MM

    def get_statistical_variance_as_ndarray(self, shape=None, expected_weight=1., nuisance_indices=None, impossible_indices=None, truth_indices=None):
        """Return the statistical variance of the single ResponseMatrix elements as ndarray.

        The variance is estimated from the actual bin contents in a Bayesian
        motivated way.

        The response matrix creation is modeled as a three step process:

        1.  Reconstruction efficiency according to a binomial process.
        1.  Distribution of truth events among the reco bins according to a
            multinomial distribution.
        2.  Correction of the categorical probabilities according to the mean
            weights of the events in each bin.

        So the response matrix element can be written like this:

            R_ij = m_ij * p_ij * eff_j

        where eff_j is the total efficiency of events in truth bin j, p_ij is the
        unweighted multinomial reconstruction probability in reco bin i and
        m_ij the weight correction. The variance of R_ij is estimated by
        estimating the variances of these values separately.

        The variance of eff_j is estimated by using the Bayesian conjugate
        prior for biinomial distributions: the Bets distribution. We assume a
        prior that is uniform in the reconstruction efficiency. We then update
        it with the simulated events. The variance of the posterior
        distribution is taken as the variance of the efficiency.

        The variance of p_ij is estimated by using the Bayesian conjugate prior
        for multinomial distributions: the Dirichlet distribution. We assume a
        prior that is uniform in the ignorant about reconstruction
        probabilities. We then update it with the simulated events. The
        variance of the posterior distribution is taken as the variance of the
        transition probability.

        If a list of `nuisance_indices` is provided, the probabilities of *not*
        reconstructing events in the respective truth categories will be fixed
        to 0. This is useful for background categories where one is not
        interested in the true number of events.

        If a list of `impossible_indices` is provided, the probabilities of
        reconstructing events in the respective reco categories will be fixed
        to 0. This is useful for bins that are impossible to have any events
        in them by theiur definition.

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

        If `truth_indices` are provided, a sliced matrix with only the given
        columns will be returned.
        """

        beta1, beta2, alpha, mu, sigma = self._get_stat_error_parameters(expected_weight=expected_weight, nuisance_indices=nuisance_indices, impossible_indices=impossible_indices, truth_indices=truth_indices)

        # Unweighted binomial reconstructed probability (efficiency)
        # Posterior mean estimate = beta1 / (beta1 + beta2)
        beta0 = beta1 + beta2
        effj = beta1 / beta0
        # Posterior variance
        effj_var = beta1*beta2 / (beta0**2 * (beta0+1))

        # Unweighted (multinomial) transistion probabilty
        # Posterior mean estimate = alpha / alpha0
        alpha0 = np.sum(alpha, axis=0)
        pij = np.asfarray(alpha) / alpha0
        # Posterior variance
        pij_var = np.asfarray(alpha0 - alpha)
        pij_var *= alpha
        pij_var /= (alpha0**2 * (alpha0+1))

        # Weight correction
        wij = mu[:-1]
        wj = mu[-1]
        mij = wij / wj

        # Standard error propagation
        #
        #     var(m_ij) = var(w_ij) / w_j**2 + (w_ij/w_j**2)**2 * var(w_j)
        wj2 = wj**2
        var = sigma**2
        mij_var = var[:-1]/wj2 + (wij/wj2)**2 * var[-1]

        # Combine uncertainties
        effj2 = effj**2
        pij2 = pij**2
        mij2 = mij**2
        MM = mij2 * pij2 * effj_var + mij2 * pij_var * effj2 + mij_var * pij2 * effj2

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

        # Fix rounding errors
        xs[xs<0] = 0

        return xs

    def generate_random_response_matrices(self, size=None, shape=None, expected_weight=1., nuisance_indices=None, impossible_indices=None, truth_indices=None):
        """Generate random response matrices according to the estimated variance.

        This is a three step process:

        1.  Draw the binomal efficiencies from Beta distributions
        2.  Draw the multinomial reconstruction probabilities from a Dirichlet
            distribution.
        3.  Draw weight corrections from normal distributions.

        If no shape is specified, it will be set to `(N_reco, N_truth)`.

        If `truth_indices` are provided, a sliced matrix with only the given
        columns will be returned.
        """

        beta1, beta2, alpha, mu, sigma = self._get_stat_error_parameters(expected_weight=expected_weight, nuisance_indices=nuisance_indices, impossible_indices=impossible_indices, truth_indices=truth_indices)

        # Generate efficiencies
        if size is None:
            eff_size = beta1.shape
        else:
            try:
                eff_size = tuple(size)
            except TypeError:
                eff_size = (size,)
            eff_size = eff_size + beta1.shape
        effj = np.random.beta(beta1, beta2, eff_size)

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
        wj = wij[...,-1,:]
        wij = wij[...,:-1,:]
        mij = (wij / wj[...,np.newaxis,:])

        response = mij * pij * effj[...,np.newaxis,:]
        # Re-normalise after weight corrections
        response = ResponseMatrix._normalize_matrix(response)

        # Adjust shape
        if shape is None:
            if truth_indices is None:
                shape = (len(self.reco_binning.bins), len(self.truth_binning.bins))
            else:
                shape = (len(self.reco_binning.bins), len(truth_indices))
        response = response.reshape(list(response.shape[:-2]) + list(shape))

        return response

    def get_in_bin_variation_as_ndarray(self, shape=None, truth_only=True, ignore_variables=[], variable_slices={}, truth_indices=None):
        """Returns an estimate for the variation of the response within a bin.

        The in-bin variation is estimated from the maximum difference to the
        surrounding bins. The differences are normalized to the estimated
        statistical errors, so values close to one indicate a statistically
        dominated variation.

        If `truth_only` is true, only the neighbouring bins along the
        truth-axes will be considered.

        Variables specified in `ignore_variables` will not be considered.  This
        is useful to exclude categorical variables, where the response is not
        expected to vary smoothly.

        For variables in `variable_slices` only the specified slice will be rotated,
        e.g. `variable_slices = {'var_A': slice(1,5)}`.

        If `truth_indices` are provided, a sliced matrix with only the given
        columns will be returned.
        """

        nbins = self.response_binning.nbins
        resp_vars = self.response_binning.variables
        truth_vars = self.truth_binning.variables
        resp = self.get_mean_response_matrix_as_ndarray(shape=nbins)
        stat = self.get_statistical_variance_as_ndarray(shape=nbins)
        ret = np.zeros_like(resp, dtype=float)

        # Generate the shifted matrices
        for i, var in enumerate(resp_vars):
            # Ignore non-truth variables if flag says so
            if truth_only and var not in truth_vars:
                continue

            # Ignore other specified variables
            if var in ignore_variables:
                continue

            # Ignore single-bin variables
            if resp.shape[i] == 1:
                continue

            if var in variable_slices:
                sl = variable_slices[var]
                # Copy the array
                shifted_resp = np.array(resp)
                shifted_stat = np.array(stat)
                # Roll the slices
                tup = (slice(None),)*i + (sl, Ellipsis)
                if resp[tup].shape[i] == 1: # Ignore single-bin slices
                    continue
                shifted_resp[tup] = np.roll(resp[tup], 1, axis=i)
                shifted_stat[tup] = np.roll(stat[tup], 1, axis=i)
                # Set the 'rolled-in' elements to the values of their neighbours
                i0 = (slice(None),)*i + (0, Ellipsis)
                i1 = (slice(None),)*i + (1, Ellipsis)
                shifted_resp[tup][i0] = shifted_resp[tup][i1]
                shifted_stat[tup][i0] = shifted_stat[tup][i1]
            else:
                # Roll the array
                shifted_resp = np.roll(resp, 1, axis=i)
                shifted_stat = np.roll(stat, 1, axis=i)
                # Set the 'rolled-in' elements to the values of their neighbours
                i0 = (slice(None),)*i + (0, Ellipsis)
                i1 = (slice(None),)*i + (1, Ellipsis)
                shifted_resp[i0] = shifted_resp[i1]
                shifted_stat[i0] = shifted_stat[i1]

            # Get maximum difference
            ret = np.maximum(ret, np.abs(resp - shifted_resp) / np.sqrt(stat + shifted_stat))

            # Same in other direction
            if var in variable_slices:
                sl = variable_slices[var]
                # Copy the array
                shifted_resp = np.array(resp)
                shifted_stat = np.array(stat)
                # Roll the slices
                tup = (slice(None),)*i + (sl, Ellipsis)
                shifted_resp[tup] = np.roll(resp[tup], -1, axis=i)
                shifted_stat[tup] = np.roll(stat[tup], -1, axis=i)
                # Set the 'rolled-in' elements to the values of their neighbours
                im1 = (slice(None),)*i + (-1, Ellipsis)
                im2 = (slice(None),)*i + (-2, Ellipsis)
                shifted_resp[tup][im1] = shifted_resp[tup][im2]
                shifted_stat[tup][im1] = shifted_stat[tup][im2]
            else:
                # Roll the array
                shifted_resp = np.roll(resp, -1, axis=i)
                shifted_stat = np.roll(stat, -1, axis=i)
                # Set the 'rolled-in' elements to the values of their neighbours
                im1 = (slice(None),)*i + (-1, Ellipsis)
                im2 = (slice(None),)*i + (-2, Ellipsis)
                shifted_resp[im1] = shifted_resp[im2]
                shifted_stat[im1] = shifted_stat[im2]

            # Get maximum difference
            ret = np.maximum(ret, np.abs(resp - shifted_resp) / np.sqrt(stat + shifted_stat))

        ret.shape = (len(self.reco_binning.bins), len(self.truth_binning.bins))

        # Slice the truth bins
        if truth_indices is not None:
            ret = np.array(ret[:,truth_indices])

        # Adjust shape
        if shape is not None:
            ret.shape = shape

        return ret

    @staticmethod
    def _max_step(resp, select, ignore_variables, variable_slices, kwargs):
        variables = resp.truth_binning.variables
        projection = {}
        summed = False

        # Get projections of the entries on all variable axes
        for var in variables:
            if select == 'entries':
                projection[var] = resp.truth_binning.project([var]).get_entries_as_ndarray()
            elif select == 'entries_sum':
                projection[var] = resp.truth_binning.project([var]).get_entries_as_ndarray()
                summed = True
            elif select == 'in-bin':
                inbin = resp.get_in_bin_variation_as_ndarray(ignore_variables=ignore_variables, variable_slices=variable_slices, **kwargs)
                temp_binning = deepcopy(resp.truth_binning)
                temp_binning.set_values_from_ndarray(inbin)
                projection[var] = temp_binning.project([var], reduction_function=np.max).get_values_as_ndarray()
            elif select == 'in-bin_sum':
                inbin = resp.get_in_bin_variation_as_ndarray(ignore_variables=ignore_variables, variable_slices=variable_slices, **kwargs)
                temp_binning = deepcopy(resp.truth_binning)
                temp_binning.set_values_from_ndarray(inbin)
                projection[var] = temp_binning.project([var], reduction_function=np.max).get_values_as_ndarray()
                summed = True
            else:
                raise ValueError("Unknown selection method.")

        # Get projected bin with lowest number of entries
        lowest = (None, -1, np.inf, None)
        for var in projection:
            if var in ignore_variables:
                continue
            if var in variable_slices:
                sl = variable_slices[var]
                proj = projection[var][sl]
            else:
                sl = slice(None)
                proj = projection[var]
            if len(proj) <= 1:
                continue
            if summed:
                proj = np.convolve(proj, [1,1], mode='valid')
            i = np.argmin(proj)
            if proj[i] < lowest[2]:
                lowest = (var, i, proj[i], sl)

        if lowest[0] is None:
            return None

        # Get lowest neighbour
        var, i, entries, sl = lowest
        projection = projection[var][sl]
        if summed:
            neighbour = i+1
        else:
            if i > 0:
                neighbour = i-1
                if i < len(projection)-1 and projection[i+1] < projection[i-1]:
                    neighbour = i+1
            else:
                neighbour = i+1

        # Which binedge to remove
        i = max(i, neighbour)
        if sl.start is not None:
            i += sl.start

        return resp.rebin({var: [i]})

    def maximize_stats_by_rebinning(self, in_bin_variation_limit=5., select='entries', ignore_variables=[], variable_slices={}, **kwargs):
        """Maximize the number of events per bin by rebinning the matrix.

        Bins will only be merged if the maximum in-bin variation of the
        resulting matrix does not exceed the `in_bin_variation_limit`.

        The argument `select` determines how the merging candidate is selected:

            entries: the bin with the lowest number of truth entries
            entries_sum: the pair of bins with the lowest number of truth entries
            in-bin: the bin with the lowest maximum in-bin variation
            in-bin_sum: the pair of bins with the lowest sum of maximum in-bin variations

        Additional keyword arguments will be passed to the method
        `get_in_bin_variation_as_ndarray`.
        """

        resp = deepcopy(self)
        last_resp = deepcopy(self)
        var = np.max(resp.get_in_bin_variation_as_ndarray(ignore_variables=ignore_variables, variable_slices=variable_slices, **kwargs))

        while var < in_bin_variation_limit:
            last_resp = resp
            resp = ResponseMatrix._max_step(resp, select, ignore_variables, variable_slices, kwargs)
            if resp is None:
                break
            var = np.max(resp.get_in_bin_variation_as_ndarray(ignore_variables=ignore_variables, variable_slices=variable_slices, **kwargs))

        return last_resp


    def plot_values(self, filename, variables=None, divide=True, kwargs1d={}, kwargs2d={}, figax=None):
        """Plot the values of the response binning.

        This plots the distribution of events that have *both* a truth and reco bin.
        """

        return self.response_binning.plot_values(filename, variables, divide, kwargs1d, kwargs2d, figax)

    def plot_entries(self, filename, variables=None, divide=True, kwargs1d={}, kwargs2d={}, figax=None):
        """Plot the entries of the response binning.

        This plots the distribution of events that have *both* a truth and reco bin.
        """

        return self.response_binning.plot_entries(filename, variables, divide, kwargs1d, kwargs2d, figax)

    def plot_in_bin_variation(self, filename, variables=None, kwargs1d={}, kwargs2d={}, figax=None, **kwargs):
        """Plot the maximum in-bin variation for projections on all truth variables.

        Additional `kwargs` will be passed on to `get_in_bin_variation_as_ndarray`.
        """

        truth_binning = self.truth_binning
        inbin = self.get_in_bin_variation_as_ndarray(**kwargs)
        inbin = np.max(inbin, axis=0)

        truth_binning.plot_ndarray(filename, inbin, variables=variables, kwargs1d=kwargs1d, kwargs2d=kwargs2d, figax=figax, divide=False, reduction_function=np.max)

    def plot_statistical_variation(self, filename, variables=None, kwargs1d={}, kwargs2d={}, figax=None, **kwargs):
        """Plot the maximum statistical variation for projections on all truth variables.

        Additional `kwargs` will be passed on to `get_statistical_variance_as_ndarray`.
        """

        truth_binning = self.truth_binning
        stat = self.get_statistical_variance_as_ndarray(**kwargs)
        stat = np.sqrt(np.max(stat, axis=0))

        truth_binning.plot_ndarray(filename, stat, variables=variables, kwargs1d=kwargs1d, kwargs2d=kwargs2d, figax=figax, divide=False, reduction_function=np.max)

    def plot_expected_efficiency(self, filename, variables=None, kwargs1d={}, kwargs2d={}, figax=None, nuisance_indices=None, **kwargs):
        """Plot expected efficiencies for projections on all truth variables.

        This assumes the truth values are distributed like the generator data.
        This does *not* consider the statistical uncertainty of the matrix
        elements.

        Additional `kwargs` will be passed on to `get_response/truth_values_as_ndarray`.
        """

        if nuisance_indices is None:
            nuisance_indices = self.nuisance_indices

        truth_binning = self.truth_binning
        shape = (len(self.reco_binning.bins), len(self.truth_binning.bins))
        eff = self.get_response_values_as_ndarray(shape=shape, **kwargs)
        eff = np.sum(eff, axis=0)
        eff[nuisance_indices] = 0.
        truth = self.get_truth_values_as_ndarray(**kwargs)
        truth = np.where(truth > 0, truth, 1e-50)
        truth[nuisance_indices] = 1e-50

        truth_binning.plot_ndarray(filename, eff, variables=variables, kwargs1d=kwargs1d, kwargs2d=kwargs2d, figax=figax, divide=False, reduction_function=np.sum, denominator=truth)

class ResponseMatrixArrayBuilder(object):
    """Class that generates consistent ndarrays from multiple response matrix objects.

    To save space, it only stores the truth bins that were actually filled.
    When creating the total ndarray, missing columns are filled with default values.
    The matrices must have been built using the same truth information!
    Their truth binnings may only differ in the nuisance bins!
    These truth bins are handled such that the efficiency of the most efficient matrix in each of the nuisance bins is 100%.
    """

    def __init__(self, nstat):
        """ """
        self.nstat = nstat
        self.reset()

    def reset(self):
        self.nmatrices = 0
        self._matrices = []
        self._mean_matrices = []
        self._truth_values = []
        self._truth_entries = None
        self._response_values = None
        self._filled_indices = []
        self._nuisance_indices = None

    def add_matrix(self, response_matrix):
        """Add a matrix to the collection."""

        # Check that the nuisance indices are identical
        nuisance_indices = response_matrix.nuisance_indices
        if self._nuisance_indices is None:
            self._nuisance_indices = nuisance_indices
        elif set(self._nuisance_indices) != set(nuisance_indices):
            raise RuntimeError("Matrices have different nuisance indices!")

        filled_indices = response_matrix.filled_truth_indices
        if self.nstat > 0:
            matrix = response_matrix.generate_random_response_matrices(self.nstat, truth_indices=filled_indices)
        else:
            matrix = response_matrix.get_response_matrix_as_ndarray(truth_indices=filled_indices)
        mean_matrix = response_matrix.get_mean_response_matrix_as_ndarray(truth_indices=filled_indices)
        truth_values = response_matrix.get_truth_values_as_ndarray(indices=filled_indices)
        response_values = response_matrix.get_response_values_as_ndarray() # We need *all* entries
        truth_entries = response_matrix.get_truth_entries_as_ndarray() # We need *all* entries

        self._filled_indices.append(filled_indices)
        self._matrices.append(matrix)
        self._mean_matrices.append(mean_matrix)
        self._truth_values.append(truth_values)
        if self._truth_entries is None:
            self._truth_entries = truth_entries
        else:
            self._truth_entries = np.maximum(self._truth_entries, truth_entries)
        if self._response_values is None:
            self._response_values = response_values
        else:
            self._response_values += response_values
        self.nmatrices += 1

    def _get_filled_truth_indices_set(self):
        """Return the set of filled truth indices."""
        all_indices = set()
        for i in self._filled_indices:
            all_indices.update(i)
        return all_indices

    def get_filled_truth_indices(self):
        """Return the list of filled truth indices."""
        return sorted(self._get_filled_truth_indices_set())

    def get_truth_entries_as_ndarray(self):
        """Return the array of (maximum) entries in the truh bins."""
        return self._truth_entries

    def get_response_values_as_ndarray(self):
        """Return the (mean) values of the response bins."""
        return self._response_values / self.nmatrices

    def _get_truth_value_scale(self, tv):
        """Get scale to make nuisance bins consistent.

        The nuisance bins must be scaled between the multiple matrices, because
        in each matrix their efficiency is 1 by definition.  Ideally they all
        would be scaled to the true number of true events in each truth bin,
        but this information is not available for nuisance bins.  Instead we
        use the sum of truth values over all matrices as denominator off the
        efficiency, e.g. the efficiency of nuisance truth bin j in matrix t:

            eff_tj = N_tj / sum_t( N_tj)

        This means the efficiency of the nuisance bins goes down with more
        added toy matrices.  This could be counteracted by multiplying the
        efficiency with the number of matrices, but that could lead to
        efficiencies >1, which can lead to mathematical problems further down
        the line.
        """

        all_indices = self._get_filled_truth_indices_set()
        nuisance_indices = set(self._nuisance_indices)
        filled_nuisance_indices = all_indices & nuisance_indices
        max_tv = np.sum(tv, axis=0)
        max_tv = np.where(max_tv > 0, max_tv, 1.0)
        scale = np.ones_like(tv) # Start with scales = 1
        for i in np.searchsorted(sorted(all_indices), sorted(filled_nuisance_indices)):
            scale[:,i] = tv[:,i] / max_tv[i] # Set scale of nuisance indices
        return scale

    def get_response_matrices_as_ndarray(self):
        """Get the response matrices as consistent ndarray."""

        M = []
        tv = []

        # Insert missing columns
        all_indices = self._get_filled_truth_indices_set()
        nuisance_indices = set(self._nuisance_indices)
        for indices, matrix, truth_values in zip(self._filled_indices, self._matrices, self._truth_values):
            missing_indices = all_indices - set(indices)
            # Make sure only nuisance indices are missing
            if (len(missing_indices - nuisance_indices) > 0):
                raise RuntimeError("Truth difference in non-nuisance index!")
            missing_indices = list(missing_indices)
            insert_positions = np.searchsorted(indices, missing_indices)
            extended_matrix = np.insert(matrix, insert_positions, 0., axis=-1)
            M.append(extended_matrix)
            extended_truth_values = np.insert(truth_values, insert_positions, 0., axis=-1)
            tv.append(extended_truth_values)

        M = np.array(M)
        tv = np.array(tv)

        # Scale (nuisance) truth bins so they are consistent
        scale = self._get_truth_value_scale(tv)
        if self.nstat > 0:
            M = M * scale[:,np.newaxis,np.newaxis,:]
        else:
            M = M * scale[:,np.newaxis,:]

        return M

    def get_mean_response_matrix_as_ndarray(self):
        """Get the mean response matrix as ndarray."""

        M = []
        tv = []

        # Insert missing columns
        all_indices = self._get_filled_truth_indices_set()
        nuisance_indices = set(self._nuisance_indices)
        for indices, matrix, truth_values in zip(self._filled_indices, self._mean_matrices, self._truth_values):
            missing_indices = all_indices - set(indices)
            # Make sure only nuisance indices are missing
            if (len(missing_indices - nuisance_indices) > 0):
                raise RuntimeError("Truth difference in non-nuisance index!")
            missing_indices = list(missing_indices)
            insert_positions = np.searchsorted(indices, missing_indices)
            extended_matrix = np.insert(matrix, insert_positions, 0., axis=-1)
            M.append(extended_matrix)
            extended_truth_values = np.insert(truth_values, insert_positions, 0., axis=-1)
            tv.append(extended_truth_values)

        M = np.array(M)
        tv = np.array(tv)

        # Scale (nuisance) truth bins so they are consistent
        scale = self._get_truth_value_scale(tv)
        M = M * scale[:,np.newaxis,:]

        return np.mean(M, axis=0)
