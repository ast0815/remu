"""Module handling the creation and use of migration matrices."""

import numpy as np
from scipy import stats
from scipy import linalg
from copy import copy, deepcopy
from warnings import warn

# Load matplotlib only on demand
# from matplotlib import pyplot as plt
plt = None

from .binning import Binning

class ResponseMatrix(object):
    """Matrix that describes the detector response to true events.

    Parameters
    ----------

    truth_binning : RectangularBinning
        The Binning object describing the truth categorization.
    reco_binning : RectangularBinning
        The Binning object describing the reco categorization.
    nuisance_indices : list of ints, optional
        List of indices of nuisance truth bins.
        These are treated like their efficiency is exactly 1.
    impossible_indices :list of ints, optional
        List of indices of impossible reco bins.
        These are treated like their probability is exactly 0.
    response_binning : RectangularBinning, optional
        The Binning object describing the reco and truth categorization.
        Usually this will be generated from the truth and reco binning using
        their :meth:`RectangularBinning.cartesian_product` method.

    Notes
    -----

    The truth and reco binnings will be combined with their
    `cartesian_product` method.

    The truth bins corresonding to the `nuisance_indices` will be treated
    like they have a total efficiency of 1.

    The reco bins corresonding to the `impossible_indices` will be treated
    like they are filled with a probability of 0.

    Two response matrices can be combined by adding them ``new_resp = respA +
    respB``. This yields a new matrix that is equivalent to one that has been
    filled with the data in both ``respA`` and ``respB``. The truth and reco
    binnings in ``respA`` and ``respB`` must be identical for this to make
    sense.

    """

    def __init__(self, reco_binning, truth_binning, nuisance_indices=[], impossible_indices=[], response_binning=None):
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

        The values of the bins adjacent to the removed bin edges will be
        summed up in the resulting larger bin. Please note that bin values
        are lost if the first or last binedge of a variable are removed.

        Parameters
        ----------

        remove_binedges : dict of list of ints

            A dictionary specifying the bin edge indices of each variable that
            should be removed. Binning variables that are not part of the
            dictionary are kept as is.  E.g. if you want to remove bin edge 2
            in ``var_A`` and bin edges 3, 4 and 7 in ``var_C``::

                remove_binedges = { 'var_A': [2], 'var_C': [3, 4, 7] }

        Returns
        -------

        ResponseMatrix
            The new response matrix with the given bin edges removed.

        Warnings
        --------

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

    def fill(self, *args, **kwargs):
        """Fill events into the binnings."""
        self.truth_binning.fill(*args, **kwargs)
        self.reco_binning.fill(*args, **kwargs)
        self.response_binning.fill(*args, **kwargs)
        self._update_filled_indices()

    def _fix_rounding_errors(self):
        """Fix rounding errors that cause impossible matrices."""

        resp = self.get_response_values_as_ndarray()
        truth = self.get_truth_values_as_ndarray()
        resp = resp.reshape((resp.size // truth.size, truth.size), order='C')
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

        See :meth:`Binning.fill_from_csv_file
        <remu.binning.Binning.fill_from_csv_file>`
        for a description of the parameters.

        See also
        --------

        fill_up_truth_from_csv_file : Re-fill only truth bins from different file.

        """
        Binning.fill_multiple_from_csv_file([self.truth_binning, self.reco_binning, self.response_binning], *args, **kwargs)
        self._fix_rounding_errors()
        self._update_filled_indices()

    def fill_up_truth_from_csv_file(self, *args, **kwargs):
        """Re-fill the truth bins with the given csv file.

        This can be used to get proper efficiencies if the true signal events
        are saved in a separate file from the reconstructed events.

        It takes the same parameters as :meth:`fill_from_csv_file`.

        Notes
        -----

        A new truth binning is created and filled with the events from the
        provided file. Each bin is compared to the corresponding bin in the
        already present truth binning. The larger value of the two is taken as
        the new truth. This way, event types that are not present in the pure
        truth data, e.g. background, are not affected by this. It can only
        *increase* the value of the truth bins, lowering their efficiency.

        For each truth bin, one of the following *must* be true for this
        operation to make sense:

        *   All events in the migration matrix are also present in the truth
            file. In this case, the additional truth events lower the
            efficiency of the truth bin. This is the case, for example, if not
            all true signal events are reconstructed.

        *   All events in the truth file are also present in the migration
            matrix. In this case, the events in the truth file have no
            influence on the response matrix. This is the case, for example, if
            only a subset of the reconstructed background is saved in the truth
            file.

        If there are events in the response matrix that are not in the truth
        tree *and* there are events in the truth tree that are not in the
        response matrix, this method will lead to a *wrong* efficiency of the
        affected truth bin.

        """

        new_truth_binning = deepcopy(self.truth_binning)
        new_truth_binning.reset()
        new_truth_binning.fill_from_csv_file(*args, **kwargs)
        return self._replace_smaller_truth(new_truth_binning)

    def fill_up_truth(self, *args, **kwargs):
        """Re-fill the truth bins with the given events file.

        This can be used to get proper efficiencies if the true signal events
        are stored separate from the reconstructed events.

        It takes the same parameters as :meth:`fill`.

        Notes
        -----

        A new truth binning is created and filled with the events from the
        provided events. Each bin is compared to the corresponding bin in the
        already present truth binning. The larger value of the two is taken as
        the new truth. This way, event types that are not present in the pure
        truth data, e.g. background, are not affected by this. It can only
        *increase* the value of the truth bins, lowering their efficiency.

        For each truth bin, one of the following *must* be true for this
        operation to make sense:

        *   All events in the migration matrix are also present in the new truth
            events. In this case, the additional truth events lower the
            efficiency of the truth bin. This is the case, for example, if not
            all true signal events are reconstructed.

        *   All events in the new truth events are also present in the migration
            matrix. In this case, the events in the new truth events have no
            influence on the response matrix. This is the case, for example, if
            only a subset of the reconstructed background is saved in the truth
            file.

        If there are events in the response matrix that are not in the new truth
        events *and* there are events in the new truth events that are not in the
        response matrix, this method will lead to a *wrong* efficiency of the
        affected truth bin.

        """

        new_truth_binning = deepcopy(self.truth_binning)
        new_truth_binning.reset()
        new_truth_binning.fill(*args, **kwargs)
        return self._replace_smaller_truth(new_truth_binning)

    def _replace_smaller_truth(self, new_truth_binning):
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
        """Get the values of the truth binning as `ndarray`."""
        return self.truth_binning.get_values_as_ndarray(*args, **kwargs)

    def get_truth_entries_as_ndarray(self, *args, **kwargs):
        """Get the number of entries in the truth binning as `ndarray`."""
        return self.truth_binning.get_entries_as_ndarray(*args, **kwargs)

    def get_truth_sumw2_as_ndarray(self, *args, **kwargs):
        """Get the sum of squared weights in the truth binning as `ndarray`."""
        return self.truth_binning.get_sumw2_as_ndarray(*args, **kwargs)

    def get_reco_values_as_ndarray(self, *args, **kwargs):
        """Get the values of the reco binning as `ndarray`."""
        return self.reco_binning.get_values_as_ndarray(*args, **kwargs)

    def get_reco_entries_as_ndarray(self, *args, **kwargs):
        """Get the number of entries in the reco binning as `ndarray`."""
        return self.reco_binning.get_entries_as_ndarray(*args, **kwargs)

    def get_reco_sumw2_as_ndarray(self, *args, **kwargs):
        """Get the sum of squared weights in the reco binning as `ndarray`."""
        return self.reco_binning.get_sumw2_as_ndarray(*args, **kwargs)

    def get_response_values_as_ndarray(self, *args, **kwargs):
        """Get the values of the response binning as `ndarray`."""
        return self.response_binning.get_values_as_ndarray(*args, **kwargs)

    def get_response_entries_as_ndarray(self, *args, **kwargs):
        """Get the number of entries in the response binning as `ndarray`."""
        return self.response_binning.get_entries_as_ndarray(*args, **kwargs)

    def get_response_sumw2_as_ndarray(self, *args, **kwargs):
        """Get the sum of squared weights in the response binning as `ndarray`."""
        return self.response_binning.get_sumw2_as_ndarray(*args, **kwargs)

    @staticmethod
    def _normalize_matrix(M):
        """Make sure all efficiencies are less than or equal to 1."""
        eff = np.sum(M, axis=-2)
        eff = np.where(eff < 1., 1., eff)[...,np.newaxis,:]
        return M / eff

    def get_response_matrix_as_ndarray(self, shape=None, truth_indices=None):
        """Return the ResponseMatrix as a ndarray.

        Uses the information in the truth and response binnings to calculate
        the response matrix.

        Parameters
        ----------

        shape : tuple of ints, optional
            The shape of the returned ndarray.
            Default: ``(#(reco bins), #(truth bins))``
        truth_indices : list of ints, optional
            Only return the response of the given truth bins.
            Default: Return full matrix.

        Returns
        -------

        ndarray

        Notes
        -----

        If shape is `None`, it s set to ``(#(reco bins), #(truth bins))``. The
        expected response of a truth vector can then be calculated like this::

            v_reco = response_matrix.dot(v_truth)

        If `truth_indices` are provided, a sliced matrix with only the given
        columns will be returned.

        See also
        --------

        get_mean_response_matrix_as_ndarray

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

    def get_mean_response_matrix_as_ndarray(self, shape=None, **kwargs):
        """Get the means of the posterior distributions of the response matrix elements.

        This is different from the "raw" matrix one gets from
        :meth:`get_response_matrix_as_ndarray`. The latter simply divides the
        sum of weights in the respective bins.

        Parameters
        ----------

        shape : tuple of ints, optional
            The shape of the returned matrices.
            Defaults to ``(#(reco bins), #(truth bins))``.
        expected_weight : float, optional
            The expected average weight of the events. This is used int the
            calculation of the weight variance.
            Default: 1.0
        nuisance_indices : list of ints, optional
            List of truth bin indices. These bins will be treated like their
            efficiency is exactly 1.
            Default: Use the `nuisance_indices` attribute of the ResponseMatrix.
        impossible_indices : list of ints, optional
            List of reco bin indices. These bins will be treated like their
            their probability is exactly 0.
            Default: Use the `impossible_indices` attribute of the ResponseMatrix.
        truth_indices : list of ints, optional
            List of truth bin indices. Only return the response of the given
            truth bins. Default: Return full matrices.

        Returns
        -------

        ndarray

        See also
        --------

        get_response_matrix_as_ndarray
        get_statistical_variance_as_ndarray
        generate_random_response_matrices

        """

        beta1, beta2, alpha, mu, sigma = self._get_stat_error_parameters(**kwargs)

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
            MM = MM.reshape(shape, order='C')

        return MM

    def get_statistical_variance_as_ndarray(self, shape=None, **kwargs):
        """Get the statistical variance of the single ResponseMatrix elements as ndarray.

        The variance is estimated from the actual bin contents in a Bayesian
        motivated way.

        Parameters
        ----------

        shape : tuple of ints, optional
            The shape of the returned matrix.
            Defaults to ``(#(reco bins), #(truth bins))``.
        kwargs : optional
            See :meth:`get_mean_response_matrix_as_ndarray` for a description
            of more optional `kwargs`.

        Returns
        -------

        ndarray

        Notes
        -----

        The response matrix creation is modeled as a three step process:

        1.  Reconstruction efficiency according to a binomial process.
        2.  Distribution of truth events among the reco bins according to a
            multinomial distribution.
        3.  Correction of the categorical probabilities according to the mean
            weights of the events in each bin.

        So the response matrix element can be written like this::

            R_ij = m_ij * p_ij * eff_j

        where ``eff_j`` is the total efficiency of events in truth bin ``j``,
        ``p_ij`` is the unweighted multinomial reconstruction probability in
        reco bin ``i`` and ``m_ij`` the weight correction. The variance of
        ``R_ij`` is estimated by estimating the variances of these values
        separately.

        The variance of ``eff_j`` is estimated by using the Bayesian conjugate
        prior for biinomial distributions: the Beta distribution. We assume a
        prior that is uniform in the reconstruction efficiency. We then update
        it with the simulated events. The variance of the posterior
        distribution is taken as the variance of the efficiency.

        The variance of ``p_ij`` is estimated by using the Bayesian conjugate
        prior for multinomial distributions: the Dirichlet distribution. We
        assume a prior that is uniform in the ignorant about reconstruction
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

        See also
        --------

        get_mean_response_matrix_as_ndarray
        generate_random_response_matrices

        """

        beta1, beta2, alpha, mu, sigma = self._get_stat_error_parameters(**kwargs)

        # Unweighted binomial reconstruction probability (efficiency)
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
            MM = MM.reshape(shape, order='C')

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

        if len(params) == 1:
            # Special case for response matrices with only one reco bin
            return np.ones(total_size)

        xs = np.zeros(total_size)

        xs[...,0] = np.random.beta(params[0], np.sum(params[1:]), size=size)
        for j in range(1,len(params)-1):
            phi = np.random.beta(params[j], sum(params[j+1:]), size=size)
            xs[...,j] = (1-np.sum(xs, axis=-1)) * phi
        xs[...,-1] = (1-np.sum(xs, axis=-1))

        # Fix rounding errors
        xs[xs<0] = 0

        return xs

    def generate_random_response_matrices(self, size=None, shape=None, **kwargs):
        """Generate random response matrices according to the estimated variance.

        Parameters
        ----------

        size : int or tuple of ints, optional
            How many random matrices should be generated.
        shape : tuple of ints, optional
            The shape of the returned matrices.
            Defaults to ``(#(reco bins), #(truth bins))``.
        kwargs : optional
            See :meth:`get_mean_response_matrix_as_ndarray` for a description
            of more optional `kwargs`.

        Returns
        -------

        ndarray

        Notes
        -----

        This is a three step process:

        1.  Draw the binomal efficiencies from Beta distributions
        2.  Draw the multinomial reconstruction probabilities from a Dirichlet
            distribution.
        3.  Draw weight corrections from normal distributions.

        If no shape is specified, it will be set to ``(#(reco bins, #(truth bins))``.

        If `truth_indices` are provided, a sliced matrix with only the given
        columns will be returned.

        See also
        --------

        get_mean_response_matrix_as_ndarray
        get_statistical_variance_as_ndarray

        """

        beta1, beta2, alpha, mu, sigma = self._get_stat_error_parameters(**kwargs)

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
            truth_indices = kwargs.pop('truth_indices', None)
            if truth_indices is None:
                shape = (len(self.reco_binning.bins), len(self.truth_binning.bins))
            else:
                shape = (len(self.reco_binning.bins), len(truth_indices))
        response = response.reshape(list(response.shape[:-2])+list(shape), order='C')

        return response

    def get_in_bin_variation_as_ndarray(self, shape=None, truth_only=True, ignore_variables=[], variable_slices={}, truth_indices=None):
        """Get an estimate for the variation of the response within a bin.

        The in-bin variation is estimated from the maximum difference to the
        surrounding bins. The differences are normalized to the estimated
        statistical errors, so values close to one indicate a statistically
        dominated variation.

        Parameters
        ----------

        shape : tuple of ints, optional
            The shape of the returned ndarray.
            Default: ``(#(reco bins), #(truth bins))``
        truth_only : bool, optional
            Only consider the neighbouring bins along the truth-axes.
        ignore_variables : list of strings, optional
            These variables will not be considered. This is useful to exclude
            categorical variables, where the response is not expected to vary
            smoothly.
        variable_slices : dict of slices, optional
            For variables in `variable_slices` only the specified slice will be
            used for comparison, e.g. `variable_slices = {'var_A': slice(1,5)}`.
            Useful if the response is only expected to be smooth over a given
            range of the variable.
        truth_indices : list of ints, optional
            Return a sliced matrix with only the given columns.

        Returns
        -------

        ndarray

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

        ret = ret.reshape((len(self.reco_binning.bins), len(self.truth_binning.bins)), order='C')

        # Slice the truth bins
        if truth_indices is not None:
            ret = np.array(ret[:,truth_indices])

        # Adjust shape
        if shape is not None:
            ret = ret.reshape(shape, order='C')

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

        Parameters
        ----------

        in_bin_variation_limit : float, optional
            Bins will only be merged if the maximum in-bin variation of the
            resulting matrix does not exceed the `in_bin_variation_limit`.
        select : {'entries', 'entries_sum', 'in-bin', 'in-bin_sum'}
            Determines how the merging candidate is selected:

            entries
                the bin with the lowest number of truth entries
            entries_sum
                the pair of bins with the lowest number of truth entries
            in-bin
                the bin with the lowest maximum in-bin variation
            in-bin_sum
                the pair of bins with the lowest sum of maximum in-bin variations

        kwargs : optional
            Additional keyword arguments will be passed to the method
            :meth:`get_in_bin_variation_as_ndarray`.

        See also
        --------

        get_in_bin_variation_as_ndarray

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

    @staticmethod
    def _block_mahalanobis2(X, mu, inv_cov):
        """Efficiently calculate squared Mahalanobis distance for diagonal block matrix covariances.

        It returns the squared Mahalanobis distance of each block separately.
        To get the total distance, one must sum over these numbers.

        Parameters
        ----------

        X : array_like
            The objects of which the Mahalanobis distance will be calculated.
            Must be of shape ``(n, a, b)``.
        mu : array_like
            The mean values of the distribution. The Mahalanobis distance is
            calculated with respect to these.
            Must be of shape ``(a, b)``.
        inv_cov : array_like
            The inverse of the covariance matrix of the distribution.
            It must be a diagonal block matrix of ``a`` blocks with each block
            of the shape ``(b, b)``. To save space, the off-diagonal 0s are not stored.
            Must be of shape ``(a, b, b)``.

        Returns
        -------

        D_M : ndarray
            The array of squared Mahalanobis distances of shape ``(n, a)``.

        """

        diff = np.asfarray(X) - np.asfarray(mu)
        inv_cov = np.asfarray(inv_cov)
        D_M = np.einsum('...b,...bc,...c', diff, inv_cov, diff)
        return D_M

    def distance_as_ndarray(self, other, shape=None, N=None, return_distances_from_mean=False, **kwargs):
        """Calculate the squared Mahalanobis distance of the two matrices for each truth bin.

        Parameters
        ----------

        other : ResponseMatrix
            The other ResponseMatrix for the comparison.
        shape : tuple of ints, optional
            The shape of the returned matrix.
            Defaults to ``(#(truth bins),)``.
        N : int, optional
            Number of random matrices to be generated for the calculation.
            This number must be larger than the number of *reco* bins!
            Otherwise the covariances cannot be calculated correctly.
            Defaults to ``#(reco bins) + 100)``.
        return_distances_from_mean : bool, optional
            Also return the ndarray ``distances_from_mean``.
        kwargs : optional
            Additional keyword arguments are passed through to
            `generate_random_response_matrices`.

        Returns
        -------

        distance : ndarray
            Array of shape `shape` with the squared Mahalanobis distance
            of the mean difference between the matrices for each truth bin:
            ``D^2( mean(self.random_matrices - other.random_matrices) )``
        distances_from_mean : ndarray, optional
            Array of shape ``(N,)+shape`` with the squared Mahalanobis
            distances between the randomly generated matrix differences
            and the mean matrix difference for each truth bin:
            ``D^2( (self.random_matrices - other.random_matrices)
            - mean(self.random_matrices - other.random_matrices) )``

        """

        n_reco = len(self.reco_binning.bins)
        if 'truth_indices' in kwargs:
            n_truth = len(kwargs['truth_indices'])
        else:
            n_truth = len(self.truth_binning.bins)
        n_bins = n_truth * n_reco
        if N is None:
            N = n_reco + 100

        # Get the *transposed* set of matrices
        self_matrices = self.generate_random_response_matrices(size=N, **kwargs).T

        other_matrices = other.generate_random_response_matrices(size=N, **kwargs).T
        differences = self_matrices - other_matrices

        # Since the detector response is handled completely independently for each truth index,
        # we can calculate the covariance matrices and distances for each one individually.
        inv_cov_list = []
        for i in range(n_truth):
            cov = np.cov(differences[i])
            inv_cov_list.append(np.linalg.inv(cov))

        null = np.zeros((n_truth, n_reco))
        mean = (self.get_mean_response_matrix_as_ndarray(**kwargs)
            - other.get_mean_response_matrix_as_ndarray(**kwargs)).T

        distance = self._block_mahalanobis2([null], mean, inv_cov_list)[0]

        if shape is not None:
            distance = distance.reshape(shape, order='C')

        if return_distances_from_mean:
            differences = differences.transpose((2,0,1)) # (truth, reco, N) -> (N, truth, reco)
            distances_from_mean = self._block_mahalanobis2(differences, mean, inv_cov_list)
            return distance, distances_from_mean
        else:
            return distance

    def distance(self, other, N=None, return_distances_from_mean=False, **kwargs):
        """Return the overall squared Mahalanobis distance between the two matrices.

        Parameters
        ----------

        other : ResponseMatrix
            The other ResponseMatrix for the comparison.
        N : int, optional
            Number of random matrices to be generated for the calculation.
            This number must be larger than the number of *reco* bins!
            Otherwise the covariances cannot be calculated correctly.
            Defaults to ``#(reco bins) + 100)``.
        return_distances_from_mean : bool, optional
            Also return the ndarray ``distances_from_mean``.
        kwargs : optional
            Additional keyword arguments are passed through to
            `generate_random_response_matrices`.

        Returns
        -------

        distance : flaot
            The squared Mahalanobis distance of the mean difference
            between the matrices:
            ``D^2( mean(self.random_matrices - other.random_matrices) )``
        distances_from_mean : ndarray
            Array of shape ``(N,)`` with the squared Mahalanobis
            distances between the randomly generated matrix differences
            and the mean matrix difference:
            ``D^2( (self.random_matrices - other.random_matrices)
            - mean(self.random_matrices - other.random_matrices) )``

        """

        if return_distances_from_mean:
            null_distance, distances = self.distance_as_ndarray(other, N=N, return_distances_from_mean=True, **kwargs)
            null_distance = np.sum(null_distance, axis=-1)
            distances = np.sum(distances, axis=-1)
            return null_distance, distances
        else:
            null_distance = self.distance_as_ndarray(other, N=N, **kwargs)
            return np.sum(null_distance, axis=-1)

    def compatibility(self, other, N=None, return_all=False, truth_indices=None, **kwargs):
        """Calculate the compatibility between this and another response matrix.

        Basically, this checks whether the point of "the matrices are
        identical" is an outlier in the distribution of matrix differences as
        defined by the statistical uncertainties of the matrix elements. This
        is done using the Mahalanobis distance as the test statistic. If the
        point "the matrices are identical" is not a reasonable part of the
        distribution, it is not reasonable to assume that the true matrices are
        identical.

        Parameters
        ----------

        other : ResponseMatrix
            The other response matrix.
        N : int, optional
            Number of random matrices to be generated for the calculation.
            This number must be larger than the number of *reco* bins!
            Otherwise the covariances cannot be calculated correctly.
            Defaults to ``#(reco bins) + 100)``.
        return_all : bool, optional
            If ``False``, return only `null_prob_count`, and `null_prob_chi2`.
        truth_indices : list of ints, optional
            Only use the given truth indices to calculate the compatibility.
            If this is not specified, only the indices with at least one entry
            in *both* matrices are used.

        Returns
        -------

        null_prob_count : float
            The Bayesian p-value evaluated by counting the expected number of
            random matrix differences more extreme than the mean difference.
        null_prob_chi2 : float
            The Bayesian p-value evaluated by assuming a chi-square distribution
            of the squares of Mahalanobis distances.
        null_distance : float
            The squared Mahalanobis distance of the mean differences between the
            two matrices.
        distances : ndarray
            The set of squared Mahalanobis distances between randomly generated
            matrix differences and the mean matrix difference.
        df : int
            Degrees of freedom of the assumed chi-squared distribution of the
            squared Mahalanobis distances. This is equal to the number of matrix
            elements that are considered for the calculation:
            ``df = len(truth_indices) * len(reco_bins in matrix)``.

        Notes
        -----

        The distribution of matrix differences is evaluated by generating ``N``
        random response matrices from both compared matrices and calculating
        the (n-dimensional) differences. The resulting set of matrix
        differences defines the mean ``mean(differences)`` and the covariance
        matrix ``cov(differences)``. The covariance in turn defines a metric
        for the Mahalanobis distance ``D_M(x)`` on the space of matrix
        differences, where ``x`` is a set of matrix element differences.

        The distance between the mean difference and the Null hypothesis, that
        the two true matrices are identical, is the ``null_distance``::

            null_distance = D_M(0 - mean(differences))

        The compatibility between the matrices is now defined as the Bayesian
        probability that the true difference between the matrices is more
        extreme (has a larger distance from the mean difference) than the
        Null hypothesis. For this, we can just evaluate the set of
        matrix differences that was used to calculate the covariance matrix::

            distances = D_M(differences - mean(differences))
            null_prob_count = np.sum(distances >= null_distance) / distances.size

        It will be 1 if the mean difference between the matrices is 0, and tend
        to 0 when the mean difference between the matrices is far from 0. "Far"
        in this case is determined by the uncertainty, i.e. the covariance, of
        the difference determination.

        In the case of normal distributed differences, the distribution of
        squared Mahalanobis distances becomes chi-squared distributed. The
        numbers of degrees of freedom of that distribution is the number of
        variates, i.e. the number of response matrix elements that are being
        considered. This can be used to calculate a theoretical value for the
        compatibility::

            df = len(truth_indices) * #(reco_bins)
            null_prob_chi2 = chi2.sf(null_distance**2, df)

        Since the distribution of differences is not necessarily Gaussian, this
        is only an estimate. Its advantage is that it is less dependent on the
        number of randomly drawn matrices.

        """

        if truth_indices is None:
            # If nothing else is specified, only consider truth bins that have at least one event in them in both matrices.
            # Empty bins hold no information.
            truth_indices = list(sorted(set(self.filled_truth_indices) & set(other.filled_truth_indices)))

        n_reco = len(self.reco_binning.bins)
        n_truth = len(truth_indices)
        n_bins = n_truth * n_reco

        null_distance, distances = self.distance(other, N=N, truth_indices=truth_indices, return_distances_from_mean=True, **kwargs)

        null_prob_chi2 = stats.chi2.sf(null_distance, n_bins)
        null_prob_count = float(np.sum(distances >= null_distance)) / distances.size

        if return_all:
            return null_prob_count, null_prob_chi2, null_distance, distances, n_bins
        else:
            return null_prob_count, null_prob_chi2

    def plot_compatibility(self, filename, other):
        """Plot the compatibility of the two matrices.

        Parameters
        ----------

        filename : string
            The filename where the plot will be stored.
        other : ResponseMatrix
            The other Response Matrix for the comparison.

        Returns
        -------

        fig : Figure
            The figure that was used for plotting.
        ax : Axis
            The axis that was used for plotting.

        """

        # Load matplotlib on demand
        global plt
        if plt is None:
            from matplotlib import pyplot as _pyplot
            plt = _pyplot

        prob_count, prob_chi2, dist, distances, df = self.compatibility(other, return_all=True)

        fig, ax = plt.subplots()
        ax.set_xlabel(r"$D_M^2$")
        nbins = max(min(distances.size // 100, 100), 5)
        ax.hist(distances, nbins, normed=True, histtype='stepfilled', label="actual distribution, C=%.3f"%(prob_count,))
        x = np.linspace(df-3*np.sqrt(2*df), df+5*np.sqrt(2*df), 50)
        ax.plot(x, stats.chi2.pdf(x, df), label="$\chi^2$ distribution, C=%.3f"%(prob_chi2,))
        ax.axvline(dist, color='r', label="null distance")
        ax.legend(loc='best', framealpha=0.5)
        fig.savefig(filename)

        return fig, ax

    def plot_distance(self, filename, other, expectation=True, variables=None, kwargs1d={}, kwargs2d={}, figax=None, reduction_function=np.sum, **kwargs):
        """Plot the squared mahalanobis distance between two matrices.

        Parameters
        ----------

        filename : string
            The filename of the plot to be stored.
        other : ResponseMatrix
            The other ResponseMatrix for the comparison
        expectation : bool, optional
            Plot an approximation of the expected values for identical matrices in the 1D histograms.
        variables : {None, (None, None), list, (list, list)}, optional
            ``None``, plot marginal histograms for all variables.
            ``(None, None)``, plot 2D histograms of all possible variable combinations.
            ``list``, plot marginal histograms for these variables.
            ``(list, list)``, plot 2D histograms of the Cartesian product of the two variable lists.
            2D histograms where both variables are identical are plotted as 1D histograms.
        kwargs1d, kwargs2d : dict, optional
            Additional keyword arguments for the 1D/2D histograms.
             If the key `label` is present, a legend will be drawn.
        figax : (figure, axes), optional
            Pair of figure and axes to be used for plotting.
            Can be used to plot multiple binnings on top of one another.
            Default: Create new figure and axes.
        reduction_function : function, optional
            Use this function to marginalize out variables.
        kwargs : optional
            Additional `kwargs` will be passed to `calculate_squared_mahalanobis_distance`.

        Returns
        -------

        fig : Figure
            The Figure that was drawn on.
        ax : Axis
            The axes objects.

        """

        truth_binning = self.truth_binning
        distances = self.distance_as_ndarray(other, **kwargs)

        figax = truth_binning.plot_ndarray(filename, distances, variables=variables, divide=False, kwargs1d=kwargs1d, kwargs2d=kwargs2d, figax=figax, reduction_function=reduction_function)
        if expectation:
            any_entries = (self.get_truth_entries_as_ndarray() > 0)
            any_entries &= (other.get_truth_entries_as_ndarray() > 0)
            expect = np.where(any_entries, len(self.reco_binning.bins), 0)
            figax = truth_binning.plot_ndarray(filename, expect, variables=variables, divide=False, kwargs1d={'linestyle': 'dashed'}, kwargs2d={'alpha': 0.}, figax=figax, reduction_function=reduction_function)
        return figax

    def plot_values(self, *args, **kwargs):
        """Plot the values of the response binning.

        This plots the distribution of events that have *both* a truth and reco bin.

        See also
        --------

        remu.Binning.RectangularBinning.plot_values

        """

        return self.response_binning.plot_values(*args, **kwargs)

    def plot_entries(self, *args, **kwargs):
        """Plot the entries of the response binning.

        This plots the distribution of events that have *both* a truth and reco bin.

        See also
        --------

        remu.binning.RectangularBinning.plot_entries

        """

        return self.response_binning.plot_entries(*args, **kwargs)

    def plot_in_bin_variation(self, filename, variables=None, kwargs1d={}, kwargs2d={}, figax=None, **kwargs):
        """Plot the maximum in-bin variation.

        Parameters
        ----------

        filename : string or None
            The target filename of the plot. If `None`, the plot fill not be
            saved to disk. This is only useful with the `figax` option.
        variables : optional
            One of the following:

            `list of strings`
                List of variables to plot marginal histograms for.
            `None`
                Plot marginal histograms for all variables.
            `(list of strings, list of strings)`
                Plot 2D histograms of the cartesian product of the two variable lists.
                2D histograms where both variables are identical are plotted as 1D histograms.
            `(None, None)`
                Plot 2D histograms of all possible variable combinations.
                2D histograms where both variables are identical are plotted as 1D histograms.

            Default: `None`
        kwargs1d, kwargs2d : dict, optional
            Additional keyword arguments for the 1D/2D histograms.
            If the key `label` is present, a legend will be drawn.
        figax : tuple of (Figure, list of list of Axis), optional
            Pair of figure and axes to be used for plotting.
            Can be used to plot multiple binnings on top of one another.
            Default: Create new figure and axes.
        kwargs : optional
            Additional `kwargs` will be passed on to :meth:`get_in_bin_variation_as_ndarray`.

        Returns
        -------

        fig : Figure
            The Figure that was used for plotting.
        ax : list of list of Axis
            The axes that were used for plotting.

        See also
        --------

        get_in_bin_variation_as_ndarray

        """

        truth_binning = self.truth_binning
        inbin = self.get_in_bin_variation_as_ndarray(**kwargs)
        inbin = np.max(inbin, axis=0)

        return truth_binning.plot_ndarray(filename, inbin, variables=variables, kwargs1d=kwargs1d, kwargs2d=kwargs2d, figax=figax, divide=False, reduction_function=np.max)

    def plot_statistical_variance(self, filename, variables=None, kwargs1d={}, kwargs2d={}, figax=None, **kwargs):
        """Plot the maximum statistical variation for projections on all truth variables.

        Parameters
        ----------

        filename : string or None
            The target filename of the plot. If `None`, the plot fill not be
            saved to disk. This is only useful with the `figax` option.
        variables : optional
            One of the following:

            `list of strings`
                List of variables to plot marginal histograms for.
            `None`
                Plot marginal histograms for all variables.
            `(list of strings, list of strings)`
                Plot 2D histograms of the cartesian product of the two variable lists.
                2D histograms where both variables are identical are plotted as 1D histograms.
            `(None, None)`
                Plot 2D histograms of all possible variable combinations.
                2D histograms where both variables are identical are plotted as 1D histograms.

            Default: `None`
        kwargs1d, kwargs2d : dict, optional
            Additional keyword arguments for the 1D/2D histograms.
            If the key `label` is present, a legend will be drawn.
        figax : tuple of (Figure, list of list of Axis), optional
            Pair of figure and axes to be used for plotting.
            Can be used to plot multiple binnings on top of one another.
            Default: Create new figure and axes.
        kwargs : optional
            Additional `kwargs` will be passed on to :meth:`get_statistical_variance_as_ndarray`.

        Returns
        -------

        fig : Figure
            The Figure that was used for plotting.
        ax : list of list of Axis
            The axes that were used for plotting.

        See also
        --------

        get_statistical_variance_as_ndarray

        """

        truth_binning = self.truth_binning
        stat = self.get_statistical_variance_as_ndarray(**kwargs)
        stat = np.sqrt(np.max(stat, axis=0))

        return truth_binning.plot_ndarray(filename, stat, variables=variables, kwargs1d=kwargs1d, kwargs2d=kwargs2d, figax=figax, divide=False, reduction_function=np.max)

    def plot_expected_efficiency(self, filename, variables=None, kwargs1d={}, kwargs2d={}, figax=None, **kwargs):
        """Plot expected efficiencies for projections on all truth variables.

        This assumes the truth values are distributed like the generator data.
        This does *not* consider the statistical uncertainty of the matrix
        elements.

        Parameters
        ----------

        filename : string or None
            The target filename of the plot. If `None`, the plot fill not be
            saved to disk. This is only useful with the `figax` option.
        variables : optional
            One of the following:

            `list of strings`
                List of variables to plot marginal histograms for.
            `None`
                Plot marginal histograms for all variables.
            `(list of strings, list of strings)`
                Plot 2D histograms of the cartesian product of the two variable lists.
                2D histograms where both variables are identical are plotted as 1D histograms.
            `(None, None)`
                Plot 2D histograms of all possible variable combinations.
                2D histograms where both variables are identical are plotted as 1D histograms.

            Default: `None`
        kwargs1d, kwargs2d : dict, optional
            Additional keyword arguments for the 1D/2D histograms.
            If the key `label` is present, a legend will be drawn.
        figax : tuple of (Figure, list of list of Axis), optional
            Pair of figure and axes to be used for plotting.
            Can be used to plot multiple binnings on top of one another.
            Default: Create new figure and axes.
        kwargs : optional
            Additional `kwargs` will be passed to :meth:`get_response_values_as_ndarray`
            and :meth:`get_truth_values_as_ndarray`.

        Returns
        -------

        fig : Figure
            The Figure that was used for plotting.
        ax : list of list of Axis
            The axes that were used for plotting.

        """

        nuisance_indices = self.nuisance_indices

        truth_binning = self.truth_binning
        shape = (len(self.reco_binning.bins), len(self.truth_binning.bins))
        eff = self.get_response_values_as_ndarray(shape=shape, **kwargs)
        eff = np.sum(eff, axis=0)
        eff[nuisance_indices] = 0.
        truth = self.get_truth_values_as_ndarray(**kwargs)
        truth = np.where(truth > 0, truth, 1e-50)
        truth[nuisance_indices] = 1e-50

        return truth_binning.plot_ndarray(filename, eff, variables=variables, kwargs1d=kwargs1d, kwargs2d=kwargs2d, figax=figax, divide=False, reduction_function=np.sum, denominator=truth)

    def __add__(self, other):
        """Add two matrices together, combining their data."""
        ret = deepcopy(self)
        ret.truth_binning = self.truth_binning + other.truth_binning
        ret.reco_binning = self.reco_binning + other.reco_binning
        ret.response_binning = self.response_binning + other.response_binning
        ret._fix_rounding_errors()
        ret._update_filled_indices()
        return ret

class ResponseMatrixArrayBuilder(object):
    """Class that generates consistent ndarrays from multiple response matrix objects.

    Parameters
    ----------

    nstat : int
        The number of random matrices to be generated for each ResponseMatrix.
        If `nstat` is 0, no random matrices are generated. Instead the output
        of :meth:`ResponseMatrix.get_response_matrix_as_ndarray` is used.

    Notes
    -----

    This class is used to generate the random response matrices from multiple
    toy simulations covering the detector uncertainties. Each toy simulation
    yields one ResponseMatrix object. These ResponseMatrices are then combined
    with a `ResponseMatrixArrayBuilder`.

    The `ResponseMatrixArrayBuilder` is designed to use as little memory as
    possible. It stores only the necessary information of the
    `ResponseMatrices`, *not* the `ResponseMatrix` objects themselves. It only
    stores the truth bins that were actually filled.

    The matrices must have been built using the same truth information! The
    filled bins may only differ in the nuisance bins. When creating the final
    `ndarray`, missing nuisance columns are filled with default values (0).

    The relative efficiencies of the nuisance bins are guaranteed to be
    consistent between the different response matrices. The absolute value of
    the efficiencies is not conserved. In fact, the efficiency for
    nuisance bin ``j`` in ResponseMatrix ``t`` will be::

        eff_tj = N_tj / sum_t( N_tj)

    where ``N_tj`` is the value of nuisance truth bin ``j`` in ResponseMatrix
    ``t``. This means the efficiency of the nuisance bins decreases with the
    number of added matrices. This has to be taken into account when creating
    truth templates for the nuisance bins. They should consist of the *sum* of
    the selected nuisance events over the different toy simulations. This way,
    the template multiplied with the ndarray will re-create the number of
    selected nuisance events in each bin for all toy response matrices.

    """

    def __init__(self, nstat):
        """ """
        self.nstat = nstat
        self.reset()

    def reset(self):
        """Reset everything to 0."""
        self.nmatrices = 0
        self._matrices = []
        self._mean_matrices = []
        self._truth_values = []
        self._truth_entries = None
        self._response_values = None
        self._filled_indices = []
        self._nuisance_indices = None

    def add_matrix(self, response_matrix):
        """Add a matrix to the collection.

        Parameters
        ----------

        response_matrix : ResponseMatrix

        Notes
        -----

        This immediately triggers the generation of `nstat` random variations
        of `response_matrix`.

        See also
        --------

        ResponseMatrix.generate_random_response_matrices

        """

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
        """Return the list of filled truth indices.

        This list contains the indices of all truth bins that have been filled
        in at least one of the matrices that were added to the
        `ResponseMatrixArrayBuilder`.

        """
        return sorted(self._get_filled_truth_indices_set())

    def get_truth_entries_as_ndarray(self):
        """Return the array of maximum entries in the truth bins."""
        return self._truth_entries

    def get_response_values_as_ndarray(self):
        """Return the mean values of the response bins."""
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

    def get_random_response_matrices_as_ndarray(self):
        """Get the response matrices as consistent ndarray.

        Returns
        -------

        ndarray
            A big `ndarray` containing `nstat` generated matrices for each
            `ResponseMatrix` that has been added.

        Notes
        -----

        The returned matrices will only contain the columns, i.e. truth bins,
        that have been filled in at least one of the added ResponseMatrices.
        The shape of the returned array will be::

            (#(RepsonseMatrices), [nstat,] #(reco bins), #(filled truth bins))

        The indices of the returned (filled) truth bins can be requested with
        the :meth:`get_filled_truth_indices` method.

        See also
        --------

        get_mean_response_matrices_as_ndarray

        """

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

    def get_mean_response_matrices_as_ndarray(self):
        """Get the mean response matrices as ndarray.

        Returns
        -------

        ndarray
            A big `ndarray` containing the mean matrix for each
            `ResponseMatrix` that has been added.

        Notes
        -----

        The returned matrices will only contain the columns, i.e. truth bins,
        that have been filled in at least one of the added ResponseMatrices.
        The shape of the returned array will be::

            (#(RepsonseMatrices), #(reco bins), #(filled truth bins))

        The indices of the returned (filled) truth bins can be requested with
        the :meth:`get_filled_truth_indices` method.

        See also
        --------

        get_random_response_matrices_as_ndarray

        """

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
