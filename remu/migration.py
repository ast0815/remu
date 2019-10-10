"""Module handling the creation and use of migration matrices."""

from __future__ import division
import numpy as np
from scipy import stats
from scipy import linalg
from copy import copy, deepcopy
from warnings import warn

from .binning import Binning, CartesianProductBinning

class ResponseMatrix(object):
    """Matrix that describes the detector response to true events.

    Parameters
    ----------

    reco_binning : RectangularBinning
        The Binning object describing the reco categorization.
    truth_binning : RectangularBinning
        The Binning object describing the truth categorization.
    nuisance_indices : list of ints, optional
        List of indices of nuisance truth bins.
        These are treated like their efficiency is exactly 1.
    impossible_indices :list of ints, optional
        List of indices of impossible reco bins.
        These are treated like their probability is exactly 0.
    response_binning : CartesianProductBinning, optional
        The Binning object describing the reco and truth categorization.
        Usually this will be generated from the truth and reco binning.

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

    Attributes
    ----------

    truth_binning : Binning
        The :class:`.Binning` object for the truth information of the events.
    reco_binning : Binning
        The :class:`.Binning` object for the reco information of the events.
    response_binning : CartesianProductBinning
        The :class:`.CartesianProductBinning` of reco and truth binning.
    nuisance_indices : list of int
        The truth data indices that will be handled as nuisance bins.
    impossible_indices : list of int
        The reco data indices that will be treated as impossible to occur.
    filled_truth_indices : list of int
        The data indices of truth bins that have at least one event in them.

    """

    def __init__(self, reco_binning, truth_binning, nuisance_indices=[], impossible_indices=[], response_binning=None):
        self.truth_binning = truth_binning
        self.reco_binning = reco_binning
        if response_binning is None:
            self.response_binning = CartesianProductBinning([reco_binning.clone(dummy=True), truth_binning.clone(dummy=True)])
        else:
            self.response_binning = response_binning
        self.nuisance_indices=nuisance_indices
        self.impossible_indices=impossible_indices
        self._update_filled_indices()

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

    def set_truth_values_from_ndarray(self, *args, **kwargs):
        """Set the values of the truth binning as `ndarray`."""
        self.truth_binning.set_values_from_ndarray(*args, **kwargs)

    def set_truth_entries_from_ndarray(self, *args, **kwargs):
        """Set the number of entries in the truth binning as `ndarray`."""
        self.truth_binning.set_entries_from_ndarray(*args, **kwargs)
        self._update_filled_indices()

    def set_truth_sumw2_from_ndarray(self, *args, **kwargs):
        """Set the sum of squared weights in the truth binning as `ndarray`."""
        self.truth_binning.set_sumw2_from_ndarray(*args, **kwargs)

    def set_reco_values_from_ndarray(self, *args, **kwargs):
        """Set the values of the reco binning as `ndarray`."""
        self.reco_binning.set_values_from_ndarray(*args, **kwargs)

    def set_reco_entries_from_ndarray(self, *args, **kwargs):
        """Set the number of entries in the reco binning as `ndarray`."""
        self.reco_binning.set_entries_from_ndarray(*args, **kwargs)

    def set_reco_sumw2_from_ndarray(self, *args, **kwargs):
        """Set the sum of squared weights in the reco binning as `ndarray`."""
        self.reco_binning.set_sumw2_from_ndarray(*args, **kwargs)

    def set_response_values_from_ndarray(self, *args, **kwargs):
        """Set the values of the response binning as `ndarray`."""
        self.response_binning.set_values_from_ndarray(*args, **kwargs)

    def set_response_entries_from_ndarray(self, *args, **kwargs):
        """Set the number of entries in the response binning as `ndarray`."""
        self.response_binning.set_entries_from_ndarray(*args, **kwargs)

    def set_response_sumw2_from_ndarray(self, *args, **kwargs):
        """Set the sum of squared weights in the response binning as `ndarray`."""
        self.response_binning.set_sumw2_from_ndarray(*args, **kwargs)

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

        original_shape = (self.reco_binning.data_size, self.truth_binning.data_size)

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

        N_reco = self.reco_binning.data_size
        N_truth = self.truth_binning.data_size
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
        n_vars = len(self.reco_binning.phasespace)
        prior = min(1., 3.**n_vars / (N_reco - len(impossible_indices)))
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
            probability is exactly 0.
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

        # Unweighted (multinomial) smearing probabilty
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
        in them by their definition.

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
        get_in_bin_variation_as_ndarray

        """

        beta1, beta2, alpha, mu, sigma = self._get_stat_error_parameters(**kwargs)

        # Unweighted binomial reconstruction probability (efficiency)
        # Posterior mean estimate = beta1 / (beta1 + beta2)
        beta0 = beta1 + beta2
        effj = beta1 / beta0
        # Posterior variance
        effj_var = beta1*beta2 / (beta0**2 * (beta0+1))

        # Unweighted (multinomial) smearing probabilty
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

    def get_in_bin_variation_as_ndarray(self, shape=None, truth_indices=None, normalize=True, **kwargs):
        """Get an estimate for the variation of the response within a bin.

        The in-bin variation is estimated from the maximum difference to the
        surrounding truth bins. The differences can be normalized to the
        estimated statistical errors, so values close to one indicate a
        statistically dominated variation.

        Parameters
        ----------

        shape : tuple of ints, optional
            The shape of the returned ndarray.
            Default: ``(#(reco bins), #(truth bins))``

        truth_indices : list of ints, optional
            Return a sliced matrix with only the given columns.

        normalize : bool, optional
            Divide the variation by the statistical variance

        **kwargs : optional
            Additional keyword arguments are passed to
            :meth:`get_mean_response_matrix_as_ndarray` and
            :meth:`get_statistical_variance_as_ndarray`.

        Returns
        -------

        ndarray

        See also
        --------

        get_statistical_variance_as_ndarray

        """
        response = self.get_mean_response_matrix_as_ndarray(**kwargs)
        if normalize:
            variance = self.get_statistical_variance_as_ndarray(**kwargs)
        adjacent = self.truth_binning.get_adjacent_data_indices()
        variation = np.zeros_like(response)
        for reco_index in range(response.shape[0]):
            for truth_index in range(response.shape[1]):
                diff = response[reco_index, truth_index]
                diff = diff - response[reco_index, adjacent[truth_index]]
                diff = diff**2
                if normalize:
                    var = variance[reco_index, truth_index]
                    var = var + variance[reco_index, adjacent[truth_index]]
                    diff = diff/var
                variation[reco_index, truth_index] = np.sqrt(np.max(diff))

        if truth_indices is not None:
            variation = variation[:,truth_indices]

        if shape is not None:
            variation.shape = shape

        return variation

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
                shape = (self.reco_binning.data_size, self.truth_binning.data_size)
            else:
                shape = (self.reco_binning.data_size, len(truth_indices))
        response = response.reshape(list(response.shape[:-2])+list(shape), order='C')

        return response

    def __add__(self, other):
        """Add two matrices together, combining their data."""
        ret = deepcopy(self)
        ret.truth_binning = self.truth_binning + other.truth_binning
        ret.reco_binning = self.reco_binning + other.reco_binning
        ret.response_binning = self.response_binning + other.response_binning
        ret._fix_rounding_errors()
        ret._update_filled_indices()
        return ret

    def export(self, filename, compress=False, nstat=None, sparse=True):
        """Save all necessary information for using the response matrix.

        Saves all necessary information for using the response matrix`
        in a NumPy ``.npz`` archive.

        Parameters
        ----------

        filename : str or file
            Where to store the arrays.
        compress : bool, optional
            Whether to use compression.
        nstat : int, optional
            How many random variations of the matrix to generate.
            Default: Export mean matrix, no random variation
        sparse : bool, optional
            Should a sparse version be exported, or the full matrix.

        See also
        --------

        ResponseMatrixArrayBuilder.export

        """

        if nstat is None:
            matrices = self.get_mean_response_matrix_as_ndarray()[np.newaxis,...]
        else:
            matrices = self.generate_random_response_matrices(size=nstat)
        truth_entries = self.get_truth_entries_as_ndarray()

        if sparse:
            sparse_indices = np.flatnonzero(truth_entries)
            matrices = matrices[...,sparse_indices]
            data = {
                'matrices': matrices,
                'truth_entries': truth_entries,
                'sparse_indices': sparse_indices,
                'is_sparse': True,
                }
        else:
            data = {
                'matrices': matrices,
                'truth_entries': truth_entries,
                }

        if compress:
            np.savez_compressed(filename, **data)
        else:
            np.savez(filename, **data)

    def clone(self):
        """Create a functioning copy of the response matrix."""
        reco_binning = self.reco_binning.clone()
        response_binning = self.response_binning.clone()
        truth_binning = self.truth_binning.clone()
        nuisance_indices = deepcopy(self.nuisance_indices)
        impossible_indices = deepcopy(self.impossible_indices)
        return ResponseMatrix(reco_binning, truth_binning, nuisance_indices=nuisance_indices, impossible_indices=impossible_indices, response_binning=response_binning)

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
    yields one :class:`ResponseMatrix` object. These are then combined with
    the :class:`ResponseMatrixArrayBuilder`.

    The :class:`ResponseMatrixArrayBuilder` is designed to use as little memory
    as possible. It stores only the necessary information of the
    :class:`ResponseMatrix`, *not* the :class:`ResponseMatrix` objects
    themselves. It only stores the truth bins that were actually filled.

    The matrices must have been built using the same truth information! The
    filled bins may only differ in the nuisance bins. When creating the final
    :class:`ndarray`, missing nuisance columns that were filled in some but not
    all matrices are filled with default values (0).

    The relative efficiencies of the nuisance bins are guaranteed to be
    consistent between the different response matrices. The absolute value of
    the efficiencies is not conserved. In fact, the efficiency for
    nuisance bin ``j`` in :class:`ResponseMatrix` ``t`` will be::

        eff_tj = N_tj / max_t( N_tj)

    where ``N_tj`` is the value of nuisance truth bin ``j`` in
    :class:`ResponseMatrix` ``t``. This means the efficiency of the nuisance
    bins can decrease with the number of added matrices. This has to be taken
    into account when creating truth templates for the nuisance bins. They
    should consist of the *max* of the selected nuisance events over the
    different toy simulations. This way, the template multiplied with the
    ndarray will re-create the number of selected nuisance events in each bin
    for all toy response matrices.

    Attributes
    ----------

    nstat : int
        The number of statistical throws to generate for each added matrix.
    nmatrices : int
        The number of matrices that were added.

    """

    def __init__(self, nstat):
        """ """
        self.nstat = nstat
        self.reset()

    def reset(self):
        """Reset everything to 0."""
        self.nmatrices = 0
        self._matrices = []
        self._weights = []
        self._mean_matrices = []
        self._truth_values = []
        self._truth_entries = None
        self._filled_indices = []
        self._nuisance_indices = None

    def add_matrix(self, response_matrix, weight=1.):
        """Add a matrix to the collection.

        Parameters
        ----------

        response_matrix : :class:`ResponseMatrix`
        weight : float, optional

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
            matrix = response_matrix.get_response_matrix_as_ndarray(truth_indices=filled_indices)[np.newaxis,...]
        mean_matrix = response_matrix.get_mean_response_matrix_as_ndarray(truth_indices=filled_indices)
        truth_values = response_matrix.get_truth_values_as_ndarray(indices=filled_indices)
        truth_entries = response_matrix.get_truth_entries_as_ndarray() # We need *all* entries

        self._filled_indices.append(filled_indices)
        self._matrices.append(matrix)
        self._weights.append(weight)
        self._mean_matrices.append(mean_matrix)
        self._truth_values.append(truth_values)
        if self._truth_entries is None:
            self._truth_entries = truth_entries
        else:
            self._truth_entries = np.maximum(self._truth_entries, truth_entries)
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

    def _get_truth_value_scale(self, tv):
        """Get scale to make nuisance bins consistent.

        The nuisance bins must be scaled between the multiple matrices, because
        in each matrix their efficiency is 1 by definition.  Ideally they all
        would be scaled to the true number of true events in each truth bin,
        but this information is not available for nuisance bins.  Instead we
        use the sum of truth values over all matrices as denominator off the
        efficiency, e.g. the efficiency of nuisance truth bin j in matrix t:

            eff_tj = N_tj / max_t( N_tj)

        This means the efficiency of the nuisance bins can decrease with the
        number of added matrices. This has to be taken into account when
        creating truth templates for the nuisance bins. They should consist of
        the *max* of the selected nuisance events over the different toy
        simulations. This way, the template multiplied with the ndarray will
        re-create the number of selected nuisance events in each bin for all
        toy response matrices.

        """

        all_indices = self._get_filled_truth_indices_set()
        nuisance_indices = set(self._nuisance_indices)
        filled_nuisance_indices = all_indices & nuisance_indices
        max_tv = np.max(tv, axis=0)
        max_tv = np.where(max_tv > 0, max_tv, 1.0)
        scale = np.ones_like(tv) # Start with scales = 1
        for i in np.searchsorted(sorted(all_indices), sorted(filled_nuisance_indices)):
            scale[:,i] = tv[:,i] / max_tv[i] # Set scale of nuisance indices
        return scale

    def get_random_response_matrices_as_ndarray(self):
        """Get the response matrices as consistent ndarray.

        Returns
        -------

        M, weights : ndarray
            A big :class:`ndarray` containing `nstat` generated matrices for
            each :class:`ResponseMatrix` that has been added, and a vector of
            weights for corresponding the matrices.

        Notes
        -----

        The returned matrices will only contain the columns, i.e. truth bins,
        that have been filled in at least one of the added
        :class:`ResponseMatrix` objects. The shape of the returned array will
        be::

            (#(RepsonseMatrices) x nstat, #(reco bins), #(filled truth bins))

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
        M = M * scale[:,np.newaxis,np.newaxis,:]

        # Broadcast weights
        weights = np.broadcast_to(np.array(self._weights)[:,np.newaxis], M.shape[:2])

        # Reshape to vector of matrices.
        M.shape = (max(self.nstat, 1) * self.nmatrices,) + M.shape[2:]
        weights = weights.flatten()

        return M, weights

    def get_mean_response_matrices_as_ndarray(self):
        """Get the mean response matrices as ndarray.

        Returns
        -------

        M, weights : ndarray
            A big :class:`ndarray` containing `nstat` generated matrices for
            each :class:`ResponseMatrix` that has been added, and a vector of
            weights for corresponding the matrices.

        Notes
        -----

        The returned matrices will only contain the columns, i.e. truth bins,
        that have been filled in at least one of the added
        :class:`ResponseMatrix` objects. The shape of the returned array will
        be::

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
        weights = np.array(self._weights)

        return M, weights

    def export(self, filename, compress=False):
        """Save all necessary information for using the response matrix.

        Saves all necessary information for using the response matrix
        in a NumPy ``.npz`` archive.

        Parameters
        ----------

        filename : str or file
            Where to store the arrays
        compress : bool, optional
            Whether to use compression

        See also
        --------

        ResponseMatrix.export

        """

        matrices, weights =  self.get_random_response_matrices_as_ndarray()
        truth_entries = self.get_truth_entries_as_ndarray()

        data = {
            'matrices': matrices,
            'weights': weights,
            'truth_entries': truth_entries,
            'sparse_indices': np.flatnonzero(truth_entries),
            'is_sparse': True,
            }

        if compress:
            np.savez_compressed(filename, **data)
        else:
            np.savez(filename, **data)
