"""Module that deals with the calculations of likelihoods."""

from __future__ import division
from copy import deepcopy
from six.moves import map, zip
import numpy as np
from scipy import stats
from scipy import optimize
from scipy.misc import derivative
import inspect
from warnings import warn

# Load multiprocess, matplotlib, and pymc on demand
#rom multiprocess import Pool
Pool = None
#from matplotlib import pyplot as plt
plt = None
#import pymc
pymc = None

class CompositeHypothesis(object):
    """A CompositeHypothesis translates a set of parameters into a truth vector.

    Parameters
    ----------

    translation_function : function
        The function to translate a vector of parameters into a vector of truth
        expectation values::

            truth_vector = translation_function(parameter_vector)

        It must support translating arrays of parameter vectors into arrays of
        truth vectors::

            [truth_vector, ...] = translation_function([parameter_vector, ...])

    parameter_limits : iterable of tuples of floats, optional
        An iterable of lower and upper limits of the hypothesis' parameters.
        The number of limits determines the number of parameters. Parameters
        can be `None`. This sets no limit in that direction.
        ::

            [ (x1_min, x1_max), (x2_min, x2_max), ... ]

        Parameter limits are used in likelihood maximization.

    parameter_priors : iterable of functions, optional
        An iterable of prior probability density functions. The number of
        priors determines the number of parameters. Each function must return
        the logarithmic probability density, given a value of the corresponding
        parameter::

            prior(value=default) = log( pdf(value) )

        They should return ``-numpy.inf`` for excluded values. The function's
        argument *must* be named `value` and a default *must* be provided.
        Parameter priors are used in Marcov Chain Monte Carlo evaluations.

    parameter_names : iterable of strings, optional
        Iterable of the parameter names. These names will be used in some
        plotting comvenience functions.

    Notes
    -----

    Depending on the use case, one can provide `parameter_limits` and/or
    `parameter_priors`, but they are *not* checked for consistency!

    """

    def __init__(self, translation_function, parameter_limits=None, parameter_priors=None, parameter_names=None):
        if parameter_limits is None and parameter_priors is None:
            raise TypeError("Must provide at least one of `parameter_lmits` and/or `parameter_priors`")

        self.parameter_limits = parameter_limits
        self.parameter_priors = parameter_priors
        self.parameter_names = parameter_names
        self._translate = translation_function

    def translate(self, parameters):
        """Translate the parameter vector to a truth vector.

        Parameters
        ----------

        parameters : ndarray like
            Vector of the hypothesis parameters.

        Returns
        -------

        ndarray
            Vector of the corresponding truth space expectation values.

        """
        return self._translate(parameters)

    def fix_parameters(self, fix_values):
        """Return a new CompositeHypothesis by fixing some parameters.

        Parameters
        ----------

        fix_values : iterable of values

            This iterable must have the same length as the vector of parameters
            of the CompositeHypothesis. The parameters of the new
            CompositeHypothesis are fixed to the given values. Parameters that
            should not be fixed must be specified with ``None``. For example,
            to fix the first and third parameter of a 3-parameter hypothesis,
            `fix_values` must look like this::

                (1.23, None, 9.87)

            The resulting CompositeHypothesis has one free parameter, the
            second parameter of the original hypothesis.

        """

        fix_values = np.array(fix_values, dtype=float)
        unfixed = np.where(np.isnan(fix_values))

        def new_translation_function(new_parameters,
                _fix_values=fix_values, _unfixed=unfixed,
                _old_translation_function=self._translate):
            _fix_values[unfixed] = new_parameters
            return _old_translation_function(_fix_values)

        if self.parameter_limits is None:
            new_parameter_limits = None
        else:
            new_parameter_limits = []
            for i, v in enumerate(fix_values):
                if np.isnan(v):
                    new_parameter_limits.append(self.parameter_limits[i])

        if self.parameter_priors is None:
            new_parameter_priors = None
        else:
            new_parameter_priors = []
            for i, v in enumerate(fix_values):
                if np.isnan(v):
                    new_parameter_priors.append(self.parameter_priors[i])

        if self.parameter_names is None:
            new_parameter_names = None
        else:
            new_parameter_names = []
            for i, v in enumerate(fix_values):
                if np.isnan(v):
                    new_parameter_names.append(self.parameter_names[i])

        return CompositeHypothesis(
            translation_function=new_translation_function,
            parameter_limits=new_parameter_limits,
            parameter_priors=new_parameter_priors,
            parameter_names=new_parameter_names)

class LinearHypothesis(CompositeHypothesis):
    """Special case of CompositeHypothesis for linear combinations.

    Parameters
    ----------

    M : 2-dimensional ndarray
        The matrix translating the parameter vector into a truth vector::

            truth = M.dot(parameters)

    b : ndarray, optional
        A constant (vector) to be added to the truth vector::

            truth = M.dot(parameters) + b

    *args, **kwargs : optional
        Other arguments are passed on to the `CompositeHypothesis` init method.

    See also
    --------

    TemplateHypothesis

    """

    def __init__(self, M, b=None, *args, **kwargs):
        self.M = np.array(M, dtype=float)
        if b is None:
            self.b = None
        else:
            self.b = np.array(b, dtype=float)

        if b is None:
            translate = lambda par: np.tensordot(par, self.M, axes=(-1,-1))
        else:
            if self.M.size == 0:
                translate = lambda par: self.b
            else:
                translate = lambda par: np.tensordot(par, self.M, axes=(-1,-1)) + self.b

        CompositeHypothesis.__init__(self, translate, *args, **kwargs)

    def fix_parameters(self, fix_values):
        """Return a new LinearHypothesis by fixing some parameters.

        Parameters
        ----------

        fix_values : iterable of values

            This iterable must have the same length as the vector of parameters
            of the LinearHypothesis. The parameters of the new LinearHypothesis
            are fixed to the given values. Parameters that should not be fixed
            must be specified with ``None``. For example, to fix the first and
            third parameter of a 3-parameter hypothesis, `fix_values` must look
            like this::

                (1.23, None, 9.87)

            The resulting LinearHypothesis has one free parameter, the second
            parameter of the original hypothesis.

        """


        fix_values = np.array(fix_values, dtype=float)
        unfixed = np.isnan(fix_values)

        if self.parameter_limits is None:
            new_parameter_limits = None
        else:
            new_parameter_limits = []
            for i, v in enumerate(fix_values):
                if np.isnan(v):
                    new_parameter_limits.append(self.parameter_limits[i])

        if self.parameter_priors is None:
            new_parameter_priors = None
        else:
            new_parameter_priors = []
            for i, v in enumerate(fix_values):
                if np.isnan(v):
                    new_parameter_priors.append(self.parameter_priors[i])

        if self.parameter_names is None:
            new_parameter_names = None
        else:
            new_parameter_names = []
            for i, v in enumerate(fix_values):
                if np.isnan(v):
                    new_parameter_names.append(self.parameter_names[i])

        fix_values[unfixed] = 0.
        new_b = self.translate(fix_values)
        new_M = np.array(self.M[:,unfixed])

        return LinearHypothesis(M=new_M, b=new_b,
            parameter_limits=new_parameter_limits,
            parameter_priors=new_parameter_priors,
            parameter_names=new_parameter_names)

class TemplateHypothesis(LinearHypothesis):
    """Convenience class to turn truth templates into a CompositeHypothesis.

    Parameters
    ----------

    templates : iterable of ndarrays
        Iterable of truth vector templates.
    constant : ndarray, optional
        Constant offset to be added to the truth vector.
    parameter_limits : iterable of tuple of floats, optional
        An iterable of lower and upper limits of the hypothesis' parameters.
        Defaults to non-negative parameter values.
    *args, **kwargs : optional
        Other arguments are passed to the `LinearHypothesis` init method.

    See also
    --------

    LinearHypothesis

    """

    def __init__(self, templates, constant=None, parameter_limits=None, *args, **kwargs):
        M = np.array(templates, dtype=float).T
        if parameter_limits is None:
            parameter_limits = [(0,None)]*M.shape[-1]

        LinearHypothesis.__init__(self, M, constant, parameter_limits, *args, **kwargs)

class JeffreysPrior(object):
    """Universal non-informative prior for use in Bayesian MCMC analysis.

    Parameters
    ----------

    response_matrix : ndarray
        Response matrix that translates truth into reco bins.
        Can be an array of matrices.

    translation_function : function
        The function to translate a vector of parameters into a vector of truth
        expectation values::

            truth_vector = translation_function(parameter_vector)

        It must support translating arrays of parameter vectors into arrays of
        truth vectors::

            [truth_vector, ...] = translation_function([parameter_vector, ...])

    parameter_limits : iterable of tuple of floats
        An iterable of lower and upper limits of the hypothesis' parameters.
        The number of limits determines the number of parameters. Parameters
        can be `None`. This sets no limit in that direction.
        ::

            [ (x1_min, x1_max), (x2_min, x2_max), ... ]

    default_values : iterable of floats
        The default values of the parameters

    dx : array like
        Array of step sizes to be used in numerical differentiation.
        Default: ``numpy.full(len(parameter_limits), 1e-3)``

    total_truth_limit : float
        Maximum total number of truth events to consider in the prior. This can
        be used to make priors proper in a consistent way, since the limit is
        defined in the truth space, rather than the prior parameter space.

    Notes
    -----

    The `JeffreysPrior` object can be called like a function. It will return
    the prior log-likliehood of the given set of parameters::

        prior_likelihood = jeffreys_prior(parameter_vector)

    If the prior was constructed with more than one response matrix,
    the matrix to be used for the calculation can be chosen with the
    `toy_index` argument::

        jeffreys_prior(parameter_vector, toy_index=5)

    By construction, the JeffreysPrior will return the log probability
    `-inf`, a probability of 0, when the expected *reco* values do not
    depend on one of the parameters. In this case the "useless" parameter
    should be removed. It simply cannot be constrained with the given
    detector response.

    """

    def __init__(self, response_matrix, translation_function, parameter_limits, default_values, dx=None, total_truth_limit=None):
        old_shape = response_matrix.shape
        new_shape = (int(np.prod(old_shape[:-2])), old_shape[-2], old_shape[-1])
        self.response_matrix = response_matrix.reshape(new_shape)

        self.translate = translation_function

        limits = list(zip(*parameter_limits))
        self.lower_limits = np.array([ x if x is not None else -np.inf for x in limits[0] ])
        self.upper_limits = np.array([ x if x is not None else np.inf for x in limits[1] ])

        self.total_truth_limit = total_truth_limit or np.inf

        self.default_values = np.array(default_values)

        self._npar = len(parameter_limits)
        self._nreco = response_matrix.shape[-2]
        self._i_diag = np.diag_indices(self._npar)
        self.dx = dx or np.full(self._npar, 1e-3)

    def fisher_matrix(self, parameters, toy_index=0):
        """Calculate the Fisher information matrix for the given parameters.

        Parameters
        ----------

        parameters : aray like
            The parameters of the translation function.
        toy_index : int, optional
            The index of the response matrix to be used for the calculation

        Returns
        -------

        ndarray
            The Fisher information matrix.

        """

        resp = self.response_matrix[toy_index]
        npar = self._npar
        nreco = self._nreco
        par_mat = np.array(np.broadcast_to(parameters, (npar, npar)))
        i_diag = self._i_diag

        # list of parameter sets -> list of reco values
        def reco_expectation(theta):
            ret = np.tensordot(self.translate(theta), resp, [[-1], [-1]])
            return ret

        # npar by npar matrix of reco values
        expect = np.broadcast_to(reco_expectation(par_mat), (npar, npar, nreco))

        # Diff expects the last axis to be the parameters, not the reco values!
        # varied parameter set -> transposed list of reco values
        def curried(x):
            par_mat[i_diag] = x
            return reco_expectation(par_mat).T

        diff = derivative(curried, x0=parameters, dx=self.dx).T

        diff_i = np.broadcast_to(diff, (npar, npar, nreco))
        diff_j = np.swapaxes(diff_i, 0, 1)

        # Nansum to ignore 0. * 0. / 0.
        # Equivalent to ignoring those reco bins
        fish = np.nansum(diff_i * diff_j / expect, axis=-1)
        return fish

    def __call__(self, value, toy_index=0):
        """Calculate the prior probability of the given parameter set."""

        # Out of bounds?
        if np.any(value < self.lower_limits) or np.any(value > self.upper_limits):
            return -np.inf

        # Out of total truth bound?
        if np.sum(self.translate(value)) > self.total_truth_limit:
            return -np.inf

        fish = self.fisher_matrix(value, toy_index)
        with np.errstate(under='ignore'):
            sign, log_det = np.linalg.slogdet(fish)

        return 0.5*log_det

class LikelihoodMachine(object):
    """Class that calculates likelihoods for truth vectors.

    Parameters
    ----------

    data_vector : array like
        The vector with the actual data
    response_matrix : array like
        One or more response matrices as ndarrays
    truth_limits : array like, optional
        The upper limits of the truth bins up to which the response matrix
        stays valid.
    limit_method : {'raise', 'prohibit'}, optional
        How to handle truth values outside their limits.
    eff_threshold : float, optional
        Only consider truth bins with an efficienct above this threshold.
    eff_indices : list of ints, optional
        Use only these truth bins for likelihood calculations.
    is_sparse : bool, optional
        Assume that the response matrix is already a sparse matrix with only
        the `eff_indices` being present.

    Notes
    -----

    The optional `truth_limits` tells the LikelihoodMachine up to which
    truth bin value the response matrix stays valid. If the machine is
    asked to calculate the likelihood of an out-of-bounds truth vector, it
    is handled according to `limit_method`:

    'raise' (default)
        An exception is raised.
    'prohibit'
        A likelihood of 0 (log likelihood of `-inf`)is returned.

    This can be used to constrain the testable theories to events that have
    been simulated enough times in the detector Monte Carlo data. I.e. if
    one wants demands a 10x higher MC statistic::

        truth_limits = generator_truth_vector / 10.

    The `eff_threshold` determines above which total reconstruction
    efficiency a truth bin counts as efficient. Is the total efficiency
    equal to or below the threshold, the bin is ignored in all likelihood
    calculations.

    Alternatively, a list of `eff_indices` can be provided. Only the
    specified truth bins are used for likelihood calculations in that case.
    If the flag `is_sparse` is set to `True`, the provided
    `response_matrix` is *not* sliced according to the `eff_indices`.
    Instead it is assumed that the given matrix already only contains the
    columns as indicated by the `eff_indices` array, i.e. it must fulfill
    the following condition::

        response_matrix.shape[-1] == len(eff_indixes)

    If the matrix is sparse, the vector of truth limits *must* be provided.
    Its length must be that of the non-sparse response matrix, i.e. the
    number of truth bins irrespective of efficient indices.

    """

    def __init__(self, data_vector, response_matrix, truth_limits=None, limit_method='raise', eff_threshold=0., eff_indices=None, is_sparse=False):
        self.data_vector = np.array(data_vector)
        self.response_matrix = np.array(response_matrix)
        if self.response_matrix.ndim > 3:
            # Multiple matrices could come arranged as a n-dimensional array.
            # In principle this should not be a problem, but some places in the
            # code seem to assume at most a 1D list of matrices. It is simplest
            # to ensure that shape here. Might revisit this decision if a
            # n-dimensional array of matrices ever actually becomes
            # necessary/useful.
            self.response_matrix.shape = (np.prod(self.response_matrix.shape[:-2]),) + self.response_matrix.shape[-2:]
        if truth_limits is None:
            self.truth_limits = np.full(self.response_matrix.shape[-1], np.inf)
        else:
            self.truth_limits = np.array(truth_limits)
        self.limit_method = limit_method

        # Calculate the reduced response matrix for speedier calculations
        if eff_indices is None:
            self._reduced_response_matrix, self._i_eff = LikelihoodMachine._reduce_response_matrix(self.response_matrix, threshold=eff_threshold)
            self._n_eff = np.size(self._i_eff)
        else:
            self._i_eff = np.array(eff_indices)
            self._n_eff = np.size(self._i_eff)
            if is_sparse:
                if truth_limits is None:
                    raise ValueError("Must provide truth limits for sparse arrays.")
                self._reduced_response_matrix = self.response_matrix
            else:
                self._reduced_response_matrix = np.array(self.response_matrix[...,self._i_eff])

    @staticmethod
    def _reduce_response_matrix(response_matrix, threshold=0.):
        """Calculate a reduced response matrix, eliminating columns with 0. efficiency.

        Parameters
        ----------

        response_matrix : ndarray
        threshold : float, optional

        Returns
        -------

        reduced_response_matrix : view of ndarray
            A view of the matrix with reduced number of columns.
        efficiency_vector : ndarray
            A vector of boolean values, describing which columns were kept.

        Notes
        -----

        How to use the reduced reposne matrix:

            reco = reduced_response_matrix.dot(truth_vector[efficiency_vector])

        """
        # Only deal with truth bins that have a efficiency > 0 in any of the response matrices
        eff = np.sum(response_matrix, axis=-2)
        if eff.ndim > 1:
            eff = np.max(eff, axis=tuple(range(eff.ndim-1)))
        eff = np.argwhere( eff > threshold ).flatten()

        reduced_response_matrix = response_matrix[...,eff]

        return reduced_response_matrix, eff

    @staticmethod
    def _create_vector_array(vector, shape, append=True):
        """Create an array containing multiple copies of a vector.

        Notes
        -----

        The resulting shape depends on the `append` parameter.
        ::

            vector.shape = (a,b,...)
            shape = (c,d,...)

        If `append` is `True`::

            ret.shape = (c,d,...,a,b,...)

        If `append` is `False`::

            ret.shape = (a,b,...,c,d,...)

        """

        if append:
            arr = np.broadcast_to(vector, list(shape) + list(vector.shape))
        else:
            arr = np.broadcast_to(vector.T, list(shape[::-1]) + list(vector.shape)[::-1]).T

        return arr

    @staticmethod
    def _translate(response_matrix, truth_vector):
        """Use the response matrix to translate the truth values into reco values."""

        # We need to set the terms of the einstein sum according to the number of axes.
        # N-dimensional case: 'a...dkl,e...fl->a...de...fk'
        ax_resp = response_matrix.ndim
        if ax_resp == 2:
            ein_resp = ''
        elif ax_resp == 3:
            ein_resp = 'd'
        elif ax_resp == 4:
            ein_resp = 'ad'
        elif ax_resp > 4:
            ein_resp = 'a...d'
        ax_truth = truth_vector.ndim
        if ax_truth == 1:
            ein_truth = ''
        elif ax_truth == 2:
            ein_truth = 'f'
        elif ax_truth == 3:
            ein_truth = 'ef'
        elif ax_truth > 3:
            ein_truth = 'e...f'
        reco = np.einsum(ein_resp+'kl,'+ein_truth+'l->'+ein_resp+ein_truth+'k', response_matrix, truth_vector)

        return reco

    class _LazyLogProbabilityCalculator(object):
        """Class for lazy probability computations.

        Assumes that data and response matrix do not change to avoid re-calculating things.
        ::

            Poisson PMF:
            p(k, mu) = (mu^k / k!) * exp(-mu)
            ln(p(k, mu)) = k*ln(mu) - mu - ln(k!)
            ln(k!) = -ln(p(k,mu)) + k*ln(mu) - mu = -ln(p(k,1.)) - 1.

        """

        def __init__(self, data_vector, response_matrix, _constant=None):
            self.data_shape = data_vector.shape
            self.response_shape = response_matrix.shape
            self._constant = _constant

            # Extend response_matrix to shape (a,b,...,c,d,...,n_data,n_truth)
            self.resp = LikelihoodMachine._create_vector_array(response_matrix, self.data_shape[:-1])

            # Save constants for PMF calculation
            with np.errstate(divide='ignore', invalid='ignore'):
                self.k = data_vector
                self.k0 = (data_vector == 0)
                self.ln_k_factorial = -(stats.poisson.logpmf(self.k, np.ones_like(self.k, dtype=float)) + 1.)

        @staticmethod
        def _poisson_logpmf(k, k0, ln_k_factorial, mu):
            with np.errstate(divide='ignore', invalid='ignore'):
                pmf = k*np.log(mu) - ln_k_factorial - mu
            # Need to take care of special case mu=0:  k=0 -> logpmf=0,  k>=1 -> logpmf=-inf
            mu0 = (mu==0)
            pmf[mu0 & k0] = 0
            pmf[mu0 & ~k0] = -np.inf
            pmf[~np.isfinite(pmf)] = -np.inf
            pmf = np.sum(pmf, axis=-1)
            return pmf

        def __call__(self, truth_vector):
            truth_shape = truth_vector.shape

            # Reco expectation values of shape (a,b,...,c,d,...,e,f,...,n_data)
            if self._constant is None:
                reco = LikelihoodMachine._translate(self.resp, truth_vector)
            else:
                reco = LikelihoodMachine._translate(self.resp, truth_vector) + self._constant

            # Create a data vector of the shape (a,b,...,n_data,c,d,...,e,f,...)
            shape = list(self.response_shape[:-2])+list(truth_shape[:-1])
            k = LikelihoodMachine._create_vector_array(self.k, shape, append=False)
            k0 = LikelihoodMachine._create_vector_array(self.k0, shape, append=False)
            ln_k_factorial = LikelihoodMachine._create_vector_array(
                self.ln_k_factorial, shape, append=False)

            # Move axis so we get (a,b,...,c,d,...,e,f,...,n_data)
            shape = len(self.data_shape)-1
            k              = np.moveaxis(k,              shape, -1)
            k0             = np.moveaxis(k0,             shape, -1)
            ln_k_factorial = np.moveaxis(ln_k_factorial, shape, -1)

            # Calculate the log probabilities and sum over the axis `n_data`.
            return self._poisson_logpmf(k, k0, ln_k_factorial, reco)

    @staticmethod
    def log_probability(data_vector, response_matrix, truth_vector, _constant=None):
        """Calculate the log probabilty of some data given a response matrix and truth vector.

        Parameters
        ----------

        data_vector : array like
            The measured data.
        response_matrix : array like
            The detector response matrix as ndarray.
        truth_vector : array like
            The vector of truth expectation values.

        Returns
        -------

        p : ndarray
            The log likelihood of `data_vector` given the `response_matrix`
            and `truth_vector`.

        Notes
        -----

        Each of the three objects can actually be an array of vectors/matrices::

            data_vector.shape = (a,b,...,n_data)
            response_matrix.shape = (c,d,...,n_data,n_truth)
            truth_vector.shape = (e,f,...,n_truth)

        In this case, the return value will have the following shape::

            p.shape = (a,b,...,c,d,...,e,f,...)

        """

        return LikelihoodMachine._LazyLogProbabilityCalculator(data_vector, response_matrix, _constant)(truth_vector)

    @staticmethod
    def _collapse_systematics_axes(ll, systaxis, systematics):
        """Collapse the given axes according to the systematics mode."""

        if type(systematics) is tuple:
            # Return specific result
            index = tuple([ slice(None) ] * min(systaxis) + list(systematics) + [Ellipsis])
            ll = ll[index]
        elif isinstance(systematics, np.ndarray):
            # Return specific result for each non-systematics index
            # The shape of `systematics` must match the shape of the non-systemtic axes:
            #
            #     systematics.shape = (a,b,c,...,len(systaxis))
            oi = np.indices(systematics.shape[:-1])
            index = tuple([ i for i in oi[:min(systaxis)] ] + [ systematics[...,i] for i in range(len(systaxis)) ] + [ i for i in oi[min(systaxis):] ])
            ll = ll[index]
        elif systematics == 'profile' or systematics == 'maximum':
            # Return maximum
            ll = np.max(ll, axis=systaxis)
        elif systematics == 'marginal' or systematics == 'average':
            # Return average
            N = np.prod(np.array(ll.shape)[np.array(systaxis, dtype=int)])
            ll = np.logaddexp.reduce(ll, axis=systaxis) - np.log(N)
        else:
            raise ValueError("Unknown systematics method!")

        return ll

    def _reduced_log_likelihood(self, reduced_truth_vector, systematics='marginal'):
        """Calculate a more efficient log likelihood using only truth values that have an influence."""
        ll = LikelihoodMachine.log_probability(self.data_vector, self._reduced_response_matrix, reduced_truth_vector)

        # Deal with systematics, i.e. multiple response matrices
        systaxis = tuple(range(self.data_vector.ndim-1, self.data_vector.ndim-1+self._reduced_response_matrix.ndim-2))
        if systematics is not None and len(systaxis) > 0:
            ll = LikelihoodMachine._collapse_systematics_axes(ll, systaxis, systematics)

        return ll

    def _reduce_truth_vector(self, truth_vector):
        """Return a reduced truth vector view."""
        return np.array(np.asarray(truth_vector)[...,self._i_eff])

    def _reduce_matrix(self, truth_vector):
        """Return a reduced matrix view."""
        return np.array(np.asarray(truth_vector)[...,self._i_eff,:])

    def log_likelihood(self, truth_vector, systematics='marginal'):
        """Calculate the log likelihood of a vector of truth expectation values.

        Parameters
        ----------

        truth_vector : array like
            Array of truth expectation values. Can be a multidimensional array
            of truth vectors. The shape of the array must be `(a, b, c, ...,
            n_truth_values)`.

        systematics : {'profile', 'marginal'} or tuple or ndarray or None
            How to deal with detector systematics, i.e. multiple response
            matrices:

            'profile', 'maximum'
                Choose the response matrix that yields the highest probability.
            'marginal', 'average'
                Take the arithmetic average probability yielded by the
                matrices.
            `tuple(index)`
                Select one specific matrix.
            `array(indices)`
                Select a specific matrix for each truth vector. Must have the
                shape ``(a, b, c, ..., len(index))``.
            `None`
                Do nothing, return multiple likelihoods.

        """

        if np.any(truth_vector > self.truth_limits):
            if self.limit_method == 'raise':
                i = np.argwhere(truth_vector > self.truth_limits)[0,-1]
                raise RuntimeError("Truth value %d is above allowed limits!"%(i,))
            elif self.limit_method == 'prohibit':
                return -np.inf
            else:
                raise ValueError("Unknown limit method: '%s'"%(self.limit_method))

        # Use reduced truth values for efficient calculations.
        reduced_truth_vector = self._reduce_truth_vector(truth_vector)

        ll = self._reduced_log_likelihood(reduced_truth_vector, systematics=systematics)

        return ll

    @staticmethod
    def max_log_probability(data_vector, response_matrix, composite_hypothesis, systematics='marginal', disp=False, method='basinhopping', kwargs={}):
        """Calculate the maximum possible probability of the data in a CompositeHypothesis.

        Parameters
        ----------

        data_vector : array like
            Vector of measured values.

        response_matrix : array like
            The response matrix that translates truth into reco space. Can be
            an arbitrarily shaped array of response matrices.

        composite_hypothesis : CompositeHypothesis
            The hypothesis to be evaluated.

        systematics : {'profile', 'marginal'}, optional
            How to deal with detector systematics, i.e. multiple response
            matrices:

            'profile', 'maximum'
                Choose the response matrix that yields the highest probability.
            'marginal', 'average'
                Take the arithmetic average probability yielded by the matrices.

        disp : bool, optional
            Display status messages during optimization.

        method : {'differential_evolution', 'basinhopping'}, optional
            Select the method to be used for maximization.

        kwargs : dict, optional
            Keyword arguments to be passed to the minimizer.
            If empty, reasonable default values will be used.

        Returns
        -------

        res : OptimizeResult
            Object containing the maximum log probability ``res.P``.
            In case of ``systematics=='profile'``, it also contains the index
            of the response matrix that yielded the maximum likelihood
            ``res.i``.

        """
        if isinstance(composite_hypothesis, LinearHypothesis):
            # Special case!
            # Since the parameter translation is just a matrix multiplication,
            # we can save a lot of computing time by pre-calculating the combined
            # matrix.
            R = np.tensordot(response_matrix, composite_hypothesis.M, axes=(-1,-2))
            b = composite_hypothesis.b
            if b is None:
                likfun = LikelihoodMachine._LazyLogProbabilityCalculator(data_vector, R)
            else:
                const = LikelihoodMachine._translate(response_matrix, b)
                likfun = LikelihoodMachine._LazyLogProbabilityCalculator(data_vector, R, _constant=const)
        else:
            probfun = LikelihoodMachine._LazyLogProbabilityCalculator(data_vector, response_matrix)
            likfun = lambda x : probfun(composite_hypothesis.translate(x))

        # Negative log probability function
        if systematics == 'profile' or systematics == 'maximum':
            nll = lambda x: -np.max(likfun(x))
        elif systematics == 'marginal' or systematics == 'average':
            N_resp = np.prod(response_matrix.shape[:-2])
            nll = lambda x: -(np.logaddexp.reduce(likfun(x)) - np.log(N_resp))
        else:
            raise ValueError("Unknown systematics method!")

        if len(composite_hypothesis.parameter_limits) == 0:
            # Special case!
            # We seem to be dealing with a degenerate CompositeHypothesis with no free parameters.
            # Just return a dummy optimisation result.
            res = optimize.OptimizeResult()
            res.x = np.ndarray(0)
            res.fun = nll(res.x)
            res.P = -res.fun
            if systematics == 'profile' or systematics == 'maximum':
                res.i = np.argmax(LikelihoodMachine.log_probability(data_vector, response_matrix, composite_hypothesis.translate(res.x)))
            return res

        # Parameter limits
        bounds = composite_hypothesis.parameter_limits
        def limit(bounds):
            l=[0.,0.]
            if bounds[0] is None:
                l[0] = -np.inf
            else:
                l[0] = bounds[0]
            if bounds[1] is None:
                l[1] = np.inf
            else:
                l[1] = bounds[1]

            return l

        limits = np.array(list(map(limit, bounds)), dtype=float).transpose()
        ranges = limits[1]-limits[0]

        if method == 'differential_evolution':
            kw = {}
            kw.update(kwargs)

            res = optimize.differential_evolution(nll, bounds, disp=disp, **kw)
        elif method == 'basinhopping':
            # Start values
            def start_value(limits):
                if None in limits:
                    if limits[0] is not None:
                        # No upper limit
                        # Start at lower limit
                        return limits[0]
                    elif limits[1] is not None:
                        # No lower limit
                        # Start at upper limit
                        return limits[1]
                    else:
                        # Neither lower nor upper limit
                        # Start at 0.
                        return 0.
                else:
                    # Start at halfway point between limits
                    return (limits[1]+limits[0]) / 2.

            x0 = np.array(list(map(start_value, bounds)))
            if 'x0' in kwargs:
                if len(kwargs['x0']) == len(bounds):
                    x0 = np.array(kwargs.pop('x0'))
                else:
                    warn("Length of `x0` does not correspond to number of parameters!", stacklevel=2)
                    kwargs.pop('x0')

            # Step length for basin hopping
            def step_value(limits):
                if None in limits:
                    return 1.
                else:
                    # Step size in the order of the parameter range
                    return (limits[1]-limits[0]) / 10.
            step = np.array(list(map(step_value, bounds)))

            # Number of parameters
            n = len(bounds)

            # Define a step function that does *not* produce illegal parameter values
            def step_fun(x):
                # Vary parameters according to their distance from the expectation,
                # but at least by a minimum amount.
                dx = np.random.randn(n) * np.maximum(np.abs(x - x0) / 2., step)

                # Make sure the new values are within bounds
                ret = x + dx
                with np.errstate(invalid='ignore'):
                    ret = np.where(ret > limits[1], limits[1]-((ret-limits[1]) % ranges), ret)
                    ret = np.where(ret < limits[0], limits[0]+((limits[0]-ret) % ranges), ret)

                return ret

            # Minimizer options
            kw = {
                'take_step': step_fun,
                'T': n,
                'niter': 100,
                'minimizer_kwargs': {
                    'method': 'L-BFGS-B',
                    'bounds': bounds,
                    'options': {
                        #'disp': True,
                        'ftol' : 1e-12,
                    }
                }
            }
            kw.update(kwargs)
            with np.errstate(invalid='ignore'):
                res = optimize.basinhopping(nll, x0, disp=disp, **kw)
        else:
            raise ValueError("Unknown method: %s"%(method,))

        res.P = -res.fun
        if systematics == 'profile' or systematics == 'maximum':
            res.i = np.argmax(LikelihoodMachine.log_probability(data_vector, response_matrix, composite_hypothesis.translate(res.x)))

        return res

    def _composite_hypothesis_wrapper(self, composite_hypothesis):
        """Return a new composite hypothesis, that translates to reduced truth vectors."""
        if isinstance(composite_hypothesis, LinearHypothesis):
            # Special case! Save computing time by removing the unneeded rows.
            M = self._reduce_matrix(composite_hypothesis.M)
            if composite_hypothesis.b is None:
                b = None
            else:
                b = self._reduce_truth_vector(composite_hypothesis.b)
            H0 = LinearHypothesis(M, b,
                                    parameter_limits=composite_hypothesis.parameter_limits,
                                    parameter_priors=composite_hypothesis.parameter_priors,
                                    parameter_names=composite_hypothesis.parameter_names)
        else:
            fun = lambda x: self._reduce_truth_vector(composite_hypothesis.translate(x))
            H0 = CompositeHypothesis(translation_function=fun,
                                    parameter_limits=composite_hypothesis.parameter_limits,
                                    parameter_priors=composite_hypothesis.parameter_priors,
                                    parameter_names=composite_hypothesis.parameter_names)
        return H0

    def max_log_likelihood(self, composite_hypothesis, *args, **kwargs):
        """Calculate the maximum possible likelihood in the given CompositeHypothesis.

        Parameters
        ----------

        composite_hypothesis : CompositeHypothesis
            The hypothesis to be evaluated.

        systematics : {'profile', 'marginal'}, optional
            How to deal with detector systematics, i.e. multiple response
            matrices:

            'profile', 'maximum'
                Choose the response matrix that yields the highest probability.
            'marginal', 'average'
                Take the arithmetic average probability yielded by the matrices.

        disp : bool, optional
            Display status messages during optimization.

        method : {'differential_evolution', 'basinhopping'}, optional
            Select the method to be used for maximization.

        kwargs : dict, optional
            Keyword arguments to be passed to the minimizer.
            If empty, reasonable default values will be used.

        Returns
        -------

        res : OptimizeResult
            Object containing the maximum log likelihood ``res.L``.
            In case of ``systematics=='profile'``, it also contains the index
            of the response matrix that yielded the maximum likelihood
            ``res.i``.

        """

        resp = self._reduced_response_matrix
        # Wrapping composite hypothesis to produce reduced truth vectors
        H0 = self._composite_hypothesis_wrapper(composite_hypothesis)
        ret = LikelihoodMachine.max_log_probability(self.data_vector, resp, H0, *args, **kwargs)
        ret.L = ret.P
        del ret.P
        return ret

    @staticmethod
    def generate_random_data_sample(response_matrix, truth_vector, size=None, each=False):
        """Generate random data samples from the provided truth_vector.

        Parameters
        ----------

        response_matrix : array like
            The matrix that translates the truth vector to reco-space
            expecation values.
        truth_vector : array like
            The truth-space expectation values used to generate the data.
        size : int or tuple of ints, optional
            The number of data vectors to be generated.
        each : bool, optional
            Generate `size` vectors for each response matrix.
            Otherwise `size` determines the total number of generated data
            vectors and the response matrices are chosen randomly.

        """

        mu = response_matrix.dot(truth_vector)

        if each:
            # One set per matrix
            if size is not None:
                # Append truth vector shape to requested shape of data sets
                try:
                    shape = list(size)
                except TypeError:
                    shape = [size]
                shape.extend(mu.shape)
                size = shape

            return np.random.poisson(mu, size=size)
        else:
            # Randomly choose matrices
            # Flatten expectation values
            if mu.ndim > 1:
                mu.shape = (np.prod(mu.shape[:-1]), mu.shape[-1])
            else:
                mu.shape = (1, mu.shape[-1])

            if size is not None:
                # Append truth vector shape to requested shape of data sets
                try:
                    shape = list(size)
                except TypeError:
                    shape = [size]
                i = np.random.randint(mu.shape[0], size=shape)
                mu = mu[i,...]
            else:
                i = np.random.randint(mu.shape[0])
                mu = mu[i,...]

            return np.random.poisson(mu)

    def likelihood_p_value(self, truth_vector, N=2500, generator_matrix_index=None, systematics='marginal', **kwargs):
        """Calculate the likelihood p-value of a truth vector.

        The likelihood p-value is the probability of the data yielding a lower
        likelihood than the actual data, given that the simple hypothesis of
        `truth_vector` is true.

        Parameters
        ----------

        truth_vector : array like
            The evaluated hypotheis expressed as a vector of truth expectation
            values.

        N : int, optional
            The number of MC evaluations of the hypothesis.

        generator_matrix_index : int or tuple, optional
            The index of the response matrix to be used to generate the fake
            data. This needs to be specified only if the LikelihoodMachine
            contains more than one response matrix. If it is `None`, N data
            sets are thrown for *each* matrix, and a p-value is evaluated for
            all of them. The return value thus becomes an array of p-values.

        systematics : {'profile', 'marginal'}, optional
            How to deal with detector systematics, i.e. multiple response
            matrices:

            'profile', 'maximum'
                Choose the response matrix that yields the highest likelihood.
            'marginal', 'average'
                Take the arithmetic average probability yielded by the matrices.

        **kwargs : optional
            Additional keyword arguments will be passed to :meth:`log_likelihood`.

        Returns
        -------

        p : float or ndarray
            The likelihood p-value.

        Notes
        -----

        The p-value is estimated by randomly creating `N` data samples
        according to the given `truth_vector`. The number of data-sets that
        yield a likelihood as bad as, or worse than the likelihood given the
        actual data, `n`, are counted. The estimate for `p` is then::

            p = n/N.

        The variance of the estimator follows that of binomial statistics::

                     var(n)   Np(1-p)      1
            var(p) = ------ = ------- <= ---- .
                      N^2       N^2       4N

        The expected uncertainty can thus be directly influenced by choosing an
        appropriate number of evaluations.

        See also
        --------

        max_likelihood_p_value
        max_likelihood_ratio_p_value
        mcmc

        """
        # Reduce truth vectors to efficient values
        reduced_truth_vector = self._reduce_truth_vector(truth_vector)

        # Get likelihood of actual data
        p0 = self._reduced_log_likelihood(reduced_truth_vector, systematics=systematics, **kwargs)

        # Decide which matrix to use for data generation
        if self._reduced_response_matrix.ndim > 2 and generator_matrix_index is not None:
            resp = self._reduced_response_matrix[generator_matrix_index]
        else:
            resp = self._reduced_response_matrix

        # Draw N fake data distributions
        fake_data = LikelihoodMachine.generate_random_data_sample(resp, reduced_truth_vector, N)
        # shape = N, resp_shape-2, data_shape

        # Calculate probabilities of each generated sample
        prob = LikelihoodMachine.log_probability(fake_data, self._reduced_response_matrix, reduced_truth_vector, **kwargs)
        # shape = N, resp_shape-2, resp_shape-2

        # Deal with systematics, i.e. multiple response matrices
        if prob.ndim > (1+resp.ndim-2):
            systaxis = tuple(range(1+resp.ndim-2, prob.ndim))
            prob = LikelihoodMachine._collapse_systematics_axes(prob, systaxis, systematics)
        # shape = N, resp_shape-2

        # Count number of probabilities lower than or equal to the likelihood of the real data
        n = np.sum(prob <= p0, axis=0, dtype=float)

        # Return the quotient
        return n / N

    def max_likelihood_p_value(self, composite_hypothesis, parameters=None, N=250, generator_matrix_index=None, systematics='marginal', nproc=0, **kwargs):
        """Calculate the maximum-likelihood p-value of a composite hypothesis.

        The maximum-likelihood p-value is the probability of the data yielding
        a lower maximum likelihood (over the possible parameter space of the
        composite hypothesis) than the actual data, given that the best-fit
        parameter set of the composite hypothesis is true.

        Parameters
        ----------

        composite_hypothesis : CompositeHypothesis
            The evaluated composite hypothesis.

        parameters : array like, optional
            The assumed true parameters of the composite hypothesis. If no
            parameters are given, they will be determined with the maximum
            likelihood method.

        N : int, optional
            The number of MC evaluations of the hypothesis.

        generator_matrix_index : int or tuple, optional
            The index of the response matrix to be used to generate the fake
            data. This needs to be specified only if the LikelihoodMachine
            contains more than one response matrix. If it is `None`, N data
            sets are thrown for *each* matrix, and a p-value is evaluated for
            all of them. The return value thus becomes an array of p-values.

        systematics : {'profile', 'marginal'}, optional
            How to deal with detector systematics, i.e. multiple response
            matrices:

            'profile', 'maximum'
                Choose the response matrix that yields the highest likelihood.
            'marginal', 'average'
                Take the arithmetic average probability yielded by the matrices.

        nproc : int, optional
            If this parameters is >= 1, the according number of processes are
            spawned to calculate the p-value in parallel.

        **kwargs : optional
            Additional keyword arguments will be passed to :meth:`max_log_likelihood`.

        Returns
        -------

        p : float or ndarray
            The maximum-likelihood p-value.

        Notes
        -----

        When used to reject composite hypotheses, this p-value is somtime
        called the "profile plug-in p-value", as one "plugs in" the maximum
        likelihood estimate of the hypothesis parameters to calculate it. It's
        coverage properties are not exact, so care has to be taken to make sure
        it performs as expected (e.g. by testing it with simulated data)..

        The p-value is estimated by randomly creating `N` data samples
        according to the given `truth_vector`. The number of data-sets that
        yield a likelihood ratio as bad as, or worse than the likelihood given
        the actual data, `n`, are counted. The estimate for `p` is then::

            p = n/N.

        The variance of the estimator follows that of binomial statistics::

                     var(n)   Np(1-p)      1
            var(p) = ------ = ------- <= ---- .
                      N^2       N^2       4N

        The expected uncertainty can thus be directly influenced by choosing an
        appropriate number of evaluations.

        See also
        --------

        likelihood_p_value
        max_likelihood_ratio_p_value
        mcmc

        """

        # Get truth vector from assumed true hypothesis
        if parameters is None:
            parameters = self.max_log_likelihood(composite_hypothesis, **kwargs).x

        truth_vector = self._reduce_truth_vector(composite_hypothesis.translate(parameters))

        # Decide which matrix to use for data generation
        if self._reduced_response_matrix.ndim > 2 and generator_matrix_index is not None:
            resp = self._reduced_response_matrix[generator_matrix_index]
        else:
            resp = self._reduced_response_matrix

        # Draw N fake data distributioxxns
        fake_data = LikelihoodMachine.generate_random_data_sample(resp, truth_vector, N)
        # shape = N, resp_shape-2, data_shape
        # Flatten the fake data sets from possibly multiple response matrices
        fake_shape = fake_data.shape
        fake_data.shape = (np.prod(fake_data.shape[:-1]), fake_data.shape[-1])
        # shape = N * resp_shape-2, data_shape

        # Wrapping composite hypothesis to produce reduced truth vectors
        H0 = self._composite_hypothesis_wrapper(composite_hypothesis)

        # Calculate the maximum probabilities
        def prob_fun(data):
            try:
                prob = LikelihoodMachine.max_log_probability(data, self._reduced_response_matrix, H0, systematics=systematics, **kwargs).P
            except KeyboardInterrupt:
                raise Exception("Terminated.")
            return prob

        if nproc >= 1:
            # Load multiprocess on demand
            global Pool
            if Pool is None:
                from multiprocess import Pool as _Pool
                Pool = _Pool
            p = Pool(nproc)
            prob = np.fromiter(p.map(prob_fun, fake_data), dtype=float)
            p.terminate()
            p.join()
            del p
        else:
            prob = np.fromiter(map(prob_fun, fake_data), dtype=float)
        # shape = N*resp_shape-2
        prob.shape = fake_shape[0:-1] + prob.shape[1:]
        # shape = N, resp_shape-2

        # Get likelihood of actual data
        p0 = self._reduced_log_likelihood(truth_vector, systematics=systematics)

        # Count number of probabilities lower than or equal to the likelihood of the real data
        n = np.sum(prob <= p0, axis=0, dtype=float)

        # Return the quotient
        return n / N

    def max_likelihood_ratio_p_value(self, H0, H1, par0=None, par1=None, N=250, generator_matrix_index=None, systematics='marginal', nproc=0, nested=True, **kwargs):
        """Calculate the maximum-likelihood-ratio p-value of two composite hypotheses.

        The maximum-likelihood-ratio p-value is the probability of the data
        yielding a lower ratio of maximum likelihoods (over the possible
        parameter spaces of the composite hypotheses) than the actual data,
        given that the best-fit parameter set of hypothesis `H0` is true.

        Parameters
        ----------

        H0 : CompositeHypothesis
            The tested composite hypothesis. Usually a subset of `H1`.

        H1 : CompositeHypothesis
            The alternative composite hypothesis.

        par0 : array like, optional
            The assumed true parameters of the tested hypothesis.If no
            parameters are given, they will be determined with the maximum
            likelihood method.

        par1 : array like, optional
            The maximum likelihood parameters of the alternative hypothesis.
            If no parameters are given, they will be determined with the
            maximum likelihood method.

        N : int, optional
            The number of MC evaluations of the hypothesis.

        generator_matrix_index : int or tuple, optional
            The index of the response matrix to be used to generate the fake
            data. This needs to be specified only if the LikelihoodMachine
            contains more than one response matrix. If it is `None`, N data
            sets are thrown for *each* matrix, and a p-value is evaluated for
            all of them. The return value thus becomes an array of p-values.

        systematics : {'profile', 'marginal'}, optional
            How to deal with detector systematics, i.e. multiple response
            matrices:

            'profile', 'maximum'
                Choose the response matrix that yields the highest likelihood.
            'marginal', 'average'
                Take the arithmetic average probability yielded by the matrices.

        nproc : int, optional
            If this parameters is >= 1, the according number of processes are
            spawned to calculate the p-value in parallel.

        nested : bool or 'ignore', optional
            Is H0 a nested theory, i.e. does it cover a subset of H1? In this
            case, the maximum likelihoods must always follow ``L0 <= L1``. If
            `True` or `'ignore'`, the calculation of likelihood ratios is
            re-tried a couple of times if no valid likelihood ratio is found.
            If `True` and no valid value was found after 10 tries, an errors is
            raised. If `False`, those cases are just accepted.

        **kwargs : optional
            Additional keyword arguments will be passed to :meth:`max_log_likelihood`.

        Returns
        -------

        p : float or ndarray
            The maximum-likelihood-ratio p-value.

        Notes
        -----

        When used to reject composite hypotheses, this p-value is somtime
        called the "profile plug-in p-value", as one "plugs in" the maximum
        likelihood estimate of the hypothesis parameters to calculate it. It's
        coverage properties are not exact, so care has to be taken to make sure
        it performs as expected (e.g. by testing it with simulated data)..

        The p-value is estimated by randomly creating `N` data samples
        according to the given `truth_vector`. The number of data-sets that
        yield a likelihood ratio as bad as, or worse than the likelihood given
        the actual data, `n`, are counted. The estimate for `p` is then::

            p = n/N.

        The variance of the estimator follows that of binomial statistics::

                     var(n)   Np(1-p)      1
            var(p) = ------ = ------- <= ---- .
                      N^2       N^2       4N

        The expected uncertainty can thus be directly influenced by choosing an
        appropriate number of evaluations.

        See also
        --------

        likelihood_p_value
        max_likelihood_p_value
        mcmc

        """
        # Get truth vector from assumed true hypothesis
        if par0 is None:
            par0 = self.max_log_likelihood(H0, systematics=systematics, **kwargs).x
        if par1 is None:
            par1 = self.max_log_likelihood(H1, systematics=systematics, **kwargs).x

        truth_vector = self._reduce_truth_vector(H0.translate(par0))
        alternative_truth = self._reduce_truth_vector(H1.translate(par1))

        # Get likelihood of actual data
        p0 = self._reduced_log_likelihood(truth_vector, systematics=systematics)
        p1 = self._reduced_log_likelihood(alternative_truth, systematics=systematics)
        r0 = p0-p1 # difference because log

        # If we assume a nested hypothesis, we should try to fix impossible likelihood ratios
        nested_tolerance = 1e-2
        if nested is True or nested == 'ignore':
            ttl = 10
            while r0 > nested_tolerance and ttl > 0:
                try:
                    p0 = np.maximum(p0, self.max_log_likelihood(H0, systematics=systematics, **kwargs).L)
                    p1 = np.maximum(p1, self.max_log_likelihood(H1, systematics=systematics, **kwargs).L)
                except KeyboardInterrupt:
                    raise Exception("Terminated.")
                r0 = p0 - p1
                ttl -= 1
        if r0 > nested_tolerance and (nested is True):
            raise RuntimeError("Could not find a valid likelihood ratio! Is H0 a subset of H1?")
        if r0 > nested_tolerance and (nested == 'ignore'):
            warn("Could not find a valid likelihood ratio! Is H0 a subset of H1?", stacklevel=2)
        if r0 > 0 and (nested is True):
            r0 = 0.

        # Decide which matrix to use for data generation
        if self._reduced_response_matrix.ndim > 2 and generator_matrix_index is not None:
            resp = self._reduced_response_matrix[generator_matrix_index]
        else:
            resp = self._reduced_response_matrix

        # Draw N fake data distributions
        fake_data = LikelihoodMachine.generate_random_data_sample(resp, truth_vector, N)
        # shape = N, resp_shape-2, data_shape
        # Flatten the fake data sets from possibly multiple response matrices
        fake_shape = fake_data.shape
        fake_data.shape = (np.prod(fake_data.shape[:-1]), fake_data.shape[-1])
        # shape = N * resp_shape-2, data_shape

        # Wrapping composite hypothesis to produce reduced truth vectors
        wH0 = self._composite_hypothesis_wrapper(H0)
        wH1 = self._composite_hypothesis_wrapper(H1)

        # Calculate the maximum probabilities
        kwargs0 = {}
        kwargs1 = {}
        if 'kwargs' in kwargs:
            fitkwargs = kwargs.pop('kwargs')
            kwargs0.update(fitkwargs)
            kwargs1.update(fitkwargs)
            if 'x1' in kwargs0:
                kwargs0.pop('x1')
            if 'x0' in kwargs1:
                kwargs1.pop('x0')
            if 'x1' in kwargs1:
                kwargs1['x0'] = kwargs1.pop('x1')

        def ratio_fun(data):
            r = 1.
            p0 = -np.inf
            p1 = -np.inf
            if nested is True or nested == 'ignore':
                ttl = 10
            else:
                ttl = 1
            # If we assume a nested hypothesis, we should try to fix impossible likelihood ratios
            while r > nested_tolerance and ttl > 0:
                try:
                    p0 = np.maximum(p0, LikelihoodMachine.max_log_probability(data, self._reduced_response_matrix, wH0, systematics=systematics, kwargs=kwargs0, **kwargs).P)
                    p1 = np.maximum(p1, LikelihoodMachine.max_log_probability(data, self._reduced_response_matrix, wH1, systematics=systematics, kwargs=kwargs1, **kwargs).P)
                except KeyboardInterrupt:
                    raise Exception("Terminated.")
                r = p0 - p1
                ttl -= 1
            if r > nested_tolerance and (nested is True):
                raise RuntimeError("Could not find a valid likelihood ratio! Is H0 a subset of H1?")
            if r > 0 and (nested is True):
                r = 0.
            return r # difference because log

        if nproc >= 1:
            # Load multiprocess on demand
            global Pool
            if Pool is None:
                from multiprocess import Pool as _Pool
                Pool = _Pool
            p = Pool(nproc)
            ratio = np.fromiter(p.map(ratio_fun, fake_data), dtype=float)
            p.terminate()
            p.join()
            del p
        else:
            ratio = np.fromiter(map(ratio_fun, fake_data), dtype=float)
        # shape = N*resp_shape-2
        ratio.shape = fake_shape[0:-1] + ratio.shape[1:]
        # shape = N, resp_shape-2

        # Count number of probabilities lower than or equal to the likelihood of the real data
        n = np.sum(ratio <= r0, axis=0, dtype=float)

        # Return the quotient
        return n / N

    def mcmc(self, composite_hypothesis, prior_only=False):
        """Return a Marcov Chain Monte Carlo object for the hypothesis.

        The hypothesis must define priors for its parameters.

        Parameters
        ----------

        composite_hypothesis : CompositeHypothesis
        prior_only : bool, optional
            Use only the prior infomation. Ignore the data. Useful for testing
            purposes.

        Notes
        -----

        See documentation of `PyMC` for a description of the `MCMC` class.

        See also
        --------

        plr
        plot_truth_bin_traces
        plot_reco_bin_traces
        likelihood_p_value
        max_likelihood_p_value
        max_likelihood_ratio_p_value

        """

        # Load pymc on demand
        global pymc
        if pymc is None:
            import pymc as _pymc
            pymc = _pymc

        priors = composite_hypothesis.parameter_priors

        names = composite_hypothesis.parameter_names
        if names is None:
            names = [ 'par_%d'%(i,) for i in range(len(priors)) ]

        # Toy index as additional stochastic
        n_toys = np.prod(self.response_matrix.shape[:-2])
        toy_index = pymc.DiscreteUniform('toy_index', lower=0, upper=(n_toys-1))

        # The parameter pymc stochastics
        parameters = []
        names_priors = list(zip(names, priors))
        for n,p in names_priors:
            # Get default value of prior
            if isinstance(p, JeffreysPrior):
                # Jeffreys prior?
                default = p.default_values
                parents = {'toy_index': toy_index}
            else:
                # Regular function
                default = inspect.getargspec(p).defaults[0]
                parents = {}

            parameters.append(pymc.Stochastic(
                logp = p,
                doc = '',
                name = n,
                parents = parents,
                value = default,
                dtype=float))

        # The data likelihood
        if prior_only:
            def logp(value=self.data_vector, parameters=parameters, toy_index=toy_index):
                """Do not consider the data likelihood."""
                return 0
        else:
            def logp(value=self.data_vector, parameters=parameters, toy_index=toy_index):
                """The reconstructed data."""
                return self.log_likelihood(composite_hypothesis.translate(parameters), systematics=(toy_index,))
        data = pymc.Stochastic(
            logp = logp,
            doc = '',
            name = 'data',
            parents = {'parameters': parameters, 'toy_index': toy_index},
            value = self.data_vector,
            dtype = int,
            observed = True)

        M = pymc.MCMC({'data': data, 'parameters': parameters, 'toy_index': toy_index})
        M.use_step_method(pymc.DiscreteMetropolis, toy_index, proposal_distribution='Prior')

        return M

    def marginal_log_likelihood(self, composite_hypothesis, parameters, toy_indices):
        """Calculate the marginal likelihood.

        Parameters
        ----------

        composite_hypothesis : CompositeHypothesis
            The composite hypotheses for which the likelihood will be calculated.

        parameters : array like
            Array of parameter vectors, drawn from the prior or posterior distribution
            of the hypothesis, e.g. with the MCMC objects::

                parameters = [
                    [1.0, 2.0, 3.0],
                    [1.1, 1.9, 2.8],
                    ...
                    ]

        toy_indices, : array_like
            The corresponding systematic toy indices, in an array of equal
            dimensionality. That means, even if the toy index is just a single
            integer, it must be provided as arrays of length 1::

                toy_indices0 = [
                    [0],
                    [3],
                    ...
                    ]

        Returns
        -------

        L : float
            The marginal log-likelihood.

        Notes
        -----

        The marginal likelihood is used in the construction of bayes factors,
        when comparing the evidence in the data for two hypotheses::

            bayes_factor = marginal_likelihood0 / marginal_likelihood1

        or in the case of log-likelihoods::

            log_bayes_factor = marginal_log_likelihood0 - marginal_log_likelihood1

        """

        L = self.log_likelihood(composite_hypothesis.translate(parameters),
            systematics=toy_indices)
        return np.logaddexp.reduce(L) - np.log(len(L))

    def plr(self, H0, parameters0, toy_indices0, H1, parameters1, toy_indices1):
        """Calculate the Posterior distribution of the log Likelihood Ratio.

        Parameters
        ----------

        H0, H1 : CompositeHypothesis
            Composite Hypotheses to be compared.

        parameters0, parameters1 : array like
            Arrays of parameter vectors, drawn from the posterior distribution
            of the hypotheses, e.g. with the MCMC objects::

                parameters0 = [
                    [1.0, 2.0, 3.0],
                    [1.1, 1.9, 2.8],
                    ...
                    ]

        toy_indices0, toy_indices1 : array_like
            The corresponding systematic toy indices, in an array of equal
            dimensionality. That means, even if the toy index is just a single
            integer, it must be provided as arrays of length 1::

                toy_indices0 = [
                    [0],
                    [3],
                    ...
                    ]

        Returns
        -------

        PLR : ndarray
            A sample from the PLR as calculated from the parameter sets.
        model_preference : float
            The resulting model preference.

        Notes
        -----

        The model preference is calculated as the fraction of likelihood ratios
        in the PLR that prefer H1 over H0::

            model_preference = N(PLR > 0) / N(PLR)

        It can be interpreted as the posterior probability for the data
        prefering H1 over H0.

        The PLR is symmetric::

            PLR(H0, H1) = -PLR(H1, H0)
            preference(H0, H1) = 1. - preference(H1, H0) # modulo the cases of PLR = 0.

        See also
        --------

        mcmc
        plot_truth_bin_traces
        plot_reco_bin_traces

        """

        L0 = self.log_likelihood(H0.translate(parameters0), systematics=toy_indices0)
        L1 = self.log_likelihood(H1.translate(parameters1), systematics=toy_indices1)
        # Build all possible combinations
        # Assumes posteriors are independent, I guess
        PLR = np.array(np.meshgrid(L1, -L0)).T.sum(axis=-1).flatten()
        preference = float(np.count_nonzero(PLR > 0)) / PLR.size
        return PLR, preference

    def plot_bin_efficiencies(self, filename, plot_limits=False, bins_per_plot=20):
        """Plot bin by bin efficiencies.

        Uses Matplotlibs ``boxplot``, showing the median (line), quartiles
        (box), 5% and 95% percentile (error bars), and mean (point) of the
        efficiencies over all matrices.

        Parameters
        ----------

        filename : string
            Where to save the plot.
        plot_limits : bool, optional
            Also plot the truth limits for each bin on a second axis.
        bins_per_plot : int, optional
            How many bins are combined into a single plot.

        Returns
        -------

        fig : Figure
        ax : list of Axis

        """

        # Load matplotlib on demand
        global plt
        if plt is None:
            from matplotlib import pyplot as _plt
            plt = _plt

        eff = self.response_matrix.sum(axis=-2)
        eff.shape = (np.prod(eff.shape[:-1], dtype=int), eff.shape[-1])
        if eff.shape[0] == 1:
            # Trick boxplot into working even if there is only one efficiency per bin
            eff = np.broadcast_to(eff, (2, eff.shape[1]))

        nplots = int(np.ceil(eff.shape[-1] / bins_per_plot))
        fig, ax= plt.subplots(nplots, squeeze=False, figsize=(8,nplots*3), sharey=True)
        ax = ax[:,0]
        for i in range(nplots):
            x = np.arange(i*bins_per_plot, min((i+1)*bins_per_plot, eff.shape[-1]), dtype=int)
            ax[i].set_ylabel("Efficiency")
            ax[i].boxplot(eff[:,i*bins_per_plot:(i+1)*bins_per_plot], whis=[5., 95.], sym='|', showmeans=True, whiskerprops={'linestyle': 'solid'}, positions=x)
            if plot_limits:
                ax2 = ax[i].twinx()
                ax2.plot(x, self.truth_limits[i*bins_per_plot:(i+1)*bins_per_plot], drawstyle='steps-mid', color='green')
                ax2.set_ylabel("Truth limits")
        ax[-1].set_xlabel("Truth bin #")
        fig.tight_layout()
        fig.savefig(filename)

        return fig, ax

    def plot_truth_bin_traces(self, filename, trace, plot_limits=False, bins_per_plot=20):
        """Plot bin by bin MCMC truth traces.

        Uses Matplotlibs ``boxplot``, showing the traces' median (line),
        quartiles (box), 5% and 95% percentile (error bars), and mean (point).

        Parameters
        ----------

        filename : string
            Where to save the plot.
        trace : array like
            The posterior trace of the truth bin values of an MCMC.
        plot_limits : bool or 'relative', optional
            Also plot the truth limits.
            If 'relative', the values are divided by the limits before plotting.
        bins_per_plot : int, optional
            How many bins are combined into a single plot.

        Returns
        -------

        fig : Figure
        ax : list of Axis

        See also
        --------

        mcmc
        plot_reco_bin_traces

        """

        # Load matplotlib on demand
        global plt
        if plt is None:
            from matplotlib import pyplot as _plt
            plt = _plt

        trace = trace.reshape( (np.prod(trace.shape[:-1], dtype=int), trace.shape[-1]) )
        if trace.shape[0] == 1:
            # Trick boxplot into working even if there is only one trace entry per bin
            trace = np.broadcast_to(trace, (2, trace.shape[1]))

        if plot_limits == 'relative':
            trace = trace / np.where(self.truth_limits > 0, self.truth_limits, 1.)

        nplots = int(np.ceil(trace.shape[-1] / bins_per_plot))
        fig, ax= plt.subplots(nplots, squeeze=False, figsize=(8,nplots*3), sharey=True)
        ax = ax[:,0]
        for i in range(nplots):
            x = np.arange(i*bins_per_plot, min((i+1)*bins_per_plot, trace.shape[-1]), dtype=int)
            ax[i].boxplot(trace[:,i*bins_per_plot:(i+1)*bins_per_plot], whis=[5., 95.], sym='|', showmeans=True, whiskerprops={'linestyle': 'solid'}, positions=x)
            if plot_limits == 'relative':
                ax[i].set_ylabel("Value / Truth limit")
            else:
                ax[i].set_ylabel("Value")
                if plot_limits:
                    ax[i].plot(x, self.truth_limits[i*bins_per_plot:(i+1)*bins_per_plot], drawstyle='steps-mid', color='green', label="Truth limit")
                    ax[i].legend(loc='best')
        ax[-1].set_xlabel("Truth bin #")
        fig.tight_layout()
        fig.savefig(filename)

        return fig, ax

    def plot_reco_bin_traces(self, filename, trace, toy_index=None, plot_data=False, bins_per_plot=20):
        """Plot bin by bin MCMC reco traces.

        Uses Matplotlibs ``boxplot``, showing the traces' median (line),
        quartiles (box), 5% and 95% percentile (error bars), and mean (point).

        Parameters
        ----------

        filename : string
            Where to save the plot.
        trace : array like
            The posterior trace of the *truth* bin values of an MCMC.
        toy_index : array like, optional
            The posterior trace of the chosen toy matrices of an MCMC.
        plot_data : bool or 'relative', optional
            Also plot the actual data content of the reco bins.
            If 'relative', the values are divided by the data before plotting.
        bins_per_plot : int, optional
            How many bins are combined into a single plot.

        Returns
        -------

        fig : Figure
        ax : list of Axis

        See also
        --------

        mcmc
        plot_truth_bin_traces

        """

        # Load matplotlib on demand
        global plt
        if plt is None:
            from matplotlib import pyplot as _plt
            plt = _plt

        resp = self._reduced_response_matrix
        if toy_index is not None:
            resp = resp[toy_index,...]

        trace = self._reduce_truth_vector(trace)[...,np.newaxis,:]
        trace = np.einsum('...i,...i->...', resp, trace)

        # Reshape for boxplotting
        trace = trace.reshape( (np.prod(trace.shape[:-1], dtype=int), trace.shape[-1]) )

        # Trick boxplot into working even if there is only one trace entry per bin
        if trace.shape[0] == 1:
            trace = np.broadcast_to(trace, (2,) + trace.shape[1:])

        if plot_data == 'relative':
            trace = trace / np.where(self.data_vector > 0, self.data_vector, 1.)

        nplots = int(np.ceil(trace.shape[-1] / bins_per_plot))
        fig, ax= plt.subplots(nplots, squeeze=False, figsize=(8,nplots*3), sharey=True)
        ax = ax[:,0]
        for i in range(nplots):
            x = np.arange(i*bins_per_plot, min((i+1)*bins_per_plot, trace.shape[-1]), dtype=int)
            ax[i].boxplot(trace[:,i*bins_per_plot:(i+1)*bins_per_plot], whis=[5., 95.], sym='|', showmeans=True, whiskerprops={'linestyle': 'solid'}, positions=x)
            if plot_data == 'relative':
                ax[i].set_ylabel("Value / Data")
            else:
                ax[i].set_ylabel("Value")
                if plot_data:
                    ax[i].plot(x, self.data_vector[i*bins_per_plot:(i+1)*bins_per_plot], drawstyle='steps-mid', color='green', label="Data")
                    ax[i].legend(loc='best')
        ax[-1].set_xlabel("Reco bin #")
        fig.tight_layout()
        fig.savefig(filename)

        return fig, ax
