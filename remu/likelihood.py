from __future__ import division
from six.moves import map, zip
import numpy as np
from scipy.stats import poisson
from scipy import optimize
from scipy.misc import derivative
from matplotlib import pyplot as plt
import pymc
import inspect
from warnings import warn

class CompositeHypothesis(object):
    """A CompositeHypothesis translates a set of parameters into a truth vector."""

    def __init__(self, translation_function, parameter_limits=None, parameter_priors=None, parameter_names=None):
        """Initialize the CompositeHypothesis.

        Arguments
        ---------

        translation_function : The function to translate a vector of parameters into
                               a vector of truth expectation values:

                                   truth_vector = translation_function(parameter_vector)

                               It must support translating arrays of parameter vectors into
                               arrays of truth vectors:

                                   [truth_vector, ...] = translation_function([parameter_vector, ...])

        parameter_limits : An iterable of lower and upper limits of the hypothesis' parameters.
                           The number of limits determines the number of parameters.
                           Parameters can be `None`. This sets no limit in that direction.

                               [ (x1_min, x1_max), (x2_min, x2_max), ... ]

                           Parameter limits are used in likelihood maximization.

        parameter_priors : An iterable of prior probability density functions.
                           The number of priors determines the number of parameters.
                           Each function must return the logarithmic probability density,
                           given a value of the corresponding parameter.

                               prior(value=default) = log( pdf(value) )

                           They should return `-numpy.inf` for excluded values.
                           The function's argument *must* be named `value` and a default *must* be provided.
                           Parameter priors are used in Marcov Chain Monte Carlo evaluations.

        parameter_names : Optional. Iterable of the parameter names.
                          These names will be used in some plotting comvenience functions.

        Depending on the use case, one can provide `parameter_limits` and/or `parameter_priors`,
        but they are *not* checked for consistency!

        """

        if parameter_limits is None and parameter_priors is None:
            raise TypeError("Must provide at least one of `parameter_lmits` and/or `parameter_priors`")

        self.parameter_limits = parameter_limits
        self.parameter_priors = parameter_priors
        self.parameter_names = parameter_names
        self._translate = translation_function

    def translate(self, parameters):
        """Translate the parameter vector to a truth vector."""
        return self._translate(parameters)

class LinearHypothesis(CompositeHypothesis):
    """Special case of CompositeHypothesis for linear combinations."""

    def __init__(self, M, b=None, *args, **kwargs):
        """Initialise the LinearHypothesis.

        Arguments
        ---------

        M : The matrix translating the parameter vector into a truth vector:

                truth = M.dot(parameters)

        b : Optional. A constant (vector) to be added to the truth vector:

                truth = M.dot(parameters) + b

        Other arguments are passed on to the CompositeHypothesis init method.
        """

        self.M = np.array(M, dtype=float)
        if b is None:
            self.b = None
        else:
            self.b = np.array(b, dtype=float)

        if b is None:
            translate = lambda par: np.tensordot(self.M, par, axes=(-1,-1))
        else:
            translate = lambda par: np.tensordot(self.M, par, axes=(-1,-1)) + self.b

        CompositeHypothesis.__init__(self, translate, *args, **kwargs)

class TemplateHypothesis(LinearHypothesis):
    """Convenience class to turn truth templates into a CompositeHypothesis."""

    def __init__(self, templates, constant=None, parameter_limits=None, *args, **kwargs):
        """Initialise the TemplateHypothesis.

        Arguments
        ---------

        templates : Iterable of truth vector templates.
        constant : Optional. Constant offset to be added to the truth vector.
        parameter_limits : Optional. An iterable of lower and upper limits of the hypothesis' parameters.
                           Defaults to non-negative parameter values.

        Other arguments are passed to the LinearHypothesis init method.
        """

        M = np.array(templates, dtype=float).T
        if parameter_limits is None:
            parameter_limits = [(0,None)]*M.shape[-1]

        LinearHypothesis.__init__(self, M, constant, parameter_limits, *args, **kwargs)

class JeffreysPrior(object):
    """Universal non-informative prior for use in Bayesian MCMC analysis."""

    def __init__(self, response_matrix, translation_function, parameter_limits, default_values, dx=None, total_truth_limit=None):
        """Initilize a JeffreysPrior.

        Arguments
        ---------

        response_matrix : Response matrix that translates truth into reco bins.
                          Can be an array of matrices.

        translation_function : The function to translate a vector of parameters into
                               a vector of truth expectation values:

                                   truth_vector = translation_function(parameter_vector)

                               It must support translating arrays of parameter vectors into
                               arrays of truth vectors:

                                   [truth_vector, ...] = translation_function([parameter_vector, ...])

        parameter_limits : An iterable of lower and upper limits of the hypothesis' parameters.
                           The number of limits determines the number of parameters.
                           Parameters can be `None`. This sets no limit in that direction.

                               [ (x1_min, x1_max), (x2_min, x2_max), ... ]

        default_values : The default values of the parameters

        dx : Array of step sizes to be used in numerical differentiation.
             Default: `numpy.full(len(parameter_limits), 1e-3)`

        total_truth_limit : Maximum total number of truth events to consider in the prior.
                            This can be used to make priors proper in a consistent way,
                            since the limit is defined in the truth space, rather than
                            the prior parameter space.

        Pitfalls
        --------

        By construction, the JeffreysPrior will return the log probability
        `-inf`, a probability of 0, when the expected *reco* values do not
        depend on one of the parameters. In this case the "useless" parameter
        should be removed. It simply cannot be constrained with the given
        detector response.

        """

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
        """Calculate the Fisher information matrix for the given parameters."""

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
    """Class that calculates likelihoods for truth vectors."""

    def __init__(self, data_vector, response_matrix, truth_limits=None, limit_method='raise', eff_threshold=0., eff_indices=None, is_sparse=False):
        """Initialize the LikelihoodMachine with the given data and response matrix.

        The optional `truth_limits` tells the LikelihoodMachine up to which
        truth bin value the response matrix stays valid. If the machine is
        asked to calculate the likelihood of an out-of-bounds truth vector, it
        is handled according to `limit_method`.

            'raise' (default) : An exception is raised.
            'prohibit' : A likelihood of 0 is returned.

        This can be used to constrain the testable theories to events that have
        been simulated enough times in the detector Monte Carlo data. I.e. if
        one wants demands a 10x higher MC statistic:

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
        the following condition:

            response_matrix.shape[-1] == len(eff_indixes)

        If the matrix is sparse, the vector of truth limits *must* be provided.
        Its length must be that of the non-sparse response matrix, i.e. the
        number of truth bins irrespective of efficient indices.

        """

        self.data_vector = np.array(data_vector)
        self.response_matrix = np.array(response_matrix)
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

        Returns
        -------

        reduced_response_matrix : A view of the matrix with reduced number of columns.
        efficiency_vector : A vector of boolean values, describing which columns were kept.

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
        """Create an array of `shape` containing n `vector`s.

        The resulting shape depends on the `append` parameter.

            vector.shape = (a,b,...)
            shape = (c,d,...)

        If `append` is `True`:

            ret.shape = (c,d,...,a,b,...)

        If `append` is `False`:

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

    @staticmethod
    def log_probability(data_vector, response_matrix, truth_vector, _constant=None):
        """Calculate the log probabilty of measuring `data_vector`, given `response_matrix` and `truth_vector`.

        Each of the three objects can actually be an array of vectors/matrices:

            data_vector.shape = (a,b,...,n_data)
            response_matrix.shape = (c,d,...,n_data,n_truth)
            truth_vector.shape = (e,f,...,n_truth)

        In this case, the return value will have the following shape:

            p.shape = (a,b,...,c,d,...,e,f,...)

        """

        data_shape = data_vector.shape
        response_shape = response_matrix.shape
        truth_shape = truth_vector.shape

        # Extend response_matrix to shape (a,b,...,c,d,...,n_data,n_truth)
        resp = LikelihoodMachine._create_vector_array(response_matrix, data_shape[:-1])

        # Reco expectation values of shape (a,b,...,c,d,...,e,f,...,n_data)
        if _constant is None:
            reco = LikelihoodMachine._translate(resp, truth_vector)
        else:
            reco = LikelihoodMachine._translate(resp, truth_vector) + _constant

        # Create a data vector of the shape (a,b,...,n_data,c,d,...,e,f,...)
        data = LikelihoodMachine._create_vector_array(data_vector, list(response_shape[:-2])+list(truth_shape[:-1]), append=False)
        # Move axis so we get (a,b,...,c,d,...,e,f,...,n_data)
        data = np.moveaxis(data, len(data_shape)-1, -1)

        # Calculate the log probabilities and sum over the axis `n_data`.
        lp = np.sum(poisson.logpmf(data, reco), axis=-1)
        # Catch NaNs.
        lp = np.where(np.isfinite(lp), lp, -np.inf)

        return lp

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

        Arguments
        ---------

        truth_vector : Array of truth expectation values.
                       Can be a multidimensional array of truth vectors.
                       The shape of the array must be `(a, b, c, ..., n_truth_values)`.
        systematics : How to deal with detector systematics, i.e. multiple response matrices.
                      'profile', 'maximum': Choose the response matrix that yields the highest probability.
                      'marginal', 'average': Sum the probabilites yielded by the matrices.
                      `tuple(index)`: Select one specific matrix.
                      `array(indices)`: Select a specific matrix for each truth vector.
                                        Must have shape `(a, b, c, ..., len(index))`.
                      `None` : Do nothing, return multiple likelihoods.
                      Defaults to `marginal`.
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
        """Calculate the maximum possible probability in the given CompositeHypothesis, given `response_matrix` and `data_vector`.

        Arguments
        ---------

        data_vector : Vector of measured values.
        response_matrix : The response matrix that translates truth into reco space.
                          Can be an arbitrarily shaped array of response matrices.
        composite_hypothesis : The hypothesis to be evaluated.
        systematics : How to deal with detector systematics, i.e. multiple response matrices.
                      'profile', 'maximum': Choose the response matrix that yields the highest probability.
                      'marginal', 'average': Sum the probabilites yielded by the matrices.
                      Defaults to 'marginal'.
        disp : Display status messages during optimization.
        method : Select the method to be used for maximization,
                 either 'differential_evolution' or 'basinhopping'.
                 Default: 'basinhopping'
        kwargs : Keyword arguments to be passed to the minimizer.
                 If empty, reasonable default values will be used.

        Returns
        -------

        res : OptimizeResult object containing the maximum log probability `res.P`.
              In case of `systematics=='profile'`, it also contains the index of
              the response matrix that yielded the maximum likelihood `res.i`
        """

        if isinstance(composite_hypothesis, LinearHypothesis):
            # Special case!
            # Since the parameter translation is just a matrix multiplication,
            # we can save a lot of computing time by pre-calculating the combined
            # matrix.
            R = np.tensordot(response_matrix, composite_hypothesis.M, axes=(-1,-2))
            b = composite_hypothesis.b
            if b is None:
                likfun = lambda x : LikelihoodMachine.log_probability(data_vector, R, x)
            else:
                const = LikelihoodMachine._translate(response_matrix, b)
                likfun = lambda x : LikelihoodMachine.log_probability(data_vector, R, x, _constant=const)
        else:
            likfun = lambda x : LikelihoodMachine.log_probability(data_vector, response_matrix, composite_hypothesis.translate(x))

        # Negative log probability function
        if systematics == 'profile' or systematics == 'maximum':
            nll = lambda x: -np.max(likfun(x))
        elif systematics == 'marginal' or systematics == 'average':
            N_resp = np.prod(response_matrix.shape[:-2])
            nll = lambda x: -(np.logaddexp.reduce(likfun(x)) - np.log(N_resp))
        else:
            raise ValueError("Unknown systematics method!")

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
        """Calculate the maximum possible Likelihood in the given CompositeHypothesis, given the measured data.

        Arguments
        ---------

        composite_hypothesis : The hypothesis to be evaluated.
        systematics : How to deal with detector systematics, i.e. multiple response matrices.
                      'profile', 'maximum': Choose the response matrix that yields the highest likelihood.
                      'marginal', 'average': Sum the probabilites yielded by the matrices.
                      Defaults to 'marginal'.
        disp : Display status messages during optimization.
        method : Select the method to be used for maximization,
                 either 'differential_evolution' or 'basinhopping'.
                 Default: 'basinhopping'
        kwargs : Keyword arguments to be passed to the minimizer.
                 If empty, reasonable default values will be used.

        Returns
        -------

        res : OptimizeResult object containing the maximum log probability `res.L`.
              In case of `systematics=='profile'`, it also contains the index of
              the response matrix that yielded the maximum likelihood `res.i`
        """

        resp = self._reduced_response_matrix
        # Wrapping composite hypothesis to produce reduced truth vectors
        H0 = self._composite_hypothesis_wrapper(composite_hypothesis)
        ret = LikelihoodMachine.max_log_probability(self.data_vector, resp, H0, *args, **kwargs)
        ret.L = ret.P
        del ret.P
        return ret

    def absolute_max_log_likelihood(self, systematics='marginal', disp=False, kwargs={}):
        """Calculate the maximum log likelihood achievable with the given data.

        Arguments
        ---------

        systematics : How to deal with detector systematics, i.e. multiple response matrices.
                      'profile', 'maximum': Choose the response matrix that yields the highest likelihood.
                      'marginal', 'average': Sum the probabilites yielded by the matrices.
                      Defaults to 'marginal'.
        disp : Display status messages during optimization.
        method : Select the method to be used for maximization,
                 either 'differential_evolution' or 'basinhopping'.
                 Default: 'basinhopping'
        kwargs : Keyword arguments to be passed to the minimizer.
                 If empty, reasonable default values will be used.

        Returns
        -------

        res : OptimizeResult object containing the maximum likelihood `res.L`.
        """

        # Create a CompositeHypothesis that uses only the efficient truth values
        n = self._n_eff
        bounds = [(0,None)]*n
        eff_to_all = np.eye(self.response_matrix.shape[-1])[:,self._i_eff]
        translate = lambda x: eff_to_all.dot(x)
        super_hypothesis = CompositeHypothesis(translate, bounds)

        res = self.max_log_likelihood(super_hypothesis, systematics=systematics, disp=disp, method='basinhopping', kwargs=kwargs)

        # Translate vector of efficient truth values back to complete vector
        res.x = translate(res.x)
        return res

    @staticmethod
    def generate_random_data_sample(response_matrix, truth_vector, size=None, each=False):
        """Generate random data samples from the provided truth_vector.

        If `each` is `True`, the data is generated for each matrix in
        `response_matrix`.  Otherwise `size` determines the total number of
        generated data sets.
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
        """Calculate the likelihood p-value of a truth vector given the measured data.

        Arguments
        ---------

        truth_vector : The evaluated theory.
        N : The number of MC evaluations of the theory.
        generator_matrix_index : The index of the response matrix to be used to generate
                                 the fake data. This needs to be specified only if the
                                 LikelihoodMachine contains more than one response matrix.
                                 If it is `None`, N data sets are thrown for *each* matrix,
                                 and a p-value is evaluated for all of them.
                                 The return value thus becomes an array of p-values.
        systematics : How to deal with detector systematics, i.e. multiple response matrices.
                      'profile', 'maximum': Choose the response matrix that yields the highest likelihood.
                      'marginal', 'average': Sum the probabilites yielded by the matrices.
                      Defaults to 'marginal'.

        Additional keyword arguments will be passed to the likelihood method.

        Returns
        -------

        p : The probability of measuring data as unlikely or more unlikely than
            the actual data.

        The p-value is estimated by randomly creating `N` data samples
        according to the given theory. The number of data-sets that yield a
        likelihood as bad as, or worse than the likelihood given the actual
        data, `n`, are counted. The estimate for p is then

            p = n/N.

        The variance of the estimator follows that of binomial statistics:

                     var(n)   Np(1-p)      1
            var(p) = ------ = ------- <= ---- .
                      N^2       N^2       4N

        The expected uncertainty can thus be directly influenced by choosing an
        appropriate number of evaluations.
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
        """Calculate the maximum likelihood p-value of a composite hypothesis given the measured data.

        Arguments
        ---------

        composite_hypothesis : The evaluated theory.
        parameters : The assumed true parameters of the composite hypothesis.
                     If no parameters are given, they will be calculated with
                     the maximum likelihood method.
        N : The number of MC evaluations of the theory.
        generator_matrix_index : The index of the response matrix to be used to generate
                                 the fake data. This needs to be specified only if the
                                 LikelihoodMachine contains more than one response matrix.
                                 If it is `None`, N data sets are thrown for *each* matrix,
                                 and a p-value is evaluated for all of them.
                                 The return value thus becomes an array of p-values.
        nproc : How many processes to use in parallel.
                Default: 0

        Additional keyword arguments will be passed to the likelihood maximizer.

        Returns
        -------

        p : The probability of measuring data that yields a lower maximum
            likelihood than the actual data.

        The p-value is estimated by randomly creating `N` data samples
        according to the given theory. The number of data-sets that yield a
        maximum likelihood as bad as, or worse than the likelihood given the
        actual data, `n`, are counted. The estimate for p is then

            p = n/N.

        The variance of the estimator follows that of binomial statistics:

                     var(n)   Np(1-p)      1
            var(p) = ------ = ------- <= ---- .
                      N^2       N^2       4N

        The expected uncertainty can thus be directly influenced by choosing an
        appropriate number of evaluations.
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
            from multiprocess import Pool
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
        """Calculate the maximum likelihood ratio p-value of a two composite hypotheses given the measured data.

        Arguments
        ---------

        H0 : The tested composite hypothesis. Usually a subset of H1.
        H1 : The alternative composite hypothesis.
        par0 : The assumed true parameters of the tested hypothesis.
        par1 : The maximum likelihood parameters of the alternative hypothesis.
        N : The number of MC evaluations of the theory.
        generator_matrix_index : The index of the response matrix to be used to generate
                                 the fake data. This needs to be specified only if the
                                 LikelihoodMachine contains more than one response matrix.
                                 If it is `None`, N data sets are thrown for *each* matrix,
                                 and a p-value is evaluated for all of them.
                                 The return value thus becomes an array of p-values.
        nproc : How many processes to use in parallel.
                Default: 0
        nested : Is H0 a nested theory, i.e. does it cover a subset of H1?
                 In this case the likelihood maximum likelihood values must always be L0 <= L1.
                 If `True` or `'ignore'`, the calculation of likelihood ratios is re-tried
                 a couple of times if no valid likelihood ratio is found.
                 If `True` and no valid value was found after 10 tries, an errors is raised.
                 If `False`, those cases are just accepted.

        Additional keyword arguments will be passed to the likelihood maximizer.

        Special case for maximizer argument `kwargs['x0']`: The two different hypotheses
        have different parameter spaces. So the (optional) elements 'x0' and 'x1' are handled
        as the starting points for the `H0` and `H1` fit respectively.

        Returns
        -------

        p : The probability of measuring data that yields a lower maximum
            likelihood ratio than the actual data.

        The p-value is estimated by randomly creating `N` data samples
        according to the given theory. The number of data-sets that yield a
        maximum likelihood as bad as, or worse than the likelihood given the
        actual data, `n`, are counted. The estimate for p is then

            p = n/N.

        The variance of the estimator follows that of binomial statistics:

                     var(n)   Np(1-p)      1
            var(p) = ------ = ------- <= ---- .
                      N^2       N^2       4N

        The expected uncertainty can thus be directly influenced by choosing an
        appropriate number of evaluations.
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
            from multiprocess import Pool
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

        See documentation of PyMC.
        """

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

    def plr(self, H0, parameters0, toy_indices0, H1, parameters1, toy_indices1):
        """Calculate the Posterior distribution of the log Likelihood Ratio.

        Arguments
        ---------

        H0/1 : Composite Hypotheses to be compared.

        parameters0/1 : Arrays of parameter vectors, drawn from the posterior
                        distribution of the hypotheses, e.g. with the MCMC objects.

                            parameters0 = [ [1.0, 2.0, 3.0],
                                            [1.1, 1.9, 2.8],
                                            ...
                                          ]

        toy_indices0/1 : The corresponding systematic toy indices, in an
                         array of equal dimensionality.  That means, even if the toy index is
                         just a single integer, it must be provided as arrays of length 1.

                             toy_indices0 = [ [0],
                                              [3],
                                              ...
                                            ]

        Returns
        -------

        PLR, model_preference : A sample from the PLR as calculated from the parameter sets
                                and the resulting model preference.

        The model preference is calculated as the fraction of likelihood ratios
        in the PLR that prefer H1 over H0:

            model_preference = N(PLR > 0) / N(PLR)

        It can be interpreted as the posterior probability for the data
        prefering H1 over H0.

        The PLR is symmetric:

            PLR(H0, H1) = -PLR(H1, H0)
            preference(H0, H1) = 1. - preference(H1, H0) # modulo the cases of PLR = 0.

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

        Also plots bin truth limits if `plot_limits` is `True`.
        """

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

    def plot_truth_bin_traces(self, filename, trace, plot_limits=False, bins_per_plot=20):
        """Plot bin by bin MCMC truth traces.

        Also plots bin truth limits if `plot_limits` is `True`.  If it is set
        to the string 'relative', the values are divided by the limit before
        plotting.
        """

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

    def plot_reco_bin_traces(self, filename, trace, toy_index=None, plot_data=False, bins_per_plot=20):
        """Plot bin by bin MCMC reco traces.

        Also plots bin data if `plot_data` is `True`.  If it is set to the
        string 'relative', the values are divided by the data before plotting.
        """

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
