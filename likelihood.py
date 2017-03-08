import numpy as np
from scipy.stats import poisson
from scipy import optimize
from scipy.misc import derivative
from matplotlib import pyplot as plt
import pymc
import inspect

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
        self.translate = translation_function

class JeffreysPrior(object):
    """Universal non-informative prior for use in Bayesian MCMC analysis."""

    def __init__(self, response_matrix, translation_function, parameter_limits, default_values, dx=None):
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

        """

        old_shape = response_matrix.shape
        new_shape = (int(np.prod(old_shape[:-2])), old_shape[-2], old_shape[-1])
        self.response_matrix = response_matrix.reshape(new_shape)

        self.translate = translation_function

        limits = zip(*parameter_limits)
        self.lower_limits = np.array([ x if x is not None else -np.inf for x in limits[0] ])
        self.upper_limits = np.array([ x if x is not None else np.inf for x in limits[1] ])

        self.default_values = np.array(default_values)

        self._npar = len(parameter_limits)
        self._nreco = response_matrix.shape[-2]
        self.dx = dx or np.full(self._npar, 1e-3)

    def fisher_matrix(self, parameters, toy_index=0):
        """Calculate the Fisher information matrix for the given parameters."""

        resp = self.response_matrix[toy_index]
        par_mat = np.array(np.broadcast_to(parameters, (self._npar, self._npar)))
        i_diag = np.diag_indices(self._npar)

        # list of parameter sets -> list of reco values
        def reco_expectation(theta):
            ret = np.tensordot(self.translate(theta), resp, [[-1], [-1]])
            return ret

        # npar by npar matrix of reco values
        expect = np.broadcast_to(reco_expectation(par_mat), (self._npar, self._npar, self._nreco))

        # Diff expects the last axis to be the parameters, not the reco values!
        # varied parameter set -> transposed list of reco values
        def curried(x):
            par_mat[i_diag] = x
            return reco_expectation(par_mat).T

        diff = derivative(curried, x0=parameters, dx=self.dx).T

        diff_i = np.broadcast_to(diff, (self._npar, self._npar, self._nreco))
        diff_j = np.swapaxes(diff_i, 0, 1)

        fish = (diff_i * diff_j / expect).sum(-1)
        return fish

    def __call__(self, value, toy_index=0):
        """Calculate the prior probability of the given parameter set."""

        # Out of bounds?
        if np.any(value < self.lower_limits) or np.any(value > self.upper_limits):
            return -np.inf

        sign, log_det = np.linalg.slogdet(self.fisher_matrix(value, toy_index))

        return 0.5*log_det

class LikelihoodMachine(object):
    """Class that calculates likelihoods for truth vectors."""

    def __init__(self, data_vector, response_matrix):
        """Initialize the LikelihoodMachine with the given data and response matrix."""

        self.data_vector = np.array(data_vector)
        self.response_matrix = np.array(response_matrix)

        # Calculte the reduced response matrix for speedier calculations
        self._reduced_response_matrix, self._eff = LikelihoodMachine._reduce_response_matrix(self.response_matrix)
        self._n_eff = np.sum(self._eff)

    @staticmethod
    def _reduce_response_matrix(response_matrix):
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
        eff = ( eff > 0. )

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

        flat_vector = vector.flat

        if append:
            arr = np.broadcast_to(vector, list(shape) + list(vector.shape))
        else:
            n = np.prod(shape, dtype=int)
            arr = np.repeat(flat_vector,n)
            arr = arr.reshape( list(vector.shape) + list(shape) )

        return arr

    @staticmethod
    def log_probability(data_vector, response_matrix, truth_vector):
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
        # We need to set the terms of the einstein sum according to the number of axes.
        # N-dimensional case: 'a...dkl,e...fl->a...de...fk'
        ax_resp = resp.ndim
        if ax_resp == 2:
            ein_resp = ''
        elif ax_resp == 3:
            ein_resp = 'd'
        elif ax_resp == 4:
            ein_resp = 'ad'
        elif ax_resp > 4:
            ein_resp = 'a...d'
        ax_truth = len(truth_shape)
        if ax_truth == 1:
            ein_truth = ''
        elif ax_truth == 2:
            ein_truth = 'f'
        elif ax_truth == 3:
            ein_truth = 'ef'
        elif ax_truth > 3:
            ein_truth = 'e...f'
        reco = np.einsum(ein_resp+'kl,'+ein_truth+'l->'+ein_resp+ein_truth+'k', resp, truth_vector)

        # Create a data vector of the shape (a,b,...,n_data,c,d,...,e,f,...)
        data = LikelihoodMachine._create_vector_array(data_vector, list(response_shape[:-2])+list(truth_shape[:-1]), append=False)
        # Move axis so we get (a,b,...,c,d,...,e,f,...,n_data)
        data = np.moveaxis(data, len(data_shape)-1, -1)

        # Calculate the log probabilities and sum over the axis `n_data`.
        lp = np.sum(poisson.logpmf(data, reco), axis=-1)
        # Catch NaNs.
        lp = np.where(np.isfinite(lp), lp, -np.inf)

        return lp

    def _reduced_log_likelihood(self, reduced_truth_vector, systematics='profile'):
        """Calculate a more efficient log likelihood using only truth values that have an influence."""
        ll = LikelihoodMachine.log_probability(self.data_vector, self._reduced_response_matrix, reduced_truth_vector)

        # Deal with systematics, i.e. multiple response matrices
        systaxis = tuple(range(self.data_vector.ndim-1, self.data_vector.ndim-1+self._reduced_response_matrix.ndim-2))
        if systematics is not None and len(systaxis) > 0:
            if systematics == 'profile' or systematics == 'maximum':
                # Return maximum
                ll = np.max(ll, axis=systaxis)
            elif systematics == 'marginal' or systematics == 'average':
                # Return average
                N_resp = np.prod(self._reduced_response_matrix.shape[:-2])
                ll = np.log(np.sum(np.exp(ll), axis=systaxis) / N_resp)
            elif type(systematics) is tuple:
                # Return specific result
                index = tuple([ slice(None) ] * (self.data_vector.ndim-1) + list(systematics) + [Ellipsis])
                ll = ll[index]
            else:
                raise ValueError("Unknown systematics method!")

        return ll

    def _reduce_truth_vector(self, truth_vector):
        """Return a reduced truth vector view."""

        truth_vector = np.array(truth_vector)
        truth_shape = truth_vector.shape
        eff_index = LikelihoodMachine._create_vector_array(self._eff, truth_shape[:-1])

        # Reduced vectors in correct shape
        reduced_shape = np.copy(truth_shape)
        reduced_shape[-1] = self._n_eff
        reduced_truth_vector = truth_vector[eff_index].reshape(reduced_shape)

        return reduced_truth_vector

    def log_likelihood(self, truth_vector, systematics='profile'):
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
                      `None` : Do nothing, return multiple likelihoods.
                      Defaults to `profile`.
        """

        # Use reduced truth values for efficient calculations.
        reduced_truth_vector = self._reduce_truth_vector(truth_vector)

        ll = self._reduced_log_likelihood(reduced_truth_vector, systematics=systematics)

        return ll

    @staticmethod
    def max_log_probability(data_vector, response_matrix, composite_hypothesis, systematics='profile', disp=False, method='basinhopping', kwargs={}):
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
                      Defaults to 'profile'.
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

        # Negative log probability function
        if systematics == 'profile' or systematics == 'maximum':
            nll = lambda x: -np.max(LikelihoodMachine.log_probability(data_vector, response_matrix, composite_hypothesis.translate(x)))
        elif systematics == 'marginal' or systematics == 'average':
            N_resp = np.prod(response_matrix.shape[:-2])
            nll = lambda x: -np.log(np.sum(np.exp(LikelihoodMachine.log_probability(data_vector, response_matrix, composite_hypothesis.translate(x)))) / N_resp)
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

        limits = np.array(map(limit, bounds), dtype=float).transpose()
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

            x0 = np.array(map(start_value, bounds))

            # Step length for basin hopping
            def step_value(limits):
                if None in limits:
                    return 1.
                else:
                    # Step size in the order of the parameter range
                    return (limits[1]-limits[0]) / 2.
            step = np.array(map(step_value, bounds))

            # Number of parameters
            n = len(bounds)

            # Define a step function that does *not* produce illegal parameter values
            def step_fun(x):
                # Vary parameters according to their value,
                # but at least by a minimum amount.
                dx = np.random.randn(n) * np.maximum(np.abs(x - x0), step)

                # Make sure the new values are within bounds
                ret = x + dx
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
                    }
                }
            }
            kw.update(kwargs)

            res = optimize.basinhopping(nll, x0, disp=disp, **kw)
        else:
            raise ValueError("Unknown method: %s"%(method,))

        res.P = -res.fun
        if systematics == 'profile' or systematics == 'maximum':
            res.i = np.argmax(LikelihoodMachine.log_probability(data_vector, response_matrix, composite_hypothesis.translate(res.x)))

        return res

    def max_log_likelihood(self, composite_hypothesis, *args, **kwargs):
        """Calculate the maximum possible Likelihood in the given CompositeHypothesis, given the measured data.

        Arguments
        ---------

        composite_hypothesis : The hypothesis to be evaluated.
        systematics : How to deal with detector systematics, i.e. multiple response matrices.
                      'profile', 'maximum': Choose the response matrix that yields the highest likelihood.
                      'marginal', 'average': Sum the probabilites yielded by the matrices.
                      Defaults to 'profile'.
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
        ret = LikelihoodMachine.max_log_probability(self.data_vector, self.response_matrix, composite_hypothesis, *args, **kwargs)
        ret.L = ret.P
        del ret.P
        return ret

    def absolute_max_log_likelihood(self, systematics='profile', disp=False, kwargs={}):
        """Calculate the maximum log likelihood achievable with the given data.

        Arguments
        ---------

        systematics : How to deal with detector systematics, i.e. multiple response matrices.
                      'profile', 'maximum': Choose the response matrix that yields the highest likelihood.
                      'marginal', 'average': Sum the probabilites yielded by the matrices.
                      Defaults to 'profile'.
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
        eff_to_all = np.eye(len(self._eff))[:,self._eff]
        translate = lambda x: eff_to_all.dot(x)
        super_hypothesis = CompositeHypothesis(translate, bounds)

        res = self.max_log_likelihood(super_hypothesis, systematics=systematics, disp=disp, method='basinhopping', kwargs=kwargs)

        # Translate vector of efficient truth values back to complete vector
        res.x = translate(res.x)
        return res

    @staticmethod
    def generate_random_data_sample(response_matrix, truth_vector, size=None):
        """Generate random data samples from the provided truth_vector."""

        mu = response_matrix.dot(truth_vector)
        if size is not None:
            # Append truth vector shape to requested shape of data sets
            try:
                shape = list(size)
            except TypeError:
                shape = [size]
            shape.extend(mu.shape)
            size = shape

        return np.random.poisson(mu, size=size)

    def likelihood_p_value(self, truth_vector, N=2500, systematics='profile'):
        """Calculate the likelihood p-value of a truth vector given the measured data.

        Arguments
        ---------

        truth_vector : The evaluated theory.
        N : The number of MC evaluations of the theory.
        systematics : How to deal with detector systematics, i.e. multiple response matrices.
                      'profile', 'maximum': Choose the response matrix that yields the highest likelihood.
                      'marginal', 'average': Sum the probabilites yielded by the matrices.
                      Defaults to 'profile'.

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

        # Draw N fake data distributions
        fake_data = LikelihoodMachine.generate_random_data_sample(self.response_matrix, truth_vector, N)

        # Reduce truth vectors to efficient values
        reduced_truth_vector = self._reduce_truth_vector(truth_vector)

        # Calculate probabilities of each generated sample
        prob = LikelihoodMachine.log_probability(fake_data, self._reduced_response_matrix, reduced_truth_vector)

        # Get likelihood of actual data
        p0 = self._reduced_log_likelihood(reduced_truth_vector, systematics=systematics)

        # Count number of probabilities lower than or equal to the likelihood of the real data
        n = np.sum(prob <= p0)

        # Return the quotient
        return float(n) / N

    def max_likelihood_p_value(self, composite_hypothesis, parameters=None, N=250, systematics='profile', **kwargs):
        """Calculate the maximum likelihood p-value of a composite hypothesis given the measured data.

        Arguments
        ---------

        composite_hypothesis : The evaluated theory.
        parameters : The assumed true parameters of the composite hypothesis.
                     If no parameters are given, they will be calculated with
                     the maximum likelihood method.
        N : The number of MC evaluations of the theory.

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

        truth_vector = composite_hypothesis.translate(parameters)

        # Draw N fake data distributions
        fake_data = LikelihoodMachine.generate_random_data_sample(self.response_matrix, truth_vector, N)

        # Calculate the maximum probabilities
        def prob_fun(data):
            return LikelihoodMachine.max_log_probability(data, self.response_matrix, composite_hypothesis, **kwargs).P
        prob = map(prob_fun, fake_data)

        # Get likelihood of actual data
        p0 = self.log_likelihood(truth_vector, systematics=systematics)

        # Count number of probabilities lower than or equal to the likelihood of the real data
        n = np.sum(prob <= p0)

        # Return the quotient
        return float(n) / N

    def max_likelihood_ratio_p_value(self, H0, H1, par0=None, par1=None, N=250, systematics='profile', **kwargs):
        """Calculate the maximum likelihood ratio p-value of a two composite hypotheses given the measured data.

        Arguments
        ---------

        H0 : The tested composite hypothesis.
        H1 : The alternative composite hypothesis.
        par0 : The assumed true parameters of the tested hypothesis.
        par1 : The maximum likelihood parameters of the alternative hypothesis.
        N : The number of MC evaluations of the theory.

        Additional keyword arguments will be passed to the likelihood maximizer.

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

        truth_vector = H0.translate(par0)
        alternative_truth = H1.translate(par1)

        # Draw N fake data distributions
        fake_data = LikelihoodMachine.generate_random_data_sample(self.response_matrix, truth_vector, N)

        # Calculate the maximum probabilities
        def ratio_fun(data):
             p0 = LikelihoodMachine.max_log_probability(data, self.response_matrix, H0, systematics=systematics, **kwargs).P
             p1 = LikelihoodMachine.max_log_probability(data, self.response_matrix, H1, systematics=systematics, **kwargs).P
             return p0-p1 # difference because log
        ratio = map(ratio_fun, fake_data)

        # Get likelihood of actual data
        p0 = self.log_likelihood(truth_vector, systematics=systematics)
        p1 = self.log_likelihood(alternative_truth, systematics=systematics)
        r0 = p0-p1 # difference because log

        # Count number of probabilities lower than or equal to the likelihood of the real data
        n = np.sum(ratio <= r0)

        # Return the quotient
        return float(n) / N

    def MCMC(self, composite_hypothesis):
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
        names_priors = zip(names, priors)
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

    def plot_bin_efficiencies(self, filename):
        """Plot bin by bin efficiencies."""

        eff = self.response_matrix.sum(axis=-2)
        eff.shape = (np.prod(eff.shape[:-1]), eff.shape[-1])

        fig, ax = plt.subplots(1)
        ax.set_xlabel("Truth bin #")
        ax.set_ylabel("Efficiency")
        ax.boxplot(eff, whis=[5., 95.], showmeans=True, whiskerprops={'linestyle': 'solid'}, positions=range(eff.shape[1]))
        fig.savefig(filename)
