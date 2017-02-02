import numpy as np
from scipy.stats import poisson
from scipy import optimize

class CompositeHypothesis(object):
    """A CompositeHypothesis translates a set of parameters into a truth vector."""

    def __init__(self, parameter_limits, translation_function):
        """Initialize the CompositeHypothesis.

        Arguments
        ---------

        parameter_limits : An iterable of lower and upper limits of the hypothesis.
                          The number of limits determines the number of parameters.
                          Parameters can be `None`. This sets no limit in that direction.

                              [ (x1_min, x1_max), (x2_min, x2_max), ... ]

        translation_function : The function to translate a vector of parameters into
                               a vector of truth expectation values:

                                   truth_vector = translation_function(parameter_vector)

        """

        self.parameter_limits = parameter_limits
        self.translate = translation_function

class LikelihoodMachine(object):
    """Class that calculates likelihoods for truth vectors."""

    def __init__(self, data_vector, response_matrix):
        """Initialize the LikelihoodMachine with the given data and response matrix."""

        self.data_vector = data_vector
        self.response_matrix = response_matrix

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
        eff = ( response_matrix.sum(axis=0) > 0. )
        reduced_response_matrix = response_matrix[:,eff]

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
    def _reduced_log_probability(data_vector, reduced_response_matrix, reduced_truth_vector):
        """Calculate the log probabilty of measuring `data_vector`, given `reduced_response_matrix` and `reduced_truth_vector`.

        Each of the three objects can actually be an array of vectors/matrices:

            data_vector.shape = (a,b,...,n_data)
            reduced_response_matrix.shape = (c,d,...,n_data,n_truth)
            reduced_truth_vector.shape = (e,f,...,n_truth)

        In this case, the return value will have the following shape:

            p.shape = (a,b,...,c,d,...,e,f,...)

        """

        data_shape = data_vector.shape
        response_shape = reduced_response_matrix.shape
        truth_shape = reduced_truth_vector.shape

        # Extend response_matrix to shape (a,b,...,c,d,...,n_data,n_truth)
        resp = LikelihoodMachine._create_vector_array(reduced_response_matrix, data_shape[:-1])

        # Reco expectation values of shape (a,b,...,c,d,...,e,f,...,n_data)
        # We need to set the terms of the einstein sum according to the number of axes.
        # N-dimensional case: 'a...dkl,e...fl->a...de...fk'
        ax_resp = len(resp.shape)
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
        reco = np.einsum(ein_resp+'kl,'+ein_truth+'l->'+ein_resp+ein_truth+'k', resp, reduced_truth_vector)

        # Create a data vector of the shape (a,b,...,n_data,c,d,...,e,f,...)
        data = LikelihoodMachine._create_vector_array(data_vector, list(response_shape[:-2])+list(truth_shape[:-1]), append=False)
        # Move axis so we get (a,b,...,c,d,...,e,f,...,n_data)
        data = np.moveaxis(data, len(data_shape)-1, -1)

        # Calculate the log probabilities and sum over the axis `n_data`.
        lp = np.sum(poisson.logpmf(data, reco), axis=-1)
        # Catch NaNs.
        lp = np.where(np.isfinite(lp), lp, -np.inf)

        return lp

    def _reduced_log_likelihood(self, reduced_truth_vector):
        """Calculate a more efficient log likelihood using only truth values that have an influence."""
        return LikelihoodMachine._reduced_log_probability(self.data_vector, self._reduced_response_matrix, reduced_truth_vector)

    @staticmethod
    def log_probability(data_vector, response_matrix, truth_vector):
        """Calculate the log probabilty of measuring `data_vector`, given `reduced_response_matrix` and `reduced_truth_vector`."""

        # TODO: Think of a good way to reduce multiple response_matrices at once.
        #       Just pass the full response matrices and truth vectors for now.

        return LikelihoodMachine._reduced_log_probability(data_vector, response_matrix, truth_vector)

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

    def log_likelihood(self, truth_vector):
        """Calculate the log likelihood of a vector of truth expectation values.

        Arguments
        ---------

        truth_vector : Array of truth expectation values.
                       Can be a multidimensional array of truth vectors.
                       The shape of the array must be `(a, b, c, ..., n_truth_values)`.
        """

        # Use reduced truth values for efficient calculations.
        reduced_truth_vector = self._reduce_truth_vector(truth_vector)

        return self._reduced_log_likelihood(reduced_truth_vector)

    def max_log_likelihood(self, composite_hypothesis, disp=False, method='basinhopping', kwargs={}):
        """Calculate the maximum log likelihood in the given CompositeHypothesis.

        Arguments
        ---------

        composite_hypothesis : The hypothesis to be evaluated.
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

        # Negative log likelihood function
        nll = lambda x: -self.log_likelihood(composite_hypothesis.translate(x))

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

        res.L = -res.fun
        res.x

        return res

    def absolute_max_log_likelihood(self, disp=False, kwargs={}):
        """Calculate the maximum log likelihood achievable with the given data.

        Arguments
        ---------

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
        super_hypothesis = CompositeHypothesis(bounds, translate)

        res = self.max_log_likelihood(super_hypothesis, disp=disp, method='basinhopping', kwargs=kwargs)

        # Translate vector of efficient truth values back to complete vector
        res.x = translate(res.x)
        return res

    def generate_random_data_sample(self, truth_vector, size=None):
        """Generate random data samples from the provided truth_vector."""

        mu = self.response_matrix.dot(truth_vector)
        if size is not None:
            # Append truth vector shape to requested shape of data sets
            try:
                shape = list(size)
            except TypeError:
                shape = [size]
            shape.extend(mu.shape)
            size = shape

        return np.random.poisson(mu, size=size)

    def likelihood_p_value(self, truth_vector, N=2500):
        """Calculate the likelihood p-value of a truth vector given the measured data.

        Arguments
        ---------

        truth_vector : The evaluated theory.
        N : The number of MC evaluations of the theory.

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
        fake_data = self.generate_random_data_sample(truth_vector, N)

        # Reduce truth vectors to efficient values
        reduced_truth_vector = self._reduce_truth_vector(truth_vector)

        # Calculate probabilities of each generated sample
        prob = LikelihoodMachine._reduced_log_probability(fake_data, self._reduced_response_matrix, reduced_truth_vector)

        # Get likelihood of actual data
        p0 = self._reduced_log_likelihood(reduced_truth_vector)

        # Count number of probabilities lower than or equal to the likelihood of the real data
        n = np.sum(prob <= p0)

        # Return the quotient
        return float(n) / N
