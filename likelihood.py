import numpy as np
from scipy.stats import poisson
from scipy import optimize

class LikelihoodMachine(object):
    """Class that calculates likelihoods for truth vectors."""

    def __init__(self, data_vector, response_matrix):
        """Initialize the LikelihoodMachine with the given data and response matrix."""

        self.data_vector = data_vector
        self.response_matrix = response_matrix

        # Calculte the truth bins that have any influence at all
        self._eff = ( self.response_matrix.sum(axis=0) > 0. )
        self._reduced_response_matrix = self.response_matrix[:,self._eff]
        # How to use the reduced reposne matrix:
        # reco = reduced_response.dot(truth[eff])

    @staticmethod
    def _create_vector_array(vector, shape):
        """Create an array of `shape` containing n `vector`s."""

        n = np.prod(shape, dtype=int)
        m = len(vector)
        arr = np.ndarray((n, m), dtype=vector.dtype)
        for i in range(n):
            arr[i,:] = vector
        arr = arr.reshape( list(shape) + [m] )

        return arr

    def _reduced_log_likelihood(self, reduced_truth_vector):
        """Calculate a more efficient log likelihood using only truth values that have an influence."""
        reco = np.tensordot(reduced_truth_vector, self._reduced_response_matrix, axes=([-1],[-1]))

        # Create a data vector of the right shape
        data = LikelihoodMachine._create_vector_array(self.data_vector, reco.shape[:-1])

        ll = np.sum(poisson.logpmf(data, reco), axis=-1)
        ll = np.where(np.isfinite(ll), ll, -np.inf)
        return ll

    def log_likelihood(self, truth_vector):
        """Calculate the log likelihood of a vector of truth expectation values.

        Arguments
        ---------

        truth_vector : Array of truth expectation values.
                       Can be a multidimensional array of truth vectors.
                       The shape of the array must be `(a, b, c, ..., n_truth_values)`.
        """

        # Calculate the mask of efficient truth values.
        # For single truth vectors, it is just the _eff vector.
        # For arrays of truth vectors, we need to create an identical array of _eff vectors.
        truth_vector = np.array(truth_vector)
        truth_shape = truth_vector.shape
        eff_index = LikelihoodMachine._create_vector_array(self._eff, truth_shape[:-1])

        # Reduced vectors in correct shape
        reduced_shape = np.copy(truth_shape)
        reduced_shape[-1] = np.sum(self._eff)
        reduced_truth_vector = truth_vector[eff_index].reshape(reduced_shape)

        ignored_truth_values = truth_vector[np.logical_not(eff_index)]
        if np.sum(ignored_truth_values) > 0:
            print("Warning: Truth contains expectation values for bins with zero efficiency!")

        return self._reduced_log_likelihood(reduced_truth_vector)

    def absolute_max_log_likelihood(self, disp=False, method='basinhopping', kwargs=None):
        """Calculate the maximum achievable log likelihood.

        Arguments
        ---------

        disp : Display status messages during optimization.
        method : Select the method to be used for maximization,
                 either 'differential_evolution' or 'basinhopping'.
                 Default: 'basinhopping'
        kwargs : Keyword arguments to be passed to the minimizer.
                 If `None`, reasonable default values will be used.

        Returns
        -------

        res : OptimizeResult object containing the maximum likelihood `res.L`.
        """

        # Negative log likelihood function
        nll = lambda x: -self._reduced_log_likelihood(x)

        # Number of effective parameters
        n = np.sum(self._eff)

        # Number of data events
        ndata = np.sum(self.data_vector)

        if method == 'differential_evolution':
            # Set upper bound for truth bins to the total number of events in the data
            bounds = [(0,ndata)] * n
            if kwargs is None:
                kwargs = {}

            res = optimize.differential_evolution(nll, bounds, disp=disp, **kwargs)
        elif method == 'basinhopping':
            # Start with a flat distribution of truth values
            mu0 = float(ndata) / n
            x0 = [mu0] * n

            # Define a step function that does *not* produce negative bin values
            def step_fun(x):
                # Vary bins according to their filling,
                # but at least by a minimum amount.
                dx = np.random.randn(n) * (np.sqrt(x) + mu0)
                return np.abs(x+dx)

            # Local minimizer options
            if kwargs is None:
                kwargs = {
                    'take_step': step_fun,
                    'T': n,
                    'minimizer_kwargs': {
                        'method': 'L-BFGS-B',
                        'bounds': [(0, None)]*n, # No negative values!
                        'options': {
                            #'disp': True,
                        }
                    }
                }

            res = optimize.basinhopping(nll, x0, disp=disp, **kwargs)
        else:
            raise ValueError("Unknown method: %s"%(method,))

        res.L = -res.fun
        reduced_x = res.x
        res.x = np.zeros(self.response_matrix.shape[1], dtype=float)
        res.x[self._eff] = reduced_x

        return res

