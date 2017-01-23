import numpy as np
from scipy.stats import poisson
from scipy import optimize

class LikelihoodMachine(object):
    """Class that calculates likelihoods for truth vectors."""

    def __init__(self, data_vector, response_matrix):
        """Initialize the LikelihoodMachine with the given data and response matrix."""

        self.data_vector = data_vector
        self.response_matrix = response_matrix

        # Calculte the reduced response matrix for speedier calculations
        self._reduced_response_matrix, self._eff = LikelihoodMachine._reduce_response_matrix(self.response_matrix)

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

        n = np.prod(shape, dtype=int)
        m = len(flat_vector)
        if append:
            arr = np.ndarray((n, m), dtype=vector.dtype)
            for i in range(n):
                arr[i,:] = flat_vector
            arr = arr.reshape( list(shape) + list(vector.shape) )
        else:
            arr = np.ndarray((m, n), dtype=vector.dtype)
            for i in range(n):
                arr[:,i] = flat_vector
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
