"""Module that deals with the calculations of likelihoods."""

import numpy as np
from scipy import optimize, stats


def derivative(func, x0, dx=1.0e-6, n=1, args=(), order=3):
    """Calculate the n-th derivative of a function at a point.

    This is a replacement for the deprecated scipy.misc.derivative.
    """
    # Handle array dx properly - treat each element separately
    if hasattr(dx, "__iter__") and not isinstance(dx, str):
        dx = np.asarray(dx)
        if dx.size == 1:
            dx = float(dx)
        else:
            # For array dx, we need to handle each derivative separately
            # Use vectorized approach for efficiency
            x0 = np.asarray(x0)
            result = np.empty_like(x0, dtype=float)
            for i, h in enumerate(dx):
                x0_i = x0[i] if x0.ndim > 0 else x0
                result[i] = (func(x0_i + h, *args) - func(x0_i - h, *args)) / (2 * h)
            return result

    if n == 1:
        # Use central difference for first derivative
        h = dx
        return (func(x0 + h, *args) - func(x0 - h, *args)) / (2 * h)
    else:
        # For higher derivatives, use finite differences with central difference weights
        h = dx
        if n == 2 and order >= 2:
            # Second derivative central difference
            return (func(x0 + h, *args) - 2 * func(x0, *args) + func(x0 - h, *args)) / (
                h**2
            )
        else:
            # Fall back to forward difference for other cases
            points = [x0 + i * h for i in range(order + 1)]
            values = [func(p, *args) for p in points]
            # Simple finite difference approximation
            result = 0.0
            for i in range(1, min(n + 1, len(values))):
                result += ((-1) ** (i + 1)) * values[i] / (h**i)
            return result


# Use this function/object for parallelization where possible
mapper = map

# Empty slices
_SLICE = slice(None)


class JeffreysPrior:
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

    Attributes
    ----------

    response_matrix : ndarray
        The response matrix used to calculate the Fisher matrix.
    default_values : ndarray
        The default values of the parameters.
    lower_limits : ndarray
        Lower limits of the parameters.
    upper_limits : ndarray
        Upper limits of the parameters.
    total_truth_limit : float
        The upper limit of total number of true events.
    dx : ndarray
        Array of step sizes to be used in numerical differentiation.

    """

    def __init__(
        self,
        response_matrix,
        translation_function,
        parameter_limits,
        default_values,
        dx=None,
        total_truth_limit=None,
    ):
        old_shape = response_matrix.shape
        new_shape = (int(np.prod(old_shape[:-2])), old_shape[-2], old_shape[-1])
        self.response_matrix = response_matrix.reshape(new_shape)

        self._translate = translation_function

        limits = list(zip(*parameter_limits))
        self.lower_limits = np.array(
            [x if x is not None else -np.inf for x in limits[0]]
        )
        self.upper_limits = np.array(
            [x if x is not None else np.inf for x in limits[1]]
        )

        self.total_truth_limit = total_truth_limit or np.inf

        self.default_values = np.array(default_values)

        self._npar = len(parameter_limits)
        self._nreco = response_matrix.shape[-2]
        self._i_diag = np.diag_indices(self._npar)
        self.dx = dx or np.full(self._npar, 1e-3)

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

    def fisher_matrix(self, parameters, toy_index=0):
        """Calculate the Fisher information matrix for the given parameters.

        Parameters
        ----------

        parameters : array like
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
            ret = np.tensordot(self.translate(theta), resp, ([-1], [-1]))
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
        with np.errstate(under="ignore"):
            sign, log_det = np.linalg.slogdet(fish)

        return 0.5 * log_det


class DataModel:
    """Base class for representation of data statistical models.

    The resulting object can be called like a function::

        log_likelihood = likelihood_calculator(prediction)

    Parameters
    ----------

    data_vector : array_like
        Shape: ``([a,b,...,]n_reco_bins,)``

    Attributes
    ----------

    data_vector : ndarray
        The data vector ``k``.

    """

    def __init__(self, data_vector):
        self.data_vector = np.asarray(data_vector)

    def log_likelihood(self, reco_vector):
        """Calculate the likelihood of the provided expectation values.

        The reco vector can have a shape ``([c,d,]n_reco_bins,)``. Assuming the
        data is of shape ``([a,b,...,]n_reco_bins,)``, the output will be of
        shape ``([a,b,...,][c,d,...,])``.

        """

        raise NotImplementedError("Must be implemented in a subclass!")

    @classmethod
    def generate_toy_data(cls, reco_vector, size=None):
        """Generate toy data according to the expectation values.

        The reco vector can have a shape ``([c,d,]n_reco_bins,)``. Assuming the
        requested size is ``(a,b,...,)``, the output will be of shape
        ``([a,b,...,][c,d,...,])``.

        """

        raise NotImplementedError("Must be implemented in a subclass!")

    @classmethod
    def generate_toy_data_model(cls, *args, **kwargs):
        """Generate toy data model according to the expectation values.

        All arguments are passed to :meth:`generate_toy_data`.

        """

        return cls(cls.generate_toy_data(*args, **kwargs))

    def __call__(self, *args, **kwargs):
        return self.log_likelihood(*args, **kwargs)


class PoissonData(DataModel):
    """Class for fast Poisson likelihood calculations.

    The resulting object can be called like a function::

        log_likelihood = likelihood_calculator(prediction)

    Parameters
    ----------

    data_vector : array_like int
        Shape: ``([a,b,...,]n_reco_bins,)``

    Notes
    -----

    Assumes that data does not change to avoid re-calculating things::

        Poisson PMF:
        p(k, mu) = (mu^k / k!) * exp(-mu)
        ln(p(k, mu)) = k*ln(mu) - mu - ln(k!)
        ln(k!) = -ln(p(k,mu)) + k*ln(mu) - mu = -ln(p(k,1.)) - 1.

    Attributes
    ----------

    data_vector : ndarray
        The data vector ``k``.
    k0 : ndarray
        Mask where ``k == 0``.
    ln_k_factorial : ndarray
        The precomputed ``ln(k!)``.

    """

    def __init__(self, data_vector):
        # Save constants for PMF calculation
        with np.errstate(divide="ignore", invalid="ignore"):
            self.data_vector = np.asarray(data_vector, dtype=int)
            k = self.data_vector
            self.k0 = k == 0
            self.ln_k_factorial = -(
                stats.poisson.logpmf(k, np.ones_like(k, dtype=float)) + 1.0
            )

    @staticmethod
    def _poisson_logpmf(k, k0, ln_k_factorial, mu):
        with np.errstate(divide="ignore", invalid="ignore"):
            pmf = k * np.log(mu) - ln_k_factorial - mu
        # Need to take care of special case mu=0:  k=0 -> logpmf=0,  k>=1 -> logpmf=-inf
        mu0 = mu == 0
        pmf[mu0 & k0] = 0
        pmf[mu0 & ~k0] = -np.inf
        pmf[~np.isfinite(pmf)] = -np.inf
        pmf = np.sum(pmf, axis=-1)
        return pmf

    def log_likelihood(self, reco_vector):
        """Calculate the likelihood of the provided expectation values.

        The reco vector can have a shape ``([c,d,]n_reco_bins,)``. Assuming the
        data is of shape ``([a,b,...,]n_reco_bins,)``, the output will be of
        shape ``([a,b,...,][c,d,...,])``.

        """

        reco_vector = np.asarray(reco_vector, dtype=float)

        data_shape = self.data_vector.shape  # = ([a,b,...,]n_reco_bins,)
        reco_shape = reco_vector.shape  # = ([c,d,...,]n_reco_bins,)

        # Cast vectors to the shape ([a,b,...,][c,d,...,]n_reco_bins,)
        data_index = (
            (_SLICE,) * (len(data_shape) - 1)
            + (np.newaxis,) * (len(reco_shape) - 1)
            + (_SLICE,)
        )
        reco_index = (
            (np.newaxis,) * (len(data_shape) - 1)
            + (_SLICE,) * (len(reco_shape) - 1)
            + (_SLICE,)
        )

        # Calculate the log probabilities of shape ([a,b,...,][c,d,...,]).
        return self._poisson_logpmf(
            self.data_vector[data_index],
            self.k0[data_index],
            self.ln_k_factorial[data_index],
            reco_vector[reco_index],
        )

    @classmethod
    def generate_toy_data(cls, reco_vector, size=None):
        """Generate toy data according to the expectation values.

        The reco vector can have a shape ``([c,d,]n_reco_bins,)``. Assuming the
        requested size is ``(a,b,...,)``, the output will be of shape
        ``([a,b,...,][c,d,...,])``.

        """

        mu = reco_vector
        if size is not None:
            # Append truth vector shape to requested shape of data sets
            try:
                shape = list(size)
            except TypeError:
                shape = [size]
            shape.extend(mu.shape)
            size = shape

        data = np.random.poisson(mu, size=size)
        return data


class SystematicsConsumer:
    """Class that consumes the systematics axis on an array of log likelihoods."""

    @staticmethod
    def consume_axis(log_likelihood, weights=None):
        """Collapse the systematic axes according to the systematics mode."""
        raise NotImplementedError("Must be implemented in a subclass!")

    def __call__(self, *args, **kwargs):
        return self.consume_axis(*args, **kwargs)


class NoSystematics(SystematicsConsumer):
    """SystematicsConsumer that does nothing."""

    @staticmethod
    def consume_axis(log_likelihood, weights=None):
        return log_likelihood


class MarginalLikelihoodSystematics(SystematicsConsumer):
    """SystematicsConsumer that averages over the systematic axis.

    Optionally applies weights.

    """

    @staticmethod
    def consume_axis(log_likelihood, weights=None):
        if weights is None:
            weights = np.ones_like(log_likelihood)
        log_weights = np.log(weights / np.sum(weights, axis=-1, keepdims=True))
        weighted = log_likelihood + log_weights
        # Avoid numerical problems by using this "trick"
        max_weighted = np.max(weighted, axis=-1, keepdims=True)
        i_inf = ~np.isfinite(max_weighted)
        max_weighted[i_inf] = 0.0
        weighted = weighted - max_weighted
        with np.errstate(under="ignore"):
            ret = max_weighted[..., 0] + np.logaddexp.reduce(weighted, axis=-1)
        return ret


class ProfileLikelihoodSystematics(SystematicsConsumer):
    """SystematicsConsumer that maximises over the systematic axes."""

    @staticmethod
    def consume_axis(log_likelihood, weights=None):
        return np.max(log_likelihood, axis=-1)


class Predictor:
    """Base class for objects that turn sets of parameters to predictions.

    Parameters
    ----------

    bounds : ndarray
        Lower and upper bounds for all parameters. Can be ``+/- np.inf``.
    defaults : ndarray
        "Reasonable" default values for the parameters. Used in optimisations.

    Attributes
    ----------

    bounds : ndarray
        Lower and upper bounds for all parameters. Can be ``+/- np.inf``.
    defaults : ndarray
        "Reasonable" default values for the parameters. Used in optimisations.

    """

    def __init__(self, bounds, defaults):
        self.bounds = np.asarray(bounds)
        self.defaults = np.asarray(defaults)

    def check_bounds(self, parameters):
        """Check that all parameters are within bounds."""
        parameters = np.asarray(parameters, dtype=float)
        check = (parameters >= self.bounds[:, 0]) & (parameters <= self.bounds[:, 1])
        return np.all(check, axis=-1)

    def prediction(self, parameters, systematics_index=_SLICE):
        """Turn a set of parameters into an ndarray of reco predictions.

        Parameters
        ----------

        parameters : ndarray
        systematics_index : int, optional

        Returns
        -------

        prediction, weights : ndarray

        Notes
        -----

        The regular output must have at least two dimensions. The last
        dimension corresponds to the number of predictions, e.g. reco bin
        number. The second to last dimension corresponds to systematic
        prediction uncertainties of a single parameter set.

        Parameters can be arbitrarily shaped ndarrays. The method must support
        turning a multiple sets of parameters into an array of predictions::

            parameters.shape == ([c,d,...,]n_parameters,)
            prediction.shape == ([c,d,...,]n_systematics,n_predictions,)
            weights.shape == ([c,d,...,]n_systematics,)

        If the `systematics_index` is specified, only the respective value on
        the systematics axis should be returned::

            prediction.shape == ([c,d,...,]n_predictions,)
            weights.shape == ([c,d,...,],)

        """

        raise NotImplementedError("This function must be implemented in a subclass!")

    def fix_parameters(self, fix_values):
        """Return a new Predictor with fewer free parameters.

        Parameters
        ----------

        fix_values : iterable
            List of the parameter values that the parameters should be fixed at.
            The list must be of the same length as the number of parameters of `predictor`.
            Any parameters that should remain unfixed should be specified with ``None`` or
            ``np.nan``.

        See also
        --------

        FixedParameterPredictor

        """

        return FixedParameterPredictor(self, fix_values)

    def compose(self, other):
        """Return a new Predictor that is a composition with `other`.

        ::

            new_predictor(parameters) = self(other(parameters))

        """

        return ComposedPredictor([self, other])

    def __call__(self, *args, **kwargs):
        """See :meth:`prediction`."""
        return self.prediction(*args, **kwargs)


class ComposedPredictor(Predictor):
    """Wrapper class that composes different Predictors into one.

    Parameters
    ----------

    predictors : list of Predictor
        The Predictors will be composed in turn. The second must predict
        parameters for the first, the third for the second, etc. The last
        Predictor defines what parameters will be accepted by the resulting
        Predictor.
    combine_systematics : string, optional
        The strategy how to combine the systematics of the Predictors.
        Default: "cartesian"

    Notes
    -----

    Systematics will be handled according to the `combine_systematics`
    parameter. Possible values are:

    ``"cartesian"``
        Combine systematics in a Cartesian product. There will be one
        output prediction for each possible combination of intermediate
        systematics.

    ``"same"``
        Assume that the systamtics are the same and no combination is done.
        The dimensions and weights for the systematics _must_ be identical
        for all provided Predictors.

    """

    def __init__(self, predictors, combine_systematics="cartesian"):
        self.predictors = predictors

        # TODO: Currently taking the bounds from the last predictor
        # This might still run into the bounds of intermediate predictors
        self.bounds = predictors[-1].bounds
        self.defaults = predictors[-1].defaults

        self.combine_systematics = combine_systematics

    def _compose_cartesian(self, parameters, systematics_index):
        """Apply cartesian combination of systematics."""

        parameters = np.asarray(parameters)
        orig_shape = parameters.shape
        orig_len = len(orig_shape)
        weights = np.ones(parameters.shape[:-1])

        for pred in reversed(self.predictors):
            parameters, w = pred.prediction(parameters)
            weights = weights[..., np.newaxis] * w

        # Flatten systematics
        # original_parameters.shape = ([c,d,...,]n_parameters)
        # chained_output.shape = ([c,d,...,]systn,...,syst0,n_reco)
        # reordered_output.shape = ([c,d,...,]syst0,...,systn,n_reco)
        # desired_output.shape = ([c,d,...,]syst,n_reco)
        shape = parameters.shape
        shape_len = len(shape)
        new_order = (
            tuple(range(orig_len - 1))  # c,d,...
            + tuple(range(shape_len - 2, orig_len - 2, -1))  # syst0,syst1,...
            + (shape_len - 1,)  # n_reco
        )
        parameters = np.transpose(parameters, new_order)
        parameters = parameters.reshape(
            orig_shape[:-1] + (np.prod(shape[orig_len - 1 : -1]),) + shape[-1:]
        )

        #           c,d,...                    syst0,syst1,...
        new_order = tuple(range(orig_len - 1)) + tuple(
            range(shape_len - 2, orig_len - 2, -1)
        )
        weights = np.transpose(weights, new_order)
        weights = weights.reshape(
            orig_shape[:-1] + (np.prod(shape[orig_len - 1 : -1]),)
        )

        return parameters[..., systematics_index, :], weights[..., systematics_index]

    def _compose_same(self, parameters, systematics_index):
        """Apply 'same' combination of systematics."""

        parameters = np.asarray(parameters)

        for i, pred in enumerate(reversed(self.predictors)):
            if isinstance(systematics_index, int):
                # Just use a single systematic index
                parameters, w = pred.prediction(
                    parameters, systematics_index=systematics_index
                )
                continue

            # Need to handle all systematics
            if i == 0:
                # First predictor
                # Just do the regular thing
                parameters, w = pred.prediction(
                    parameters, systematics_index=systematics_index
                )
            else:
                # Second and on, need to make sure to apply the right systematics
                # to the right parameters
                temp_pars = []
                for s in range(parameters.shape[-2]):
                    # Loop over all systematic indices
                    p, _ = pred.prediction(parameters[..., s, :], systematics_index=s)
                    temp_pars.append(p[..., np.newaxis, :])  # Insert systematics axis
                parameters = np.concatenate(temp_pars, axis=-2)

        weights = w
        return parameters, weights

    def prediction(self, parameters, systematics_index=_SLICE):
        """Turn a set of parameters into an ndarray of predictions.

        Parameters
        ----------

        parameters : ndarray
        systematics_index : int, optional

        Returns
        -------

        prediction, weights : ndarray

        Notes
        -----

        The optional argument `systematics_index` will be applied to the final
        output of the composed predictions, e.g. the flattened Cartesian
        product of the intermediate systematics if the combination strategy is
        "cartesian".

        """

        if self.combine_systematics == "cartesian":
            return self._compose_cartesian(parameters, systematics_index)
        elif self.combine_systematics == "same":
            return self._compose_same(parameters, systematics_index)
        else:
            raise ValueError(
                "%s is not a valid systematics combination strategy."
                % (self.combine_systematics)
            )


class SummedPredictor(Predictor):
    """Wrapper class that sums predictions of multiple Predictors.

    Parameters
    ----------

    predictors : list of Predictor
        The Predictors whos output will be added. The resulting Predictor will
        accept the concatenated list of the separate original input parameters
        in the same order as they appear in the list.
    scale_factors : iterable of float or arrays: optional
        Provide a scaling factor for each predictor. The output will be
        multiplied with the factor before summing up the predictions.
        Each factor can also be an array of factors so that each systematic
        variation gets a different scaling factor. But it needs to be applicable
        by multiplication, i.e. ``scaled_out = factor * out``. This can be achieved
        by giving each matrix the shape ``(n_systematics, 1)``.
    combine_systematics : string, optional
        The strategy how to combine the systematics of the Predictors.
        Default: "cartesian"

    Notes
    -----

    Systematics will be handled according to the `combine_systematics`
    parameter. Possible values are:

    ``"cartesian"``
        Combine systematics in a Cartesian product. There will be one
        output prediction for each possible combination of intermediate
        systematics.

    ``"same"``
        Assume that the systamtics are the same and no combination is done.
        The dimensions and weights for the systematics _must_ be identical
        for all provided Predictors.

    See also
    --------

    ConcatenatedPredictor

    """

    def __init__(self, predictors, scale_factors=None, combine_systematics="cartesian"):
        self.predictors = predictors

        self.bounds = np.concatenate([p.bounds for p in predictors], axis=0)
        self.defaults = np.concatenate([p.defaults for p in predictors], axis=0)
        self.scale_factors = scale_factors
        self.combine_systematics = combine_systematics

    def prediction(self, parameters, systematics_index=_SLICE):
        """Turn a set of parameters into an ndarray of predictions.

        Parameters
        ----------

        parameters : ndarray
        systematics_index : int, optional

        Returns
        -------

        prediction, weights : ndarray

        Notes
        -----

        The optional argument `systematics_index` will be applied to the final
        output of the summed predictions, i.e. the flattened combination of the
        original systematics.

        """

        parameters = np.asarray(parameters)
        orig_shape = parameters.shape
        orig_len = len(orig_shape)
        weights = np.ones(parameters.shape[:-1])

        prediction = np.array([0.0])
        i_par = 0

        # Get the scale factors or generate ones
        scale_factors = self.scale_factors
        if scale_factors is None:
            scale_factors = np.ones_like(self.predictors)

        if len(scale_factors) != len(self.predictors):
            raise ValueError(
                "Must provide same number of predictors and scale factors!"
            )

        for pred, factor in zip(self.predictors, scale_factors):
            # Consume the given parameters in order.
            j_par = i_par + len(pred.defaults)
            par = parameters[..., i_par:j_par]
            i_par = j_par
            p, w = pred.prediction(par)
            p = np.asarray(factor) * p

            if self.combine_systematics == "cartesian":
                # Sum up predictions and create new axis for systematics
                # p.shape = ([c,d,...,]syst_n+1,n_reco)
                # prediction.shape = ([c,d,...,]syst_0,...,syst_n-1,n_reco)
                # new.shape = ([c,d,...,]syst_0,...,syst_n-1,syst_n,n_reco)
                ndim_diff = prediction.ndim - p.ndim
                slices = (
                    (Ellipsis,) + (np.newaxis,) * (ndim_diff + 1) + (slice(None),) * 2
                )
                p = p[slices]
                prediction = prediction[..., np.newaxis, :] + p

                slices = (Ellipsis,) + (np.newaxis,) * (ndim_diff + 1) + (slice(None),)
                w = w[slices]
                weights = weights[..., np.newaxis] * w
            elif self.combine_systematics == "same":
                # All weights and predictions have the same shape
                # Weights have the same values
                prediction = prediction + p
                weights = w
            else:
                raise ValueError(
                    "%s is not a valid systematics combination strategy."
                    % (self.combine_systematics)
                )

        # Flatten output
        # original_parameters.shape = ([c,d,...,]n_parameters)
        # summed_output.shape = ([c,d,...,]syst0,...,systn,n_reco)
        # desired_output.shape = ([c,d,...,]syst,n_reco)
        shape = prediction.shape
        prediction = prediction.reshape(
            orig_shape[:-1] + (np.prod(shape[orig_len - 1 : -1]),) + shape[-1:]
        )
        weights = weights.reshape(
            orig_shape[:-1] + (np.prod(shape[orig_len - 1 : -1]),)
        )

        return prediction[..., systematics_index, :], weights[..., systematics_index]


class ConcatenatedPredictor(Predictor):
    """Wrapper class that concatenates predictions of multiple Predictors.

    Parameters
    ----------

    predictors : list of Predictor
        The Predictors whos output will be concateanted.
    share_parameters : bool, default=False
        If ``True``, the `predictors` must accept the same number of
        parameters. The resulting predictor will then accept the same number
        and apply them to all. Otherwise the resulting Predictor will accept
        the concatenated list of the separate original input parameters in the
        same order as they appear in `predictors`.
    combine_systematics : string, optional
        The strategy how to combine the systematics of the Predictors.
        Default: "cartesian"

    Notes
    -----

    Systematics will be handled according to the `combine_systematics`
    parameter. Possible values are:

    ``"cartesian"``
        Combine systematics in a Cartesian product. There will be one
        output prediction for each possible combination of intermediate
        systematics.

    ``"same"``
        Assume that the systamtics are the same and no combination is done.
        The dimensions and weights for the systematics _must_ be identical
        for all provided Predictors.

    See also
    --------

    SummedPredictor

    """

    def __init__(
        self, predictors, share_parameters=False, combine_systematics="cartesian"
    ):
        self.predictors = predictors

        if share_parameters:
            self.bounds = predictors[0].bounds
            self.defaults = predictors[0].defaults
        else:
            self.bounds = np.concatenate([p.bounds for p in predictors], axis=0)
            self.defaults = np.concatenate([p.defaults for p in predictors], axis=0)
        self.share_parameters = share_parameters
        self.combine_systematics = combine_systematics

    def _concatenate_cartesian(self, parameters):
        """Calculate predictions using the "cartesian" systematics strategy."""

        orig_shape = parameters.shape

        weights = np.ones(parameters.shape[:-1])
        predictions = []
        i_par = 0

        for pred in self.predictors:
            if not self.share_parameters:
                # Consume the given parameters in order.
                j_par = i_par + len(pred.defaults)
                par = parameters[..., i_par:j_par]
                i_par = j_par
            else:
                par = parameters

            p, w = pred.prediction(par)
            predictions.append(p)

            # Bring weights in right shape
            # w.shape = ([c,d,...,]syst_n)
            # weights.shape = ([c,d,...,]syst_0*...*syst_n-1)
            # new.shape = ([c,d,...,]syst_0*...*syst_n-1*syst_n)
            ndim_diff = weights.ndim - w.ndim
            slices = (Ellipsis,) + (np.newaxis,) * (ndim_diff + 1) + (slice(None),)
            w = w[slices]
            weights = weights[..., np.newaxis] * w
            weights = weights.reshape(orig_shape[:-1] + (-1,))

        # Concatenate predictions with right systematic axes
        # pred shapes = ([c,d,...,]syst_i,n_reco_i)
        # desired output shape = ([c,d,...,]syst0*...*systn,n_reco)

        prediction = predictions[0]
        for i in range(1, len(predictions)):
            # prediction.shape = ([c,d,...,]syst_X,n_reco_X)
            # p1.shape = ([c,d,...,]syst_n,n_reco_n)
            # intermediate.shape = ([c,d,...,]sys_X,syst_n,n_reco)
            # final.shape = ([c,d,...,]sys_X*syst_n,n_reco)
            p1 = predictions[i]
            p0 = np.broadcast_to(
                prediction[..., np.newaxis, :],
                orig_shape[:-1]
                + (prediction.shape[-2], p1.shape[-2], prediction.shape[-1]),
            )
            p1 = np.broadcast_to(
                p1[..., np.newaxis, :, :],
                orig_shape[:-1] + (prediction.shape[-2], p1.shape[-2], p1.shape[-1]),
            )
            prediction = np.concatenate([p0, p1], axis=-1)
            prediction = prediction.reshape(
                orig_shape[:-1] + (-1, prediction.shape[-1])
            )

        return prediction, weights

    def _concatenate_same(self, parameters):
        """Calculate predictions using the "same" systematics strategy."""

        weights = np.ones(parameters.shape[:-1])
        predictions = []
        i_par = 0

        for pred in self.predictors:
            if not self.share_parameters:
                # Consume the given parameters in order.
                j_par = i_par + len(pred.defaults)
                par = parameters[..., i_par:j_par]
                i_par = j_par
            else:
                par = parameters

            p, w = pred.prediction(par)
            predictions.append(p)
            weights = w

        prediction = np.concatenate(predictions, axis=-1)

        return prediction, weights

    def prediction(self, parameters, systematics_index=_SLICE):
        """Turn a set of parameters into an ndarray of predictions.

        Parameters
        ----------

        parameters : ndarray
        systematics_index : int, optional

        Returns
        -------

        prediction, weights : ndarray

        Notes
        -----

        The optional argument `systematics_index` will be applied to the final
        output of the summed predictions, i.e. the flattened combination of the
        original systematics.

        """

        parameters = np.asarray(parameters)

        if self.combine_systematics == "cartesian":
            prediction, weights = self._concatenate_cartesian(parameters)
        elif self.combine_systematics == "same":
            prediction, weights = self._concatenate_same(parameters)
        else:
            raise ValueError(
                "%s is not a valid systematics combination strategy."
                % (self.combine_systematics)
            )

        return prediction[..., systematics_index, :], weights[..., systematics_index]


class FixedParameterPredictor(Predictor):
    """Wrapper class that fixes parameters of another predictor.

    The resulting `Predictor` will have fewer parameters, namely those that have
    not been fixed.

    Paramters
    ---------

    predictor : Predictor
        The original predictor which will have some of its parameters fixed.
    fix_values : iterable
        List of the parameter values that the parameters should be fixed at.
        The list must be of the same length as the number of parameters of `predictor`.
        Any parameters that should remain unfixed should be specified with ``None`` or
        ``np.nan``.

    """

    def __init__(self, predictor, fix_values):
        self.predictor = predictor
        self.fix_values = np.array(fix_values, dtype=float)
        insert_mask = ~np.isnan(self.fix_values)
        insert_indices = np.argwhere(insert_mask).flatten()
        self.insert_values = self.fix_values[insert_indices]
        # Indices must be provided as indices on the array with the missing values
        self.insert_indices = insert_indices - np.arange(insert_indices.size)

        self.bounds = predictor.bounds[~insert_mask]
        self.defaults = predictor.defaults[~insert_mask]

    def insert_fixed_parameters(self, parameters):
        """Insert the fixed parameters into a vector of free parameters."""
        parameters = np.asarray(parameters)
        return np.insert(parameters, self.insert_indices, self.insert_values, axis=-1)

    def prediction(self, parameters, systematics_index=_SLICE):
        """Turn a set of parameters into an ndarray of predictions.

        Parameters
        ----------

        parameters : ndarray
        systematics_index : int, optional

        Returns
        -------

        prediction, weights : ndarray

        """

        parameters = self.insert_fixed_parameters(parameters)
        return self.predictor(parameters, systematics_index)


class LinearEinsumPredictor(Predictor):
    """Predictor calcuating a linear combination of the paramters using `np.einsum`.

    Paramters
    ---------

    subscripts : str
        The subscripts string provided to `np.einsum`.
    coefficients : ndarray or tuple of
        The first operand to be supplies to `np.einsum`.
        Shape: ``(n_systematics, [I, J, ...,] K)``
    constants : ndarray, optional
        Constants to be added to the result of `np.einsum` after flattening it.
        Shape: ``([n_systematics,] n_output)``
    weights : ndarray, optional
        Shape: ``([n_systematics,])``
    bounds : ndarray, optional
        Lower and upper bounds for all parameters. Can be ``+/- np.inf``.
    defaults : ndarray, optional
        "Reasonable" default values for the parameters. Used in optimisation.
    reshape_parameters : tuple of int, optional
        Reshape the last axis of the parameters to the given shape before
        the einsum contraction.

    Notes
    -----

    At least one of `bounds`, `defaults`, or `reshape_parameters` must be
    provided in order to calculate the expected number of parameters.

    The prediction will be calculated using `np.einsum` using the provided
    `subscripts` string. The two operands will be the provided `coefficients`
    as well as the parameters of the prediction function call.

    The parameters will have the shape ``[(A,B,...,)+]reshape_parameters``, if
    `reshape_parameters` is set, or ``([A,B,...,]n_parameters)`` otherwise.

    The output of the `np.einsum` _must_ have the shape
    ``([A,B,...,]n_systematics,[X,Y,...,]Z)``, no matter whethre the dimensions
    ``A, B, ...`` actually exist.

    Examples
    --------

    This predictor will calculate the product of the parameters with the
    systematically varied matrices:

    >>> M = [ [[1,0], [0,1]], [[0.9, 0], [0, 0.8]] ]
    >>> p = LinearEinsumPredictor("ijk,...k->...ij", M, defaults=[1,1])
    >>> p([1.0, 2.0])
    (array([[1. , 2. ],
        [0.9, 1.6]]),
     array([1., 1.]))


    Interpret parameters as matrix and do a matrix multiplication:

    >>> M = [ [[1,0], [0,1]], [[0.9, 0], [0, 0.8]] ]
    >>> p = LinearEinsumPredictor("ijk,...kl->...ijl", M, reshape_parameters=(2,2))
    >>> p([1.0, 2.0, 3.0, 4.0])
    (array([[1. , 2. , 3. , 4. ],
        [0.9, 1.8, 2.4, 3.2]]),
     array([1., 1.]))

    Do an element-wise multiplication:

    >>> M = [ [1, 2, 3, 4] ]
    >>> p = LinearEinsumPredictor("ij,...j->...ij", M, reshape_parameters=(4,))
    >>> p([1.0, 1.0, 1.0, 1.0])
    (array([[1., 2., 3., 4.]]), array([1.]))

    """

    def __init__(
        self,
        subscripts,
        coefficients,
        constants=0.0,
        weights=1.0,
        bounds=None,
        defaults=None,
        reshape_parameters=None,
    ):
        self.subscripts = subscripts
        self.coefficients = np.asarray(coefficients, dtype=float)
        self.constants = np.asarray(constants, dtype=float)
        while self.constants.ndim < 2:
            self.constants = self.constants[np.newaxis, ...]
        self.weights = np.asarray(weights, dtype=float)
        while self.weights.ndim < 1:
            self.weights = self.weights[np.newaxis, ...]

        if bounds is not None:
            n_parameters = len(bounds)
        elif defaults is not None:
            n_parameters = len(defaults)
        elif reshape_parameters is not None:
            n_parameters = np.prod(reshape_parameters)
        else:
            raise RuntimeError(
                "At least on of `bounds`, `defaults`, or `reshape_parameters` must be provided!"
            )

        if bounds is None:
            bounds = np.array([(-np.inf, np.inf)] * n_parameters)
        if defaults is None:
            defaults = np.array([1.0] * n_parameters)
        self.reshape_parameters = reshape_parameters
        Predictor.__init__(self, bounds, defaults)

    def prediction(self, parameters, systematics_index=_SLICE):
        """Turn a set of parameters into a reco prediction.

        Returns
        -------

        prediction : ndarray
        weights : ndarray

        """

        if isinstance(systematics_index, int):
            if systematics_index >= len(self.coefficients):
                raise IndexError("Systematic index is out of bounds.")
            systematics_index = slice(systematics_index, systematics_index + 1)
            remove_systematics = True
        else:
            remove_systematics = False

        coefficients = self.coefficients[systematics_index]

        weights = self.weights
        if len(weights) > 1:
            # If there is only one, it applies to all systematics
            weights = weights[systematics_index]

        constants = self.constants
        if len(constants) > 1:
            # If there is only one, it applies to all systematics
            constants = constants[systematics_index]

        # Get parameters as vector
        parameters = np.asarray(parameters)
        orig_shape = parameters.shape
        # Reshape parameters if requested
        if self.reshape_parameters is not None:
            parameters = parameters.reshape(orig_shape[:-1] + self.reshape_parameters)
        # Calculate einsum
        prediction = np.einsum(self.subscripts, coefficients, parameters)
        # Flatten output
        shape = prediction.shape[: len(orig_shape)] + (-1,)
        prediction = prediction.reshape(shape)
        # Apply constants
        prediction = prediction + constants
        # Reshape weights
        weights = np.broadcast_to(weights, prediction.shape[:-1])
        if remove_systematics:
            prediction.shape = prediction.shape[:-2] + prediction.shape[-1:]
            weights.shape = weights.shape[:-1]
        return prediction, weights


class MatrixPredictor(LinearEinsumPredictor):
    """Predictor that uses a matrix to fold parameters into reco space.

    ::

        output = matrix X parameters [+ constants]

    Parameters
    ----------

    matrices : ndarray
        Shape: ``([n_systematics,]n_reco_bins,n_parameters)``
    constants : ndarray, optional
        Shape: ``([n_systematics,]n_reco_bins)``
    weights : ndarray, optional
        Shape: ``(n_systematics)``
    bounds : ndarray, optional
        Lower and upper bounds for all parameters. Can be ``+/- np.inf``.
    defaults : ndarray, optional
        "Reasonable" default values for the parameters. Used in optimisation.
    sparse_indices : list or array of int, optional
        Used with sparse matrices that provide only the specified columns.
        All other columns are assumed to be 0, i.e. the parameters corresponding
        to these have no effect.

    See also
    --------

    Predictor

    Attributes
    ----------

    bounds : ndarray
        Lower and upper bounds for all parameters. Can be ``+/- np.inf``.
    defaults : ndarray
        "Reasonable" default values for the parameters. Used optimisation.
    sparse_indices : list or array of int or slice
        Used with sparse matrices that provide only the specified columns.
        All other columns are assumed to be 0, i.e. the parameters corresponding
        to these have no effect.

    """

    def __init__(self, matrices, constants=0.0, *, sparse_indices=_SLICE, **kwargs):
        self.matrices = np.asarray(matrices, dtype=float)
        while self.matrices.ndim < 3:
            self.matrices = self.matrices[np.newaxis, ...]
        self.sparse_indices = sparse_indices
        LinearEinsumPredictor.__init__(
            self,
            "ijk,...k->...ij",
            self.matrices,
            constants=constants,
            reshape_parameters=self.matrices.shape[-1:],
            **kwargs,
        )

    def prediction(self, parameters, *args, **kwargs):
        """Turn a set of parameters into a reco prediction.

        Returns
        -------

        prediction : ndarray
        weights : ndarray

        """
        # Get parameters as vector
        parameters = np.asarray(parameters)
        # Account for spare matrices
        sparse_parameters = parameters[..., self.sparse_indices]
        pred, weight = LinearEinsumPredictor.prediction(
            self, sparse_parameters, *args, **kwargs
        )
        return pred, weight

    def compose(self, other):
        """Return a new Predictor that is a composition with `other`.

        ::

            new_predictor(parameters) = self(other(parameters))

        """

        if isinstance(other, MatrixPredictor):
            return ComposedMatrixPredictor([self, other])
        else:
            return Predictor.compose(self, other)

    def fix_parameters(self, fix_values):
        """Return a new Predictor with fewer free parameters.

        Parameters
        ----------

        fix_values : iterable
            List of the parameter values that the parameters should be fixed at.
            The list must be of the same length as the number of parameters of `predictor`.
            Any parameters that should remain unfixed should be specified with ``None`` or
            ``np.nan``.

        See also
        --------

        FixedParameterMatrixPredictor

        """

        return FixedParameterMatrixPredictor(self, fix_values)


class ComposedMatrixPredictor(MatrixPredictor, ComposedPredictor):
    """Composition of MatrixPredictors.

    Parameters
    ----------

    predictors : list of Predictor
        The Predictors will be composed in turn. The second must predict
        parameters for the first, the third for the second, etc. The last
        Predictor defines what parameters will be accepted by the resulting
        Predictor.
    evaluation_point : array_like float, optional
        Shape: ``(n_parameters,)``
        Anchor point for the linearisation.
    evaluation_steps : array_like float, optional
        Shape: ``(n_parameters,)``
        Step away from the anchor point for the linearisation.
    combine_systematics : string, optional
        The strategy how to combine the systematics of the Predictors.
        Default: "cartesian"

    Notes
    -----

    This :class:`MatrixPredictor` will calculate a linear coefficients and
    offsets by evaluating the composed provided predictors at
    `evaluation_point`. The gradient at that point will be calculated by adding
    the `evaluation_steps` for each parameter separately.

    For a composition of multiple :class:`MatrixPredictor`, this will still
    lead to an exact representation (modulo variations from numerical
    accuracy). For non-linear, generic :class:`Predictor`, this means we are
    generating a linear approximation at `evaluation_point`.

    If no `evaluation_point` or `evaluation_steps` is provided, they will be
    generated from the last predictor's defaults and bounds.

    .. warning:: The auto-generated steps should work for linear functions,
                 but might be completely unsuitable, i.e. too large, for linear
                 approximations of general functions.

    Systematics will be handled according to the `combine_systematics`
    parameter. Possible values are:

    ``"cartesian"``
        Combine systematics in a Cartesian product. There will be one
        output prediction for each possible combination of intermediate
        systematics.

    ``"same"``
        Assume that the systamtics are the same and no combination is done.
        The dimensions and weights for the systematics _must_ be identical
        for all provided Predictors.

    See also
    --------

    MatrixPredictor
    ComposedPredictor

    """

    def __init__(
        self, predictors, evaluation_point=None, evaluation_steps=None, **kwargs
    ):
        # Init like ComposedPredictor
        ComposedPredictor.__init__(self, predictors, **kwargs)

        if evaluation_point is None:
            evaluation_point = self.defaults

        if evaluation_steps is None:
            evaluation_steps = (self.bounds[:, 1] - self.bounds[:, 0]) / 100.0

            # Use step size 1 if bounds are infinite
            evaluation_steps[~np.isfinite(evaluation_steps)] = 1.0

            # Flip steps that would go out of upper bound
            i = evaluation_point + evaluation_steps > self.bounds[:, 1]
            evaluation_steps[i] *= -1.0

        # Use methods of regular ComposedPredictor to calculate everything
        y0, weights = ComposedPredictor.prediction(self, evaluation_point)
        columns = []
        for i, dx in enumerate(evaluation_steps):
            par = np.array(evaluation_point)
            par[i] += dx
            y1, _ = ComposedPredictor.prediction(self, par)
            columns.append(((y1 - y0) / dx)[..., np.newaxis])

        matrices = np.concatenate(columns, axis=-1)
        constants = y0 - (matrices @ evaluation_point)

        MatrixPredictor.__init__(
            self,
            matrices,
            constants=constants,
            weights=weights,
            bounds=self.bounds,
            defaults=self.defaults,
            sparse_indices=_SLICE,
        )


def LinearizedPredictor(predictor, evaluation_point=None, evaluation_steps=None):
    """Thin wrapper to create a linearized predictor.

    Parameters
    ----------

    predictor : Predictor
        The predictor that should be linearized.
    evaluation_point : array_like float, optional
        Shape: ``(n_parameters,)``
        Anchor point for the linearisation.
    evaluation_steps : array_like float, optional
        Shape: ``(n_parameters,)``
        Step away from the anchor point for the linearisation.

    Returns
    -------

    predictor : ComposedMatrixPredictor

    Notes
    -----

    This is just a thin wrapper function to create a
    :class:`ComposedMatrixPredictor`. Instead of multiple :class:`Predictor`,
    it only accepts a single one.

    See also
    --------

    ComposedMatrixPredictor

    """

    return ComposedMatrixPredictor(
        [
            predictor,
        ],
        evaluation_point=evaluation_point,
        evaluation_steps=evaluation_steps,
    )


class FixedParameterMatrixPredictor(MatrixPredictor, FixedParameterPredictor):
    """Wrapper class that fixes parameters of a linear predictor.

    Speeds things up considerably compared to the universal `FixedParameterPredictor`,
    but only works with `MatrixPredictor`.

    Paramters
    ---------

    predictor : MatrixPredictor
        The original predictor which will have some of its parameters fixed.
    fix_values : iterable
        List of the parameter values that the parameters should be fixed at.
        The list must be of the same length as the number of parameters of `predictor`.
        Any parameters that should remain unfixed should be specified with ``None`` or
        ``np.nan``.

    See also
    --------

    FixedParameterPredictor
    MatrixPredictor

    """

    def __init__(self, predictor, fix_values):
        FixedParameterPredictor.__init__(self, predictor, fix_values)

        matrices = self.predictor.matrices[..., np.isnan(self.fix_values)]

        const_par = self.fix_values
        const_par[np.isnan(self.fix_values)] = 0.0
        constants, weights = self.predictor(const_par)

        MatrixPredictor.__init__(
            self,
            matrices,
            constants=constants,
            weights=weights,
            bounds=self.bounds,
            defaults=self.defaults,
            sparse_indices=_SLICE,
        )


class ResponseMatrixPredictor(MatrixPredictor):
    """Event rate predictor from ResponseMatrix objects.

    Parameters
    ----------

    filename : str or file object
        The exported information of a ResponseMatrix or
        ResponseMatrixArrayBuilder.

    """

    def __init__(self, filename):
        data = np.load(filename)
        matrices = data["matrices"]
        constants = 0.0
        weights = data.get("weights", 1.0)
        if data.get("is_sparse", False):
            sparse_indices = data["sparse_indices"]
        else:
            sparse_indices = _SLICE
        eps = np.finfo(
            float
        ).eps  # Add epsilon so there is a very small allowed range for empty bins
        bounds = [(0.0, x + eps) for x in data["truth_entries"]]
        defaults = data["truth_entries"] / 2.0
        MatrixPredictor.__init__(
            self,
            matrices,
            constants=constants,
            weights=weights,
            bounds=bounds,
            defaults=defaults,
            sparse_indices=sparse_indices,
        )


class TemplatePredictor(MatrixPredictor):
    """MatrixPredictor from templates.

    Parameters
    ----------

    templates : array like
        The templates to be combined together.
        Each template gets its own weight parameter.
        Shape: ``([n_systematics,]n_templates,len(template))``
    constants : ndarray, optional
        Shape: ``([n_systematics,]n_reco_bins)``
    **kwargs : optional
        Additional keyword arguments are passed to the MatrixPredictor.

    """

    def __init__(self, templates, constants=0, **kwargs):
        matrices = np.asarray(templates)
        matrices = np.array(np.swapaxes(matrices, -1, -2))
        bounds = kwargs.pop("bounds", [(0.0, np.inf)] * matrices.shape[-1])
        defaults = kwargs.pop("defaults", [1.0] * matrices.shape[-1])
        MatrixPredictor.__init__(
            self,
            matrices,
            constants=constants,
            bounds=bounds,
            defaults=defaults,
            **kwargs,
        )


class LikelihoodCalculator:
    """Class that calculates the likelihoods of parameter sets.

    Parameters
    ----------

    data_model : DataModel
        Object that describes the statistical model of the data.
    predictor : Predictor
        The object that translates parameter sets into reco expectation values.
    systematics : {'marginal', 'profile'} or SystematicsConsumer, optional
        Specifies how to handle systematic prediction uncertainties, i.e. multiple
        predictions from a single parameter set.

    Notes
    -----

    TODO

    """

    def __init__(self, data_model, predictor, systematics="marginal"):
        self.data_model = data_model
        self.predictor = predictor
        if systematics == "marginal" or systematics == "average":
            self.systematics = MarginalLikelihoodSystematics
        elif systematics == "profile" or systematics == "maximum":
            self.systematics = ProfileLikelihoodSystematics
        else:
            self.systematics = systematics

    def log_likelihood(self, *args, **kwargs):
        """Calculate the log likelihood of the given parameters.

        Passes all arguments to the `predictor`.

        """

        prediction, weights = self.predictor.prediction(*args, **kwargs)
        log_likelihood = self.data_model.log_likelihood(prediction)
        log_likelihood = self.systematics.consume_axis(log_likelihood, weights)
        # Fix out of bounds to -inf
        check = self.predictor.check_bounds(*args)
        if check.ndim == 0:
            if not check:
                log_likelihood = -np.inf
        else:
            log_likelihood[..., ~check] = -np.inf
        return log_likelihood

    def generate_toy_likelihood_calculators(self, parameters, N=1, **kwargs):
        """Generate LikelihoodCalculator objects with randomly varied data.

        Accepts only single set of parameters.

        Returns
        -------

        toy_calculators : list of LikelihoodCalculator

        """

        parameters = np.asarray(parameters)
        if parameters.ndim != 1:
            raise ValueError("Parameters must be 1D array!")

        predictor = self.predictor
        systematics = self.systematics

        prediction, weights = predictor(parameters, **kwargs)
        weights = weights / np.sum(weights, axis=-1)

        toys = []
        for _i in range(N):
            j = np.random.choice(len(weights), p=weights)
            data_model = self.data_model.generate_toy_data_model(prediction[j])
            toys.append(LikelihoodCalculator(data_model, predictor, systematics))

        return toys

    def fix_parameters(self, fix_values):
        """Return a new LikelihoodCalculator with fewer free parameters.

        Parameters
        ----------

        fix_values

        """

        data = self.data_model
        pred = self.predictor.fix_parameters(fix_values)
        syst = self.systematics
        return LikelihoodCalculator(data, pred, syst)

    def compose(self, predictor):
        """Return a new LikelihoodCalculator with the composed Predictor.

        Parameters
        ----------

        predictor : Predictor
            The predictor of the calculator will be composed with this.

        """

        data = self.data_model
        pred = self.predictor.compose(predictor)
        syst = self.systematics
        return LikelihoodCalculator(data, pred, syst)

    def __call__(self, *args, **kwargs):
        return self.log_likelihood(*args, **kwargs)


class LikelihoodMaximizer:
    """Class to maximise the likelihood over a parameter space."""

    def minimize(self, fun, x0, bounds, **kwargs):
        """General minimisation function."""
        raise NotImplementedError("Must be implemented in a subclass!")

    def maximize_log_likelihood(self, likelihood_calculator, **kwargs):
        """Maximise the likelihood"""

        def fun(x):
            fun = -likelihood_calculator(x)
            return fun

        bounds = likelihood_calculator.predictor.bounds
        x0 = likelihood_calculator.predictor.defaults
        if len(x0) == 0:
            # Nothing to optimise, return dummy result
            opt = optimize.OptimizeResult()
            opt.x = np.ndarray(0)
            opt.fun = -likelihood_calculator(opt.x)
            opt.log_likelihood = -opt.fun
        else:
            opt = self.minimize(fun, x0, bounds, **kwargs)
            opt.log_likelihood = -opt.fun
        return opt

    def __call__(self, *args, **kwargs):
        return self.maximize_log_likelihood(*args, **kwargs)


class BasinHoppingMaximizer(LikelihoodMaximizer):
    """Class to maximise the likelihood over a parameter space.

    Uses SciPy's Basin Hopping algorithm.

    Parameters
    ----------

    **kwargs : optional
        Arguments to be passed to the basin hopping function.

    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def minimize(self, fun, x0, bounds, **kwargs):
        minimizer_kwargs = {
            "bounds": bounds,
        }
        # expected log likelihood variation in the order of degrees of freedom
        args = {
            "T": len(x0),
            "niter": 10,
            "minimizer_kwargs": minimizer_kwargs,
        }
        args.update(self.kwargs)
        return optimize.basinhopping(fun, x0, **args)


class HypothesisTester:
    """Class for statistical tests of hypotheses."""

    _default_maximizer = BasinHoppingMaximizer()

    def __init__(self, likelihood_calculator, maximizer=_default_maximizer):
        self.likelihood_calculator = likelihood_calculator
        self.maximizer = maximizer

    def likelihood_p_value(self, parameters, N=2500, **kwargs):
        """Calculate the likelihood p-value of a set of parameters.

        The likelihood p-value is the probability of hypothetical alternative
        data yielding a lower likelihood than the actual data, given that the
        simple hypothesis described by the parameters is true.

        Parameters
        ----------

        parameters : array like
            The evaluated hypotheis expressed as a vector of its parameters.
            Can be a multidimensional array of vectors. The p-value for each
            vector is calculated independently.

        N : int, optional
            The number of MC evaluations of the hypothesis.

        **kwargs : optional
            Additional keyword arguments will be passed to the likelihood
            calculator.

        Returns
        -------

        p : float or ndarray
            The likelihood p-value.

        Notes
        -----

        The p-value is estimated by creating ``N`` data samples according to
        the given ``parameters``. The data is varied by both the statistical
        and systematic uncertainties resulting from the prediction. The number
        of data-sets that yield a likelihood as bad as, or worse than the
        likelihood given the actual data, ``n``, are counted. The estimate for
        ``p`` is then::

            p = n/N.

        The variance of the estimator follows that of binomial statistics::

                     var(n)   Np(1-p)      1
            var(p) = ------ = ------- <= ---- .
                      N^2       N^2       4N

        The expected uncertainty can thus be directly influenced by choosing an
        appropriate number of toy data sets.

        See also
        --------

        max_likelihood_p_value
        max_likelihood_ratio_p_value

        """

        parameters = np.array(parameters)
        shape = parameters.shape[:-1]
        parameters.shape = (np.prod(shape, dtype=int), parameters.shape[-1])

        LC = self.likelihood_calculator  # Calculator

        p_values = []

        for par in parameters:
            L0 = LC(par, **kwargs)  # Likelihood given data

            toy_LC = LC.generate_toy_likelihood_calculators(par, N=N, **kwargs)
            toy_L = list(
                mapper(lambda C, par=par, kwargs=kwargs: C(par, **kwargs), toy_LC)
            )
            toy_L = np.array(toy_L)

            p_values.append(np.sum(L0 >= toy_L, axis=-1) / N)

        p_values = np.array(p_values, dtype=float)
        p_values.shape = shape
        return p_values

    def max_likelihood_p_value(self, fix_parameters=None, N=250):
        """Calculate the maximum-likelihood p-value.

        The maximum-likelihood p-value is the probability of the data yielding
        a lower maximum likelihood (over the possible parameter space of the
        likelihood calculator) than the actual data, given that the best-fit
        parameter set of is true.

        Parameters
        ----------

        fix_parameters : array like, optional
            Optionally fix some or all of the paramters of the
            :class:`LikelihoodCalculator`.

        N : int, optional
            The number of MC evaluations of the hypothesis.

        **kwargs : optional
            Additional keyword arguments will be passed to the maximiser.

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
        LikelihoodCalculator.fix_parameters

        """

        if fix_parameters is None:
            LC = self.likelihood_calculator  # Calculator
        else:
            LC = self.likelihood_calculator.fix_parameters(fix_parameters)  # Calculator
        maxer = self.maximizer  # Maximiser

        opt = maxer(LC)
        opt_par = opt.x
        L0 = opt.log_likelihood

        toy_LC = LC.generate_toy_likelihood_calculators(opt_par, N=N)
        toy_opt = list(mapper(lambda C, maxer=maxer: maxer(C), toy_LC))
        toy_L = np.asarray([topt.log_likelihood for topt in toy_opt], dtype=float)

        p_value = np.sum(L0 >= toy_L, axis=-1) / N

        return p_value

    def _max_log_likelihood_ratio(
        self, LC, fix_parameters, alternative_fix_parameters, return_parameters=False
    ):
        # Calculator 0
        LC0 = LC.fix_parameters(fix_parameters)

        # Calculator 1
        if alternative_fix_parameters is None:
            LC1 = LC
        else:
            LC1 = LC.fix_parameters(alternative_fix_parameters)

        maxer = self.maximizer  # Maximiser

        opt0 = maxer(LC0)
        opt1 = maxer(LC1)

        L0 = opt0.log_likelihood
        L1 = opt1.log_likelihood

        if return_parameters:
            # "Unfix" the parameters and generate toy calculators
            full_parameters = LC0.predictor.insert_fixed_parameters(opt0.x)
            return L0 - L1, full_parameters
        else:
            return L0 - L1

    def max_likelihood_ratio_p_value(
        self, fix_parameters, alternative_fix_parameters=None, N=250, **kwargs
    ):
        """Calculate the maximum-likelihood-ratio p-value.

        The maximum-likelihood-ratio p-value is the probability of the data
        yielding a lower ratio of maximum likelihoods (over the possible
        parameter spaces of the composite hypotheses) than the actual data,
        given that the best-fit parameter set of the tested hypothesis `H0` is
        true.

        Parameters
        ----------

        fix_parameters : array like
            Fix some or all of the paramters of the
            :class:`LikelihoodCalculator`. This defines the tested hypothesis
            `H0`.

        alternative_fix_parameters : array like, optional
            Optionally fix some of the paramters of the
            :class:`LikelihoodCalculator` to define the alternative Hypothesis
            `H1`. If this is not specified, `H1` is the fully unconstrained
            calculator.

        N : int, optional
            The number of MC evaluations of the hypothesis.

        **kwargs : optional
            Additional keyword arguments will be passed to the maximiser.

        Returns
        -------

        p : float or ndarray
            The maximum-likelihood-ratio p-value.

        Notes
        -----

        When used to reject composite hypotheses, this p-value is sometimes
        called the "profile plug-in p-value", as one "plugs in" the maximum
        likelihood estimate of the hypothesis parameters to calculate it. It's
        coverage properties are not exact, so care has to be taken to make sure
        it performs as expected (e.g. by testing it with simulated data).

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

        wilks_max_likelihood_ratio_p_value
        likelihood_p_value
        max_likelihood_p_value

        """

        LC = self.likelihood_calculator
        ratio0, parameters = self._max_log_likelihood_ratio(
            LC, fix_parameters, alternative_fix_parameters, return_parameters=True
        )

        # Generate toy data
        toy_LC = LC.generate_toy_likelihood_calculators(parameters, N=N)

        # Calculate ratios for toys
        def fun(
            LC,
            fix_parameters=fix_parameters,
            alternative_fix_parameters=alternative_fix_parameters,
        ):
            return self._max_log_likelihood_ratio(
                LC, fix_parameters, alternative_fix_parameters
            )

        toy_ratios = np.array(list(mapper(fun, toy_LC)))

        # Callculate p-value
        p_value = np.sum(ratio0 >= toy_ratios, axis=-1) / N

        return p_value

    def wilks_max_likelihood_ratio_p_value(
        self, fix_parameters, alternative_fix_parameters=None, **kwargs
    ):
        """Calculate the maximum-likelihood-ratio p-value using Wilk's theorem.

        The maximum-likelihood-ratio p-value is the probability of the data
        yielding a lower ratio of maximum likelihoods (over the possible
        parameter spaces of the composite hypotheses) than the actual data,
        given that the best-fit parameter set of the tested hypothesis `H0` is
        true. This method assumes that Wilk's theorem holds.

        Parameters
        ----------


        fix_parameters : array like
            Fix some or all of the paramters of the
            :class:`LikelihoodCalculator`. This defines the tested hypothesis
            `H0`.

        alternative_fix_parameters : array like, optional
            Optionally fix some of the paramters of the
            :class:`LikelihoodCalculator` to define the alternative Hypothesis
            `H1`. If this is not specified, `H1` is the fully unconstrained
            calculator.

        **kwargs : optional
            Additional keyword arguments will be passed to the maximiser.

        Returns
        -------

        p : float
            The maximum-likelihood-ratio p-value.

        Notes
        -----

        This method assumes that Wilks' theorem holds, i.e. that the logarithm
        of the maximum likelihood ratio of the two hypothesis is distributed
        like a chi-squared distribution::

            ndof = number_of_parameters_of_H1 - number_of_parameters_of_H0
            p_value = scipy.stats.chi2.sf(-2*log_likelihood_ratio, df=ndof)

        See also
        --------

        max_likelihood_ratio_p_value
        likelihood_p_value
        max_likelihood_p_value

        """

        LC = self.likelihood_calculator
        ratio0 = self._max_log_likelihood_ratio(
            LC, fix_parameters, alternative_fix_parameters
        )

        # Likelihood ratio should be distributed like a chi2 distribution
        if alternative_fix_parameters is None:
            # All parameters - unfixed parameters in H0
            ndof = len(fix_parameters) - np.sum(
                np.isnan(np.array(fix_parameters, dtype=float))
            )
        else:
            # Unfixed parameters in H1 - unfixed parameters in H0
            ndof = np.sum(
                np.isnan(np.array(alternative_fix_parameters, dtype=float))
            ) - np.sum(np.isnan(np.array(fix_parameters, dtype=float)))
        return stats.chi2.sf(-2.0 * ratio0, df=ndof)
