"""Module to define "real" toy experiment as well as "MC" experiments."""

import numpy as np
from scipy.special import ndtr # 1/2[1 + erf(z/sqrt(2))]

def build_array(dictionary):
    """Turn a dict of arrays into a structured array."""
    keys = dictionary.keys()
    dtype = []
    for k in keys:
        dtype.append((k, dictionary[k].dtype))
    dtype = np.dtype(dtype)
    arr = np.empty(len(dictionary[k]), dtype=dtype)
    for k in keys:
        arr[k] = dictionary[k]
    return arr

class Generator(object):
    """Generates "true" events according to model."""

    def __init__(self, cross_section=100.):
        self.cross_section = cross_section

    def generate_exposed(self, exposure, **kwargs):
        """Generate events according to experiment exposure."""
        n = np.random.poisson(lam=self.cross_section * exposure)
        return self.generate(n, **kwargs)

    def generate(self, n, **kwargs):
        """Generate n events."""
        return self._generate(n, **kwargs)

class ModelAGenerator(Generator):
    """Model A

    x, y ~ Normal(mean=[0.1, 0.20], cov=[[1.0,0.0],[0.0,1.0]])
    """

    def _generate(self, n):
        x, y = np.random.multivariate_normal(mean=[0.1, 0.20], cov=[[1.0,0.0],[0.0,1.0]], size=n).T
        return build_array({'true_x': x, 'true_y': y})

class ModelBGenerator(Generator):
    """Model B

    x, y ~ Normal(mean=[0.0, 0.0], cov=[[1.0,0.5],[0.5,1.0]])
    """

    def _generate(self, n):
        x, y = np.random.multivariate_normal(mean=[0.0, 0.0], cov=[[1.0,0.5],[0.5,1.0]], size=n).T
        return build_array({'true_x': x, 'true_y': y})

class BGGenerator(Generator):
    """Background

    x, y ~ Normal(mean=[0.5, 0.5], cov=[[0.2,0.0],[0.0,0.2]])
    """

    def _generate(self, n):
        x, y = np.random.multivariate_normal(mean=[0.5, 0.5], cov=[[0.5,0.0],[0.0,0.5]], size=n).T
        return build_array({'true_x': x, 'true_y': y})

class NoiseGenerator(Generator):
    """Noise

    x, y ~ Normal(mean=[0.0, 0.0], cov=[[4.0,0.0],[0.0,4.0]])
    """

    def _generate(self, n):
        x, y = np.random.multivariate_normal(mean=[0.0, 0.0], cov=[[4.0,0.0],[0.0,4.0]], size=n).T
        return build_array({'reco_x': x, 'reco_y': y})

class Detector(object):
    """Turn truth data into reconstructed events.

    ``x`` is smeared with a normal ``sigma=1``.
    The efficienct depends on ``y``: ``eff = ndtr(slope*(y-y0))`` with ``slope=1.``

    """

    def __init__(self, smear_sigma=1, eff_slope=1., eff_offset=0., max_eff=0.9):
        self.smear_sigma = smear_sigma
        self.eff_slope = eff_slope
        self.eff_offset = eff_offset
        self.max_eff = max_eff

    def reconstruct(self, events, keep_truth=False):
        """Turn events into reconstructed events."""
        reconstruction_probability = self.efficiency(events)
        reconstructed_events = events[np.random.uniform(low=0., high=1., size=events.shape) <= reconstruction_probability]
        return self.smear(reconstructed_events, keep_truth=keep_truth)

    def efficiency(self, events):
        """Return efficiency of given true events."""
        eff = self.max_eff * ndtr(self.eff_slope * (events['true_y'] - self.eff_offset))
        return eff

    def smear(self, events, keep_truth=False):
        y = np.array(events['true_y'])
        x = np.array(events['true_x'])
        n = len(x)
        x += np.random.normal(loc=0., scale=self.smear_sigma, size=n)
        dic = {'reco_x': x, 'reco_y': y}
        if keep_truth:
            dic.update({'true_x': events['true_x'], 'true_y': events['true_y']})
        return build_array(dic)
