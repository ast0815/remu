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

    def __init__(self, cross_section=100., boost_factor=0.):
        self.boost_factor = boost_factor
        self.cross_section = cross_section

    def generate_exposed(self, exposure, **kwargs):
        """Generate events according to experiment exposure."""
        n = np.random.poisson(lam=self.cross_section * exposure)
        return self.generate(n, **kwargs)

    def generate(self, n, **kwargs):
        """Generate n events."""
        return self.boost(self.generate_unboosted(n, **kwargs))

    def boost(self, events):
        """Boost the events with the given beta factor."""
        if self.boost_factor == 0.:
            return events
        else:
            longitudinal_momentum = events['true_momentum'] * events['true_costheta']
            transverse_momentum = events['true_momentum'] * np.sqrt(1 - events['true_costheta']**2)
            beta = self.boost_factor
            gamma = 1. / np.sqrt(1. - beta**2)
            # Lorentz boost for massless particles:
            longitudinal_momentum = gamma * longitudinal_momentum + gamma*beta*events['true_momentum']
            events['true_momentum'] = np.sqrt(longitudinal_momentum**2 + transverse_momentum**2)
            events['true_costheta'] = longitudinal_momentum / events['true_momentum']
            return events

    def generate_unboosted(self, n, **kwargs):
        """Generate n events in the rest frame."""
        raise NotImplementedError()

class BackgroundGenerator(Generator):
    """Background model.

    p ~ exponential(mean=50)
    costheta ~ uniform(min=-1, max=1)
    """

    def __init__(self, mean_momentum=50., **kwargs):
        self.mean_momentum = mean_momentum
        super(BackgroundGenerator, self).__init__(**kwargs)

    def generate_unboosted(self, n):
        momentum = np.random.exponential(scale=self.mean_momentum, size=n)
        costheta = np.random.uniform(low=-1.0, high=+1.0, size=n)
        return build_array({'true_momentum': momentum, 'true_costheta': costheta})

class ModelAGenerator(Generator):
    """Model A

    p ~ exponential(mean=100)
    costheta ~ uniform(min=-1, max=1)
    """

    def __init__(self, mean_momentum=50., **kwargs):
        self.mean_momentum = mean_momentum
        super(ModelAGenerator, self).__init__(**kwargs)

    def generate_unboosted(self, n, mean_momentum=100.):
        momentum = np.random.exponential(scale=mean_momentum, size=n)
        costheta = np.random.uniform(low=-1.0, high=+1.0, size=n)
        return build_array({'true_momentum': momentum, 'true_costheta': costheta})

class ModelBGenerator(Generator):
    """Model B

    p ~ |normal(mean=100, sigma=100)|
    costheta ~ uniform(min=-1, max=1)
    """

    def __init__(self, mean_momentum=50., **kwargs):
        self.mean_momentum = mean_momentum
        super(ModelBGenerator, self).__init__(**kwargs)

    def generate_unboosted(self, n, mean_momentum=100.):
        momentum = np.random.normal(loc=mean_momentum, scale=mean_momentum, size=n)
        costheta = np.random.uniform(low=-1.0, high=+1.0, size=n)
        return build_array({'true_momentum': momentum, 'true_costheta': costheta})

class Detector(object):
    """Turn truth data into reconstructed events.

    The detector is cylindrical symmetric around the beam axis.
    It is defined by a turn-on momentum below which the reconstruction efficiency drops to 0.
    This drop is characterized by the momentum turn-on width.
    The maximal efficiency depends whether the event is in the cap or barrel region.
    In between those two is a gap that is not instrumented.
    Momentum is measured in the transverse direction.
    The transverse momentum resolution is proportional to the momentum and given in %/MeV.
    Angular resolution is absolute in rad.
    """

    def __init__(self, momentum_threshold=50., momentum_turnon=10., cap_efficiency=0.5, barrel_efficiency=0.9, gap_costheta=0.7, gap_width=0.03, gap_turnon=0.01, momentum_resolution=0.01, angular_resolution=0.01):
        self.momentum_threshold = momentum_threshold
        self.momentum_turnon = momentum_turnon
        self.cap_efficiency = cap_efficiency
        self.barrel_efficiency = barrel_efficiency
        self.gap_costheta = gap_costheta
        self.gap_width = gap_width
        self.gap_turnon = gap_turnon
        self.momentum_resolution = momentum_resolution
        self.angular_resolution = angular_resolution

    def reconstruct(self, events, keep_truth=False):
        """Turn events into reconstructed events."""
        reconstruction_probability = self.efficiency(events)
        reconstructed_events = events[np.random.uniform(low=0., high=1., size=events.shape) <= reconstruction_probability]
        return self.smear(reconstructed_events, keep_truth=keep_truth)

    def momentum_efficiency(self, momentum):
        """Efficiency goes from 0 to 1 at turn-on momentum."""
        return ndtr( (momentum - self.momentum_threshold) / self.momentum_turnon )

    def gap_efficiency(self, costheta):
        """Efficiency drops to 0 at gap between barrel and cap."""
        xp = (costheta - self.gap_costheta + self.gap_width) / self.gap_turnon
        xm = (costheta - self.gap_costheta - self.gap_width) / self.gap_turnon
        return 1. - (ndtr( xp ) * ndtr( -xm ))

    def efficiency(self, events):
        """Return efficiency of given true events."""
        barrel = np.abs(events['true_costheta'] < self.gap_costheta)
        eff = np.where(barrel, self.barrel_efficiency, self.cap_efficiency)
        eff *= self.momentum_efficiency(events['true_momentum'])
        eff *= self.gap_efficiency(events['true_costheta'])
        return eff

    def smear(self, events, keep_truth=False):
        theta = np.arccos(events['true_costheta'])
        n = len(theta)
        theta += np.random.normal(loc=0, scale=self.angular_resolution, size=n)
        costheta = np.cos(theta)
        momentum = np.array(events['true_momentum'])
        momentum *= 1 + np.random.normal(loc=0, scale=self.momentum_resolution*momentum, size=n)
        dic = {'reco_momentum': momentum, 'reco_costheta': costheta}
        if keep_truth:
            dic.update({'true_momentum': events['true_momentum'], 'true_costheta': events['true_costheta']})
        return build_array(dic)
