"""Module to define "real" toy experiment as well as "MC" experiments."""

import numpy as np

class Generator(object):
    """Generates "true" events according to model."""

    def __init__(self, boost_factor=0.)
        self.boost_factor = boost_factor

    def generate_exposed(self, exposure, **kwargs):
        """Generate events according to experiment exposure."""
        n = np.random.poisson(lam=self.cross_section * exposure)
        return self.generate(n, **kwargs)

    def genrerate(self, n, **kwargs):
        """Generate n events."""
        return self.boost(self.generate_unboosted(n, **kwargs)):

    def boost(self, events):
        """Boost the events with the given beta factor."""
        if self.boost_factor == 0.:
            return events
        else
            longitudinal_momentum = events['true_momentum'] * events['true_costheta']
            transverse_momentum = events['true_momentum'] * np.sqrt(1 - events['true_costheta']**2)
            beta = self.boost_factor
            gamma = 1. / np.sqrt(1. - beta**2))
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

    cross_section = 100.

    def generate_unboosted(self, n, mean_momentum=50.):
        momentum = np.random.exponential(scale=mean_momentum, size=n)
        costheta = np.random.uniform(low=-1.0, high=+1.0, size=n)

class ModelAGenerator(Generator):
    """Model A

    p ~ exponential(mean=100)
    costheta ~ uniform(min=-1, max=1)
    """

    cross_section = 200.

    def generate_unboosted(self, n, mean_momentum=100.):
        momentum = np.random.exponential(scale=mean_momentum, size=n)
        costheta = np.random.uniform(low=-1.0, high=+1.0, size=n)

class ModelBGenerator(Generator):
    """Model B

    p ~ |normal(mean=100, sigma=100)|
    costheta ~ uniform(min=-1, max=1)
    """

    cross_section = 100.

    def generate_unboosted(self, n, mean_momentum=100.):
        momentum = np.random.normal(loc=mean_momentum, scale=mean_momentum, size=n)
        costheta = np.random.uniform(low=-1.0, high=+1.0, size=n)

class Detector(object):
    """Turn truth data into reconstructed events.

    The detector is cylindrical symmetric around the beam axis.
    It is defined by a turn on momentum below which the which the reconstruction efficiency drops to 0.
    This drop is characterized by the momentum turnon width.
    The maximal efficiency depends whether the event is in the cap or barrel region.
    In between those two is a gap that is not instrumented.
    """

    def __init__(self, momentum_threshold=50., momentum_turnon=10., longitudinal_efficiency=0.5, transverse_efficiency=0.9, gap_costheta=0.7, gap_width=0.03):
        self.momentum_threshold = momentum_threshold
        self.momentum_turnon = momentum_turnon
        self.longitudinal_efficiency = longitudinal_efficiency
        self.transverse_efficiency = transverse_efficiency
        self.gap_costheta = gap_costheta
        self.gap_width = gap_width

    def reconstruct(self, events):
        """Turn events into reconstructed events."""
        reconstruction_probability = self.efficiency(events)
        reconstructed_events = events[np.random.uniform(low=0., high=1., size=events.shape) <= reconstruction_probability]
        return self.smear(reconstructed_events)

    def efficiency(self, events):
        """Return efficiency of given true events."""
