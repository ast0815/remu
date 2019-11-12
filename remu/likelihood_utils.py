from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from remu import likelihood

def emcee_sampler(likelihood_calculator, nwalkers=None):
        import emcee

        defaults = likelihood_calculator.predictor.defaults
        ndim = len(defaults)
        if nwalkers is None:
            nwalkers = 2*ndim

        sampler = emcee.EnsembleSampler(nwalkers, ndim, likelihood_calculator)

        return sampler

def emcee_initial_guess(likelihood_calculator, nwalkers=None):
        bounds = likelihood_calculator.predictor.bounds
        defaults = likelihood_calculator.predictor.defaults
        ndim = len(defaults)
        if nwalkers is None:
            nwalkers = 2*ndim

        guess = []
        for i, b in enumerate(bounds):
            fin = np.isfinite(b)
            if fin[0] and fin[1]:
                # Upper and lower bound
                # -> Uniform
                guess.append(np.random.uniform(low=b[0], high=b[1], size=nwalkers))
            elif fin[0] and not fin[1]:
                # Only lower bound
                # -> Exponential
                scale = defaults[i] - b[0]
                if scale <= 0.:
                    scale = 1.
                guess.append(b[0] + np.random.exponential(scale=scale, size=nwalkers))
            elif not fin[0] and fin[1]:
                # Only upper bound
                # -> Exponential
                scale = b[1] - defaults[i]
                if scale <= 0.:
                    scale = 1.
                guess.append(b[1] - np.random.exponential(scale=scale, size=nwalkers))
            elif not fin[0] and not fin[1]:
                # No bounds
                # -> Normal
                scale = abs(defaults[i])
                loc = defaults[i]
                if scale <= 0.:
                    scale = 1.
                guess.append(np.random.normal(loc=loc, scale=scale, size=nwalkers))

        # Reorder things
        guess = np.array(guess).T

        # TODO: Get rid off impossible guesses w/ logL = -inf

        return guess
