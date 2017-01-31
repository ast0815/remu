import numpy as np
import csv

class ToyModel(object):
    """Base class for toy models."""

    def generate_events(self, N=1, weight=1.):
        """Generates N events.

        Must be implemented in inheriting classes.
        Returns a list of dicts.
        """

        raise NotImplementedError()

    def generate_csv_file(self, filename, N=1, weight=1.):
        "Generate N events and save them in a csv file."

        events = self.generate_events(N=N, weight=weight)

        with open(filename, 'w') as f:
            fieldnames = events[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=',')

            f.write(",".join(fieldnames) + "\r\n")
            for e in events:
                writer.writerow(e)

class MultivariateNormalModel(ToyModel):
    """ToyModel that generates N-dimensional data according to (correlated) normal distributions."""

    def __init__(self, mu, cov):
        """Initialize the model with the vector of mean values and covariance matrix."""

        self.mu = mu
        self.cov = cov

    def generate_events(self, N=1, weight=1.):
        """Generate the events according to the multivariate normal distribution."""

        varnames = ["x_%d_truth"%(i,) for i in range(len(self.mu))]

        data = np.random.multivariate_normal(self.mu, self.cov, N)

        events = []

        for i in range(N):
            events.append( dict(zip(varnames, data[i,:])) )
            events[-1]['weight'] = weight

        return events
