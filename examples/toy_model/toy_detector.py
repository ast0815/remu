import numpy as np
import csv
import copy

class ToyDetector(object):
    """Base class fpr toy detectors."""

    def propagate_event(self, event):
        """Propagate an event through the detector.

        Must be implemented in the inheriting classes.
        Returns an event with both the truth and reconstructed variables.
        """

        raise NotImplementedError()

    def propagate_csv_file(self, input_file, output_file):
        """Propagate all events in a csv file and save output in new csv file."""

        # Get fieldnames
        with open(input_file, 'r') as inf:
            reader = csv.DictReader(inf, delimiter=',')
            for row in reader:
                fieldnames = row.keys()
                newnames = []
                for fname in fieldnames:
                    if fname.endswith('_truth'):
                        newnames.append(fname[:-5]+'reco')
                fieldnames.extend(newnames)
                break

        with open(input_file, 'r') as inf:
            reader = csv.DictReader(inf, delimiter=',')
            with open(output_file, 'w') as outf:
                outf.write(",".join(fieldnames) + "\r\n")
                writer = csv.DictWriter(outf, fieldnames=fieldnames, delimiter=',')
                for row in reader:
                    event = self.propagate_event(row)
                    writer.writerow(event)

class NormalSmearingDetector(ToyDetector):
    """Toy detector that smears all variables according to a normal distribution."""

    def __init__(self, sigma):
        """Initialize the detector with a set of sigmas for the smearing."""

        self.sigma = sigma

    def propagate_event(self, event):
        """Smear with normal distribution."""

        e = copy.copy(event)
        for i in range(len(self.sigma)):
            e['x_%d_reco'%(i,)] = np.random.normal(float(e['x_%d_truth'%(i,)]), self.sigma[i])

        return e
