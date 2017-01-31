import unittest
import tempfile
import os
import csv
from toy_model import *
from toy_detector import *

class TestToyModels(unittest.TestCase):
    """Test the basic functionality of the ToyModels"""

    def setUp(self):
        self.mvnm = MultivariateNormalModel(mu=[0,1,2,3], cov=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

    def test_event_generation(self):
        """Test basic event generation."""
        events = self.mvnm.generate_events(10)
        self.assertEqual(len(events), 10)
        self.assertTrue('weight' in events[0])
        self.assertTrue('x_0_truth' in events[0])
        self.assertTrue('x_1_truth' in events[0])
        self.assertTrue('x_2_truth' in events[0])
        self.assertTrue('x_3_truth' in events[0])

    def test_csv_generation(self):
        """Test writing events to csv file."""
        tf = tempfile.NamedTemporaryFile(delete=False)
        filename = tf.name
        tf.close()
        self.mvnm.generate_csv_file(filename, 10)
        with open(filename, 'r') as f:
            n = 0
            for line in f:
                n += 1
        os.remove(filename)
        self.assertEqual(n, 11) # 10 Events + 1 header

class TestToyDetectors(unittest.TestCase):
    """Test basic functionality of toy detectors."""

    def setUp(self):
        self.mvnm = MultivariateNormalModel(mu=[0,1,2,3], cov=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        self.nsd = NormalSmearingDetector([0.1,0.2,0.3,0.4])

    def test_event_propagation(self):
        """Test propagating an event through the detector."""
        event_truth = self.mvnm.generate_events(1)[0]
        event_reco = self.nsd.propagate_event(event_truth)
        self.assertTrue('weight' in event_reco)
        self.assertTrue('x_0_truth' in event_reco)
        self.assertTrue('x_0_reco' in event_reco)
        self.assertTrue('x_1_truth' in event_reco)
        self.assertTrue('x_1_reco' in event_reco)
        self.assertTrue('x_2_truth' in event_reco)
        self.assertTrue('x_2_reco' in event_reco)
        self.assertTrue('x_3_truth' in event_reco)
        self.assertTrue('x_3_reco' in event_reco)

    def test_csv_propagation(self):
        """Test the propagation of a whole csv file."""
        tf = tempfile.NamedTemporaryFile(delete=False)
        ftruth = tf.name
        tf.close()
        tf = tempfile.NamedTemporaryFile(delete=False)
        freco = tf.name
        tf.close()
        self.mvnm.generate_csv_file(ftruth, 10)
        self.nsd.propagate_csv_file(ftruth, freco)
        with open(freco, 'r') as inf:
            reader = csv.DictReader(inf, delimiter=',')
            for row in reader:
                self.assertTrue('weight' in row)
                self.assertTrue('x_0_truth' in row)
                self.assertTrue('x_0_reco' in row)
                self.assertTrue('x_1_truth' in row)
                self.assertTrue('x_1_reco' in row)
                self.assertTrue('x_2_truth' in row)
                self.assertTrue('x_2_reco' in row)
                self.assertTrue('x_3_truth' in row)
                self.assertTrue('x_3_reco' in row)
        os.remove(ftruth)
        os.remove(freco)

if __name__ == '__main__':
    unittest.main()
