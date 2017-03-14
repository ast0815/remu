import unittest
import ruamel.yaml as yaml
from binning import *
from migration import *
from likelihood import *
import numpy as np

class TestPhasSpaces(unittest.TestCase):
    def setUp(self):
        self.psX = PhaseSpace(variables=['x'])
        self.psY = PhaseSpace(variables=['y'])
        self.psXY = PhaseSpace(variables=['x', 'y'])
        self.psXYZ = PhaseSpace(variables=['x', 'y', 'z'])

    def test_contains(self):
        """Test behaviour of 'in' operator."""
        self.assertTrue('x' in self.psX)
        self.assertFalse('x' in self.psY)

    def test_product(self):
        """Test the carthesian product of phase spaces."""
        psXY = self.psX * self.psY
        self.assertTrue('x' in psXY)
        self.assertTrue('y' in psXY)
        self.assertFalse('z' in psXY)

    def test_division(self):
        """Test the reduction of phase spaces."""
        psXYX = (self.psX * self.psY) / self.psY
        self.assertTrue('x' in psXYX)
        self.assertFalse('y' in psXYX)

    def test_equality(self):
        """Test the equlaity of phase spaces."""
        ps = PhaseSpace(variables=['x'])
        self.assertTrue(self.psX == ps)
        self.assertTrue(self.psY != ps)
        self.assertFalse(self.psX != ps)
        self.assertFalse(self.psY == ps)

    def test_comparisons(self):
        """Test whether phase spaces are subsets of other phase spaces."""
        self.assertTrue(self.psX < self.psXY)
        self.assertTrue(self.psX <= self.psXY)
        self.assertTrue(self.psXY > self.psY)
        self.assertTrue(self.psXY >= self.psY)
        self.assertFalse(self.psX < self.psX)
        self.assertFalse(self.psY > self.psY)
        self.assertFalse(self.psX < self.psY)
        self.assertFalse(self.psX > self.psY)
        self.assertFalse(self.psX <= self.psY)
        self.assertFalse(self.psX >= self.psY)

    def test_repr(self):
        """Test whether the repr reproduces same object."""
        self.assertEqual(self.psX, eval(repr(self.psX)))
        self.assertEqual(self.psXY, eval(repr(self.psXY)))
        self.assertEqual(self.psXYZ, eval(repr(self.psXYZ)))

    def test_yaml_representation(self):
        """Test whether the text parsing can reproduce the original object."""
        self.assertEqual(self.psX, yaml.load(yaml.dump(self.psX)))
        self.assertEqual(self.psXY, yaml.load(yaml.dump(self.psXY)))
        self.assertEqual(self.psXYZ, yaml.load(yaml.dump(self.psXYZ)))

class TestBins(unittest.TestCase):
    def setUp(self):
        ps = PhaseSpace(['x'])
        self.b0 = Bin(phasespace=ps)
        self.b1 = Bin(phasespace=ps, value=1.)
        self.b2 = Bin(phasespace=ps, value=2.)

    def test_init_values(self):
        """Test initialization values."""
        self.assertEqual(self.b0.value, 0.)
        self.assertEqual(self.b1.value, 1.)
        self.assertEqual(self.b2.value, 2.)

    def test_bin_arithmetic(self):
        """Test math with bins."""
        self.assertEqual((self.b1 + self.b2).value, 3.)
        self.assertEqual((self.b1 - self.b2).value, -1.)
        self.assertEqual((self.b2 * self.b2).value, 4.)
        self.assertEqual((self.b1 / self.b2).value, 0.5)

    def test_bin_filling(self):
        """Test filling of single and multiple weights"""
        self.b0.fill()
        self.assertEqual(self.b0.value, 1.0)
        self.assertEqual(self.b0.entries, 1)
        self.b0.fill(0.5)
        self.assertEqual(self.b0.value, 1.5)
        self.assertEqual(self.b0.entries, 2)
        self.b0.fill([0.5, 0.5, 0.5])
        self.assertEqual(self.b0.value, 3.0)
        self.assertEqual(self.b0.entries, 5)

    def test_equality(self):
        """Test equality comparisons between bins."""
        self.assertTrue(self.b0 == self.b0)
        self.assertFalse(self.b0 != self.b0)
        self.assertTrue(self.b0 == self.b1)
        self.assertFalse(self.b0 != self.b1)
        self.b1.phasespace *= PhaseSpace(['abc'])
        self.assertFalse(self.b0 == self.b1)
        self.assertTrue(self.b0 != self.b1)

    def test_repr(self):
        """Test whether the repr reproduces same object."""
        self.assertEqual(self.b0.phasespace, eval(repr(self.b0)).phasespace)
        self.assertEqual(self.b0.value, eval(repr(self.b0)).value)
        self.assertEqual(self.b1.value, eval(repr(self.b1)).value)
        self.assertEqual(self.b2.value, eval(repr(self.b2)).value)

    def test_yaml_representation(self):
        """Test whether the yaml parsing can reproduce the original object."""
        self.assertEqual(self.b0.phasespace, yaml.load(yaml.dump(self.b0)).phasespace)
        self.assertEqual(self.b0.value, yaml.load(yaml.dump(self.b0)).value)
        self.assertEqual(self.b1.value, yaml.load(yaml.dump(self.b1)).value)
        self.assertEqual(self.b2.value, yaml.load(yaml.dump(self.b2)).value)

class TestRectangularBins(unittest.TestCase):
    def setUp(self):
        self.b = RectangularBin(edges={'x':(0,1), 'y':(5,float('inf'))})
        self.c = RectangularBin(edges={'x':(1,2), 'y':(5,float('inf'))})

    def test_inclusion(self):
        """Test basic inclusion."""
        self.assertTrue({'x': 0.5, 'y': 10} in self.b)
        self.assertFalse({'x': -0.5, 'y': 10} in self.b)
        self.assertFalse({'x': 0.5, 'y': -10} in self.b)
        self.assertFalse({'x': -0.5, 'y': -10} in self.b)

    def test_include_lower(self):
        """Test inclusion of lower bounds."""
        self.b.include_lower=True
        self.assertTrue({'x': 0, 'y': 10} in self.b)
        self.assertTrue({'x': 0.5, 'y': 10} in self.b)
        self.assertFalse({'x': -0.5, 'y': 10} in self.b)
        self.b.include_lower=False
        self.assertFalse({'x': 0, 'y': 10} in self.b)
        self.assertTrue({'x': 0.5, 'y': 10} in self.b)
        self.assertFalse({'x': -0.5, 'y': 10} in self.b)

    def test_include_upper(self):
        """Test inclusion of upper bounds."""
        self.b.include_upper=True
        self.assertTrue({'x': 1, 'y': 10} in self.b)
        self.assertTrue({'x': 0.5, 'y': 10} in self.b)
        self.assertFalse({'x': 1.5, 'y': 10} in self.b)
        self.b.include_upper=False
        self.assertFalse({'x': 1, 'y': 10} in self.b)
        self.assertTrue({'x': 0.5, 'y': 10} in self.b)
        self.assertFalse({'x': 1.5, 'y': 10} in self.b)

    def test_bin_centers(self):
        """Test calculation of bin centers."""
        c = self.b.get_center()
        self.assertEqual(c['x'], 0.5)
        self.assertEqual(c['y'], float('inf'))

    def test_equality(self):
        """Test equality comparisons between bins."""
        self.assertTrue(self.b == self.b)
        self.assertFalse(self.b != self.b)
        self.assertTrue(self.b != self.c)
        self.assertFalse(self.b == self.c)

    def test_yaml_representation(self):
        """Test whether the yaml parsing can reproduce the original object."""
        orig = self.b
        reco = yaml.load(yaml.dump(orig))
        self.assertEqual(orig.phasespace, reco.phasespace)
        self.assertEqual(orig.value, reco.value)
        self.assertEqual(orig.edges, reco.edges)
        self.assertEqual(orig.include_lower, reco.include_lower)
        self.assertEqual(orig.include_upper, reco.include_upper)

class TestBinnings(unittest.TestCase):
    def setUp(self):
        self.b0 = RectangularBin(edges={'x':(0,1), 'y':(5,float('inf'))})
        self.b1 = RectangularBin(edges={'x':(1,2), 'y':(5,float('inf'))})
        self.binning = Binning(phasespace=self.b0.phasespace, bins=[self.b0 ,self.b1])
        self.binning0 = Binning(phasespace=self.b0.phasespace, bins=[self.b0])

    def test_get_bin_numbers(self):
        """Test the translation of events to bin numbers."""
        self.assertEqual(self.binning.get_event_bin_number({'x': 0, 'y': 10}), 0)
        self.assertEqual(self.binning.get_event_bin_number({'x': 1, 'y': 10}), 1)
        self.assertTrue(self.binning.get_event_bin_number({'x': 2, 'y': 10}) is None)

    def test_get_bin(self):
        """Test the translation of events to bins."""
        self.assertTrue(self.binning.get_event_bin({'x': 0, 'y': 10}) is self.b0)
        self.assertTrue(self.binning.get_event_bin({'x': 1, 'y': 10}) is self.b1)
        self.assertTrue(self.binning.get_event_bin({'x': 2, 'y': 10}) is None)

    def test_fill(self):
        """Test bin filling"""
        self.binning.fill({'x': 0.5, 'y': 10})
        self.assertEqual(self.b0.value, 1)
        self.assertEqual(self.b0.entries, 1)
        self.assertEqual(self.b1.value, 0)
        self.assertEqual(self.b1.entries, 0)
        self.binning.fill({'x': 1.5, 'y': 10}, 2)
        self.assertEqual(self.b0.value, 1)
        self.assertEqual(self.b0.entries, 1)
        self.assertEqual(self.b1.value, 2)
        self.assertEqual(self.b1.entries, 1)
        self.binning.fill([{'x': 0.5, 'y': 10}, {'x': 0.5, 'y': 20}], 2)
        self.assertEqual(self.b0.value, 5)
        self.assertEqual(self.b0.entries, 3)
        self.assertEqual(self.b1.value, 2)
        self.assertEqual(self.b1.entries, 1)
        self.binning.fill([{'x': 0.5, 'y': 10}, {'x': 1.5, 'y': 10}], [1, 2])
        self.assertEqual(self.b0.value, 6)
        self.assertEqual(self.b0.entries, 4)
        self.assertEqual(self.b1.value, 4)
        self.assertEqual(self.b1.entries, 2)
        self.binning.fill([{'x': -0.5, 'y': 10}, {'x': 1.5, 'y': 10}], [1, 2])
        self.assertEqual(self.b0.value, 6)
        self.assertEqual(self.b0.entries, 4)
        self.assertEqual(self.b1.value, 6)
        self.assertEqual(self.b1.entries, 3)
        self.assertRaises(ValueError, lambda: self.binning.fill({'x': -0.5, 'y': 10}, raise_error=True))
        self.assertEqual(self.b0.value, 6)
        self.assertEqual(self.b0.entries, 4)
        self.assertEqual(self.b1.value, 6)
        self.assertEqual(self.b1.entries, 3)
        self.binning.fill({'x': 0.5, 'y': 10, 'z': 123})
        self.assertEqual(self.b0.value, 7)
        self.assertEqual(self.b0.entries, 5)
        self.assertEqual(self.b1.value, 6)
        self.assertEqual(self.b1.entries, 3)
        self.assertRaises(KeyError, lambda: self.binning.fill({'x': 0.5}))
        str_arr = np.array([(0.5, 10), (0.5, 20)], dtype=[('x', float), ('y', float)])
        self.binning.fill(str_arr)
        self.assertEqual(self.b0.value, 9)
        self.assertEqual(self.b0.entries, 7)
        self.assertEqual(self.b1.value, 6)
        self.assertEqual(self.b1.entries, 3)
        self.binning.reset()
        self.assertEqual(self.b0.value, 0)
        self.assertEqual(self.b0.entries, 0)
        self.assertEqual(self.b1.value, 0)
        self.assertEqual(self.b1.entries, 0)
        self.binning.reset(123)
        self.assertEqual(self.b0.value, 123)
        self.assertEqual(self.b1.value, 123)

    def test_fill_from_csv(self):
        """Test filling the Binning from a csv file."""
        self.binning.fill_from_csv_file('testdata/csv-test.csv')
        self.assertEqual(self.b0.value, 2)
        self.assertEqual(self.b1.value, 1)
        self.binning.fill_from_csv_file('testdata/weighted-csv-test.csv', weightfield='w')
        self.assertEqual(self.b0.value, 8)
        self.assertEqual(self.b1.value, 2)
        self.binning.fill_from_csv_file('testdata/weighted-csv-test.csv')
        self.assertEqual(self.b0.value, 10)
        self.assertEqual(self.b1.value, 3)

    def test_ndarray(self):
        """Test conversion from and to ndarrays."""
        self.b1.fill([0.5, 0.5])
        arr = self.binning.get_values_as_ndarray()
        self.assertEqual(arr.shape, (2,))
        self.assertEqual(arr[0], 0)
        self.assertEqual(arr[1], 1)
        arr = self.binning.get_entries_as_ndarray()
        self.assertEqual(arr.shape, (2,))
        self.assertEqual(arr[0], 0)
        self.assertEqual(arr[1], 2)
        arr[0] = 5
        arr[1] = 10
        self.binning.set_values_from_ndarray(arr)
        self.assertEqual(self.b0.value, 5)
        self.assertEqual(self.b1.value, 10)
        arr[0] = 50
        arr[1] = 100
        self.binning.set_entries_from_ndarray(arr)
        self.assertEqual(self.b0.entries, 50)
        self.assertEqual(self.b1.entries, 100)
        arr = self.binning.get_values_as_ndarray((1,2))
        self.assertEqual(arr.shape, (1,2))
        self.assertEqual(arr[0,0], 5)
        self.assertEqual(arr[0,1], 10)
        arr[0,0] = 50
        arr[0,1] = 100
        self.binning.set_values_from_ndarray(arr)
        self.assertEqual(self.b0.value, 50)
        self.assertEqual(self.b1.value, 100)
        arr = self.binning.get_entries_as_ndarray((2,1))
        self.assertEqual(arr.shape, (2,1))

    def test_inclusion(self):
        """Test checking whether an event is binned."""
        self.assertTrue({'x': 0.5, 'y': 10} in self.binning)
        self.assertTrue({'x': 1.5, 'y': 10} in self.binning)
        self.assertFalse({'x': 2.5, 'y': 10} in self.binning)

    def test_equality(self):
        """Test equality comparisons."""
        self.assertTrue(self.binning == self.binning)
        self.assertFalse(self.binning != self.binning)
        self.assertTrue(self.binning != self.binning0)
        self.assertFalse(self.binning == self.binning0)

    def test_yaml_representation(self):
        """Test whether the yaml parsing can reproduce the original object."""
        orig = self.binning
        reco = yaml.load(yaml.dump(orig))
        self.assertEqual(orig, reco)

class TestRectangularBinnings(unittest.TestCase):
    def setUp(self):
        self.bl = RectangularBinning(binedges={'x': [0,1,2], 'y': (-10,0,10,20,float('inf'))}, variables=['x', 'y'])
        self.bl0 = RectangularBinning(binedges={'x': [0,1], 'y': (-10,0,10,20,float('inf'))}, variables=['x', 'y'])
        self.bu = RectangularBinning(binedges={'x': np.linspace(0,2,3), 'y': (-10,0,10,20,float('inf'))}, variables=['x', 'y'], include_upper=True)
        self.bxyz = RectangularBinning(binedges={'x': [0,1,2], 'y': (-10,0,10,20,float('inf')), 'z': (0,1,2)}, variables=['x', 'y', 'z'])

    def test_tuples(self):
        """Test the translation of tuples to bin numbers and back."""
        self.assertEqual(self.bl.get_tuple_bin_number(self.bl.get_bin_number_tuple(None)), None)
        self.assertEqual(self.bl.get_tuple_bin_number(self.bl.get_bin_number_tuple(-1)), None)
        self.assertEqual(self.bl.get_tuple_bin_number(self.bl.get_bin_number_tuple(0)), 0)
        self.assertEqual(self.bl.get_tuple_bin_number(self.bl.get_bin_number_tuple(5)), 5)
        self.assertEqual(self.bl.get_tuple_bin_number(self.bl.get_bin_number_tuple(7)), 7)
        self.assertEqual(self.bl.get_tuple_bin_number(self.bl.get_bin_number_tuple(8)), None)
        self.assertEqual(self.bl.get_bin_number_tuple(self.bl.get_tuple_bin_number((1,3))), (1,3))
        self.assertEqual(self.bxyz.get_bin_number_tuple(self.bxyz.get_tuple_bin_number((1,3,0))), (1,3,0))
        self.assertEqual(self.bl.get_tuple_bin_number((1,2)), 6)
        self.assertEqual(self.bl.get_event_bin_number({'x': 0, 'y': -10}), 0)
        self.assertEqual(self.bl.get_event_bin_number({'x': 0, 'y': -0}), 1)
        self.assertEqual(self.bl.get_event_bin_number({'x': 1, 'y': -10}), 4)
        self.assertEqual(self.bl.get_event_bin_number({'x': -1, 'y': -10}), None)
        self.assertEqual(self.bu.get_event_bin_number({'x': 2, 'y': 30}), 7)

    def test_fill(self):
        """Test bin filling"""
        self.bl.fill({'x': 0.5, 'y': -10}, 1)
        self.bl.fill({'x': 0.5, 'y': 0}, 2)
        self.assertEqual(self.bl.bins[0].value, 1)
        self.assertEqual(self.bl.bins[0].entries, 1)
        self.assertEqual(self.bl.bins[1].value, 2)
        self.assertEqual(self.bl.bins[1].entries, 1)
        self.bl.fill([{'x': 0.5, 'y': -10}, {'x': 0.5, 'y': 0}], [1, 2])
        self.assertEqual(self.bl.bins[0].value, 2)
        self.assertEqual(self.bl.bins[0].entries, 2)
        self.assertEqual(self.bl.bins[1].value, 4)
        self.assertEqual(self.bl.bins[1].entries, 2)
        self.bl.fill({'x': 0.5, 'y': -10, 'z': 123})
        self.assertEqual(self.bl.bins[0].value, 3)
        self.assertEqual(self.bl.bins[0].entries, 3)
        self.assertEqual(self.bl.bins[1].value, 4)
        self.assertEqual(self.bl.bins[1].entries, 2)
        self.assertRaises(KeyError, lambda: self.bl.fill({'x': 0.5}))

    def test_ndarray(self):
        """Test ndarray representations."""
        self.bl.bins[5].fill([0.5, 0.5])
        arr = self.bl.get_values_as_ndarray()
        self.assertEqual(arr.shape, (8,))
        self.assertEqual(arr[0], 0)
        self.assertEqual(arr[5], 1)
        arr = self.bl.get_entries_as_ndarray()
        self.assertEqual(arr.shape, (8,))
        self.assertEqual(arr[0], 0)
        self.assertEqual(arr[5], 2)
        arr = self.bl.get_values_as_ndarray((2,4))
        self.assertEqual(arr.shape, (2,4))
        self.assertEqual(arr[0,0], 0)
        self.assertEqual(arr[1,1], 1)
        arr = self.bl.get_entries_as_ndarray((2,4))
        self.assertEqual(arr.shape, (2,4))
        self.assertEqual(arr[0,0], 0)
        self.assertEqual(arr[1,1], 2)
        arr[0,0] = 7
        arr[1,1] = 11
        self.bl.set_entries_from_ndarray(arr)
        self.assertEqual(self.bl.bins[0].entries, 7)
        self.assertEqual(self.bl.bins[5].entries, 11)
        arr[0,0] = 5
        arr[1,1] = 6
        self.bl.set_values_from_ndarray(arr)
        self.assertEqual(self.bl.bins[0].value, 5)
        self.assertEqual(self.bl.bins[5].value, 6)

    def test_inclusion(self):
        """Test checking whether an event is binned."""
        self.assertTrue({'x': 0.5, 'y': 10} in self.bl)
        self.assertTrue({'x': 1.5, 'y': 10} in self.bl)
        self.assertFalse({'x': 2.5, 'y': 10} in self.bl)
        self.assertTrue({'x': 0, 'y': 10} in self.bl)
        self.assertFalse({'x': 2, 'y': 10} in self.bl)
        self.assertFalse({'x': 0, 'y': 10} in self.bu)
        self.assertTrue({'x': 2, 'y': 10} in self.bu)

    def test_equality(self):
        """Test equality comparisons."""
        self.assertTrue(self.bl == self.bl)
        self.assertFalse(self.bl != self.bl)
        self.assertTrue(self.bl != self.bl0)
        self.assertFalse(self.bl == self.bl0)
        self.assertTrue(self.bl != self.bu)
        self.assertFalse(self.bl == self.bu)

    def test_cartesian_product(self):
        """Test combining disjunct binnings."""
        bz = RectangularBinning(binedges={'z': [0,1,2]}, variables=['z'])
        bt = self.bl.cartesian_product(bz)
        self.assertEqual(bt, self.bxyz)

    def test_marginalization(self):
        """Test marginalizations of rectangular binnings."""
        self.bxyz.fill({'x':0, 'y':0, 'z':0})
        self.bxyz.fill({'x':0, 'y':0, 'z':1}, weight=2.)
        nb = self.bxyz.marginalize(['z'])
        self.assertEqual(nb, self.bl)
        self.assertEqual(nb.bins[1].entries, 2)
        self.assertEqual(nb.bins[1].value, 3.)

    def test_projection(self):
        """Test projections of rectangular binnings."""
        self.bxyz.fill({'x':0, 'y':0, 'z':0})
        self.bxyz.fill({'x':0, 'y':0, 'z':1}, weight=2.)
        nb = self.bxyz.project(['x', 'y'])
        self.assertEqual(nb, self.bl)
        self.assertEqual(nb.bins[1].entries, 2)
        self.assertEqual(nb.bins[1].value, 3.)

    def test_plots(self):
        """Test plots."""
        f = open('/dev/null', 'w')
        figax = self.bl.plot_entries(f, kwargs1d={'label': 'entries'})
        self.bl.plot_values(f, figax=figax)
        self.bl.plot_values(f, variables=(None,None), kwargs2d={'label': 'values'})

    def test_yaml_representation(self):
        """Test whether the yaml parsing can reproduce the original object."""
        orig = self.bl
        reco = yaml.load(yaml.dump(orig))
        self.assertEqual(orig, reco)

class TestResponseMatrices(unittest.TestCase):
    def setUp(self):
        with open('testdata/test-truth-binning.yml', 'r') as f:
            self.tb = yaml.load(f)
        with open('testdata/test-reco-binning.yml', 'r') as f:
            self.rb = yaml.load(f)
        self.rm = ResponseMatrix(self.rb, self.tb)

    def test_plots(self):
        """Test plots."""
        f = open('/dev/null', 'w')
        self.rm.plot_entries(f)
        self.rm.plot_values(f)

    def test_fill(self):
        self.rm.fill_from_csv_file('testdata/test-data.csv', weightfield='w')
        reco = self.rm.get_reco_entries_as_ndarray((2,2))
        self.assertEqual(reco[0,0], 3)
        self.assertEqual(reco[1,0], 1)
        self.assertEqual(reco[0,1], 2)
        self.assertEqual(reco[1,1], 2)
        truth = self.rm.get_truth_entries_as_ndarray((2,2))
        self.assertEqual(truth[0,0], 2)
        self.assertEqual(truth[1,0], 2)
        self.assertEqual(truth[0,1], 3)
        self.assertEqual(truth[1,1], 2)
        resp = self.rm.get_response_entries_as_ndarray((2,2,2,2))
        self.assertEqual(resp[0,0,0,0], 2)
        self.assertEqual(resp[0,1,0,0], 0)
        self.assertEqual(resp[1,0,0,0], 0)
        self.assertEqual(resp[1,1,0,0], 0)
        self.assertEqual(resp[0,0,0,1], 1)
        self.assertEqual(resp[0,1,0,1], 1)
        self.assertEqual(resp[1,0,0,1], 0)
        self.assertEqual(resp[1,1,0,1], 0)
        self.assertEqual(resp[0,0,1,0], 0)
        self.assertEqual(resp[0,1,1,0], 0)
        self.assertEqual(resp[1,0,1,0], 1)
        self.assertEqual(resp[1,1,1,0], 1)
        self.assertEqual(resp[0,0,1,1], 0)
        self.assertEqual(resp[0,1,1,1], 1)
        self.assertEqual(resp[1,0,1,1], 0)
        self.assertEqual(resp[1,1,1,1], 1)
        reco = self.rm.get_reco_values_as_ndarray((2,2))
        self.assertEqual(reco[0,0], 3)
        self.assertEqual(reco[1,0], 1)
        self.assertEqual(reco[0,1], 2)
        self.assertEqual(reco[1,1], 2)
        truth = self.rm.get_truth_values_as_ndarray((2,2))
        self.assertEqual(truth[0,0], 2)
        self.assertEqual(truth[1,0], 2)
        self.assertEqual(truth[0,1], 4)
        self.assertEqual(truth[1,1], 2)
        resp = self.rm.get_response_values_as_ndarray((2,2,2,2))
        self.assertEqual(resp[0,0,0,0], 2)
        self.assertEqual(resp[0,1,0,0], 0)
        self.assertEqual(resp[1,0,0,0], 0)
        self.assertEqual(resp[1,1,0,0], 0)
        self.assertEqual(resp[0,0,0,1], 1)
        self.assertEqual(resp[0,1,0,1], 1)
        self.assertEqual(resp[1,0,0,1], 0)
        self.assertEqual(resp[1,1,0,1], 0)
        self.assertEqual(resp[0,0,1,0], 0)
        self.assertEqual(resp[0,1,1,0], 0)
        self.assertEqual(resp[1,0,1,0], 1)
        self.assertEqual(resp[1,1,1,0], 1)
        self.assertEqual(resp[0,0,1,1], 0)
        self.assertEqual(resp[0,1,1,1], 1)
        self.assertEqual(resp[1,0,1,1], 0)
        self.assertEqual(resp[1,1,1,1], 1)
        resp = self.rm.get_response_matrix_as_ndarray(16)
        self.assertEqual(resp.shape, (16,))
        self.rm.reset()
        reco = self.rm.get_reco_values_as_ndarray((2,2))
        self.assertEqual(reco[0,0], 0)
        truth = self.rm.get_truth_values_as_ndarray((2,2))
        self.assertEqual(truth[0,0], 0)
        resp = self.rm.get_response_values_as_ndarray((2,2,2,2))
        self.assertEqual(resp[0,0,0,0], 0)

    def test_matrix_consistency(self):
        """Test that matrix and truth vector reproduce the reco vector."""
        self.rm.fill_from_csv_file('testdata/test-data.csv', weightfield='w')
        reco = self.rm.get_reco_values_as_ndarray()
        truth = self.rm.get_truth_values_as_ndarray()
        resp = self.rm.get_response_matrix_as_ndarray()
        self.assertTrue(np.all(reco == resp.dot(truth)))

class TestLikelihoodMachines(unittest.TestCase):
    def setUp(self):
        with open('testdata/test-truth-binning.yml', 'r') as f:
            tb = yaml.load(f)
        with open('testdata/test-reco-binning.yml', 'r') as f:
            rb = yaml.load(f)
        rm = ResponseMatrix(rb, tb)
        rm.fill_from_csv_file('testdata/test-data.csv', weightfield='w')
        data_vector = rm.get_reco_entries_as_ndarray() # Entries because we need integer event numbers
        self.truth_vector = rm.get_truth_values_as_ndarray()
        response_matrix = []
        response_matrix.append(rm.get_response_matrix_as_ndarray())
        # Create a second response matric for systematics stuff
        rm._truth_binning.fill_from_csv_file('testdata/test-data.csv')
        response_matrix.append(rm.get_response_matrix_as_ndarray())
        self.L = LikelihoodMachine(data_vector, response_matrix[0])
        self.L2 = LikelihoodMachine(data_vector, response_matrix)

    def test_log_probabilities(self):
        """Test n-dimensional calculation of probabilities."""
        # Three data sets
        data = np.array([
                [2,4],
                [2,1],
                [2,0],
               ])

        # Two simple response matrices
        resp = np.array([
                [[1.,0.],
                 [0.,1.]],

                [[0.,1.],
                 [1.,0.]],
               ])

        # Five theories
        truth = np.array([
                 [2.,4.],
                 [2.,3.],
                 [2.,2.],
                 [2.,1.],
                 [2.,0.],
                ])

        # Calculate *all* probabilities:
        ret = LikelihoodMachine.log_probability(data, resp, truth)
        self.assertEqual(ret.shape, (3,2,5))
        np.testing.assert_allclose(ret, np.array(
            [[[-2.93972921, -3.0904575,  -3.71231793, -5.48490665,     -np.inf],
              [-4.32602357, -3.90138771, -3.71231793, -4.09861229,     -np.inf]],

             [[-3.92055846, -3.20824053, -2.61370564, -2.30685282,     -np.inf],
              [-3.22741128, -2.80277542, -2.61370564, -3.,             -np.inf]],

             [[-5.30685282, -4.30685282, -3.30685282, -2.30685282, -1.30685282],
              [-3.92055846, -3.4959226,  -3.30685282, -3.69314718,     -np.inf]]]
            ))

    def test_log_likelihood(self):
        """Test basic likelihood computation."""
        self.assertAlmostEqual(self.L.log_likelihood(self.truth_vector), -5.1096282421)
        ret = self.L.log_likelihood([self.truth_vector, self.truth_vector])
        self.assertAlmostEqual(ret[0], -5.1096282421)
        self.assertAlmostEqual(ret[1], -5.1096282421)
        ret = self.L.log_likelihood([[self.truth_vector]*2]*3)
        self.assertAlmostEqual(ret[0][0], -5.1096282421)
        self.assertAlmostEqual(ret[1][1], -5.1096282421)
        self.assertAlmostEqual(ret[2][1], -5.1096282421)
        ret = self.L2.log_likelihood(self.truth_vector, systematics='profile')
        self.assertAlmostEqual(ret, -5.1096282421)
        ret = self.L2.log_likelihood(self.truth_vector, systematics='marginal')
        self.assertAlmostEqual(ret, -5.58442415546)
        ret = self.L2.log_likelihood(self.truth_vector, systematics=None)
        self.assertAlmostEqual(ret[0], -5.1096282421)
        self.assertAlmostEqual(ret[1], -6.52011704)
        ret = self.L2.log_likelihood(self.truth_vector, systematics=(0,))
        self.assertAlmostEqual(ret, -5.1096282421)
        ret = self.L2.log_likelihood([self.truth_vector, self.truth_vector], systematics=(1,))
        self.assertAlmostEqual(ret[0], -6.52011704)
        self.assertAlmostEqual(ret[1], -6.52011704)
        ret = self.L2.log_likelihood([self.truth_vector, self.truth_vector, self.truth_vector], systematics=np.array([[0],[1],[0]]))
        self.assertAlmostEqual(ret[0], -5.1096282421)
        self.assertAlmostEqual(ret[1], -6.52011704)
        self.assertAlmostEqual(ret[2], -5.1096282421)
        self.truth_vector[0] += 1
        self.assertAlmostEqual(self.L.log_likelihood(self.truth_vector), -5.2465820247)

    def test_max_log_likelihood(self):
        """Test maximum likelihood calculation with CompositeHypotheses"""
        fun = lambda x: x
        H = CompositeHypothesis(fun, [(0,None)]*4)
        ret = self.L.max_log_likelihood(H, method='basinhopping')
        ll, x = ret.L, ret.x
        self.assertAlmostEqual(ll, -5.110, places=3)
        self.assertAlmostEqual(x[0], 2, places=2)
        self.assertAlmostEqual(x[1], 4, places=2)
        self.assertAlmostEqual(x[2], 2, places=2)
        self.assertAlmostEqual(x[3], 2, places=2)
        ret = self.L2.max_log_likelihood(H, systematics='profile')
        ll, x = ret.L, ret.x
        self.assertAlmostEqual(ll, -5.110, places=3)
        ret = self.L2.max_log_likelihood(H, systematics='marginal')
        ll, x = ret.L, ret.x
        self.assertAlmostEqual(ll, -5.518, places=3)
        self.assertAlmostEqual(x[0], 2.217, places=2)
        self.assertAlmostEqual(x[1], 5.410, places=2)
        self.assertAlmostEqual(x[2], 2.507, places=2)
        self.assertAlmostEqual(x[3], 2.181, places=2)
        fun = lambda x: np.repeat(x,2)
        H = CompositeHypothesis(fun, [(0,5),(0,5)])
        ret = self.L.max_log_likelihood(H, method='differential_evolution')
        ll, x, s = ret.L, ret.x, ret.success
        self.assertTrue(s)
        self.assertAlmostEqual(ll, -5.145, places=3)
        self.assertAlmostEqual(x[0], 2.49, places=2)
        self.assertAlmostEqual(x[1], 2.13, places=2)
        ret = self.L.max_log_likelihood(H, method='basinhopping')
        ll, x = ret.L, ret.x
        self.assertAlmostEqual(ll, -5.145, places=3)
        self.assertAlmostEqual(x[0], 2.49, places=2)
        self.assertAlmostEqual(x[1], 2.13, places=2)

    def test_absolute_max_log_likelihood(self):
        """Test absolute likelihood maximisation."""
        ret = self.L.absolute_max_log_likelihood()
        ll, x, s = ret.L, ret.x, ret.lowest_optimization_result.success
        self.assertTrue(s)
        self.assertAlmostEqual(ll, -5.110, places=3)
        self.assertAlmostEqual(x[0], 2, places=2)
        self.assertAlmostEqual(x[1], 4, places=2)
        self.assertAlmostEqual(x[2], 2, places=2)
        self.assertAlmostEqual(x[3], 2, places=2)
        self.L.data_vector[0] += 1
        self.L.data_vector[1] -= 1
        self.L.data_vector[2] += 1
        self.L.data_vector[3] -= 1
        ret = self.L.absolute_max_log_likelihood(kwargs={'niter': 10})
        s = ret.lowest_optimization_result.success
        self.assertTrue(s)

    def test_data_sample_generation(self):
        """Test the generatrion of random samples."""
        truth = np.array([1.,2.,3.,4.])
        mc = LikelihoodMachine.generate_random_data_sample(self.L.response_matrix, truth)
        self.assertEqual(mc.shape, (4,))
        mc = LikelihoodMachine.generate_random_data_sample(self.L.response_matrix, truth, 3)
        self.assertEqual(mc.shape, (3,4))
        mc = LikelihoodMachine.generate_random_data_sample(self.L.response_matrix, truth, (5,6))
        self.assertEqual(mc.shape, (5,6,4))

    def test_likelihood_p_value(self):
        """Test the calculation of p-values."""
        p = self.L.likelihood_p_value(self.truth_vector)
        self.assertEqual(p, 1.0)
        self.truth_vector[0] += 4
        p = self.L.likelihood_p_value(self.truth_vector, N=250000)
        self.assertAlmostEqual(p, 0.612, places=2)
        p = self.L.likelihood_p_value(self.truth_vector, N=250000)
        self.assertAlmostEqual(p, 0.612, places=2)
        p = self.L.likelihood_p_value(self.truth_vector, N=250000)
        self.assertAlmostEqual(p, 0.612, places=2)
        p = self.L.likelihood_p_value(self.truth_vector, N=250000)
        self.assertAlmostEqual(p, 0.612, places=2)

    def test_max_likelihood_p_value(self):
        """Test the calculation of the p-value of composite hypotheses."""
        fun = lambda x: x
        H = CompositeHypothesis(fun, [(0,None)]*4)
        ret = self.L.max_log_likelihood(H, kwargs={'niter':2})
        p = self.L.max_likelihood_p_value(H, ret.x, kwargs={'niter':2})
        self.assertTrue(0.3 < p and p < 0.5)
        fun = lambda x: np.repeat(x,4)
        H = CompositeHypothesis(fun, [(0,None)])
        ret = self.L.max_log_likelihood(H, kwargs={'niter':2})
        p = self.L.max_likelihood_p_value(H, ret.x, kwargs={'niter':2})
        self.assertTrue(0.7 < p and p < 0.9)

    def test_max_likelihood_ratio_p_value(self):
        """Test the calculation of the p-value of composite hypotheses comparisons."""
        fun1 = lambda x: x
        H1 = CompositeHypothesis(fun1, [(0,None)]*4)
        ret1 = self.L.max_log_likelihood(H1, kwargs={'niter':2})
        fun = lambda x: np.repeat(x,2)
        H = CompositeHypothesis(fun, [(0,None),(0,None)])
        ret = self.L.max_log_likelihood(H, kwargs={'niter':2})
        p = self.L.max_likelihood_ratio_p_value(H, H1, par0=ret.x, par1=ret1.x, kwargs={'niter':2})
        self.assertTrue(0.8 < p and p <= 1.0)
        fun = lambda x: np.repeat(x,4)
        H = CompositeHypothesis(fun, [(0,None)])
        p = self.L.max_likelihood_ratio_p_value(H, H1, kwargs={'niter':2})
        self.assertTrue(0.8 < p and p <= 1.0)

    def test_mcmc(self):
        """Test Marcov Chain Monte Carlo."""
        fun = lambda x: np.repeat(x, 2, axis=-1)
        pri = JeffreysPrior(self.L.response_matrix, fun, [(0,100), (0,100)], (50,50))
        H = CompositeHypothesis(fun, parameter_priors=[pri], parameter_names=['x'])
        M = self.L.mcmc(H)
        M.sample(100, burn=50, thin=10, tune_interval=10, progress_bar=False)

    def test_plr(self):
        fun = lambda x: np.repeat(x, 2, axis=-1)
        pri = JeffreysPrior(self.L.response_matrix, fun, [(0,100), (0,100)], (50,50))
        H0 = CompositeHypothesis(fun, parameter_priors=[pri], parameter_names=['x'])
        fun = lambda x: np.repeat(x, 4, axis=-1)
        pri = JeffreysPrior(self.L.response_matrix, fun, [(0,100),], (50,))
        H1 = CompositeHypothesis(fun, parameter_priors=[pri], parameter_names=['x'])
        PLR, pref = self.L.plr(H0, [[50,50], [51,49]], [[0], [0]], H1, [[50,], [51,]], [[0], [0]])
        self.assertEqual(PLR.size, 4)

    def test_plots(self):
        """Test plots."""
        f = open('/dev/null', 'w')
        self.L.plot_bin_efficiencies(f)

if __name__ == '__main__':
    unittest.main()
