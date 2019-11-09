from __future__ import division
import sys
import unittest2 as unittest
import yaml
from remu.binning import *
from remu.migration import *
from remu.likelihood import *
from remu.plotting import *
from remu.matrix_utils import *
from remu.likelihood_utils import *
import numpy as np
from numpy import array, inf
import pandas as pd

if __name__ == '__main__':
    # Parse arguments for skipping tests
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--noproc", help="do not test multiprocess", action='store_true')
    args, testargs = parser.parse_known_args()
    noproc = args.noproc
    testargs = sys.argv[0:1] + testargs
else:
    noproc = False

class TestPhaseSpaces(unittest.TestCase):
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

    def test_clone(self):
        """Test whether the repr reproduces same object."""
        obj = self.psXYZ
        self.assertEqual(obj, obj.clone())

    def test_repr(self):
        """Test whether the repr reproduces same object."""
        self.assertEqual(self.psX, eval(repr(self.psX)))
        self.assertEqual(self.psXY, eval(repr(self.psXY)))
        self.assertEqual(self.psXYZ, eval(repr(self.psXYZ)))

    def test_yaml_representation(self):
        """Test whether the text parsing can reproduce the original object."""
        self.assertEqual(self.psX, yaml.full_load(yaml.dump(self.psX)))
        self.assertEqual(self.psXY, yaml.full_load(yaml.dump(self.psXY)))
        self.assertEqual(self.psXYZ, yaml.full_load(yaml.dump(self.psXYZ)))

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
        self.assertEqual(self.b0.sumw2, 1.0)
        self.b0.fill(0.5)
        self.assertEqual(self.b0.value, 1.5)
        self.assertEqual(self.b0.entries, 2)
        self.assertEqual(self.b0.sumw2, 1.25)
        self.b0.fill([0.5, 0.5, 0.5])
        self.assertEqual(self.b0.value, 3.0)
        self.assertEqual(self.b0.entries, 5)
        self.assertEqual(self.b0.sumw2, 2.0)

    def test_equality(self):
        """Test equality comparisons between bins."""
        self.assertTrue(self.b0 == self.b0)
        self.assertFalse(self.b0 != self.b0)
        self.assertTrue(self.b0 == self.b1)
        self.assertFalse(self.b0 != self.b1)
        self.b1.phasespace *= PhaseSpace(['abc'])
        self.assertFalse(self.b0 == self.b1)
        self.assertTrue(self.b0 != self.b1)

    def test_clone(self):
        """Test whether the repr reproduces same object."""
        obj = self.b0
        self.assertEqual(obj, obj.clone())

    def test_repr(self):
        """Test whether the repr reproduces same object."""
        self.assertEqual(self.b0.phasespace, eval(repr(self.b0)).phasespace)
        self.assertEqual(self.b0.value, eval(repr(self.b0)).value)
        self.assertEqual(self.b1.value, eval(repr(self.b1)).value)
        self.assertEqual(self.b2.value, eval(repr(self.b2)).value)

    def test_yaml_representation(self):
        """Test whether the yaml parsing can reproduce the original object."""
        orig = self.b0
        yml = yaml.dump(orig)
        reco = yaml.full_load(yml)
        self.assertEqual(orig, reco)

class TestRectangularBins(unittest.TestCase):
    def setUp(self):
        self.b = RectangularBin(['x', 'y'], [(0,1), (5,float('inf'))])
        self.c = RectangularBin(['x', 'y'], [(1,2), (5,float('inf'))])

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
        self.assertEqual(c[0], 0.5)
        self.assertEqual(c[1], float('inf'))

    def test_equality(self):
        """Test equality comparisons between bins."""
        self.assertTrue(self.b == self.b)
        self.assertFalse(self.b != self.b)
        self.assertTrue(self.b != self.c)
        self.assertFalse(self.b == self.c)

    def test_clone(self):
        """Test whether the repr reproduces same object."""
        obj = self.b
        self.assertEqual(obj, obj.clone())

    def test_repr(self):
        """Test whether the repr reproduces same object."""
        obj = self.b
        self.assertEqual(obj, eval(repr(obj)))

    def test_yaml_representation(self):
        """Test whether the yaml parsing can reproduce the original object."""
        orig = self.b
        yml = yaml.dump(orig)
        reco = yaml.full_load(yml)
        self.assertEqual(orig, reco)

class TestCartesianProductBins(unittest.TestCase):
    def setUp(self):
        self.x0 = RectangularBin(variables=['x'], edges=[(0,1)], dummy=True)
        self.x1 = RectangularBin(variables=['x'], edges=[(1,2)], dummy=True)
        self.y0 = RectangularBin(variables=['y'], edges=[(0,1)], dummy=True)
        self.y1 = RectangularBin(variables=['y'], edges=[(1,2)], dummy=True)
        self.z0 = RectangularBin(variables=['z'], edges=[(0,1)], dummy=True)
        self.z1 = RectangularBin(variables=['z'], edges=[(1,2)], dummy=True)
        self.bx = Binning(bins=[self.x0, self.x1], dummy=True)
        self.by = Binning(bins=[self.y0, self.y1], dummy=True)
        self.bz = Binning(bins=[self.z0, self.z1], dummy=True)
        self.b = CartesianProductBin([self.bx, self.by, self.bz], [0, 1, 0])
        self.c = CartesianProductBin([self.bx, self.by, self.bz], [0, 1, 1])

    def test_inclusion(self):
        """Test basic inclusion."""
        self.assertTrue({'x': 0, 'y': 1, 'z': 0} in self.b)
        self.assertTrue({'x': 0, 'y': 0, 'z': 0} not in self.b)
        self.assertTrue({'x': 0, 'y': 1, 'z': 1} in self.c)
        self.assertTrue({'x': 0, 'y': 0, 'z': 1} not in self.c)

    def test_equality(self):
        """Test equality comparisons between bins."""
        self.assertTrue(self.b == self.b)
        self.assertFalse(self.b != self.b)
        self.assertTrue(self.b != self.c)
        self.assertFalse(self.b == self.c)

    def test_clone(self):
        """Test whether the repr reproduces same object."""
        obj = self.b
        self.assertEqual(obj, obj.clone())

    def test_repr(self):
        """Test whether the repr reproduces same object."""
        obj = self.b
        self.assertEqual(obj, eval(repr(obj)))

    def test_yaml_representation(self):
        """Test whether the yaml parsing can reproduce the original object."""
        orig = self.b
        yml = yaml.dump(orig)
        reco = yaml.full_load(yml)
        self.assertEqual(orig, reco)

class TestBinnings(unittest.TestCase):
    def setUp(self):
        self.b0 = RectangularBin(variables=['x', 'y'], edges=[(0,1), (5,float('inf'))])
        self.b1 = RectangularBin(variables=['x', 'y'], edges=[(1,2), (5,float('inf'))])
        self.binning = Binning(bins=[self.b0, self.b1])
        self.binning0 = Binning(phasespace=self.b0.phasespace, bins=[self.b0.clone()])
        self.binning1 = Binning(bins=[self.b0.clone(), self.b1.clone()], subbinnings={0: self.binning.clone()})
        self.binning2 = Binning(bins=[self.b0.clone(), self.b1.clone()], subbinnings={0: self.binning1.clone()})

    def test_get_bin_indices(self):
        """Test the translation of events to bin numbers."""
        self.assertEqual(self.binning.get_event_bin_index({'x': 0, 'y': 10}), 0)
        self.assertEqual(self.binning.get_event_bin_index({'x': 1, 'y': 10}), 1)
        self.assertTrue(self.binning.get_event_bin_index({'x': 2, 'y': 10}) is None)

    def test_get_data_indices(self):
        """Test the translation of events to bin numbers."""
        self.assertEqual(self.binning1.get_event_data_index({'x': 0, 'y': 10}), 0)
        self.assertEqual(self.binning.get_event_data_index({'x': 1, 'y': 10}), 1)
        self.assertEqual(self.binning1.get_event_data_index({'x': 1, 'y': 10}), 2)
        self.assertEqual(self.binning2.get_event_data_index({'x': 1, 'y': 10}), 3)
        self.assertTrue(self.binning1.get_event_data_index({'x': 2, 'y': 10}) is None)

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
        self.binning.fill([{'x': -0.5, 'z': 10}, {'x': 1.5, 'z': 10}], [1, 2], rename={'z': 'y'})
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
        str_arr = np.array([(0.5, 10), (0.5, 20)], dtype=[('x', float), ('z', float)])
        self.binning.fill(str_arr, rename={'z': 'y'})
        self.assertEqual(self.b0.value, 11)
        self.assertEqual(self.b0.entries, 9)
        self.assertEqual(self.b1.value, 6)
        self.assertEqual(self.b1.entries, 3)
        df = pd.DataFrame({'x': [0.5, 0.5], 'z': [10, 20]})
        self.binning.fill(df, rename={'z': 'y'})
        self.assertEqual(self.b0.value, 13)
        self.assertEqual(self.b0.entries, 11)
        self.assertEqual(self.b1.value, 6)
        self.assertEqual(self.b1.entries, 3)
        self.binning.reset()
        str_arr = np.array([], dtype=[('x', float), ('y', float)])
        self.binning.fill(str_arr)
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
        self.binning.fill_from_csv_file('testdata/weighted-csv-test.csv', weightfield='w', buffer_csv_files=True)
        self.assertEqual(self.b0.value, 8)
        self.assertEqual(self.b1.value, 2)
        self.binning.fill_from_csv_file('testdata/weighted-csv-test.csv', buffer_csv_files=True)
        self.assertEqual(self.b0.value, 10)
        self.assertEqual(self.b1.value, 3)
        self.binning.fill_from_csv_file('testdata/weighted-csv-test.csv', buffer_csv_files=True, cut_function=lambda data: data[data['y'] < 15])
        self.assertEqual(self.b0.value, 11)
        self.assertEqual(self.b1.value, 4)
        self.binning.fill_from_csv_file(['testdata/csv-test.csv', 'testdata/csv-test.csv'], buffer_csv_files=True)
        self.assertEqual(self.b0.value, 15)
        self.assertEqual(self.b1.value, 6)
        self.binning.fill_from_csv_file('testdata/csv-test.csv', weight=0.5)
        self.assertEqual(self.b0.value, 16)
        self.assertAlmostEqual(self.b1.value, 6.5)
        self.binning.fill_from_csv_file(['testdata/csv-test.csv']*2, weight=[0.5, 2.0])
        self.assertEqual(self.b0.value, 21)
        self.assertAlmostEqual(self.b1.value, 9.0)

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
        arr = self.binning.get_sumw2_as_ndarray()
        self.assertEqual(arr.shape, (2,))
        self.assertEqual(arr[0], 0)
        self.assertEqual(arr[1], 0.5)
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
        arr[0] = 500
        arr[1] = 1000
        self.binning.set_sumw2_from_ndarray(arr)
        self.assertEqual(self.b0.sumw2, 500)
        self.assertEqual(self.b1.sumw2, 1000)
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
        arr = self.binning.get_sumw2_as_ndarray((2,1))
        self.assertEqual(arr.shape, (2,1))
        arr = self.binning.get_entries_as_ndarray(indices=[0])
        self.assertEqual(arr.shape, (1,))
        arr = self.binning.get_values_as_ndarray(indices=[0])
        self.assertEqual(arr.shape, (1,))
        arr = self.binning.get_sumw2_as_ndarray(indices=[0])
        self.assertEqual(arr.shape, (1,))

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

    def test_adding(self):
        """Test adding binnings."""
        new_binning = self.binning + self.binning
        self.assertTrue(np.all(self.binning.get_values_as_ndarray() * 2 == new_binning.get_values_as_ndarray()))

    def test_subbinning_modification(self):
        """Test marginalizing and inserting subbinnings."""
        orig = self.binning1
        orig.fill_from_csv_file('testdata/csv-test.csv')
        marg = orig.marginalize_subbinnings([0])
        self.assertEqual(orig.get_values_as_ndarray().sum(), marg.get_values_as_ndarray().sum())
        insrt = marg.insert_subbinning(0, orig.subbinnings[0].clone())
        self.assertEqual(orig.get_values_as_ndarray().sum(), insrt.get_values_as_ndarray().sum())
        self.assertEqual(orig, insrt)

    def test_clone(self):
        """Test whether the repr reproduces same object."""
        obj = self.binning
        self.assertEqual(obj, obj.clone())

    def test_repr(self):
        """Test whether the repr reproduces same object."""
        obj = self.binning
        self.assertEqual(obj, eval(repr(obj)))

    def test_yaml_representation(self):
        """Test whether the yaml parsing can reproduce the original object."""
        orig = self.binning
        yml = yaml.dump(orig)
        reco = yaml.full_load(yml)
        self.assertEqual(orig, reco)
        orig = self.binning0
        yml = yaml.dump(orig)
        reco = yaml.full_load(yml)
        self.assertEqual(orig, reco)
        orig = self.binning1
        yml = yaml.dump(orig)
        reco = yaml.full_load(yml)
        self.assertEqual(orig, reco)
        orig = self.binning2
        yml = yaml.dump(orig)
        reco = yaml.full_load(yml)
        self.assertEqual(orig, reco)

class TestCartesianProductBinnings(unittest.TestCase):
    def setUp(self):
        self.bx = LinearBinning('x', [0, 1, 2], dummy=True)
        self.by = LinearBinning('y', [0, 1, 2], dummy=True)
        self.bz = LinearBinning('z', [0, 1, 2], dummy=True)
        self.bynest = LinearBinning('y', [0, 1, 2], subbinnings={0: self.bz.clone()}, dummy=True)
        self.b0 = CartesianProductBinning([self.bx, self.bynest], subbinnings={2: self.bz.clone()})

    def test_get_event_data_index(self):
        """Test that events are put in the right data bins."""
        self.assertEqual(self.b0.get_event_data_index({'x': 0, 'y': 0, 'z': 0}), 0)
        self.assertEqual(self.b0.get_event_data_index({'x': 0, 'y': 0, 'z': 1}), 1)
        self.assertEqual(self.b0.get_event_data_index({'x': 0, 'y': 1, 'z': 0}), 2)
        self.assertEqual(self.b0.get_event_data_index({'x': 0, 'y': 1, 'z': 1}), 3)
        self.assertEqual(self.b0.get_event_data_index({'x': 1, 'y': 0, 'z': 0}), 4)
        self.assertEqual(self.b0.get_event_data_index({'x': 1, 'y': 0, 'z': 1}), 5)
        self.assertEqual(self.b0.get_event_data_index({'x': 1, 'y': 1, 'z': 0}), 6)
        self.assertEqual(self.b0.get_event_data_index({'x': 1, 'y': 1, 'z': 1}), 6)

    def test_adjacent_bins(self):
        """Test that all adjacent bins make sense."""
        ret = self.b0.get_adjacent_bin_indices()
        self.assertTrue(np.array_equal(ret[0], np.array([1,3])))
        self.assertTrue(np.array_equal(ret[1], np.array([0,4])))
        self.assertTrue(np.array_equal(ret[2], np.array([5])))
        self.assertTrue(np.array_equal(ret[5], np.array([2])))
        ret = self.b0.get_adjacent_data_indices()
        self.assertTrue(np.array_equal(ret[0], np.array([1,4])))
        self.assertTrue(np.array_equal(ret[1], np.array([0,5])))
        self.assertTrue(np.array_equal(ret[2], np.array([3])))
        self.assertTrue(np.array_equal(ret[3], np.array([2])))
        self.assertTrue(np.array_equal(ret[4], np.array([0,5])))
        self.assertTrue(np.array_equal(ret[5], np.array([1,4])))
        self.assertTrue(np.array_equal(ret[6], np.array([])))

    def test_bins(self):
        """Test that the bin proxy works."""
        A, B = self.b0.bins[3], CartesianProductBin([self.bx.clone(), self.bynest.clone()], [1, 0])
        self.assertEqual(A, B)

    def test_marginalization(self):
        """Test marginalizations of binnings."""
        self.b0.fill({'x': 0, 'y': 1, 'z': 0})
        self.b0.fill({'x': 0, 'y': 1, 'z': 1})
        self.assertEqual(self.b0.entries_array[2], 1)
        self.assertEqual(self.b0.entries_array[3], 1)
        self.assertEqual(self.b0.marginalize([0]).binnings[0], self.bynest)
        self.assertEqual(self.b0.marginalize(0), self.b0.marginalize([0]))
        self.assertEqual(self.b0.marginalize(0).entries_array[2], 2)

    def test_projection(self):
        """Test projections of binnings."""
        self.b0.fill({'x': 0, 'y': 1, 'z': 0})
        self.b0.fill({'x': 0, 'y': 1, 'z': 1})
        self.assertEqual(self.b0.entries_array[2], 1)
        self.assertEqual(self.b0.entries_array[3], 1)
        self.assertEqual(self.b0.project([1]).binnings[0], self.bynest)
        self.assertEqual(self.b0.project(1), self.b0.project([1]).binnings[0])
        self.assertEqual(self.b0.project(0).entries_array[0], 2)
        self.assertEqual(self.b0.project(1).entries_array[2], 2)

    def test_clone(self):
        """Test whether the repr reproduces same object."""
        obj = self.b0
        self.assertEqual(obj, obj.clone())

    def test_repr(self):
        """Test whether the repr reproduces same object."""
        obj = self.b0
        self.assertEqual(obj, eval(repr(obj)))

    def test_yaml_representation(self):
        """Test whether the yaml parsing can reproduce the original object."""
        orig = self.b0
        yml = yaml.dump(orig)
        reco = yaml.full_load(yml)
        self.assertEqual(orig, reco)

class TestRectangularBinnings(unittest.TestCase):
    def setUp(self):
        self.b0 = RectangularBinning(['x','y'], [((0,2),(0,2)), ((0,1),(2,3)), ((1,2),(2,3))])

    def test_get_event_data_index(self):
        """Test that events are put in the right data bins."""
        self.assertEqual(self.b0.get_event_data_index({'x': 0, 'y': 0}), 0)
        self.assertEqual(self.b0.get_event_data_index({'x': 1, 'y': 1}), 0)
        self.assertEqual(self.b0.get_event_data_index({'x': 0, 'y': 2}), 1)
        self.assertEqual(self.b0.get_event_data_index({'x': 1, 'y': 2}), 2)

    def test_clone(self):
        """Test whether the repr reproduces same object."""
        obj = self.b0
        self.assertEqual(obj, obj.clone())

    def test_repr(self):
        """Test whether the repr reproduces same object."""
        obj = self.b0
        self.assertEqual(obj, eval(repr(obj)))

    def test_yaml_representation(self):
        """Test whether the yaml parsing can reproduce the original object."""
        orig = self.b0
        yml = yaml.dump(orig)
        reco = yaml.full_load(yml)
        self.assertEqual(orig, reco)

class TestLinearBinnings(unittest.TestCase):
    def setUp(self):
        self.bx = LinearBinning('x', [0, 1, 2])
        self.by = LinearBinning('y', [0, 1, 2])
        self.b0 = LinearBinning('x', [0, 1, 2], subbinnings={0: self.by.clone()})

    def test_get_event_data_index(self):
        """Test that events are put in the right data bins."""
        self.assertEqual(self.b0.get_event_data_index({'x': 0, 'y': 0}), 0)
        self.assertEqual(self.b0.get_event_data_index({'x': 0, 'y': 1}), 1)
        self.assertEqual(self.b0.get_event_data_index({'x': 1, 'y': 0}), 2)
        self.assertEqual(self.b0.get_event_data_index({'x': 1, 'y': 1}), 2)

    def test_adjacent_bins(self):
        """Test that all adjacent bins make sense."""
        ret = self.b0.get_adjacent_bin_indices()
        self.assertTrue(np.array_equal(ret[0], np.array([1])))
        self.assertTrue(np.array_equal(ret[1], np.array([0])))
        ret = self.b0.get_adjacent_data_indices()
        self.assertTrue(np.array_equal(ret[0], np.array([1])))
        self.assertTrue(np.array_equal(ret[1], np.array([0])))
        self.assertTrue(np.array_equal(ret[2], np.array([])))

    def test_bins(self):
        """Test that the bin proxy works."""
        A, B = self.b0.bins[0], RectangularBin(['x'], [(0,1)])
        self.assertEqual(A, B)

    def test_slice(self):
        """Test the slicing into new binning."""
        self.b0.set_values_from_ndarray([1,2,5])
        sl = self.b0.slice(1,2)
        ret = sl.get_values_as_ndarray()
        self.assertEqual(ret[0], 5)

    def test_remove_bin_edges(self):
        """Test the merging of bins."""
        self.b0.set_values_from_ndarray([1,2,5])
        sl = self.b0.remove_bin_edges([1])
        ret = sl.get_values_as_ndarray()
        self.assertEqual(ret[0], 8)
        sl = self.b0.remove_bin_edges([0])
        ret = sl.get_values_as_ndarray()
        self.assertEqual(ret[0], 5)
        sl = self.b0.remove_bin_edges([2])
        ret = sl.get_values_as_ndarray()
        self.assertEqual(ret[0], 3)

    def test_clone(self):
        """Test whether the repr reproduces same object."""
        obj = self.b0
        self.assertEqual(obj, obj.clone())

    def test_repr(self):
        """Test whether the repr reproduces same object."""
        obj = self.b0
        self.assertEqual(obj, eval(repr(obj)))

    def test_yaml_representation(self):
        """Test whether the yaml parsing can reproduce the original object."""
        orig = self.b0
        yml = yaml.dump(orig)
        reco = yaml.full_load(yml)
        self.assertEqual(orig, reco)

class TestRectilinearBinnings(unittest.TestCase):
    def setUp(self):
        self.bl = RectilinearBinning(variables=['x', 'y'], bin_edges=[[0,1,2], (-10,0,10,20,float('inf'))])
        self.bl0 = RectilinearBinning(variables=['x', 'y'], bin_edges=[[0,1], (-10,0,10,20,float('inf'))])
        self.bu = RectilinearBinning(variables=['x', 'y'], bin_edges=[np.linspace(0,2,3), (-10,0,10,20,float('inf'))], include_upper=True)
        self.bxyz = RectilinearBinning(variables=['x', 'y', 'z'], bin_edges=[[0,1,2], (-10,0,10,20,float('inf')), (0,1,2)])

    def test_tuples(self):
        """Test the translation of tuples to bin numbers and back."""
        self.assertEqual(self.bl.get_tuple_bin_index(self.bl.get_bin_index_tuple(None)), None)
        self.assertEqual(self.bl.get_tuple_bin_index(self.bl.get_bin_index_tuple(-1)), None)
        self.assertEqual(self.bl.get_tuple_bin_index(self.bl.get_bin_index_tuple(0)), 0)
        self.assertEqual(self.bl.get_tuple_bin_index(self.bl.get_bin_index_tuple(5)), 5)
        self.assertEqual(self.bl.get_tuple_bin_index(self.bl.get_bin_index_tuple(7)), 7)
        self.assertEqual(self.bl.get_tuple_bin_index(self.bl.get_bin_index_tuple(8)), None)
        self.assertEqual(self.bl.get_bin_index_tuple(self.bl.get_tuple_bin_index((1,3))), (1,3))
        self.assertEqual(self.bxyz.get_bin_index_tuple(self.bxyz.get_tuple_bin_index((1,3,0))), (1,3,0))
        self.assertEqual(self.bl.get_tuple_bin_index((1,2)), 6)
        self.assertEqual(self.bl.get_event_bin_index({'x': 0, 'y': -10}), 0)
        self.assertEqual(self.bl.get_event_bin_index({'x': 0, 'y': -0}), 1)
        self.assertEqual(self.bl.get_event_bin_index({'x': 1, 'y': -10}), 4)
        self.assertEqual(self.bl.get_event_bin_index({'x': -1, 'y': -10}), None)
        self.assertEqual(self.bu.get_event_bin_index({'x': 2, 'y': 30}), 7)

    def test_fill(self):
        """Test bin filling"""
        self.bl.fill({'x': 0.5, 'y': -10}, 1)
        self.bl.fill({'x': 0.5, 'y': 0}, 2)
        self.assertEqual(self.bl.bins[0].value, 1)
        self.assertEqual(self.bl.bins[0].entries, 1)
        self.assertEqual(self.bl.bins[0].sumw2, 1)
        self.assertEqual(self.bl.bins[1].value, 2)
        self.assertEqual(self.bl.bins[1].entries, 1)
        self.assertEqual(self.bl.bins[1].sumw2, 4)
        self.bl.fill([{'x': 0.5, 'y': -10}, {'x': 0.5, 'y': 0}], [1, 2])
        self.assertEqual(self.bl.bins[0].value, 2)
        self.assertEqual(self.bl.bins[0].entries, 2)
        self.assertEqual(self.bl.bins[0].sumw2, 2)
        self.assertEqual(self.bl.bins[1].value, 4)
        self.assertEqual(self.bl.bins[1].entries, 2)
        self.assertEqual(self.bl.bins[1].sumw2, 8)
        self.bl.fill({'x': 0.5, 'y': -10, 'z': 123})
        self.assertEqual(self.bl.bins[0].value, 3)
        self.assertEqual(self.bl.bins[0].entries, 3)
        self.assertEqual(self.bl.bins[0].sumw2, 3)
        self.assertEqual(self.bl.bins[1].value, 4)
        self.assertEqual(self.bl.bins[1].entries, 2)
        self.assertEqual(self.bl.bins[1].sumw2, 8)
        df = pd.DataFrame({'x': [0.5, 0.5], 'z': [-10, -10]})
        self.bl.fill(df, rename={'z': 'y'})
        self.assertEqual(self.bl.bins[0].value, 5)
        self.assertEqual(self.bl.bins[0].entries, 5)
        self.assertEqual(self.bl.bins[0].sumw2, 5)
        self.assertEqual(self.bl.bins[1].value, 4)
        self.assertEqual(self.bl.bins[1].entries, 2)
        self.assertEqual(self.bl.bins[1].sumw2, 8)
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
        arr = self.bl.get_sumw2_as_ndarray()
        self.assertEqual(arr.shape, (8,))
        self.assertEqual(arr[0], 0)
        self.assertEqual(arr[5], 0.5)
        arr = self.bl.get_values_as_ndarray((2,4))
        self.assertEqual(arr.shape, (2,4))
        self.assertEqual(arr[0,0], 0)
        self.assertEqual(arr[1,1], 1)
        arr = self.bl.get_entries_as_ndarray((2,4))
        self.assertEqual(arr.shape, (2,4))
        self.assertEqual(arr[0,0], 0)
        self.assertEqual(arr[1,1], 2)
        arr = self.bl.get_sumw2_as_ndarray((2,4))
        self.assertEqual(arr.shape, (2,4))
        self.assertEqual(arr[0,0], 0)
        self.assertEqual(arr[1,1], 0.5)
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
        arr[0,0] = 9
        arr[1,1] = 13
        self.bl.set_sumw2_from_ndarray(arr)
        self.assertEqual(self.bl.bins[0].sumw2, 9)
        self.assertEqual(self.bl.bins[5].sumw2, 13)
        arr = self.bl.get_entries_as_ndarray(indices=[0])
        self.assertEqual(arr.shape, (1,))
        arr = self.bl.get_values_as_ndarray(indices=[0])
        self.assertEqual(arr.shape, (1,))
        arr = self.bl.get_sumw2_as_ndarray(indices=[0])
        self.assertEqual(arr.shape, (1,))

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
        self.assertFalse(self.bl == self.bxyz)
        self.assertFalse(self.bxyz == self.bl)

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

    def test_slice(self):
        """Test slices of rectangular binnings."""
        self.bxyz.fill({'x':0, 'y':10, 'z':0}, weight=2.)
        self.bxyz.fill({'x':1, 'y':10, 'z':1})
        self.bxyz.fill({'x':0, 'y':0, 'z':0}, weight=2.)
        nb = self.bxyz.slice({'x': (0,1), 'y': (2,-1)})
        val = nb.get_values_as_ndarray()
        ent = nb.get_entries_as_ndarray()
        self.assertEqual(val.sum(), 2)
        self.assertEqual(ent.sum(), 1)

    def test_remove_bin_edges(self):
        """Test rebinning of rectangular binnings."""
        self.bxyz.fill({'x':0, 'y':10, 'z':0}, weight=2.)
        self.bxyz.fill({'x':1, 'y':10, 'z':1})
        self.bxyz.fill({'x':0, 'y':0, 'z':0}, weight=2.)
        nb = self.bxyz.remove_bin_edges({'x': [2], 'y': [0,2]})
        val = nb.get_values_as_ndarray()
        ent = nb.get_entries_as_ndarray()
        sw2 = nb.get_sumw2_as_ndarray()
        self.assertEqual(val.shape[0], 4)
        self.assertEqual(val.sum(), 4)
        self.assertEqual(ent.sum(), 2)
        self.assertEqual(sw2.sum(), 8)

    def test_clone(self):
        """Test whether the repr reproduces same object."""
        obj = self.bl
        self.assertEqual(obj, obj.clone())

    def test_repr(self):
        """Test whether the repr reproduces same object."""
        obj = self.bl
        self.assertEqual(obj, eval(repr(obj)))

    def test_yaml_representation(self):
        """Test whether the yaml parsing can reproduce the original object."""
        orig = self.bl
        yml = yaml.dump(orig)
        reco = yaml.full_load(yml)
        self.assertEqual(orig, reco)

class TestResponseMatrices(unittest.TestCase):
    def setUp(self):
        with open('testdata/test-truth-binning.yml', 'r') as f:
            self.tb = yaml.full_load(f)
        with open('testdata/test-reco-binning.yml', 'r') as f:
            self.rb = yaml.full_load(f)
        self.rm = ResponseMatrix(self.rb, self.tb)

    def test_fill(self):
        self.rm.fill_from_csv_file('testdata/test-data.csv', weightfield='w')
        reco = self.rm.get_reco_entries_as_ndarray((2,2))
        self.assertEqual(reco[0,0], 1)
        self.assertEqual(reco[1,0], 1)
        self.assertEqual(reco[0,1], 2)
        self.assertEqual(reco[1,1], 2)
        truth = self.rm.get_truth_entries_as_ndarray((2,2))
        self.assertEqual(truth[0,0], 2)
        self.assertEqual(truth[1,0], 2)
        self.assertEqual(truth[0,1], 3)
        self.assertEqual(truth[1,1], 2)
        resp = self.rm.get_response_entries_as_ndarray((2,2,2,2))
        self.assertEqual(resp[0,0,0,0], 0)
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
        self.assertEqual(reco[0,0], 1)
        self.assertEqual(reco[1,0], 1)
        self.assertEqual(reco[0,1], 2)
        self.assertEqual(reco[1,1], 2)
        truth = self.rm.get_truth_values_as_ndarray((2,2))
        self.assertEqual(truth[0,0], 2)
        self.assertEqual(truth[1,0], 2)
        self.assertEqual(truth[0,1], 4)
        self.assertEqual(truth[1,1], 2)
        resp = self.rm.get_response_values_as_ndarray((2,2,2,2))
        self.assertEqual(resp[0,0,0,0], 0)
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
        self.assertEqual(len(self.rm.filled_truth_indices), 4)
        resp = self.rm.get_response_matrix_as_ndarray(16)
        self.assertEqual(resp.shape, (16,))
        resp = self.rm.get_response_matrix_as_ndarray((4,2), truth_indices=[0,2])
        self.assertEqual(resp.shape, (4,2))
        self.rm.reset()
        reco = self.rm.get_reco_values_as_ndarray((2,2))
        self.assertEqual(reco[0,0], 0)
        truth = self.rm.get_truth_values_as_ndarray((2,2))
        self.assertEqual(truth[0,0], 0)
        resp = self.rm.get_response_values_as_ndarray((2,2,2,2))
        self.assertEqual(resp[0,0,0,0], 0)
        self.rm.set_reco_values_from_ndarray(reco)
        self.rm.set_truth_values_from_ndarray(truth)
        self.rm.set_response_values_from_ndarray(resp)
        self.rm.set_reco_entries_from_ndarray(reco)
        self.rm.set_truth_entries_from_ndarray(truth)
        self.rm.set_response_entries_from_ndarray(resp)
        self.rm.set_reco_sumw2_from_ndarray(reco)
        self.rm.set_truth_sumw2_from_ndarray(truth)
        self.rm.set_response_sumw2_from_ndarray(resp)

    def test_matrix_consistency(self):
        """Test that matrix and truth vector reproduce the reco vector."""
        self.rm.fill_from_csv_file('testdata/test-data.csv', weightfield='w')
        reco = self.rm.get_reco_values_as_ndarray()
        truth = self.rm.get_truth_values_as_ndarray()
        self.rm.fill_up_truth_from_csv_file('testdata/test-data.csv', weightfield='w') # This should do nothing
        resp = self.rm.get_response_matrix_as_ndarray()
        self.assertTrue(np.all(reco == resp.dot(truth)))

    def test_variance(self):
        """Test the variance calculation."""
        self.rm.fill_from_csv_file('testdata/test-data.csv', weightfield='w')
        var = self.rm.get_statistical_variance_as_ndarray()
        self.assertAlmostEqual(var[0,0], 0.0046875)
        self.assertAlmostEqual(var[1,1], 0.00952183)
        self.assertAlmostEqual(var[2,2], 0.02202381)
        self.assertAlmostEqual(var[3,3], 0.02202381)
        var = self.rm.get_statistical_variance_as_ndarray(truth_indices=[0,-1])
        self.assertEqual(var.shape, (4,2))

    def test_mean(self):
        """Test the mean calculation."""
        self.rm.fill_from_csv_file('testdata/test-data.csv', weightfield='w')
        mean = self.rm.get_mean_response_matrix_as_ndarray()
        self.assertAlmostEqual(mean[0,0], 0.0625)
        self.assertAlmostEqual(mean[1,1], 0.15)
        self.assertAlmostEqual(mean[2,2], 0.25)
        self.assertAlmostEqual(mean[3,3], 0.25)
        mean = self.rm.get_mean_response_matrix_as_ndarray(truth_indices=[0,1])
        self.assertEqual(mean.shape, (4,2))

    def test_in_bin_variation(self):
        """Test the in-bin variation calculation."""
        self.rm.fill_from_csv_file('testdata/test-data.csv', weightfield='w')
        var = self.rm.get_in_bin_variation_as_ndarray()
        self.assertAlmostEqual(var[0,0], 0.7340425)
        self.assertAlmostEqual(var[1,1], 0.7340425)
        self.assertAlmostEqual(var[2,2], 1.1472384)
        self.assertAlmostEqual(var[3,3], 0.8583145)
        var = self.rm.get_in_bin_variation_as_ndarray(shape=(2,2,2,1), truth_indices=[0,-1])
        self.assertEqual(var.shape, (2,2,2,1))

    def test_random_generation(self):
        """Test generation of randomly varied matrices."""
        self.rm.fill_from_csv_file('testdata/test-data.csv', weightfield='w')
        ret = self.rm.generate_random_response_matrices()
        self.assertEqual(ret.shape, (4,4))
        ret = self.rm.generate_random_response_matrices(impossible_indices=[0,3])
        self.assertEqual(ret.shape, (4,4))
        ret = self.rm.generate_random_response_matrices(truth_indices=[0,2])
        self.assertEqual(ret.shape, (4,2))
        ret = self.rm.generate_random_response_matrices(2)
        self.assertEqual(ret.shape, (2,4,4))
        ret = self.rm.generate_random_response_matrices(2, truth_indices=[1,2,3])
        self.assertEqual(ret.shape, (2,4,3))
        ret = self.rm.generate_random_response_matrices((2,3), shape=(2,8), nuisance_indices=[0])
        self.assertEqual(ret.shape, (2,3,2,8))
        ret = self.rm.generate_random_response_matrices((2,3), shape=(4,3), nuisance_indices=[1,3], truth_indices=[0,2,3])
        self.assertEqual(ret.shape, (2,3,4,3))

    def test_add(self):
        """Test adding two response matrices."""
        self.rm.fill_from_csv_file('testdata/test-data.csv', weightfield='w')
        new_rm = self.rm + self.rm
        self.assertTrue(np.all(new_rm.get_truth_values_as_ndarray() == 2 * self.rm.get_truth_values_as_ndarray()))
        self.assertTrue(np.all(new_rm.get_reco_values_as_ndarray() == 2 * self.rm.get_reco_values_as_ndarray()))
        self.assertTrue(np.all(new_rm.get_response_values_as_ndarray() == 2 * self.rm.get_response_values_as_ndarray()))

    def test_export_and_load(self):
        """Test exporting and loading the response matrices in a single file."""
        with TemporaryFile() as f:
            self.rm.export(f, compress=False, sparse=False, nstat=10)
            f.seek(0)
            lm = LikelihoodMachine(np.array([1,1,1,1]), f)
            self.assertEqual(lm._n_eff, 4)
        with TemporaryFile() as f:
            self.rm.export(f, compress=True, sparse=True, nstat=None)
            f.seek(0)
            lm = LikelihoodMachine(np.array([1,1,1,1]), f)
            self.assertEqual(lm._n_eff, 0) # Testing with an empty matrix so there are no efficient bins

class TestResponseMatrixArrayBuilders(unittest.TestCase):
    def setUp(self):
        with open('testdata/test-truth-binning.yml', 'r') as f:
            self.tb = yaml.full_load(f)
        with open('testdata/test-reco-binning.yml', 'r') as f:
            self.rb = yaml.full_load(f)
        self.rm = ResponseMatrix(self.rb, self.tb, nuisance_indices=[2])
        self.builder = ResponseMatrixArrayBuilder(5)

    def test_mean(self):
        """Test ResponseMatrixArrayBuilder mean matrix."""
        self.builder.nstat = 0
        self.rm.fill_from_csv_file('testdata/test-data.csv', weightfield='w')
        self.builder.add_matrix(self.rm)
        self.rm.fill({'x_reco':1, 'y_reco':0, 'x_truth':1, 'y_truth':0})
        self.builder.add_matrix(self.rm)
        M = self.builder.get_mean_response_matrices_as_ndarray()
        self.assertEqual(M.shape, (4,4))

    def test_norandom(self):
        """Test ResponseMatrixArrayBuilder without random generation."""
        self.builder.nstat = 0
        self.rm.fill_from_csv_file('testdata/test-data.csv', weightfield='w')
        self.builder.add_matrix(self.rm)
        self.rm.fill({'x_reco':1, 'y_reco':0, 'x_truth':1, 'y_truth':0})
        self.builder.add_matrix(self.rm)
        M = self.builder.get_random_response_matrices_as_ndarray()
        self.assertEqual(M.shape, (2,4,4))

    def test_random(self):
        """Test ResponseMatrixArrayBuilder with random generation."""
        self.rm.fill_from_csv_file('testdata/test-data.csv', weightfield='w')
        self.builder.add_matrix(self.rm)
        self.rm.fill({'x_reco':1, 'y_reco':0, 'x_truth':1, 'y_truth':0})
        self.builder.add_matrix(self.rm)
        M = self.builder.get_random_response_matrices_as_ndarray()
        self.assertEqual(M.shape, (2,5,4,4))

    def test_truth_entries(self):
        """Test the truth entries array generation."""
        self.rm.fill_from_csv_file('testdata/test-data.csv', weightfield='w')
        self.builder.add_matrix(self.rm)
        self.rm.fill({'x_reco':1, 'y_reco':0, 'x_truth':1, 'y_truth':0})
        self.builder.add_matrix(self.rm)
        M = self.builder.get_truth_entries_as_ndarray()
        self.assertEqual(tuple(M), (2,3,3,2))

    def test_export_and_load(self):
        """Test exporting and loading the response matrices in a single file."""
        self.rm.fill_from_csv_file('testdata/test-data.csv', weightfield='w')
        self.builder.add_matrix(self.rm)
        self.rm.fill({'x_reco':1, 'y_reco':0, 'x_truth':1, 'y_truth':0})
        self.builder.add_matrix(self.rm)
        with TemporaryFile() as f:
            self.builder.export(f, compress=False)
            f.seek(0)
            LikelihoodMachine(np.array([1,1,1,1]), f)
        with TemporaryFile() as f:
            self.builder.export(f, compress=True)
            f.seek(0)
            LikelihoodMachine(np.array([1,1,1,1]), f)

class TestCompositeHypotheses(unittest.TestCase):
    def setUp(self):
        fun = lambda x: np.insert(x, 0, 0.)
        self.H = CompositeHypothesis(fun,
            parameter_limits=[(0,10), (0,20), (0,30)],
            parameter_priors=[lambda x: -np.log(x)]*3,
            parameter_names=['A','B','C'])

    def test_parameter_fixing(self):
        """Test fixing parameters to create new CompositeHypotheses."""
        H1 = self.H.fix_parameters((None, 1, 1))
        self.assertTrue(np.all(H1.translate([2]) == self.H.translate([2,1,1])))
        self.assertTrue(H1.parameter_limits == [(0,10)])
        self.assertTrue(len(H1.parameter_priors) == 1)
        self.assertTrue(H1.parameter_names == ['A'])

class TestTemplateHypotheses(unittest.TestCase):
    def setUp(self):
        self.H1 = TemplateHypothesis([[1,1,0,0],[0,0,1,1]], None, [(0,10),(0,10)])
        self.H2 = TemplateHypothesis([[1,1,0,0],[0,0,1,1]], [1,1,1,1])

    def test_parameter_fixing(self):
        """Test fixing parameters to create new CompositeHypotheses."""
        H0 = self.H1.fix_parameters((1, None))
        self.assertTrue(np.all(H0.translate([2]) == self.H1.translate([1,2])))
        H0 = self.H2.fix_parameters((1, None))
        self.assertTrue(np.all(H0.translate([2]) == self.H2.translate([1,2])))

class TestLikelihoodMachines(unittest.TestCase):
    def setUp(self):
        with open('testdata/test-truth-binning.yml', 'r') as f:
            tb = yaml.full_load(f)
        with open('testdata/test-reco-binning.yml', 'r') as f:
            rb = yaml.full_load(f)
        rm = ResponseMatrix(rb, tb)
        rm.fill_from_csv_file('testdata/test-data.csv', weightfield='w')
        data_vector = rm.get_reco_entries_as_ndarray() # Entries because we need integer event numbers
        self.truth_vector = rm.get_truth_values_as_ndarray()
        response_matrix = []
        response_matrix.append(rm.get_response_matrix_as_ndarray())
        # Create a second response matric for systematics stuff
        rm.truth_binning.fill_from_csv_file('testdata/test-data.csv')
        response_matrix.append(rm.get_response_matrix_as_ndarray())
        self.L = LikelihoodMachine(data_vector, response_matrix[0])
        self.L2 = LikelihoodMachine(data_vector, np.array([response_matrix])[...,[1,2,3]], truth_limits=[np.inf]*4, eff_indices=[1,2,3], is_sparse=True)
        self.L3 = LikelihoodMachine(data_vector, np.array([response_matrix])[...,[1,2,3]], truth_limits=[np.inf]*4, eff_indices=[1,2,3], is_sparse=True, matrix_weights=[2.,1.])
        self.L4 = LikelihoodMachine(data_vector, np.array([response_matrix[0]]+response_matrix)[...,[1,2,3]], truth_limits=[np.inf]*4, eff_indices=[1,2,3], is_sparse=True)

    def test_truth_folding(self):
        """Test folding truth vectors into reco space."""
        self.assertEqual(self.L.fold(self.truth_vector).shape, (4,))
        self.assertEqual(self.L2.fold(self.truth_vector).shape, (2,4))

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
        self.assertAlmostEqual(self.L.log_likelihood(self.truth_vector), -4.6137056388801092)
        ret = self.L.log_likelihood([self.truth_vector, self.truth_vector])
        self.assertAlmostEqual(ret[0], -4.6137056388801092)
        self.assertAlmostEqual(ret[1], -4.6137056388801092)
        ret = self.L.log_likelihood([[self.truth_vector]*2]*3)
        self.assertAlmostEqual(ret[0][0], -4.6137056388801092)
        self.assertAlmostEqual(ret[1][1], -4.6137056388801092)
        self.assertAlmostEqual(ret[2][1], -4.6137056388801092)
        ret = self.L2.log_likelihood(self.truth_vector, systematics='profile')
        self.assertAlmostEqual(ret, -4.6137056388801092)
        ret = self.L2.log_likelihood(self.truth_vector, systematics='marginal')
        self.assertAlmostEqual(ret, -5.0016299959913546)
        ret = self.L2.log_likelihood(self.truth_vector, systematics=None)
        self.assertAlmostEqual(ret[0], -4.6137056388801092)
        self.assertAlmostEqual(ret[1], -5.6439287294984988)
        ret = self.L3.log_likelihood(self.truth_vector, systematics='marginal')
        self.assertAlmostEqual(ret, -4.8549591379478764)
        ret4 = self.L4.log_likelihood(self.truth_vector, systematics='marginal')
        self.assertAlmostEqual(ret, ret4)
        ret = self.L2.log_likelihood(self.truth_vector, systematics=(0,))
        self.assertAlmostEqual(ret, -4.6137056388801092)
        ret = self.L2.log_likelihood([self.truth_vector, self.truth_vector], systematics=(1,))
        self.assertAlmostEqual(ret[0], -5.6439287294984988)
        self.assertAlmostEqual(ret[1], -5.6439287294984988)
        ret = self.L2.log_likelihood([self.truth_vector, self.truth_vector, self.truth_vector], systematics=np.array([[0],[1],[0]]))
        self.assertAlmostEqual(ret[0], -4.6137056388801092)
        self.assertAlmostEqual(ret[1], -5.6439287294984988)
        self.assertAlmostEqual(ret[2], -4.6137056388801092)
        self.truth_vector[0] += 1
        self.assertAlmostEqual(self.L.log_likelihood(self.truth_vector), -4.6137056388801092)
        self.L.truth_limits = np.ones_like(self.truth_vector)
        self.assertRaises(RuntimeError, lambda: self.L.log_likelihood(self.truth_vector))
        self.L.limit_method = 'prohibit'
        self.assertEqual(self.L.log_likelihood(self.truth_vector), -np.inf)
        self.L.limit_method = 'garbage'
        self.assertRaises(ValueError, lambda: self.L.log_likelihood(self.truth_vector))

    def test_best_possible_likelihood(self):
        self.assertAlmostEqual(self.L.best_possible_log_likelihood(self.truth_vector), -4.6137056388801092)
        ret = self.L2.best_possible_log_likelihood(self.truth_vector, systematics='marginal')
        self.assertAlmostEqual(ret, -3.5727488803986978)
        ret = self.L2.best_possible_log_likelihood([self.truth_vector, self.truth_vector*10])
        self.assertAlmostEqual(ret, -3.5727488803986978)
        ret3 = self.L3.best_possible_log_likelihood(self.truth_vector)
        ret4 = self.L4.best_possible_log_likelihood(self.truth_vector)
        self.assertAlmostEqual(ret3, ret4)

    def test_pseudo_chi2(self):
        self.assertAlmostEqual(self.L.pseudo_chi2(self.truth_vector), 0)

    def test_max_log_likelihood(self):
        """Test maximum likelihood calculation with CompositeHypotheses"""
        fun = lambda x: np.insert(x, 0, 0.)
        H = CompositeHypothesis(fun, [(0,None)]*3)
        ret = self.L.max_log_likelihood(H, method='basinhopping', systematics='profile')
        ll, x = ret.L, ret.x
        self.assertAlmostEqual(ll, -4.614, places=3)
        self.assertAlmostEqual(x[0], 4, places=2)
        self.assertAlmostEqual(x[1], 2, places=2)
        self.assertAlmostEqual(x[2], 2, places=2)
        self.assertAlmostEqual(ll, -4.614, places=3)
        H = TemplateHypothesis([[1,1,0,0],[0,0,1,1]], None, [(0,10),(0,10)])
        ret = self.L2.max_log_likelihood(H, method='differential_evolution', systematics='marginal')
        ll, x, s = ret.L, ret.x, ret.success
        self.assertTrue(s)
        self.assertAlmostEqual(ll, -4.920, places=3)
        self.assertAlmostEqual(x[0], 4.93, places=2)
        self.assertAlmostEqual(x[1], 2.53, places=2)
        H = TemplateHypothesis([[1,1,0,0],[0,0,1,1]], [1,1,1,1], [(0,10),(0,10)])
        ret = self.L2.max_log_likelihood(H, method='differential_evolution', systematics='marginal')
        ll, x, s = ret.L, ret.x, ret.success
        self.assertTrue(s)
        self.assertAlmostEqual(ll, -4.920, places=3)
        self.assertAlmostEqual(x[0], 3.93, places=2)
        self.assertAlmostEqual(x[1], 1.53, places=2)
        H = LinearHypothesis([[]], [1,1,1,1], parameter_limits=[])
        ret = self.L2.max_log_likelihood(H)
        self.assertAlmostEqual(ret.L, -7.170, places=3)
        ret = self.L3.max_log_likelihood(H)
        self.assertAlmostEqual(ret.L, -6.914, places=3)
        ret4 = self.L4.max_log_likelihood(H)
        self.assertAlmostEqual(ret.L, ret4.L)

    def test_data_sample_generation(self):
        """Test the generatrion of random samples."""
        truth = np.array([1.,2.,3.,4.])
        mc = LikelihoodMachine.generate_random_data_sample(self.L.response_matrix, truth)
        self.assertEqual(mc.shape, (4,))
        mc = LikelihoodMachine.generate_random_data_sample(self.L.response_matrix, truth, 3)
        self.assertEqual(mc.shape, (3,4))
        mc = LikelihoodMachine.generate_random_data_sample(self.L.response_matrix, truth, (5,6))
        self.assertEqual(mc.shape, (5,6,4))
        mc = LikelihoodMachine.generate_random_data_sample(self.L2.response_matrix, truth[1:], (5,6))
        self.assertEqual(mc.shape, (5,6,4))
        mc = LikelihoodMachine.generate_random_data_sample(self.L2.response_matrix, truth[1:], (5,6), each=True)
        self.assertEqual(mc.shape, (5,6,2,4))
        mc = LikelihoodMachine.generate_random_data_sample(self.L3.response_matrix, truth[1:], (5,6))
        self.assertEqual(mc.shape, (5,6,4))
        mc = LikelihoodMachine.generate_random_data_sample(self.L4.response_matrix, truth[1:], (5,6))
        self.assertEqual(mc.shape, (5,6,4))

    def test_likelihood_p_value(self):
        """Test the calculation of p-values."""
        p = self.L.likelihood_p_value(self.truth_vector)
        self.assertEqual(p, 1.0)
        self.truth_vector[2] += 4
        p = self.L.likelihood_p_value(self.truth_vector, N=250000)
        self.assertTrue(abs(p-0.725) < 0.01)
        p3 = self.L3.likelihood_p_value(self.truth_vector, N=250000)
        p4 = self.L4.likelihood_p_value(self.truth_vector, N=250000)
        self.assertTrue(abs(p3-p4) < 0.01)
        self.assertTrue(abs(p-p4) > 0.01)

    def test_max_likelihood_p_value(self):
        """Test the calculation of the p-value of composite hypotheses."""
        fun = lambda x: np.insert(x, 0, 0.)
        H = CompositeHypothesis(fun, [(0,None)]*3)
        ret = self.L.max_log_likelihood(H, kwargs={'niter':2})
        p = self.L.max_likelihood_p_value(H, ret.x, kwargs={'niter':2}, N=10)
        self.assertTrue(0. <= p <= 1.0)
        fun = lambda x: np.repeat(x,4)
        H = CompositeHypothesis(fun, [(0,None)])
        ret = self.L.max_log_likelihood(H, kwargs={'niter':2})
        p = self.L.max_likelihood_p_value(H, ret.x, kwargs={'niter':2}, N=10)
        self.assertTrue(0. <= p <= 1.0)
        p = self.L2.max_likelihood_p_value(H, ret.x, kwargs={'niter':2}, N=10)
        self.assertTrue(0. <= p <= 1.0)
        p = self.L3.max_likelihood_p_value(H, ret.x, kwargs={'niter':2}, N=10)
        self.assertTrue(0. <= p <= 1.0)

    def test_max_likelihood_ratio_p_value(self):
        """Test the calculation of the p-value of composite hypotheses comparisons."""
        fun1 = lambda x: x
        H1 = CompositeHypothesis(fun1, [(0,None)]*4)
        ret1 = self.L.max_log_likelihood(H1, kwargs={'niter':2})
        fun = lambda x: np.repeat(x,2)
        H = CompositeHypothesis(fun, [(0,None),(0,None)])
        ret = self.L.max_log_likelihood(H, kwargs={'niter':2})
        p = self.L.max_likelihood_ratio_p_value(H, H1, par0=ret.x, par1=ret1.x, kwargs={'niter':2}, N=10)
        self.assertTrue(0.0 <= p <= 1.0)
        p = self.L.max_likelihood_ratio_p_value(H, H1, par0=[10,10], par1=[1000,1000,1000,1000], kwargs={'niter':2}, N=10)
        self.assertTrue(0.0 <= p <= 1.0)
        fun = lambda x: np.repeat(x,4)
        H = CompositeHypothesis(fun, [(0,None)])
        p = self.L.max_likelihood_ratio_p_value(H, H1, kwargs={'niter':2}, N=10)
        self.assertTrue(0.0 <= p <= 1.0)
        p = self.L2.max_likelihood_ratio_p_value(H, H1, kwargs={'niter':2}, N=10)
        self.assertTrue(0.0 <= p <= 1.0)
        p = self.L3.max_likelihood_ratio_p_value(H, H1, kwargs={'niter':2}, N=10)
        self.assertTrue(0.0 <= p <= 1.0)

    def test_wilks_max_likelihood_ratio_p_value(self):
        """Test the calculation of Wilks' p-value of composite hypotheses comparisons."""
        fun1 = lambda x: x
        H1 = CompositeHypothesis(fun1, [(0,None)]*4)
        ret1 = self.L.max_log_likelihood(H1, kwargs={'niter':2})
        fun = lambda x: np.repeat(x,2)
        H = CompositeHypothesis(fun, [(0,None),(0,None)])
        ret = self.L.max_log_likelihood(H, kwargs={'niter':2})
        p = self.L.wilks_max_likelihood_ratio_p_value(H, H1, par0=ret.x, par1=ret1.x)
        self.assertTrue(0.0 <= p <= 1.0)
        p = self.L.wilks_max_likelihood_ratio_p_value(H, H1, par0=[10,10], par1=[1000,1000,1000,1000])
        self.assertTrue(0.0 <= p <= 1.0)
        fun = lambda x: np.repeat(x,4)
        H = CompositeHypothesis(fun, [(0,None)])
        p = self.L.wilks_max_likelihood_ratio_p_value(H, H1, kwargs={'niter':2})
        self.assertTrue(0.0 <= p <= 1.0)
        p = self.L2.wilks_max_likelihood_ratio_p_value(H, H1, kwargs={'niter':2})
        self.assertTrue(0.0 <= p <= 1.0)
        p3 = self.L3.wilks_max_likelihood_ratio_p_value(H, H1, kwargs={'niter':2})
        self.assertTrue(0.0 <= p3 <= 1.0)
        p4 = self.L4.wilks_max_likelihood_ratio_p_value(H, H1, kwargs={'niter':2})
        self.assertAlmostEqual(p3, p4)

    @unittest.skipIf(noproc, "Skipping multiprocess test.")
    def test_multiprocess(self):
        """Test parallelisation."""
        fun = lambda x: np.repeat(x,2)
        H = CompositeHypothesis(fun, [(0,None),(0,None)])
        self.L.max_likelihood_p_value(H, kwargs={'niter':2}, N=10, nproc=2)
        fun1 = lambda x: x
        H1 = CompositeHypothesis(fun1, [(0,None)]*4)
        self.L.max_likelihood_ratio_p_value(H, H1, kwargs={'niter':2}, N=10, nproc=2)

    def test_mcmc(self):
        """Test Marcov Chain Monte Carlo."""
        fun = lambda x: np.repeat(x, 2, axis=-1)
        pri = JeffreysPrior(self.L.response_matrix, fun, [(0,100), (0,100)], (50,50))
        H = CompositeHypothesis(fun, parameter_priors=[pri], parameter_names=['x'])
        M = self.L.mcmc(H)
        M.sample(100, burn=50, thin=10, tune_interval=10, progress_bar=False)
        fun_sparse = lambda x : np.asarray(x)[...,[0,1,1]]
        pri = JeffreysPrior(self.L2.response_matrix, fun_sparse, [(0,100), (0,100)], (50,50))
        H = CompositeHypothesis(fun, parameter_priors=[pri], parameter_names=['x'])
        M = self.L2.mcmc(H)
        M.sample(100, burn=50, thin=10, tune_interval=10, progress_bar=False)
        M = self.L3.mcmc(H)
        M.sample(100, burn=50, thin=10, tune_interval=10, progress_bar=False)

    def test_marginal_likelihood(self):
        """Test the marginal likelihood calculation"""
        fun = lambda x: np.repeat(x, 2, axis=-1)
        pri = JeffreysPrior(self.L.response_matrix, fun, [(0,100), (0,100)], (50,50))
        H0 = CompositeHypothesis(fun, parameter_priors=[pri], parameter_names=['x'])
        mll = self.L.marginal_log_likelihood(H0, [[50,50], [51,49]], [[0], [0]])
        self.assertTrue(mll < 0)

    def test_plr(self):
        fun = lambda x: np.repeat(x, 2, axis=-1)
        pri = JeffreysPrior(self.L.response_matrix, fun, [(0,100), (0,100)], (50,50))
        H0 = CompositeHypothesis(fun, parameter_priors=[pri], parameter_names=['x'])
        fun = lambda x: np.repeat(x, 4, axis=-1)
        pri = JeffreysPrior(self.L.response_matrix, fun, [(0,100),], (50,))
        H1 = CompositeHypothesis(fun, parameter_priors=[pri], parameter_names=['x'])
        PLR, pref = self.L.plr(H0, [[50,50], [51,49]], [[0], [0]], H1, [[50,], [51,]], [[0], [0]])
        self.assertEqual(PLR.size, 4)

class TestPlotting(unittest.TestCase):
    def setUp(self):
        pass

    def test_array_plotter(self):
        array = np.array([1,3,2,5])
        plt = get_plotter(array, bins_per_row=3)
        plt.plot_array(hatch=None)
        plt.plot_array(array*2, hatch=None)
        plt.plot_array([array, array*2], stack_function=np.mean, hatch=None)
        plt.plot_array([array, array*2], stack_function=0.68)
        plt.plot_array([array, array*2], stack_function=(np.mean, np.max), edgecolor='g')

    def test_binning_plotter(self):
        b0 = RectangularBin(variables=['x', 'y'], edges=[(0,1), (5,float('inf'))])
        b1 = RectangularBin(variables=['x', 'y'], edges=[(1,2), (5,float('inf'))])
        binning0 = Binning(bins=[b0, b1])
        binning = Binning(bins=[b0, b1], subbinnings={0: binning0.clone(dummy=True)})
        binning.set_values_from_ndarray([2.,1.,4.])
        binning.set_entries_from_ndarray([4,2,8])
        binning.set_sumw2_from_ndarray([1.,0.5,2.])
        plt = get_plotter(binning, marginalize_subbinnings=False)
        plt.plot_values(binning.clone())
        plt.plot_entries()
        plt.plot_sumw2()
        plt = get_plotter(binning, marginalize_subbinnings=True)
        plt.plot_sumw2(binning.clone(), label='Z')
        plt.plot_entries(label='Y', hatch='')
        plt.plot_values(label='X')
        plt.plot_array(binning.value_array)
        plt.plot_array(binning0.value_array)
        plt.legend()

    def test_cartesian_plotter(self):
        x0 = RectangularBin(variables=['x'], edges=[(0,1)], dummy=True)
        x1 = RectangularBin(variables=['x'], edges=[(1,2)], dummy=True)
        y0 = RectangularBin(variables=['y'], edges=[(0,1)], dummy=True)
        y1 = RectangularBin(variables=['y'], edges=[(1,2)], dummy=True)
        z0 = RectangularBin(variables=['z'], edges=[(0,1)], dummy=True)
        z1 = RectangularBin(variables=['z'], edges=[(1,2)], dummy=True)
        bx = Binning(bins=[x0, x1], dummy=True)
        by = Binning(bins=[y0, y1], dummy=True)
        bz = Binning(bins=[z0, z1], dummy=True)
        bynest = Binning(bins=[y0.clone(), y1.clone()], subbinnings={0: bz.clone()}, dummy=True)
        b0 = CartesianProductBinning([bx, bynest, bz], subbinnings={2: bz.clone()})
        b0.set_values_from_ndarray(np.arange(13))
        b0.set_entries_from_ndarray(np.arange(13)*2)
        b0.set_sumw2_from_ndarray(np.arange(13)*3)
        plt = get_plotter(b0)
        plt.plot_sumw2(label='Z')
        plt.plot_entries(label='Y')
        plt.plot_values(label='X')
        plt.plot_array(np.arange(26).reshape((2,13)), label='A', stack_function=0.9)
        plt.legend()

    def test_linear_plotter(self):
        b0 = LinearBinning('x', [0,1,5,7,inf])
        b0.set_values_from_ndarray(np.arange(4))
        b0.set_entries_from_ndarray(np.arange(4)*2)
        b0.set_sumw2_from_ndarray(np.arange(4)*3)
        plt = get_plotter(b0, bins_per_row=2)
        plt.plot_sumw2(label='Z')
        plt.plot_entries(label='Y')
        plt.plot_values(label='X')
        plt.legend()

    def test_rectilinear_plotter(self):
        b0 = RectilinearBinning(['x', 'y'], [[-1,0,1,5,7,inf], [-inf,0,5,10,inf]])
        b0.set_values_from_ndarray(np.arange(20))
        b0.set_entries_from_ndarray(np.arange(20)*2)
        b0.set_sumw2_from_ndarray(np.arange(20)*3)
        plt = get_plotter(b0)
        plt.plot_sumw2(label='Z', scatter=100)
        plt.plot_entries(label='Y', scatter=100)
        plt.plot_values(label='X', scatter=100)
        plt.legend()

class TestMatrixUtils(unittest.TestCase):
    def setUp(self):
        with open('testdata/test-truth-binning.yml', 'r') as f:
            tb = yaml.full_load(f)
        with open('testdata/test-reco-binning.yml', 'r') as f:
            rb = yaml.full_load(f)
        tb = tb.insert_subbinning(3, LinearBinning('y_truth', [1.0,1.5,2.0]))
        rm = ResponseMatrix(rb, tb)
        rm.fill_from_csv_file('testdata/test-data.csv', weightfield='w')
        self.rm = rm

    def test_mahalanobis_distance(self):
        rA = self.rm
        rB = rA.clone()
        rA.fill_from_csv_file('testdata/test-data.csv', weightfield='w')
        rB.fill_from_csv_file('testdata/test-data.csv', weightfield='w')
        rB.fill_from_csv_file('testdata/test-data.csv', weightfield='w')
        null_distance, distances = mahalanobis_distance(rA, rB, return_distances_from_mean=True)
        self.assertTrue(null_distance.shape == (5,))
        self.assertTrue(distances.shape == (104,5))
        self.assertTrue(np.all(null_distance >= 0.))
        self.assertTrue(np.all(distances >= 0.))

    def test_compatibility(self):
        rA = self.rm
        rB = rA.clone()
        rA.fill_from_csv_file('testdata/test-data.csv', weightfield='w')
        rB.fill_from_csv_file('testdata/test-data.csv', weightfield='w')
        rB.fill_from_csv_file('testdata/test-data.csv', weightfield='w')
        p_count, p_chi2, null_distance, distances, n_bins = compatibility(rA, rB, return_all=True, min_quality=0.0)
        self.assertTrue(p_count >= 0. and p_count <= 1.)
        self.assertTrue(p_chi2 >= 0. and p_count <= 1.)
        self.assertTrue(null_distance >= 0.)
        self.assertEqual(n_bins, 4*5)
        self.assertEqual(distances.size, 100+4)

    def test_plotting(self):
        """Test plotting matrices."""
        plot_compatibility(self.rm, self.rm.clone(), filename=None, min_quality=0.0)
        plot_mahalanobis_distance(self.rm, self.rm.clone(), plot_expectation=True, filename=None)
        plot_in_bin_variation(self.rm, filename=None)
        plot_statistical_uncertainty(self.rm, filename=None)
        plot_relative_in_bin_variation(self.rm, filename=None)
        plot_mean_efficiency(self.rm, filename=None)
        plot_mean_response_matrix(self.rm, filename=None)

    def test_improve_stats(self):
        """Test the automatic optimization of ResponseMatrices."""
        rm0 = improve_stats(self.rm)
        self.assertEqual(self.rm.response_binning.value_array.sum(),
                         rm0.response_binning.value_array.sum())
        self.assertEqual(self.rm.truth_binning.value_array.sum(),
                         rm0.truth_binning.value_array.sum())
        self.assertEqual(self.rm.reco_binning.value_array.sum(),
                         rm0.reco_binning.value_array.sum())
        rm1 = improve_stats(rm0)
        self.assertEqual(self.rm.response_binning.value_array.sum(),
                         rm1.response_binning.value_array.sum())
        self.assertEqual(self.rm.truth_binning.value_array.sum(),
                         rm1.truth_binning.value_array.sum())
        self.assertEqual(self.rm.reco_binning.value_array.sum(),
                         rm1.reco_binning.value_array.sum())
        rm1 = improve_stats(rm0, data_index=3)
        self.assertEqual(self.rm.response_binning.value_array.sum(),
                         rm1.response_binning.value_array.sum())
        self.assertEqual(self.rm.truth_binning.value_array.sum(),
                         rm1.truth_binning.value_array.sum())
        self.assertEqual(self.rm.reco_binning.value_array.sum(),
                         rm1.reco_binning.value_array.sum())
        rm1 = improve_stats(self.rm, data_index=3)
        self.assertEqual(self.rm.response_binning.value_array.sum(),
                         rm1.response_binning.value_array.sum())
        self.assertEqual(self.rm.truth_binning.value_array.sum(),
                         rm1.truth_binning.value_array.sum())
        self.assertEqual(self.rm.reco_binning.value_array.sum(),
                         rm1.reco_binning.value_array.sum())

class TestLikelihoodUtils(unittest.TestCase):
    def setUp(self):
        with open('testdata/test-truth-binning.yml', 'r') as f:
            tb = yaml.full_load(f)
        with open('testdata/test-reco-binning.yml', 'r') as f:
            rb = yaml.full_load(f)
        rm = ResponseMatrix(rb, tb)
        rm.fill_from_csv_file('testdata/test-data.csv', weightfield='w')
        data_vector = rm.get_reco_entries_as_ndarray() # Entries because we need integer event numbers
        self.truth_vector = rm.get_truth_values_as_ndarray()
        response_matrix = []
        response_matrix.append(rm.get_response_matrix_as_ndarray())
        # Create a second response matric for systematics stuff
        rm.truth_binning.fill_from_csv_file('testdata/test-data.csv')
        response_matrix.append(rm.get_response_matrix_as_ndarray())
        self.L = LikelihoodMachine(data_vector, np.array([response_matrix])[...,[1,2,3]], truth_limits=[np.inf]*4, eff_indices=[1,2,3], is_sparse=True, matrix_weights=[2.,1.])

    def test_plots(self):
        """Test the different plotting functions."""
        plot_bin_efficiencies(self.L, filename=None)
        plot_truth_bin_traces(self.L, filename=None, trace=np.random.uniform(size=(50,4)))
        plot_reco_bin_traces(self.L, filename=None, trace=np.random.uniform(size=(50,4)), toy_index=0)

if __name__ == '__main__':
    np.seterr(all='raise')
    unittest.main(argv=testargs)
