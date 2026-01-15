import sys
import unittest
from tempfile import TemporaryFile

import numpy as np
import pandas as pd
import yaml
from scipy import stats

import remu.binning as binning
import remu.likelihood as likelihood
import remu.likelihood_utils as likelihood_utils
import remu.matrix_utils as matrix_utils
import remu.migration as migration
import remu.plotting as plotting
import remu.yaml as yaml_compat

if __name__ == "__main__":
    # Parse arguments for skipping tests
    import argparse

    parser = argparse.ArgumentParser()
    args, testargs = parser.parse_known_args()
    testargs = sys.argv[0:1] + testargs


class TestPhaseSpaces(unittest.TestCase):
    def setUp(self):
        self.psX = binning.PhaseSpace(variables=["x"])
        self.psY = binning.PhaseSpace(variables=["y"])
        self.psXY = binning.PhaseSpace(variables=["x", "y"])
        self.psXYZ = binning.PhaseSpace(variables=["x", "y", "z"])

    def test_contains(self):
        """Test behaviour of 'in' operator."""
        self.assertTrue("x" in self.psX)
        self.assertFalse("x" in self.psY)

    def test_product(self):
        """Test the carthesian product of phase spaces."""
        psXY = self.psX * self.psY
        self.assertTrue("x" in psXY)
        self.assertTrue("y" in psXY)
        self.assertFalse("z" in psXY)

    def test_division(self):
        """Test the reduction of phase spaces."""
        psXYX = (self.psX * self.psY) / self.psY
        self.assertTrue("x" in psXYX)
        self.assertFalse("y" in psXYX)

    def test_equality(self):
        """Test the equlaity of phase spaces."""
        ps = binning.PhaseSpace(variables=["x"])
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
        env = vars(binning)
        env.update(vars(np))
        self.assertEqual(self.psX, eval(repr(self.psX), env))
        self.assertEqual(self.psXY, eval(repr(self.psXY), env))
        self.assertEqual(self.psXYZ, eval(repr(self.psXYZ), env))

    def test_yaml_representation(self):
        """Test whether the text parsing can reproduce the original object."""
        self.assertEqual(self.psX, yaml_compat.full_load(yaml.dump(self.psX)))
        self.assertEqual(self.psXY, yaml_compat.full_load(yaml.dump(self.psXY)))
        self.assertEqual(self.psXYZ, yaml_compat.full_load(yaml.dump(self.psXYZ)))


class TestBins(unittest.TestCase):
    def setUp(self):
        ps = binning.PhaseSpace(["x"])
        self.b0 = binning.Bin(phasespace=ps)
        self.b1 = binning.Bin(phasespace=ps, value=1.0)
        self.b2 = binning.Bin(phasespace=ps, value=2.0)

    def test_init_values(self):
        """Test initialization values."""
        self.assertEqual(self.b0.value, 0.0)
        self.assertEqual(self.b1.value, 1.0)
        self.assertEqual(self.b2.value, 2.0)

    def test_bin_arithmetic(self):
        """Test math with bins."""
        self.assertEqual((self.b1 + self.b2).value, 3.0)
        self.assertEqual((self.b1 - self.b2).value, -1.0)
        self.assertEqual((self.b2 * self.b2).value, 4.0)
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
        self.b1.phasespace *= binning.PhaseSpace(["abc"])
        self.assertFalse(self.b0 == self.b1)
        self.assertTrue(self.b0 != self.b1)

    def test_clone(self):
        """Test whether the repr reproduces same object."""
        obj = self.b0
        self.assertEqual(obj, obj.clone())

    def test_repr(self):
        """Test whether the repr reproduces same object."""
        env = vars(binning)
        env.update(vars(np))
        self.assertEqual(self.b0.phasespace, eval(repr(self.b0), env).phasespace)
        self.assertEqual(self.b0.value, eval(repr(self.b0), env).value)
        self.assertEqual(self.b1.value, eval(repr(self.b1), env).value)
        self.assertEqual(self.b2.value, eval(repr(self.b2), env).value)

    def test_yaml_representation(self):
        """Test whether the yaml parsing can reproduce the original object."""
        orig = self.b0
        yml = yaml.dump(orig)
        reco = yaml_compat.full_load(yml)
        self.assertEqual(orig, reco)


class TestRectangularBins(unittest.TestCase):
    def setUp(self):
        self.b = binning.RectangularBin(["x", "y"], [(0, 1), (5, float("inf"))])
        self.c = binning.RectangularBin(["x", "y"], [(1, 2), (5, float("inf"))])

    def test_inclusion(self):
        """Test basic inclusion."""
        self.assertTrue({"x": 0.5, "y": 10} in self.b)
        self.assertFalse({"x": -0.5, "y": 10} in self.b)
        self.assertFalse({"x": 0.5, "y": -10} in self.b)
        self.assertFalse({"x": -0.5, "y": -10} in self.b)

    def test_include_lower(self):
        """Test inclusion of lower bounds."""
        self.b.include_lower = True
        self.assertTrue({"x": 0, "y": 10} in self.b)
        self.assertTrue({"x": 0.5, "y": 10} in self.b)
        self.assertFalse({"x": -0.5, "y": 10} in self.b)
        self.b.include_lower = False
        self.assertFalse({"x": 0, "y": 10} in self.b)
        self.assertTrue({"x": 0.5, "y": 10} in self.b)
        self.assertFalse({"x": -0.5, "y": 10} in self.b)

    def test_include_upper(self):
        """Test inclusion of upper bounds."""
        self.b.include_upper = True
        self.assertTrue({"x": 1, "y": 10} in self.b)
        self.assertTrue({"x": 0.5, "y": 10} in self.b)
        self.assertFalse({"x": 1.5, "y": 10} in self.b)
        self.b.include_upper = False
        self.assertFalse({"x": 1, "y": 10} in self.b)
        self.assertTrue({"x": 0.5, "y": 10} in self.b)
        self.assertFalse({"x": 1.5, "y": 10} in self.b)

    def test_bin_centers(self):
        """Test calculation of bin centers."""
        c = self.b.get_center()
        self.assertEqual(c[0], 0.5)
        self.assertEqual(c[1], float("inf"))

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
        env = vars(binning)
        env.update(vars(np))
        self.assertEqual(obj, eval(repr(obj), env))

    def test_yaml_representation(self):
        """Test whether the yaml parsing can reproduce the original object."""
        orig = self.b
        yml = yaml.dump(orig)
        reco = yaml_compat.full_load(yml)
        self.assertEqual(orig, reco)


class TestCartesianProductBins(unittest.TestCase):
    def setUp(self):
        self.x0 = binning.RectangularBin(variables=["x"], edges=[(0, 1)], dummy=True)
        self.x1 = binning.RectangularBin(variables=["x"], edges=[(1, 2)], dummy=True)
        self.y0 = binning.RectangularBin(variables=["y"], edges=[(0, 1)], dummy=True)
        self.y1 = binning.RectangularBin(variables=["y"], edges=[(1, 2)], dummy=True)
        self.z0 = binning.RectangularBin(variables=["z"], edges=[(0, 1)], dummy=True)
        self.z1 = binning.RectangularBin(variables=["z"], edges=[(1, 2)], dummy=True)
        self.bx = binning.Binning(bins=[self.x0, self.x1], dummy=True)
        self.by = binning.Binning(bins=[self.y0, self.y1], dummy=True)
        self.bz = binning.Binning(bins=[self.z0, self.z1], dummy=True)
        self.b = binning.CartesianProductBin([self.bx, self.by, self.bz], [0, 1, 0])
        self.c = binning.CartesianProductBin([self.bx, self.by, self.bz], [0, 1, 1])

    def test_inclusion(self):
        """Test basic inclusion."""
        self.assertTrue({"x": 0, "y": 1, "z": 0} in self.b)
        self.assertTrue({"x": 0, "y": 0, "z": 0} not in self.b)
        self.assertTrue({"x": 0, "y": 1, "z": 1} in self.c)
        self.assertTrue({"x": 0, "y": 0, "z": 1} not in self.c)

    def test_equality(self):
        """Test equality comparisons between bins."""
        self.assertTrue(self.b == self.b)
        self.assertFalse(self.b != self.b)
        self.assertTrue(self.b != self.c)
        self.assertFalse(self.b == self.c)

    def test_marginal_bins(self):
        """Test returning the marginal bins."""
        self.assertEqual(self.b.get_marginal_bins(), (self.x0, self.y1, self.z0))
        self.assertEqual(
            self.b.get_marginal_subbins(), ((self.x0,), (self.y1,), (self.z0,))
        )

    def test_clone(self):
        """Test whether the repr reproduces same object."""
        obj = self.b
        self.assertEqual(obj, obj.clone())

    def test_repr(self):
        """Test whether the repr reproduces same object."""
        obj = self.b
        env = vars(binning)
        env.update(vars(np))
        self.assertEqual(obj, eval(repr(obj), env))

    def test_yaml_representation(self):
        """Test whether the yaml parsing can reproduce the original object."""
        orig = self.b
        yml = yaml.dump(orig)
        reco = yaml_compat.full_load(yml)
        self.assertEqual(orig, reco)


class TestBinnings(unittest.TestCase):
    def setUp(self):
        self.b0 = binning.RectangularBin(
            variables=["x", "y"], edges=[(0, 1), (5, float("inf"))]
        )
        self.b1 = binning.RectangularBin(
            variables=["x", "y"], edges=[(1, 2), (5, float("inf"))]
        )
        self.binning = binning.Binning(bins=[self.b0, self.b1])
        self.binning0 = binning.Binning(
            phasespace=self.b0.phasespace, bins=[self.b0.clone()]
        )
        self.binning1 = binning.Binning(
            bins=[self.b0.clone(), self.b1.clone()],
            subbinnings={0: self.binning.clone()},
        )
        self.binning2 = binning.Binning(
            bins=[self.b0.clone(), self.b1.clone()],
            subbinnings={0: self.binning1.clone()},
        )

    def test_get_bin_indices(self):
        """Test the translation of events to bin numbers."""
        self.assertEqual(self.binning.get_event_bin_index({"x": 0, "y": 10}), 0)
        self.assertEqual(self.binning.get_event_bin_index({"x": 1, "y": 10}), 1)
        self.assertTrue(self.binning.get_event_bin_index({"x": 2, "y": 10}) is None)

    def test_get_data_indices(self):
        """Test the translation of events to bin numbers."""
        self.assertEqual(self.binning1.get_event_data_index({"x": 0, "y": 10}), 0)
        self.assertEqual(self.binning.get_event_data_index({"x": 1, "y": 10}), 1)
        self.assertEqual(self.binning1.get_event_data_index({"x": 1, "y": 10}), 2)
        self.assertEqual(self.binning2.get_event_data_index({"x": 1, "y": 10}), 3)
        self.assertTrue(self.binning1.get_event_data_index({"x": 2, "y": 10}) is None)

    def test_get_bin(self):
        """Test the translation of events to bins."""
        self.assertTrue(self.binning.get_event_bin({"x": 0, "y": 10}) is self.b0)
        self.assertTrue(self.binning.get_event_bin({"x": 1, "y": 10}) is self.b1)
        self.assertTrue(self.binning.get_event_bin({"x": 2, "y": 10}) is None)

    def test_get_subbins(self):
        """Test returning all subbins of a data index."""
        b0 = self.b0
        b1 = self.b1
        self.assertEqual(self.binning2.get_subbins(0), (b0, b0, b0))
        self.assertEqual(self.binning2.get_subbins(1), (b0, b0, b1))
        self.assertEqual(self.binning2.get_subbins(2), (b0, b1))
        self.assertEqual(self.binning2.get_subbins(3), (b1,))
        self.assertEqual(
            self.binning2.get_event_subbins({"x": 0, "y": 10}), (b0, b0, b0)
        )
        self.assertEqual(self.binning2.get_event_subbins({"x": 1, "y": 10}), (b1,))
        self.assertTrue(self.binning2.get_event_subbins({"x": 2, "y": 10}) is None)

    def test_iter_subbins(self):
        """Test that all subbins are returned correctly."""
        bins = list(self.binning2.iter_subbins())
        b0 = self.b0
        b1 = self.b1
        self.assertEqual(bins, [(b0, b0, b0), (b0, b0, b1), (b0, b1), (b1,)])

    def test_fill(self):
        """Test bin filling"""
        self.binning.fill({"x": 0.5, "y": 10})
        self.assertEqual(self.b0.value, 1)
        self.assertEqual(self.b0.entries, 1)
        self.assertEqual(self.b1.value, 0)
        self.assertEqual(self.b1.entries, 0)
        self.binning.fill({"x": 1.5, "y": 10}, 2)
        self.assertEqual(self.b0.value, 1)
        self.assertEqual(self.b0.entries, 1)
        self.assertEqual(self.b1.value, 2)
        self.assertEqual(self.b1.entries, 1)
        self.binning.fill([{"x": 0.5, "y": 10}, {"x": 0.5, "y": 20}], 2)
        self.assertEqual(self.b0.value, 5)
        self.assertEqual(self.b0.entries, 3)
        self.assertEqual(self.b1.value, 2)
        self.assertEqual(self.b1.entries, 1)
        self.binning.fill([{"x": 0.5, "y": 10}, {"x": 1.5, "y": 10}], [1, 2])
        self.assertEqual(self.b0.value, 6)
        self.assertEqual(self.b0.entries, 4)
        self.assertEqual(self.b1.value, 4)
        self.assertEqual(self.b1.entries, 2)
        self.binning.fill(
            [{"x": -0.5, "z": 10}, {"x": 1.5, "z": 10}], [1, 2], rename={"z": "y"}
        )
        self.assertEqual(self.b0.value, 6)
        self.assertEqual(self.b0.entries, 4)
        self.assertEqual(self.b1.value, 6)
        self.assertEqual(self.b1.entries, 3)
        self.assertRaises(
            ValueError,
            lambda: self.binning.fill({"x": -0.5, "y": 10}, raise_error=True),
        )
        self.assertEqual(self.b0.value, 6)
        self.assertEqual(self.b0.entries, 4)
        self.assertEqual(self.b1.value, 6)
        self.assertEqual(self.b1.entries, 3)
        self.binning.fill({"x": 0.5, "y": 10, "z": 123})
        self.assertEqual(self.b0.value, 7)
        self.assertEqual(self.b0.entries, 5)
        self.assertEqual(self.b1.value, 6)
        self.assertEqual(self.b1.entries, 3)
        self.assertRaises(KeyError, lambda: self.binning.fill({"x": 0.5}))
        str_arr = np.array([(0.5, 10), (0.5, 20)], dtype=[("x", float), ("y", float)])
        self.binning.fill(str_arr)
        self.assertEqual(self.b0.value, 9)
        self.assertEqual(self.b0.entries, 7)
        self.assertEqual(self.b1.value, 6)
        self.assertEqual(self.b1.entries, 3)
        str_arr = np.array([(0.5, 10), (0.5, 20)], dtype=[("x", float), ("z", float)])
        self.binning.fill(str_arr, rename={"z": "y"})
        self.assertEqual(self.b0.value, 11)
        self.assertEqual(self.b0.entries, 9)
        self.assertEqual(self.b1.value, 6)
        self.assertEqual(self.b1.entries, 3)
        df = pd.DataFrame({"x": [0.5, 0.5], "z": [10, 20]})
        self.binning.fill(df, rename={"z": "y"})
        self.assertEqual(self.b0.value, 13)
        self.assertEqual(self.b0.entries, 11)
        self.assertEqual(self.b1.value, 6)
        self.assertEqual(self.b1.entries, 3)
        self.binning.reset()
        str_arr = np.array([], dtype=[("x", float), ("y", float)])
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
        self.binning.fill_from_csv_file("testdata/csv-test.csv")
        self.assertEqual(self.b0.value, 2)
        self.assertEqual(self.b1.value, 1)
        self.binning.fill_from_csv_file(
            "testdata/weighted-csv-test.csv", weightfield="w", buffer_csv_files=True
        )
        self.assertEqual(self.b0.value, 8)
        self.assertEqual(self.b1.value, 2)
        self.binning.fill_from_csv_file(
            "testdata/weighted-csv-test.csv", buffer_csv_files=True
        )
        self.assertEqual(self.b0.value, 10)
        self.assertEqual(self.b1.value, 3)
        self.binning.fill_from_csv_file(
            "testdata/weighted-csv-test.csv",
            buffer_csv_files=True,
            cut_function=lambda data: data[data["y"] < 15],
        )
        self.assertEqual(self.b0.value, 11)
        self.assertEqual(self.b1.value, 4)
        self.binning.fill_from_csv_file(
            ["testdata/csv-test.csv", "testdata/csv-test.csv"], buffer_csv_files=True
        )
        self.assertEqual(self.b0.value, 15)
        self.assertEqual(self.b1.value, 6)
        self.binning.fill_from_csv_file("testdata/csv-test.csv", weight=0.5)
        self.assertEqual(self.b0.value, 16)
        self.assertAlmostEqual(self.b1.value, 6.5)
        self.binning.fill_from_csv_file(
            ["testdata/csv-test.csv"] * 2, weight=[0.5, 2.0]
        )
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
        arr = self.binning.get_values_as_ndarray((1, 2))
        self.assertEqual(arr.shape, (1, 2))
        self.assertEqual(arr[0, 0], 5)
        self.assertEqual(arr[0, 1], 10)
        arr[0, 0] = 50
        arr[0, 1] = 100
        self.binning.set_values_from_ndarray(arr)
        self.assertEqual(self.b0.value, 50)
        self.assertEqual(self.b1.value, 100)
        arr = self.binning.get_entries_as_ndarray((2, 1))
        self.assertEqual(arr.shape, (2, 1))
        arr = self.binning.get_sumw2_as_ndarray((2, 1))
        self.assertEqual(arr.shape, (2, 1))
        arr = self.binning.get_entries_as_ndarray(indices=[0])
        self.assertEqual(arr.shape, (1,))
        arr = self.binning.get_values_as_ndarray(indices=[0])
        self.assertEqual(arr.shape, (1,))
        arr = self.binning.get_sumw2_as_ndarray(indices=[0])
        self.assertEqual(arr.shape, (1,))

    def test_inclusion(self):
        """Test checking whether an event is binned."""
        self.assertTrue({"x": 0.5, "y": 10} in self.binning)
        self.assertTrue({"x": 1.5, "y": 10} in self.binning)
        self.assertFalse({"x": 2.5, "y": 10} in self.binning)

    def test_equality(self):
        """Test equality comparisons."""
        self.assertTrue(self.binning == self.binning)
        self.assertFalse(self.binning != self.binning)
        self.assertTrue(self.binning != self.binning0)
        self.assertFalse(self.binning == self.binning0)

    def test_adding(self):
        """Test adding binnings."""
        new_binning = self.binning + self.binning
        self.assertTrue(
            np.all(
                self.binning.get_values_as_ndarray() * 2
                == new_binning.get_values_as_ndarray()
            )
        )

    def test_subbinning_modification(self):
        """Test marginalizing and inserting subbinnings."""
        orig = self.binning1
        orig.fill_from_csv_file("testdata/csv-test.csv")
        marg = orig.marginalize_subbinnings([0])
        self.assertEqual(
            orig.get_values_as_ndarray().sum(), marg.get_values_as_ndarray().sum()
        )
        insrt = marg.insert_subbinning(0, orig.subbinnings[0].clone())
        self.assertEqual(
            orig.get_values_as_ndarray().sum(), insrt.get_values_as_ndarray().sum()
        )
        self.assertEqual(orig, insrt)

    def test_clone(self):
        """Test whether the repr reproduces same object."""
        obj = self.binning
        self.assertEqual(obj, obj.clone())

    def test_repr(self):
        """Test whether the repr reproduces same object."""
        obj = self.binning
        env = vars(binning)
        env.update(vars(np))
        self.assertEqual(obj, eval(repr(obj), env))

    def test_yaml_representation(self):
        """Test whether the yaml parsing can reproduce the original object."""
        orig = self.binning
        yml = yaml.dump(orig)
        reco = yaml_compat.full_load(yml)
        self.assertEqual(orig, reco)
        orig = self.binning0
        yml = yaml.dump(orig)
        reco = yaml_compat.full_load(yml)
        self.assertEqual(orig, reco)
        orig = self.binning1
        yml = yaml.dump(orig)
        reco = yaml_compat.full_load(yml)
        self.assertEqual(orig, reco)
        orig = self.binning2
        yml = yaml.dump(orig)
        reco = yaml_compat.full_load(yml)
        self.assertEqual(orig, reco)


class TestCartesianProductBinnings(unittest.TestCase):
    def setUp(self):
        self.bx = binning.LinearBinning("x", [0, 1, 2], dummy=True)
        self.by = binning.LinearBinning("y", [0, 1, 2], dummy=True)
        self.bz = binning.LinearBinning("z", [0, 1, 2], dummy=True)
        self.bynest = binning.LinearBinning(
            "y", [0, 1, 2], subbinnings={0: self.bz.clone()}, dummy=True
        )
        self.b0 = binning.CartesianProductBinning(
            [self.bx, self.bynest], subbinnings={2: self.bz.clone()}
        )

    def test_get_event_data_index(self):
        """Test that events are put in the right data bins."""
        self.assertEqual(self.b0.get_event_data_index({"x": 0, "y": 0, "z": 0}), 0)
        self.assertEqual(self.b0.get_event_data_index({"x": 0, "y": 0, "z": 1}), 1)
        self.assertEqual(self.b0.get_event_data_index({"x": 0, "y": 1, "z": 0}), 2)
        self.assertEqual(self.b0.get_event_data_index({"x": 0, "y": 1, "z": 1}), 3)
        self.assertEqual(self.b0.get_event_data_index({"x": 1, "y": 0, "z": 0}), 4)
        self.assertEqual(self.b0.get_event_data_index({"x": 1, "y": 0, "z": 1}), 5)
        self.assertEqual(self.b0.get_event_data_index({"x": 1, "y": 1, "z": 0}), 6)
        self.assertEqual(self.b0.get_event_data_index({"x": 1, "y": 1, "z": 1}), 6)

    def test_adjacent_bins(self):
        """Test that all adjacent bins make sense."""
        ret = self.b0.get_adjacent_bin_indices()
        self.assertTrue(np.array_equal(ret[0], np.array([1, 3])))
        self.assertTrue(np.array_equal(ret[1], np.array([0, 4])))
        self.assertTrue(np.array_equal(ret[2], np.array([5])))
        self.assertTrue(np.array_equal(ret[5], np.array([2])))
        ret = self.b0.get_adjacent_data_indices()
        self.assertTrue(np.array_equal(ret[0], np.array([1, 4])))
        self.assertTrue(np.array_equal(ret[1], np.array([0, 5])))
        self.assertTrue(np.array_equal(ret[2], np.array([3])))
        self.assertTrue(np.array_equal(ret[3], np.array([2])))
        self.assertTrue(np.array_equal(ret[4], np.array([0, 5])))
        self.assertTrue(np.array_equal(ret[5], np.array([1, 4])))
        self.assertTrue(np.array_equal(ret[6], np.array([])))

    def test_bins(self):
        """Test that the bin proxy works."""
        A, B = self.b0.bins[3], binning.CartesianProductBin(
            [self.bx.clone(), self.bynest.clone()], [1, 0]
        )
        self.assertEqual(A, B)

    def test_marginalization(self):
        """Test marginalizations of binnings."""
        self.b0.fill({"x": 0, "y": 1, "z": 0})
        self.b0.fill({"x": 0, "y": 1, "z": 1})
        self.assertEqual(self.b0.entries_array[2], 1)
        self.assertEqual(self.b0.entries_array[3], 1)
        self.assertEqual(self.b0.marginalize([0]).binnings[0], self.bynest)
        self.assertEqual(self.b0.marginalize(0), self.b0.marginalize([0]))
        self.assertEqual(self.b0.marginalize(0).entries_array[2], 2)

    def test_projection(self):
        """Test projections of binnings."""
        self.b0.fill({"x": 0, "y": 1, "z": 0})
        self.b0.fill({"x": 0, "y": 1, "z": 1})
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
        env = vars(binning)
        env.update(vars(np))
        self.assertEqual(obj, eval(repr(obj), env))

    def test_yaml_representation(self):
        """Test whether the yaml parsing can reproduce the original object."""
        orig = self.b0
        yml = yaml.dump(orig)
        reco = yaml_compat.full_load(yml)
        self.assertEqual(orig, reco)


class TestRectangularBinnings(unittest.TestCase):
    def setUp(self):
        self.b0 = binning.RectangularBinning(
            ["x", "y"], [((0, 2), (0, 2)), ((0, 1), (2, 3)), ((1, 2), (2, 3))]
        )

    def test_get_event_data_index(self):
        """Test that events are put in the right data bins."""
        self.assertEqual(self.b0.get_event_data_index({"x": 0, "y": 0}), 0)
        self.assertEqual(self.b0.get_event_data_index({"x": 1, "y": 1}), 0)
        self.assertEqual(self.b0.get_event_data_index({"x": 0, "y": 2}), 1)
        self.assertEqual(self.b0.get_event_data_index({"x": 1, "y": 2}), 2)

    def test_clone(self):
        """Test whether the repr reproduces same object."""
        obj = self.b0
        self.assertEqual(obj, obj.clone())

    def test_repr(self):
        """Test whether the repr reproduces same object."""
        obj = self.b0
        env = vars(binning)
        env.update(vars(np))
        self.assertEqual(obj, eval(repr(obj), env))

    def test_yaml_representation(self):
        """Test whether the yaml parsing can reproduce the original object."""
        orig = self.b0
        yml = yaml.dump(orig)
        reco = yaml_compat.full_load(yml)
        self.assertEqual(orig, reco)


class TestLinearBinnings(unittest.TestCase):
    def setUp(self):
        self.bx = binning.LinearBinning("x", [0, 1, 2])
        self.by = binning.LinearBinning("y", [0, 1, 2])
        self.b0 = binning.LinearBinning(
            "x", [0, 1, 2], subbinnings={0: self.by.clone()}
        )

    def test_get_event_data_index(self):
        """Test that events are put in the right data bins."""
        self.assertEqual(self.b0.get_event_data_index({"x": 0, "y": 0}), 0)
        self.assertEqual(self.b0.get_event_data_index({"x": 0, "y": 1}), 1)
        self.assertEqual(self.b0.get_event_data_index({"x": 1, "y": 0}), 2)
        self.assertEqual(self.b0.get_event_data_index({"x": 1, "y": 1}), 2)

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
        A, B = self.b0.bins[0], binning.RectangularBin(["x"], [(0, 1)])
        self.assertEqual(A, B)

    def test_slice(self):
        """Test the slicing into new binning."""
        self.b0.set_values_from_ndarray([1, 2, 5])
        sl = self.b0.slice(1, 2)
        ret = sl.get_values_as_ndarray()
        self.assertEqual(ret[0], 5)

    def test_remove_bin_edges(self):
        """Test the merging of bins."""
        self.b0.set_values_from_ndarray([1, 2, 5])
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
        env = vars(binning)
        env.update(vars(np))
        self.assertEqual(obj, eval(repr(obj), env))

    def test_yaml_representation(self):
        """Test whether the yaml parsing can reproduce the original object."""
        orig = self.b0
        yml = yaml.dump(orig)
        reco = yaml_compat.full_load(yml)
        self.assertEqual(orig, reco)


class TestRectilinearBinnings(unittest.TestCase):
    def setUp(self):
        self.bl = binning.RectilinearBinning(
            variables=["x", "y"], bin_edges=[[0, 1, 2], (-10, 0, 10, 20, float("inf"))]
        )
        self.bl0 = binning.RectilinearBinning(
            variables=["x", "y"], bin_edges=[[0, 1], (-10, 0, 10, 20, float("inf"))]
        )
        self.bu = binning.RectilinearBinning(
            variables=["x", "y"],
            bin_edges=[np.linspace(0, 2, 3), (-10, 0, 10, 20, float("inf"))],
            include_upper=True,
        )
        self.bxyz = binning.RectilinearBinning(
            variables=["x", "y", "z"],
            bin_edges=[[0, 1, 2], (-10, 0, 10, 20, float("inf")), (0, 1, 2)],
        )

    def test_tuples(self):
        """Test the translation of tuples to bin numbers and back."""
        self.assertEqual(
            self.bl.get_tuple_bin_index(self.bl.get_bin_index_tuple(None)), None
        )
        self.assertEqual(
            self.bl.get_tuple_bin_index(self.bl.get_bin_index_tuple(-1)), None
        )
        self.assertEqual(self.bl.get_tuple_bin_index(self.bl.get_bin_index_tuple(0)), 0)
        self.assertEqual(self.bl.get_tuple_bin_index(self.bl.get_bin_index_tuple(5)), 5)
        self.assertEqual(self.bl.get_tuple_bin_index(self.bl.get_bin_index_tuple(7)), 7)
        self.assertEqual(
            self.bl.get_tuple_bin_index(self.bl.get_bin_index_tuple(8)), None
        )
        self.assertEqual(
            self.bl.get_bin_index_tuple(self.bl.get_tuple_bin_index((1, 3))), (1, 3)
        )
        self.assertEqual(
            self.bxyz.get_bin_index_tuple(self.bxyz.get_tuple_bin_index((1, 3, 0))),
            (1, 3, 0),
        )
        self.assertEqual(self.bl.get_tuple_bin_index((1, 2)), 6)
        self.assertEqual(self.bl.get_event_bin_index({"x": 0, "y": -10}), 0)
        self.assertEqual(self.bl.get_event_bin_index({"x": 0, "y": -0}), 1)
        self.assertEqual(self.bl.get_event_bin_index({"x": 1, "y": -10}), 4)
        self.assertEqual(self.bl.get_event_bin_index({"x": -1, "y": -10}), None)
        self.assertEqual(self.bu.get_event_bin_index({"x": 2, "y": 30}), 7)

    def test_fill(self):
        """Test bin filling"""
        self.bl.fill({"x": 0.5, "y": -10}, 1)
        self.bl.fill({"x": 0.5, "y": 0}, 2)
        self.assertEqual(self.bl.bins[0].value, 1)
        self.assertEqual(self.bl.bins[0].entries, 1)
        self.assertEqual(self.bl.bins[0].sumw2, 1)
        self.assertEqual(self.bl.bins[1].value, 2)
        self.assertEqual(self.bl.bins[1].entries, 1)
        self.assertEqual(self.bl.bins[1].sumw2, 4)
        self.bl.fill([{"x": 0.5, "y": -10}, {"x": 0.5, "y": 0}], [1, 2])
        self.assertEqual(self.bl.bins[0].value, 2)
        self.assertEqual(self.bl.bins[0].entries, 2)
        self.assertEqual(self.bl.bins[0].sumw2, 2)
        self.assertEqual(self.bl.bins[1].value, 4)
        self.assertEqual(self.bl.bins[1].entries, 2)
        self.assertEqual(self.bl.bins[1].sumw2, 8)
        self.bl.fill({"x": 0.5, "y": -10, "z": 123})
        self.assertEqual(self.bl.bins[0].value, 3)
        self.assertEqual(self.bl.bins[0].entries, 3)
        self.assertEqual(self.bl.bins[0].sumw2, 3)
        self.assertEqual(self.bl.bins[1].value, 4)
        self.assertEqual(self.bl.bins[1].entries, 2)
        self.assertEqual(self.bl.bins[1].sumw2, 8)
        df = pd.DataFrame({"x": [0.5, 0.5], "z": [-10, -10]})
        self.bl.fill(df, rename={"z": "y"})
        self.assertEqual(self.bl.bins[0].value, 5)
        self.assertEqual(self.bl.bins[0].entries, 5)
        self.assertEqual(self.bl.bins[0].sumw2, 5)
        self.assertEqual(self.bl.bins[1].value, 4)
        self.assertEqual(self.bl.bins[1].entries, 2)
        self.assertEqual(self.bl.bins[1].sumw2, 8)
        self.assertRaises(KeyError, lambda: self.bl.fill({"x": 0.5}))

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
        arr = self.bl.get_values_as_ndarray((2, 4))
        self.assertEqual(arr.shape, (2, 4))
        self.assertEqual(arr[0, 0], 0)
        self.assertEqual(arr[1, 1], 1)
        arr = self.bl.get_entries_as_ndarray((2, 4))
        self.assertEqual(arr.shape, (2, 4))
        self.assertEqual(arr[0, 0], 0)
        self.assertEqual(arr[1, 1], 2)
        arr = self.bl.get_sumw2_as_ndarray((2, 4))
        self.assertEqual(arr.shape, (2, 4))
        self.assertEqual(arr[0, 0], 0)
        self.assertEqual(arr[1, 1], 0.5)
        arr[0, 0] = 7
        arr[1, 1] = 11
        self.bl.set_entries_from_ndarray(arr)
        self.assertEqual(self.bl.bins[0].entries, 7)
        self.assertEqual(self.bl.bins[5].entries, 11)
        arr[0, 0] = 5
        arr[1, 1] = 6
        self.bl.set_values_from_ndarray(arr)
        self.assertEqual(self.bl.bins[0].value, 5)
        self.assertEqual(self.bl.bins[5].value, 6)
        arr[0, 0] = 9
        arr[1, 1] = 13
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
        self.assertTrue({"x": 0.5, "y": 10} in self.bl)
        self.assertTrue({"x": 1.5, "y": 10} in self.bl)
        self.assertFalse({"x": 2.5, "y": 10} in self.bl)
        self.assertTrue({"x": 0, "y": 10} in self.bl)
        self.assertFalse({"x": 2, "y": 10} in self.bl)
        self.assertFalse({"x": 0, "y": 10} in self.bu)
        self.assertTrue({"x": 2, "y": 10} in self.bu)

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
        self.bxyz.fill({"x": 0, "y": 0, "z": 0})
        self.bxyz.fill({"x": 0, "y": 0, "z": 1}, weight=2.0)
        nb = self.bxyz.marginalize(["z"])
        self.assertEqual(nb, self.bl)
        self.assertEqual(nb.bins[1].entries, 2)
        self.assertEqual(nb.bins[1].value, 3.0)

    def test_projection(self):
        """Test projections of rectangular binnings."""
        self.bxyz.fill({"x": 0, "y": 0, "z": 0})
        self.bxyz.fill({"x": 0, "y": 0, "z": 1}, weight=2.0)
        nb = self.bxyz.project(["x", "y"])
        self.assertEqual(nb, self.bl)
        self.assertEqual(nb.bins[1].entries, 2)
        self.assertEqual(nb.bins[1].value, 3.0)

    def test_slice(self):
        """Test slices of rectangular binnings."""
        self.bxyz.fill({"x": 0, "y": 10, "z": 0}, weight=2.0)
        self.bxyz.fill({"x": 1, "y": 10, "z": 1})
        self.bxyz.fill({"x": 0, "y": 0, "z": 0}, weight=2.0)
        nb = self.bxyz.slice({"x": (0, 1), "y": (2, -1)})
        val = nb.get_values_as_ndarray()
        ent = nb.get_entries_as_ndarray()
        self.assertEqual(val.sum(), 2)
        self.assertEqual(ent.sum(), 1)

    def test_remove_bin_edges(self):
        """Test rebinning of rectangular binnings."""
        self.bxyz.fill({"x": 0, "y": 10, "z": 0}, weight=2.0)
        self.bxyz.fill({"x": 1, "y": 10, "z": 1})
        self.bxyz.fill({"x": 0, "y": 0, "z": 0}, weight=2.0)
        nb = self.bxyz.remove_bin_edges({"x": [2], "y": [0, 2]})
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
        env = vars(binning)
        env.update(vars(np))
        self.assertEqual(obj, eval(repr(obj), env))

    def test_yaml_representation(self):
        """Test whether the yaml parsing can reproduce the original object."""
        orig = self.bl
        yml = yaml.dump(orig)
        reco = yaml_compat.full_load(yml)
        self.assertEqual(orig, reco)


class TestResponseMatrices(unittest.TestCase):
    def setUp(self):
        with open("testdata/test-truth-binning.yml") as f:
            self.tb = yaml.full_load(f)
        with open("testdata/test-reco-binning.yml") as f:
            self.rb = yaml.full_load(f)
        self.rm = migration.ResponseMatrix(self.rb, self.tb)

    def test_fill(self):
        self.rm.fill_from_csv_file("testdata/test-data.csv", weightfield="w")
        reco = self.rm.get_reco_entries_as_ndarray((2, 2))
        self.assertEqual(reco[0, 0], 1)
        self.assertEqual(reco[1, 0], 1)
        self.assertEqual(reco[0, 1], 2)
        self.assertEqual(reco[1, 1], 2)
        truth = self.rm.get_truth_entries_as_ndarray((2, 2))
        self.assertEqual(truth[0, 0], 2)
        self.assertEqual(truth[1, 0], 2)
        self.assertEqual(truth[0, 1], 3)
        self.assertEqual(truth[1, 1], 2)
        resp = self.rm.get_response_entries_as_ndarray((2, 2, 2, 2))
        self.assertEqual(resp[0, 0, 0, 0], 0)
        self.assertEqual(resp[0, 1, 0, 0], 0)
        self.assertEqual(resp[1, 0, 0, 0], 0)
        self.assertEqual(resp[1, 1, 0, 0], 0)
        self.assertEqual(resp[0, 0, 0, 1], 1)
        self.assertEqual(resp[0, 1, 0, 1], 1)
        self.assertEqual(resp[1, 0, 0, 1], 0)
        self.assertEqual(resp[1, 1, 0, 1], 0)
        self.assertEqual(resp[0, 0, 1, 0], 0)
        self.assertEqual(resp[0, 1, 1, 0], 0)
        self.assertEqual(resp[1, 0, 1, 0], 1)
        self.assertEqual(resp[1, 1, 1, 0], 1)
        self.assertEqual(resp[0, 0, 1, 1], 0)
        self.assertEqual(resp[0, 1, 1, 1], 1)
        self.assertEqual(resp[1, 0, 1, 1], 0)
        self.assertEqual(resp[1, 1, 1, 1], 1)
        reco = self.rm.get_reco_values_as_ndarray((2, 2))
        self.assertEqual(reco[0, 0], 1)
        self.assertEqual(reco[1, 0], 1)
        self.assertEqual(reco[0, 1], 2)
        self.assertEqual(reco[1, 1], 2)
        truth = self.rm.get_truth_values_as_ndarray((2, 2))
        self.assertEqual(truth[0, 0], 2)
        self.assertEqual(truth[1, 0], 2)
        self.assertEqual(truth[0, 1], 4)
        self.assertEqual(truth[1, 1], 2)
        resp = self.rm.get_response_values_as_ndarray((2, 2, 2, 2))
        self.assertEqual(resp[0, 0, 0, 0], 0)
        self.assertEqual(resp[0, 1, 0, 0], 0)
        self.assertEqual(resp[1, 0, 0, 0], 0)
        self.assertEqual(resp[1, 1, 0, 0], 0)
        self.assertEqual(resp[0, 0, 0, 1], 1)
        self.assertEqual(resp[0, 1, 0, 1], 1)
        self.assertEqual(resp[1, 0, 0, 1], 0)
        self.assertEqual(resp[1, 1, 0, 1], 0)
        self.assertEqual(resp[0, 0, 1, 0], 0)
        self.assertEqual(resp[0, 1, 1, 0], 0)
        self.assertEqual(resp[1, 0, 1, 0], 1)
        self.assertEqual(resp[1, 1, 1, 0], 1)
        self.assertEqual(resp[0, 0, 1, 1], 0)
        self.assertEqual(resp[0, 1, 1, 1], 1)
        self.assertEqual(resp[1, 0, 1, 1], 0)
        self.assertEqual(resp[1, 1, 1, 1], 1)
        self.assertEqual(len(self.rm.filled_truth_indices), 4)
        resp = self.rm.get_response_matrix_as_ndarray(16)
        self.assertEqual(resp.shape, (16,))
        resp = self.rm.get_response_matrix_as_ndarray((4, 2), truth_indices=[0, 2])
        self.assertEqual(resp.shape, (4, 2))
        self.rm.reset()
        reco = self.rm.get_reco_values_as_ndarray((2, 2))
        self.assertEqual(reco[0, 0], 0)
        truth = self.rm.get_truth_values_as_ndarray((2, 2))
        self.assertEqual(truth[0, 0], 0)
        resp = self.rm.get_response_values_as_ndarray((2, 2, 2, 2))
        self.assertEqual(resp[0, 0, 0, 0], 0)
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
        self.rm.fill_from_csv_file("testdata/test-data.csv", weightfield="w")
        reco = self.rm.get_reco_values_as_ndarray()
        truth = self.rm.get_truth_values_as_ndarray()
        self.rm.fill_up_truth_from_csv_file(
            "testdata/test-data.csv", weightfield="w"
        )  # This should do nothing
        resp = self.rm.get_response_matrix_as_ndarray()
        self.assertTrue(np.all(reco == resp.dot(truth)))

    def test_variance(self):
        """Test the variance calculation."""
        self.rm.fill_from_csv_file("testdata/test-data.csv", weightfield="w")
        var = self.rm.get_statistical_variance_as_ndarray()
        self.assertAlmostEqual(var[0, 0], 0.0046875)
        self.assertAlmostEqual(var[1, 1], 0.00952183)
        self.assertAlmostEqual(var[2, 2], 0.02202381)
        self.assertAlmostEqual(var[3, 3], 0.02202381)
        var = self.rm.get_statistical_variance_as_ndarray(truth_indices=[0, -1])
        self.assertEqual(var.shape, (4, 2))

    def test_mean(self):
        """Test the mean calculation."""
        self.rm.fill_from_csv_file("testdata/test-data.csv", weightfield="w")
        mean = self.rm.get_mean_response_matrix_as_ndarray()
        self.assertAlmostEqual(mean[0, 0], 0.0625)
        self.assertAlmostEqual(mean[1, 1], 0.15)
        self.assertAlmostEqual(mean[2, 2], 0.25)
        self.assertAlmostEqual(mean[3, 3], 0.25)
        mean = self.rm.get_mean_response_matrix_as_ndarray(truth_indices=[0, 1])
        self.assertEqual(mean.shape, (4, 2))

    def test_in_bin_variation(self):
        """Test the in-bin variation calculation."""
        self.rm.fill_from_csv_file("testdata/test-data.csv", weightfield="w")
        var = self.rm.get_in_bin_variation_as_ndarray()
        self.assertAlmostEqual(var[0, 0], 0.7340425)
        self.assertAlmostEqual(var[1, 1], 0.7340425)
        self.assertAlmostEqual(var[2, 2], 1.1472384)
        self.assertAlmostEqual(var[3, 3], 0.8583145)
        var = self.rm.get_in_bin_variation_as_ndarray(
            shape=(2, 2, 2, 1), truth_indices=[0, -1]
        )
        self.assertEqual(var.shape, (2, 2, 2, 1))

    def test_random_generation(self):
        """Test generation of randomly varied matrices."""
        self.rm.fill_from_csv_file("testdata/test-data.csv", weightfield="w")
        ret = self.rm.generate_random_response_matrices()
        self.assertEqual(ret.shape, (4, 4))
        ret = self.rm.generate_random_response_matrices(impossible_indices=[0, 3])
        self.assertEqual(ret.shape, (4, 4))
        ret = self.rm.generate_random_response_matrices(truth_indices=[0, 2])
        self.assertEqual(ret.shape, (4, 2))
        ret = self.rm.generate_random_response_matrices(2)
        self.assertEqual(ret.shape, (2, 4, 4))
        ret = self.rm.generate_random_response_matrices(2, truth_indices=[1, 2, 3])
        self.assertEqual(ret.shape, (2, 4, 3))
        ret = self.rm.generate_random_response_matrices(
            (2, 3), shape=(2, 8), nuisance_indices=[0]
        )
        self.assertEqual(ret.shape, (2, 3, 2, 8))
        ret = self.rm.generate_random_response_matrices(
            (2, 3), shape=(4, 3), nuisance_indices=[1, 3], truth_indices=[0, 2, 3]
        )
        self.assertEqual(ret.shape, (2, 3, 4, 3))

    def test_add(self):
        """Test adding two response matrices."""
        self.rm.fill_from_csv_file("testdata/test-data.csv", weightfield="w")
        new_rm = self.rm + self.rm
        self.assertTrue(
            np.all(
                new_rm.get_truth_values_as_ndarray()
                == 2 * self.rm.get_truth_values_as_ndarray()
            )
        )
        self.assertTrue(
            np.all(
                new_rm.get_reco_values_as_ndarray()
                == 2 * self.rm.get_reco_values_as_ndarray()
            )
        )
        self.assertTrue(
            np.all(
                new_rm.get_response_values_as_ndarray()
                == 2 * self.rm.get_response_values_as_ndarray()
            )
        )


class TestResponseMatrixArrayBuilders(unittest.TestCase):
    def setUp(self):
        with open("testdata/test-truth-binning.yml") as f:
            self.tb = yaml.full_load(f)
        with open("testdata/test-reco-binning.yml") as f:
            self.rb = yaml.full_load(f)
        self.rm = migration.ResponseMatrix(self.rb, self.tb, nuisance_indices=[2])
        self.builder = migration.ResponseMatrixArrayBuilder(5)

    def test_mean(self):
        """Test ResponseMatrixArrayBuilder mean matrix."""
        self.builder.nstat = 0
        self.rm.fill_from_csv_file("testdata/test-data.csv", weightfield="w")
        self.builder.add_matrix(self.rm)
        self.rm.fill({"x_reco": 1, "y_reco": 0, "x_truth": 1, "y_truth": 0})
        self.builder.add_matrix(self.rm)
        M, weights = self.builder.get_mean_response_matrices_as_ndarray()
        self.assertEqual(M.shape, (2, 4, 4))
        self.assertEqual(weights.shape, (2,))

    def test_norandom(self):
        """Test ResponseMatrixArrayBuilder without random generation."""
        self.builder.nstat = 0
        self.rm.fill_from_csv_file("testdata/test-data.csv", weightfield="w")
        self.builder.add_matrix(self.rm)
        self.rm.fill({"x_reco": 1, "y_reco": 0, "x_truth": 1, "y_truth": 0})
        self.builder.add_matrix(self.rm)
        M, weights = self.builder.get_random_response_matrices_as_ndarray()
        self.assertEqual(M.shape, (2, 4, 4))
        self.assertEqual(weights.shape, (2,))

    def test_random(self):
        """Test ResponseMatrixArrayBuilder with random generation."""
        self.rm.fill_from_csv_file("testdata/test-data.csv", weightfield="w")
        self.builder.add_matrix(self.rm)
        self.rm.fill({"x_reco": 1, "y_reco": 0, "x_truth": 1, "y_truth": 0})
        self.builder.add_matrix(self.rm)
        M, weights = self.builder.get_random_response_matrices_as_ndarray()
        self.assertEqual(M.shape, (10, 4, 4))
        self.assertEqual(weights.shape, (10,))

    def test_truth_entries(self):
        """Test the truth entries array generation."""
        self.rm.fill_from_csv_file("testdata/test-data.csv", weightfield="w")
        self.builder.add_matrix(self.rm)
        self.rm.fill({"x_reco": 1, "y_reco": 0, "x_truth": 1, "y_truth": 0})
        self.builder.add_matrix(self.rm)
        M = self.builder.get_truth_entries_as_ndarray()
        self.assertEqual(tuple(M), (2, 3, 3, 2))


class TestPoissonData(unittest.TestCase):
    def setUp(self):
        self.data = np.arange(4, dtype=int)
        self.calc = likelihood.PoissonData(self.data)

    def test_log_likelihood(self):
        self.assertAlmostEqual(self.calc(self.data), -3.8027754226637804)
        self.assertAlmostEqual(self.calc([1] * 4), -6.484906649788)

    def test_output_shape(self):
        ret = self.calc([[self.data] * 3, [[1] * 4] * 3])
        self.assertEqual(ret.shape, (2, 3))
        self.assertAlmostEqual(ret[0, 0], -3.8027754226637804)
        self.assertAlmostEqual(ret[1, 2], -6.484906649788)

    def test_toy_data(self):
        ret = self.calc.generate_toy_data(self.data)
        self.assertEqual(ret.shape, (4,))
        ret = self.calc.generate_toy_data(self.data, size=10)
        self.assertEqual(ret.shape, (10, 4))
        ret = self.calc.generate_toy_data(self.data, size=(3, 5))
        self.assertEqual(ret.shape, (3, 5, 4))


class TestLinearEinsumPredictor(unittest.TestCase):
    def setUp(self):
        M = [[[1, 0], [0, 1]], [[0.9, 0], [0, 0.8]]]
        self.pred = likelihood.LinearEinsumPredictor(
            "ijk,...kl->...ijl",
            M,
            constants=[0.01, 0.02, 0.03, 0.04],
            reshape_parameters=(2, 2),
        )

    def test_prediction(self):
        pred, weights = self.pred([1, 2, 3, 4])
        self.assertEqual(
            pred.tolist(), [[1.01, 2.02, 3.03, 4.04], [0.91, 1.82, 2.43, 3.24]]
        )
        self.assertEqual(weights.tolist(), [1.0, 1.0])

    def test_output_shape(self):
        pred, weights = self.pred([[1, 2, 3, 4]] * 2)
        self.assertEqual(pred.shape, (2, 2, 4))
        self.assertEqual(weights.shape, (2, 2))
        pred, weights = self.pred([[1, 2, 3, 4]] * 5)
        self.assertEqual(pred.shape, (5, 2, 4))
        self.assertEqual(weights.shape, (5, 2))
        pred, weights = self.pred([[1, 2, 3, 4]] * 5, systematics_index=1)
        self.assertEqual(pred.shape, (5, 4))
        self.assertEqual(weights.shape, (5,))


class TestMatrixPredictors(unittest.TestCase):
    def setUp(self):
        self.pred = likelihood.MatrixPredictor(
            [[[1.0, 0.0], [0.5, 1.0], [0.0, 1.0]]] * 2,
            [0.1, 0.2, 0.3],
            weights=[1.0, 0.5],
        )

    def test_prediction(self):
        pred, weights = self.pred([1, 10])
        self.assertEqual(pred.tolist(), [[1.1, 10.7, 10.3]] * 2)
        self.assertEqual(weights.tolist(), [1.0, 0.5])

    def test_output_shape(self):
        pred, weights = self.pred([[1, 10]] * 2)
        self.assertEqual(pred.shape, (2, 2, 3))
        self.assertEqual(weights.shape, (2, 2))
        pred, weights = self.pred([[1, 10]] * 5)
        self.assertEqual(pred.shape, (5, 2, 3))
        self.assertEqual(weights.shape, (5, 2))
        pred, weights = self.pred([[1, 2]] * 5, systematics_index=1)
        self.assertEqual(pred.shape, (5, 3))
        self.assertEqual(weights.shape, (5,))

    def test_parameter_fixing(self):
        fixed = self.pred.fix_parameters([None, 10])
        pred, weights = fixed([1])
        self.assertEqual(pred.tolist(), [[1.1, 10.7, 10.3]] * 2)
        self.assertEqual(weights.tolist(), [1.0, 0.5])


class TestTemplatePredictors(unittest.TestCase):
    def setUp(self):
        self.pred = likelihood.TemplatePredictor(
            [[[1.0, 0.0], [0.5, 1.0], [0.0, 1.0]]] * 2,
            [0.1, 0.2, 0.3],
            weights=[1.0, 0.5],
        )
        self.pred = likelihood.TemplatePredictor(
            [[[1.0, 0.5, 0.0], [0.0, 1.0, 1.0]]] * 2,
            constants=[0.1, 0.2, 0.3],
            weights=[1.0, 0.5],
        )

    def test_prediction(self):
        pred, weights = self.pred([1, 10])
        self.assertEqual(pred.tolist(), [[1.1, 10.7, 10.3]] * 2)
        self.assertEqual(weights.tolist(), [1.0, 0.5])

    def test_output_shape(self):
        pred, weights = self.pred([[1, 10]] * 2)
        self.assertEqual(pred.shape, (2, 2, 3))
        self.assertEqual(weights.shape, (2, 2))
        pred, weights = self.pred([[1, 10]] * 5)
        self.assertEqual(pred.shape, (5, 2, 3))
        self.assertEqual(weights.shape, (5, 2))
        pred, weights = self.pred([[1, 2]] * 5, systematics_index=1)
        self.assertEqual(pred.shape, (5, 3))
        self.assertEqual(weights.shape, (5,))


class TestFixedParameterPredictors(unittest.TestCase):
    def setUp(self):
        self.pred = likelihood.MatrixPredictor(
            [[[1.0, 0.0], [0.5, 1.0], [0.0, 1.0]]] * 2,
            [0.1, 0.2, 0.3],
            weights=[1.0, 0.5],
        )
        self.pred0 = likelihood.FixedParameterPredictor(self.pred, [1, None])
        self.pred1 = likelihood.FixedParameterPredictor(self.pred, [None, 10])
        self.pred2 = likelihood.FixedParameterPredictor(self.pred, [1, 10])

    def test_prediction(self):
        pred, weights = self.pred0([10])
        self.assertEqual(pred.tolist(), [[1.1, 10.7, 10.3]] * 2)
        self.assertEqual(weights.tolist(), [1.0, 0.5])
        pred, weights = self.pred1([1])
        self.assertEqual(pred.tolist(), [[1.1, 10.7, 10.3]] * 2)
        self.assertEqual(weights.tolist(), [1.0, 0.5])
        pred, weights = self.pred2([])
        self.assertEqual(pred.tolist(), [[1.1, 10.7, 10.3]] * 2)
        self.assertEqual(weights.tolist(), [1.0, 0.5])

    def test_output_shape(self):
        pred, weights = self.pred0([[10]] * 2)
        self.assertEqual(pred.shape, (2, 2, 3))
        self.assertEqual(weights.shape, (2, 2))
        pred, weights = self.pred0([[10]] * 5)
        self.assertEqual(pred.shape, (5, 2, 3))
        self.assertEqual(weights.shape, (5, 2))
        pred, weights = self.pred0([[10]] * 5, systematics_index=1)
        self.assertEqual(pred.shape, (5, 3))
        self.assertEqual(weights.shape, (5,))

    def test_bounds(self):
        self.assertEqual(len(self.pred.bounds), 2)
        self.assertEqual(len(self.pred0.bounds), 1)
        self.assertEqual(len(self.pred1.bounds), 1)
        self.assertEqual(len(self.pred2.bounds), 0)

    def test_defaults(self):
        self.assertEqual(len(self.pred.defaults), 2)
        self.assertEqual(len(self.pred0.defaults), 1)
        self.assertEqual(len(self.pred1.defaults), 1)
        self.assertEqual(len(self.pred2.defaults), 0)


class TestFixedParameterMatrixPredictors(unittest.TestCase):
    def setUp(self):
        self.pred = likelihood.MatrixPredictor(
            [[[1.0, 0.0], [0.5, 1.0], [0.0, 1.0]]] * 2,
            [0.1, 0.2, 0.3],
            weights=[1.0, 0.5],
        )
        self.pred0 = likelihood.FixedParameterMatrixPredictor(self.pred, [1, None])
        self.pred1 = likelihood.FixedParameterMatrixPredictor(self.pred, [None, 10])
        self.pred2 = likelihood.FixedParameterMatrixPredictor(self.pred, [1, 10])

    def test_prediction(self):
        pred, weights = self.pred0([10])
        self.assertEqual(pred.tolist(), [[1.1, 10.7, 10.3]] * 2)
        self.assertEqual(weights.tolist(), [1.0, 0.5])
        pred, weights = self.pred1([1])
        self.assertEqual(pred.tolist(), [[1.1, 10.7, 10.3]] * 2)
        self.assertEqual(weights.tolist(), [1.0, 0.5])
        pred, weights = self.pred2([])
        self.assertEqual(pred.tolist(), [[1.1, 10.7, 10.3]] * 2)
        self.assertEqual(weights.tolist(), [1.0, 0.5])

    def test_output_shape(self):
        pred, weights = self.pred0([[10]] * 2)
        self.assertEqual(pred.shape, (2, 2, 3))
        self.assertEqual(weights.shape, (2, 2))
        pred, weights = self.pred0([[10]] * 5)
        self.assertEqual(pred.shape, (5, 2, 3))
        self.assertEqual(weights.shape, (5, 2))
        pred, weights = self.pred0([[10]] * 5, systematics_index=1)
        self.assertEqual(pred.shape, (5, 3))
        self.assertEqual(weights.shape, (5,))

    def test_bounds(self):
        self.assertEqual(len(self.pred.bounds), 2)
        self.assertEqual(len(self.pred0.bounds), 1)
        self.assertEqual(len(self.pred1.bounds), 1)
        self.assertEqual(len(self.pred2.bounds), 0)

    def test_defaults(self):
        self.assertEqual(len(self.pred.defaults), 2)
        self.assertEqual(len(self.pred0.defaults), 1)
        self.assertEqual(len(self.pred1.defaults), 1)
        self.assertEqual(len(self.pred2.defaults), 0)


class TestComposedPredictors(unittest.TestCase):
    def setUp(self):
        self.w0 = np.array([1, 2])
        self.pred0 = likelihood.MatrixPredictor(
            [[[1.0, 1.0, 1.0]]] * 2, weights=self.w0
        )
        self.w1 = np.array([1, 2, 3])
        self.pred1 = likelihood.MatrixPredictor([[[1 / 3]] * 3] * 3, weights=self.w1)
        self.w2 = np.array([1, 2, 3, 4])
        self.pred2 = likelihood.MatrixPredictor(
            [np.eye(3)] * 4, [0.1, 0.2, 0.3], weights=self.w2
        )
        self.pred = likelihood.ComposedPredictor([self.pred2, self.pred1, self.pred0])

    def test_prediction(self):
        pred, weights = self.pred([1, 2, 3])
        self.assertEqual(pred.tolist(), [[2.1, 2.2, 2.3]] * 24)
        w = self.w0[np.newaxis, np.newaxis, :]
        w = w * self.w1[np.newaxis, :, np.newaxis]
        w = w * self.w2[:, np.newaxis, np.newaxis]
        self.assertEqual(weights.tolist(), w.flatten().tolist())

    def test_output_shape(self):
        pred, weights = self.pred([[1, 2, 3]] * 2)
        self.assertEqual(pred.shape, (2, 24, 3))
        self.assertEqual(weights.shape, (2, 24))
        pred, weights = self.pred([[1, 2, 3]] * 2, systematics_index=1)
        self.assertEqual(pred.shape, (2, 3))
        self.assertEqual(weights.shape, (2,))


class TestComposedPredictorsWithSameSystematics(unittest.TestCase):
    def setUp(self):
        self.w0 = np.array([1, 2])
        self.pred0 = likelihood.MatrixPredictor(
            [[[1.0, 1.0, 1.0]]] * 2, weights=self.w0
        )
        self.w1 = np.array([1, 2])
        self.pred1 = likelihood.MatrixPredictor([[[1 / 3]] * 3] * 2, weights=self.w1)
        self.w2 = np.array([1, 2])
        self.pred2 = likelihood.MatrixPredictor(
            [np.eye(3)] * 2, [0.1, 0.2, 0.3], weights=self.w2
        )
        self.pred = likelihood.ComposedPredictor(
            [self.pred2, self.pred1, self.pred0], combine_systematics="same"
        )

    def test_prediction(self):
        pred, weights = self.pred([1, 2, 3])
        self.assertEqual(pred.tolist(), [[2.1, 2.2, 2.3]] * 2)
        self.assertEqual(weights.tolist(), self.w0.flatten().tolist())

    def test_output_shape(self):
        pred, weights = self.pred([[1, 2, 3]] * 2)
        self.assertEqual(pred.shape, (2, 2, 3))
        self.assertEqual(weights.shape, (2, 2))
        pred, weights = self.pred([[1, 2, 3]] * 2, systematics_index=1)
        self.assertEqual(pred.shape, (2, 3))
        self.assertEqual(weights.shape, (2,))


class TestComposedMatrixPredictors(unittest.TestCase):
    def setUp(self):
        self.w0 = np.array([1, 2])
        self.pred0 = likelihood.MatrixPredictor(
            [[[1.0, 1.0, 1.0]]] * 2, weights=self.w0
        )
        self.w1 = np.array([1, 2, 3])
        self.pred1 = likelihood.MatrixPredictor([[[1 / 3]] * 3] * 3, weights=self.w1)
        self.w2 = np.array([1, 2, 3, 4])
        self.pred2 = likelihood.MatrixPredictor(
            [np.eye(3)] * 4, [0.1, 0.2, 0.3], weights=self.w2
        )
        self.pred = likelihood.ComposedMatrixPredictor(
            [self.pred2, self.pred1, self.pred0]
        )

    def test_prediction(self):
        pred, weights = self.pred([1, 2, 3])
        np.testing.assert_almost_equal(pred, [[2.1, 2.2, 2.3]] * 24)
        w = self.w0[np.newaxis, np.newaxis, :]
        w = w * self.w1[np.newaxis, :, np.newaxis]
        w = w * self.w2[:, np.newaxis, np.newaxis]
        self.assertEqual(weights.tolist(), w.flatten().tolist())

    def test_output_shape(self):
        pred, weights = self.pred([[1, 2, 3]] * 2)
        self.assertEqual(pred.shape, (2, 24, 3))
        self.assertEqual(weights.shape, (2, 24))
        pred, weights = self.pred([[1, 2, 3]] * 2, systematics_index=1)
        self.assertEqual(pred.shape, (2, 3))
        self.assertEqual(weights.shape, (2,))


class TestComposedMatrixPredictorsWithSameSystematics(unittest.TestCase):
    def setUp(self):
        self.w0 = np.array([1, 2])
        self.pred0 = likelihood.MatrixPredictor(
            [[[1.0, 1.0, 1.0]]] * 2, weights=self.w0
        )
        self.w1 = np.array([1, 2])
        self.pred1 = likelihood.MatrixPredictor([[[1 / 3]] * 3] * 2, weights=self.w1)
        self.w2 = np.array([1, 2])
        self.pred2 = likelihood.MatrixPredictor(
            [np.eye(3)] * 2, [0.1, 0.2, 0.3], weights=self.w2
        )
        self.pred = likelihood.ComposedMatrixPredictor(
            [self.pred2, self.pred1, self.pred0], combine_systematics="same"
        )

    def test_prediction(self):
        pred, weights = self.pred([1, 2, 3])
        np.testing.assert_almost_equal(pred, [[2.1, 2.2, 2.3]] * 2)
        w = self.w0
        self.assertEqual(weights.tolist(), w.flatten().tolist())

    def test_output_shape(self):
        pred, weights = self.pred([[1, 2, 3]] * 2)
        self.assertEqual(pred.shape, (2, 2, 3))
        self.assertEqual(weights.shape, (2, 2))
        pred, weights = self.pred([[1, 2, 3]] * 2, systematics_index=1)
        self.assertEqual(pred.shape, (2, 3))
        self.assertEqual(weights.shape, (2,))


class TestLinearizedPredictors(unittest.TestCase):
    def setUp(self):
        self.pred = likelihood.Predictor(defaults=[1.0], bounds=[[-np.inf, np.inf]])
        self.pred.prediction = lambda x: (
            np.array(x**2)[:, np.newaxis],
            np.array([1.0]),
        )
        self.lin_pred1 = likelihood.LinearizedPredictor(
            self.pred, evaluation_steps=[1.0]
        )
        self.lin_pred0 = likelihood.LinearizedPredictor(
            self.pred, evaluation_point=[0.0], evaluation_steps=[1.0]
        )

    def test_prediction(self):
        pred, weights = self.pred(np.array([[0.0], [0.5], [1.0]]))
        pred1, weights = self.lin_pred1([[0.0], [0.5], [1.0]])
        pred0, weights = self.lin_pred0([[0.0], [0.5], [1.0]])
        np.testing.assert_almost_equal(pred, [[[0.0]], [[0.25]], [[1.0]]])
        np.testing.assert_almost_equal(pred1, [[[-2.0]], [[-0.5]], [[1.0]]])
        np.testing.assert_almost_equal(pred0, [[[0.0]], [[0.5]], [[1.0]]])


class TestResponseMatrixPredictors(unittest.TestCase):
    def setUp(self):
        with open("testdata/test-truth-binning.yml") as f:
            self.tb = yaml.full_load(f)
        with open("testdata/test-reco-binning.yml") as f:
            self.rb = yaml.full_load(f)
        self.rm = migration.ResponseMatrix(self.rb, self.tb, nuisance_indices=[2])
        self.builder = migration.ResponseMatrixArrayBuilder(5)

        self.rm.fill_from_csv_file("testdata/test-data.csv", weightfield="w")
        self.builder.add_matrix(self.rm, 1.0)
        self.rm.fill({"x_reco": 1, "y_reco": 0, "x_truth": 1, "y_truth": 0})
        self.builder.add_matrix(self.rm, 0.5)
        with TemporaryFile() as f:
            self.builder.export(f)
            f.seek(0)
            self.pred = likelihood.ResponseMatrixPredictor(f)

    def test_prediction(self):
        y, w = self.pred([1, 0, 0, 0])
        self.assertEqual(y.shape, (10, 4))
        self.assertEqual(w[0], 1.0)
        self.assertEqual(w[-1], 0.5)

    def test_sparse_matrix(self):
        self.rm.reset()
        with TemporaryFile() as f:
            self.rm.export(f, sparse=True)
            f.seek(0)
            pred = likelihood.ResponseMatrixPredictor(f)
        y, w = pred([1, 0, 0, 0])
        self.assertEqual(y.shape, (1, 4))
        self.assertEqual(y[0, 0], 0.0)


class TestSummedPredictors(unittest.TestCase):
    def setUp(self):
        self.w0 = np.array([1, 2])
        self.pred0 = likelihood.MatrixPredictor(
            [np.diag([1, 2, 3])] * 2, weights=self.w0
        )
        self.w1 = np.array([1, 2, 3])
        self.pred1 = likelihood.MatrixPredictor([[[1 / 3]] * 3] * 3, weights=self.w1)
        self.w2 = np.array([1, 2, 3, 4])
        self.pred2 = likelihood.MatrixPredictor(
            [np.eye(3)] * 4, [0.1, 0.2, 0.3], weights=self.w2
        )
        self.pred = likelihood.SummedPredictor(
            [self.pred0, self.pred1, self.pred2],
            scale_factors=[[[2.0]] * 2, [[0.5]] * 3, [[1.0]] * 4],
        )

    def test_prediction(self):
        pred, weights = self.pred([2, 2, 2, 6, 4, 4, 4])
        self.assertEqual(pred.tolist(), [[9.1, 13.2, 17.3]] * 24)
        w = self.w2[np.newaxis, np.newaxis, :]
        w = w * self.w1[np.newaxis, :, np.newaxis]
        w = w * self.w0[:, np.newaxis, np.newaxis]
        self.assertEqual(weights.tolist(), w.flatten().tolist())

    def test_output_shape(self):
        pred, weights = self.pred([[2, 2, 2, 3, 4, 4, 4]] * 2)
        self.assertEqual(pred.shape, (2, 24, 3))
        self.assertEqual(weights.shape, (2, 24))
        pred, weights = self.pred([[2, 2, 2, 3, 4, 4, 4]] * 2, systematics_index=1)
        self.assertEqual(pred.shape, (2, 3))
        self.assertEqual(weights.shape, (2,))


class TestSummedPredictorsWithSameSystematics(unittest.TestCase):
    def setUp(self):
        self.w0 = np.array([1, 2])
        self.pred0 = likelihood.MatrixPredictor(
            [np.diag([1, 2, 3])] * 2, weights=self.w0
        )
        self.w1 = np.array([1, 2])
        self.pred1 = likelihood.MatrixPredictor([[[1 / 3]] * 3] * 2, weights=self.w1)
        self.w2 = np.array([1, 2])
        self.pred2 = likelihood.MatrixPredictor(
            [np.eye(3)] * 2, [0.1, 0.2, 0.3], weights=self.w2
        )
        self.pred = likelihood.SummedPredictor(
            [self.pred0, self.pred1, self.pred2], combine_systematics="same"
        )

    def test_prediction(self):
        pred, weights = self.pred([2, 2, 2, 3, 4, 4, 4])
        self.assertEqual(pred.tolist(), [[7.1, 9.2, 11.3]] * 2)
        self.assertEqual(weights.tolist(), self.w0.tolist())

    def test_output_shape(self):
        pred, weights = self.pred([[2, 2, 2, 3, 4, 4, 4]] * 2)
        self.assertEqual(pred.shape, (2, 2, 3))
        self.assertEqual(weights.shape, (2, 2))
        pred, weights = self.pred([[2, 2, 2, 3, 4, 4, 4]] * 2, systematics_index=1)
        self.assertEqual(pred.shape, (2, 3))
        self.assertEqual(weights.shape, (2,))


class TestConcatenatedPredictors(unittest.TestCase):
    def setUp(self):
        self.w0 = np.array([1, 2])
        self.pred0 = likelihood.MatrixPredictor(
            [np.diag([1, 2, 3])] * 2, weights=self.w0
        )
        self.w1 = np.array([1, 2, 3])
        self.pred1 = likelihood.MatrixPredictor([[[1 / 3]] * 3] * 3, weights=self.w1)
        self.w2 = np.array([1, 2, 3, 4])
        self.pred2 = likelihood.MatrixPredictor(
            [np.eye(3)] * 4, [0.1, 0.2, 0.3], weights=self.w2
        )
        self.pred = likelihood.ConcatenatedPredictor(
            [self.pred0, self.pred1, self.pred2]
        )

    def test_prediction(self):
        pred, weights = self.pred([2, 2, 2, 3, 4, 4, 4])
        self.assertEqual(pred.tolist(), [[2, 4, 6, 1, 1, 1, 4.1, 4.2, 4.3]] * 24)
        w = self.w2[np.newaxis, np.newaxis, :]
        w = w * self.w1[np.newaxis, :, np.newaxis]
        w = w * self.w0[:, np.newaxis, np.newaxis]
        self.assertEqual(weights.tolist(), w.flatten().tolist())

    def test_output_shape(self):
        pred, weights = self.pred([[2, 2, 2, 3, 4, 4, 4]] * 2)
        self.assertEqual(pred.shape, (2, 24, 9))
        self.assertEqual(weights.shape, (2, 24))
        pred, weights = self.pred([[2, 2, 2, 3, 4, 4, 4]] * 2, systematics_index=1)
        self.assertEqual(pred.shape, (2, 9))
        self.assertEqual(weights.shape, (2,))
        self.assertEqual(self.pred.defaults.shape, (7,))


class TestConcatenatedPredictorsWithSameParameters(unittest.TestCase):
    def setUp(self):
        self.w0 = np.array([1, 2])
        self.pred0 = likelihood.MatrixPredictor(
            [np.diag([1, 2, 3])] * 2, weights=self.w0
        )
        self.w1 = np.array([1, 2, 3, 4])
        self.pred1 = likelihood.MatrixPredictor(
            [np.eye(3)] * 4, [0.1, 0.2, 0.3], weights=self.w1
        )
        self.pred = likelihood.ConcatenatedPredictor(
            [self.pred0, self.pred1], share_parameters=True
        )
        self.pred0a = likelihood.MatrixPredictor([np.diag([1, 2, 3])])
        self.pred1a = likelihood.MatrixPredictor([np.eye(3)], [0.1, 0.2, 0.3])
        self.preda = likelihood.ConcatenatedPredictor(
            [self.pred0a, self.pred1a] * 30,
            share_parameters=True,
            combine_systematics="cartesian",
        )
        self.predb = likelihood.ConcatenatedPredictor(
            [self.pred0a, self.pred1a] * 30,
            share_parameters=True,
            combine_systematics="same",
        )

    def test_prediction(self):
        pred, weights = self.pred([1, 2, 3])
        self.assertEqual(pred.tolist(), [[1.0, 4.0, 9.0, 1.1, 2.2, 3.3]] * 8)
        w = self.w1[np.newaxis, np.newaxis, :]
        w = w * self.w0[:, np.newaxis, np.newaxis]
        self.assertEqual(weights.tolist(), w.flatten().tolist())

    def test_output_shape(self):
        pred, weights = self.pred([[1, 2, 3]] * 2)
        self.assertEqual(pred.shape, (2, 8, 6))
        self.assertEqual(weights.shape, (2, 8))
        pred, weights = self.pred([[1, 2, 3]] * 2, systematics_index=1)
        self.assertEqual(pred.shape, (2, 6))
        self.assertEqual(weights.shape, (2,))
        self.assertEqual(self.pred.defaults.shape, (3,))
        pred, weights = self.pred([1, 2, 3])
        self.assertEqual(pred.shape, (8, 6))
        self.assertEqual(weights.shape, (8,))
        pred, weights = self.preda([1, 2, 3])
        self.assertEqual(pred.shape, (1, 6 * 30))
        self.assertEqual(weights.shape, (1,))
        pred, weights = self.predb([1, 2, 3])
        self.assertEqual(pred.shape, (1, 6 * 30))
        self.assertEqual(weights.shape, (1,))


class TestConcatenatedPredictorsWithSameSystematics(unittest.TestCase):
    def setUp(self):
        self.w0 = np.array([1, 2])
        self.pred0 = likelihood.MatrixPredictor(
            [np.diag([1, 2, 3])] * 2, weights=self.w0
        )
        self.w1 = np.array([1, 2])
        self.pred1 = likelihood.MatrixPredictor([[[1 / 3]] * 3] * 2, weights=self.w1)
        self.w2 = np.array([1, 2])
        self.pred2 = likelihood.MatrixPredictor(
            [np.eye(3)] * 2, [0.1, 0.2, 0.3], weights=self.w2
        )
        self.pred = likelihood.ConcatenatedPredictor(
            [self.pred0, self.pred1, self.pred2],
            combine_systematics="same",
        )

    def test_prediction(self):
        pred, weights = self.pred([2, 2, 2, 3, 4, 4, 4])
        self.assertEqual(pred.tolist(), [[2, 4, 6, 1, 1, 1, 4.1, 4.2, 4.3]] * 2)
        self.assertEqual(weights.tolist(), self.w0.tolist())

    def test_output_shape(self):
        pred, weights = self.pred([[2, 2, 2, 3, 4, 4, 4]] * 2)
        self.assertEqual(pred.shape, (2, 2, 9))
        self.assertEqual(weights.shape, (2, 2))
        pred, weights = self.pred([[2, 2, 2, 3, 4, 4, 4]] * 2, systematics_index=1)
        self.assertEqual(pred.shape, (2, 9))
        self.assertEqual(weights.shape, (2,))


class TestSystematics(unittest.TestCase):
    def setUp(self):
        self.data = np.log(np.arange(2 * 3 * 5) + 1)
        self.data.shape = (2, 3, 5)

    def test_marginal_systematics(self):
        ret = likelihood.MarginalLikelihoodSystematics.consume_axis(self.data)

        for A, B in zip(np.exp(ret).flat, np.mean(np.exp(self.data), axis=-1).flat):
            self.assertAlmostEqual(A, B)

    def test_profile_systematics(self):
        ret = likelihood.ProfileLikelihoodSystematics.consume_axis(self.data)

        for A, B in zip(np.exp(ret).flat, np.max(np.exp(self.data), axis=-1).flat):
            self.assertAlmostEqual(A, B)


class TestLikelihoodCalculators(unittest.TestCase):
    def setUp(self):
        self.data = np.arange(4, dtype=int)
        self.data_model = likelihood.PoissonData([self.data] * 2)
        self.predictor = likelihood.TemplatePredictor([np.eye(4)] * 3)
        self.calc = likelihood.LikelihoodCalculator(self.data_model, self.predictor)

    def test_log_likelihood(self):
        test_reco = np.asarray([0.5, 0.5, 0.5, 0.5])
        ret = self.calc([test_reco] * 5)
        self.assertAlmostEqual(
            ret[0, 0], stats.poisson(test_reco).logpmf(self.data).sum()
        )
        self.assertEqual(ret.shape, (2, 5))
        ret = self.calc([test_reco, -test_reco])
        self.assertAlmostEqual(
            ret[0, 0], stats.poisson(test_reco).logpmf(self.data).sum()
        )
        self.assertTrue(ret[0, 1] == -np.inf)
        self.assertEqual(ret.shape, (2, 2))

    def test_fix_parameters(self):
        test_reco = np.array([0.5, 0.5, 0.5, 0.5])
        calc = self.calc.fix_parameters([None, np.nan, None, 0.5])
        ret = calc([test_reco[:-1]] * 5)
        self.assertAlmostEqual(
            ret[0, 0], stats.poisson(test_reco).logpmf(self.data).sum()
        )
        self.assertEqual(ret.shape, (2, 5))

    def test_compose(self):
        test_reco = np.array([0.5, 0.5, 0.5, 0.5])
        pred = likelihood.TemplatePredictor([[1, 1, 1, 1]])
        calc = self.calc.compose(pred)
        ret = calc([test_reco[:1]] * 5)
        self.assertAlmostEqual(
            ret[0, 0], stats.poisson(test_reco).logpmf(self.data).sum()
        )
        self.assertEqual(ret.shape, (2, 5))


class TestLikelihoodMaximizers(unittest.TestCase):
    def setUp(self):
        self.data = np.arange(4, dtype=int)
        self.data_model = likelihood.PoissonData(self.data)
        self.predictor = likelihood.MatrixPredictor(np.eye(4), bounds=[(0, np.inf)] * 4)
        self.calc = likelihood.LikelihoodCalculator(self.data_model, self.predictor)

    def test_basinhopping(self):
        maxer = likelihood.BasinHoppingMaximizer()
        opt = maxer(self.calc)
        for i in range(4):
            self.assertAlmostEqual(opt.x[i], self.data[i], places=3)


class TestHypothesisTesters(unittest.TestCase):
    def setUp(self):
        self.data = np.arange(4, dtype=int)
        self.data_model = likelihood.PoissonData(self.data)
        self.predictor = likelihood.MatrixPredictor(np.eye(4), bounds=[(0, np.inf)] * 4)
        self.calc = likelihood.LikelihoodCalculator(self.data_model, self.predictor)
        self.test = likelihood.HypothesisTester(self.calc)

    def test_likelihood_p_value(self):
        ret = self.test.likelihood_p_value(self.data)
        self.assertEqual(ret, 1.0)
        ret = self.test.likelihood_p_value([1, 1, 1, 1])
        self.assertTrue(0.0 < ret < 1.0)
        ret = self.test.likelihood_p_value([1, 1, 1, 0])
        self.assertEqual(ret, 0.0)
        ret = self.test.likelihood_p_value([[[1, 1, 1, 1]] * 2] * 3)
        self.assertEqual(ret.shape, (3, 2))

    def test_max_likelihood_p_value(self):
        ret = self.test.max_likelihood_p_value(N=2)
        self.assertTrue(0.0 <= ret <= 1.0)
        ret = self.test.max_likelihood_p_value(
            fix_parameters=(None, None, None, 20), N=2
        )
        self.assertAlmostEqual(ret, 0.0, places=3)

    def test_max_likelihood_ratio_p_value(self):
        ret = self.test.max_likelihood_ratio_p_value((None, None, None, 3), N=2)
        self.assertTrue(ret >= 0.5)
        ret = self.test.max_likelihood_ratio_p_value(
            (None, None, 2, 30), alternative_fix_parameters=(None, None, None, 3), N=2
        )
        self.assertAlmostEqual(ret, 0.0, places=3)

    def test_wilks_max_likelihood_ratio_p_value(self):
        ret = self.test.wilks_max_likelihood_ratio_p_value((None, None, None, 3))
        self.assertAlmostEqual(ret, 1.0, places=3)
        ret = self.test.wilks_max_likelihood_ratio_p_value(
            (None, None, 2, 30), alternative_fix_parameters=(None, None, None, 3)
        )
        self.assertAlmostEqual(ret, 0.0, places=3)
        ret = self.test.wilks_max_likelihood_ratio_p_value((0, 1, 2, 3))
        self.assertAlmostEqual(ret, 1.0, places=3)


class TestPlotting(unittest.TestCase):
    def setUp(self):
        pass

    def test_array_plotter(self):
        array = np.array([1, 3, 2, 5])
        plt = plotting.get_plotter(array, bins_per_row=3)
        plt.plot_array(hatch=None)
        plt.plot_array(array * 2, hatch=None)
        plt.plot_array([array, array * 2], stack_function=np.mean, hatch=None)
        plt.plot_array([array, array * 2], stack_function=0.68)
        plt.plot_array(
            [array, array * 2], stack_function=(np.mean, np.max), edgecolor="g"
        )

    def test_binning_plotter(self):
        b0 = binning.RectangularBin(
            variables=["x", "y"], edges=[(0, 1), (5, float("inf"))]
        )
        b1 = binning.RectangularBin(
            variables=["x", "y"], edges=[(1, 2), (5, float("inf"))]
        )
        bins0 = binning.Binning(bins=[b0, b1])
        bins = binning.Binning(bins=[b0, b1], subbinnings={0: bins0.clone(dummy=True)})
        bins.set_values_from_ndarray([2.0, 1.0, 4.0])
        bins.set_entries_from_ndarray([4, 2, 8])
        bins.set_sumw2_from_ndarray([1.0, 0.5, 2.0])
        plt = plotting.get_plotter(bins, marginalize_subbinnings=False)
        plt.plot_values(bins.clone())
        plt.plot_entries()
        plt.plot_sumw2()
        plt = plotting.get_plotter(bins, marginalize_subbinnings=True)
        plt.plot_sumw2(bins.clone(), label="Z")
        plt.plot_entries(label="Y", hatch="")
        plt.plot_values(label="X")
        plt.plot_array(bins.value_array)
        plt.plot_array(bins0.value_array)
        plt.legend()

    def test_cartesian_plotter(self):
        x0 = binning.RectangularBin(variables=["x"], edges=[(0, 1)], dummy=True)
        x1 = binning.RectangularBin(variables=["x"], edges=[(1, 2)], dummy=True)
        y0 = binning.RectangularBin(variables=["y"], edges=[(0, 1)], dummy=True)
        y1 = binning.RectangularBin(variables=["y"], edges=[(1, 2)], dummy=True)
        z0 = binning.RectangularBin(variables=["z"], edges=[(0, 1)], dummy=True)
        z1 = binning.RectangularBin(variables=["z"], edges=[(1, 2)], dummy=True)
        bx = binning.Binning(bins=[x0, x1], dummy=True)
        # by = binning.Binning(bins=[y0, y1], dummy=True)
        bz = binning.Binning(bins=[z0, z1], dummy=True)
        bynest = binning.Binning(
            bins=[y0.clone(), y1.clone()], subbinnings={0: bz.clone()}, dummy=True
        )
        b0 = binning.CartesianProductBinning(
            [bx, bynest, bz], subbinnings={2: bz.clone()}
        )
        b0.set_values_from_ndarray(np.arange(13))
        b0.set_entries_from_ndarray(np.arange(13) * 2)
        b0.set_sumw2_from_ndarray(np.arange(13) * 3)
        plt = plotting.get_plotter(b0)
        plt.plot_sumw2(label="Z")
        plt.plot_entries(label="Y")
        plt.plot_values(label="X")
        plt.plot_array(np.arange(26).reshape((2, 13)), label="A", stack_function=0.9)
        plt.legend()

    def test_linear_plotter(self):
        b0 = binning.LinearBinning("x", [0, 1, 5, 7, np.inf])
        b0.set_values_from_ndarray(np.arange(4))
        b0.set_entries_from_ndarray(np.arange(4) * 2)
        b0.set_sumw2_from_ndarray(np.arange(4) * 3)
        plt = plotting.get_plotter(b0, bins_per_row=2)
        plt.plot_sumw2(label="Z")
        plt.plot_entries(label="Y")
        plt.plot_values(label="X")
        plt.legend()

    def test_rectilinear_plotter(self):
        b0 = binning.RectilinearBinning(
            ["x", "y"], [[-1, 0, 1, 5, 7, np.inf], [-np.inf, 0, 5, 10, np.inf]]
        )
        b0.set_values_from_ndarray(np.arange(20))
        b0.set_entries_from_ndarray(np.arange(20) * 2)
        b0.set_sumw2_from_ndarray(np.arange(20) * 3)
        plt = plotting.get_plotter(b0)
        plt.plot_sumw2(label="Z", scatter=100)
        plt.plot_entries(label="Y", scatter=100)
        plt.plot_values(label="X", scatter=100)
        plt.legend()
        # Test specifiying variables for axes.
        plt = plotting.get_plotter(b0, ["x"], ["y"])
        plt.plot_sumw2(label="Z", scatter=100)
        plt.plot_entries(label="Y", scatter=100)
        plt.plot_values(label="X", scatter=100)
        plt.legend()


class TestMatrixUtils(unittest.TestCase):
    def setUp(self):
        with open("testdata/test-truth-binning.yml") as f:
            tb = yaml.full_load(f)
        with open("testdata/test-reco-binning.yml") as f:
            rb = yaml.full_load(f)
        rm_simple = migration.ResponseMatrix(rb, tb)
        tb = tb.insert_subbinning(3, binning.LinearBinning("y_truth", [1.0, 1.5, 2.0]))
        rm = migration.ResponseMatrix(rb, tb)
        rm_simple.fill_from_csv_file("testdata/test-data.csv", weightfield="w")
        rm.fill_from_csv_file("testdata/test-data.csv", weightfield="w")
        self.rm_simple = rm_simple
        self.rm = rm

    def test_mahalanobis_distance(self):
        rA = self.rm
        rB = rA.clone()
        rA.fill_from_csv_file("testdata/test-data.csv", weightfield="w")
        rB.fill_from_csv_file("testdata/test-data.csv", weightfield="w")
        rB.fill_from_csv_file("testdata/test-data.csv", weightfield="w")
        null_distance, distances = matrix_utils.mahalanobis_distance(
            rA, rB, return_distances_from_mean=True
        )
        self.assertTrue(null_distance.shape == (5,))
        self.assertTrue(distances.shape == (104, 5))
        self.assertTrue(np.all(null_distance >= 0.0))
        self.assertTrue(np.all(distances >= 0.0))

    def test_compatibility(self):
        rA = self.rm
        rB = rA.clone()
        rA.fill_from_csv_file("testdata/test-data.csv", weightfield="w")
        rB.fill_from_csv_file("testdata/test-data.csv", weightfield="w")
        rB.fill_from_csv_file("testdata/test-data.csv", weightfield="w")
        p_count, p_chi2, null_distance, distances, n_bins = matrix_utils.compatibility(
            rA, rB, return_all=True, min_quality=0.0
        )
        self.assertTrue(p_count >= 0.0 and p_count <= 1.0)
        self.assertTrue(p_chi2 >= 0.0 and p_count <= 1.0)
        self.assertTrue(null_distance >= 0.0)
        self.assertEqual(n_bins, 4 * 5)
        self.assertEqual(distances.size, 100 + 4)

    def test_plotting(self):
        """Test plotting matrices."""
        # Without subbinning
        matrix_utils.plot_compatibility(
            self.rm_simple, self.rm_simple.clone(), filename=None, min_quality=0.0
        )
        matrix_utils.plot_mahalanobis_distance(
            self.rm_simple, self.rm_simple.clone(), plot_expectation=True, filename=None
        )
        matrix_utils.plot_in_bin_variation(self.rm_simple, filename=None)
        matrix_utils.plot_statistical_uncertainty(self.rm_simple, filename=None)
        matrix_utils.plot_relative_in_bin_variation(self.rm_simple, filename=None)
        matrix_utils.plot_mean_efficiency(self.rm_simple, filename=None)
        matrix_utils.plot_mean_response_matrix(self.rm_simple, filename=None)

        # With subbinning
        matrix_utils.plot_compatibility(
            self.rm, self.rm.clone(), filename=None, min_quality=0.0
        )
        matrix_utils.plot_mahalanobis_distance(
            self.rm, self.rm.clone(), plot_expectation=True, filename=None
        )
        matrix_utils.plot_in_bin_variation(self.rm, filename=None)
        matrix_utils.plot_statistical_uncertainty(self.rm, filename=None)
        matrix_utils.plot_relative_in_bin_variation(self.rm, filename=None)
        matrix_utils.plot_mean_efficiency(self.rm, filename=None)
        matrix_utils.plot_mean_response_matrix(self.rm, filename=None)

    def test_improve_stats(self):
        """Test the automatic optimization of ResponseMatrices."""
        rm0 = matrix_utils.improve_stats(self.rm)
        self.assertEqual(
            self.rm.response_binning.value_array.sum(),
            rm0.response_binning.value_array.sum(),
        )
        self.assertEqual(
            self.rm.truth_binning.value_array.sum(), rm0.truth_binning.value_array.sum()
        )
        self.assertEqual(
            self.rm.reco_binning.value_array.sum(), rm0.reco_binning.value_array.sum()
        )
        rm1 = matrix_utils.improve_stats(rm0)
        self.assertEqual(
            self.rm.response_binning.value_array.sum(),
            rm1.response_binning.value_array.sum(),
        )
        self.assertEqual(
            self.rm.truth_binning.value_array.sum(), rm1.truth_binning.value_array.sum()
        )
        self.assertEqual(
            self.rm.reco_binning.value_array.sum(), rm1.reco_binning.value_array.sum()
        )
        rm1 = matrix_utils.improve_stats(rm0, data_index=3)
        self.assertEqual(
            self.rm.response_binning.value_array.sum(),
            rm1.response_binning.value_array.sum(),
        )
        self.assertEqual(
            self.rm.truth_binning.value_array.sum(), rm1.truth_binning.value_array.sum()
        )
        self.assertEqual(
            self.rm.reco_binning.value_array.sum(), rm1.reco_binning.value_array.sum()
        )
        rm1 = matrix_utils.improve_stats(self.rm, data_index=3)
        self.assertEqual(
            self.rm.response_binning.value_array.sum(),
            rm1.response_binning.value_array.sum(),
        )
        self.assertEqual(
            self.rm.truth_binning.value_array.sum(), rm1.truth_binning.value_array.sum()
        )
        self.assertEqual(
            self.rm.reco_binning.value_array.sum(), rm1.reco_binning.value_array.sum()
        )


class TestLikelihoodUtils(unittest.TestCase):
    def setUp(self):
        self.data = np.arange(4)
        self.pred = likelihood.TemplatePredictor(
            [np.eye(4), np.eye(4)],
            bounds=[(0, np.inf), (-np.inf, 1.001), (1.0, np.inf), (0.0, 1.0)],
        )
        self.data_model = likelihood.PoissonData(self.data)
        self.calc = likelihood.LikelihoodCalculator(self.data_model, self.pred)
        self.test = likelihood.HypothesisTester(self.calc)

    def test_emcee(self):
        sampler = likelihood_utils.emcee_sampler(self.calc)
        guess = likelihood_utils.emcee_initial_guess(self.calc)
        state = sampler.run_mcmc(guess, 500)
        sampler.reset()
        sampler.run_mcmc(state, 500)
        chain = sampler.get_chain(flat=True)
        np.mean(chain, axis=0)


if __name__ == "__main__":
    np.seterr(all="raise")
    unittest.main(argv=testargs)
