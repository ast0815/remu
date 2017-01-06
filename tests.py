import unittest
import ruamel.yaml as yaml
from binning import *

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
        self.bd = Bin(phasespace=ps, value={'a': 0, 'b': 1})

    def test_init_values(self):
        """Test initialization values."""
        self.assertEqual(self.b0.value, 0.)
        self.assertEqual(self.b1.value, 1.)
        self.assertEqual(self.b2.value, 2.)
        self.assertEqual(self.bd.value, {'a': 0, 'b': 1})

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
        self.b0.fill(0.5)
        self.assertEqual(self.b0.value, 1.5)
        self.b0.fill([0.5, 0.5, 0.5])
        self.assertEqual(self.b0.value, 3.0)

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
        self.assertEqual(self.bd.value, eval(repr(self.bd)).value)

    def test_yaml_representation(self):
        """Test whether the yaml parsing can reproduce the original object."""
        self.assertEqual(self.b0.phasespace, yaml.load(yaml.dump(self.b0)).phasespace)
        self.assertEqual(self.b0.value, yaml.load(yaml.dump(self.b0)).value)
        self.assertEqual(self.b1.value, yaml.load(yaml.dump(self.b1)).value)
        self.assertEqual(self.b2.value, yaml.load(yaml.dump(self.b2)).value)
        self.assertEqual(self.bd.value, yaml.load(yaml.dump(self.bd)).value)

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
        self.assertEqual(self.b1.value, 0)
        self.binning.fill({'x': 1.5, 'y': 10}, 2)
        self.assertEqual(self.b0.value, 1)
        self.assertEqual(self.b1.value, 2)
        self.binning.fill([{'x': 0.5, 'y': 10}, {'x': 0.5, 'y': 20}], 2)
        self.assertEqual(self.b0.value, 5)
        self.assertEqual(self.b1.value, 2)
        self.binning.fill([{'x': 0.5, 'y': 10}, {'x': 1.5, 'y': 10}], [1, 2])
        self.assertEqual(self.b0.value, 6)
        self.assertEqual(self.b1.value, 4)
        self.binning.fill([{'x': -0.5, 'y': 10}, {'x': 1.5, 'y': 10}], [1, 2])
        self.assertEqual(self.b0.value, 6)
        self.assertEqual(self.b1.value, 6)
        self.assertRaises(ValueError, lambda: self.binning.fill({'x': -0.5, 'y': 10}, raise_error=True))
        self.assertEqual(self.b0.value, 6)
        self.assertEqual(self.b1.value, 6)
        self.binning.fill({'x': 0.5, 'y': 10, 'z': 123})
        self.assertEqual(self.b0.value, 7)
        self.assertEqual(self.b1.value, 6)
        self.assertRaises(KeyError, lambda: self.binning.fill({'x': 0.5}))

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
        self.bu = RectangularBinning(binedges={'x': [0,1,2], 'y': (-10,0,10,20,float('inf'))}, variables=['x', 'y'], include_upper=True)
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
        self.assertEqual(self.bxyz.get_bin_number_tuple(self.bl.get_tuple_bin_number((1,3,0))), (1,3,0))
        self.assertEqual(self.bl.get_tuple_bin_number((1,2)), 5)
        self.assertEqual(self.bl.get_event_bin_number({'x': 0, 'y': -10}), 0)
        self.assertEqual(self.bl.get_event_bin_number({'x': 1, 'y': -10}), 1)
        self.assertEqual(self.bl.get_event_bin_number({'x': -1, 'y': -10}), None)
        self.assertEqual(self.bu.get_event_bin_number({'x': 2, 'y': 30}), 7)

    def test_fill(self):
        """Test bin filling"""
        self.bl.fill({'x': 0.5, 'y': -10}, 1)
        self.bl.fill({'x': 1.5, 'y': -10}, 2)
        self.assertEqual(self.bl.bins[0].value, 1)
        self.assertEqual(self.bl.bins[1].value, 2)
        self.bl.fill([{'x': 0.5, 'y': -10}, {'x': 1.5, 'y': -10}], [1, 2])
        self.assertEqual(self.bl.bins[0].value, 2)
        self.assertEqual(self.bl.bins[1].value, 4)
        self.bl.fill({'x': 0.5, 'y': -10, 'z': 123})
        self.assertEqual(self.bl.bins[0].value, 3)
        self.assertEqual(self.bl.bins[1].value, 4)
        self.assertRaises(KeyError, lambda: self.bl.fill({'x': 0.5}))

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

    def test_yaml_representation(self):
        """Test whether the yaml parsing can reproduce the original object."""
        orig = self.bl
        reco = yaml.load(yaml.dump(orig))
        self.assertEqual(orig, reco)

if __name__ == '__main__':
    unittest.main()
