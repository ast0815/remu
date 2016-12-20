import unittest
import binning

class TestPhasSpaces(unittest.TestCase):
    def setUp(self):
        self.psX = binning.PhaseSpace(variables=['x'])
        self.psY = binning.PhaseSpace(variables=['y'])
        self.psXY = binning.PhaseSpace(variables=['x', 'y'])

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
        ps = binning.PhaseSpace(variables=['x'])
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


class TestBins(unittest.TestCase):
    def test_init_values(self):
        """Test initialization values."""
        ps = binning.PhaseSpace(['x'])
        b = binning.Bin(phasespace=ps)
        self.assertEqual(b.value, 0.)
        b = binning.Bin(phasespace=ps, value=7.)
        self.assertEqual(b.value, 7.)
        b = binning.Bin(phasespace=ps, value={'a': 0, 'b': 1})
        self.assertEqual(b.value, {'a': 0, 'b': 1})

class TestRectangularBins(unittest.TestCase):
    def setUp(self):
        self.b = binning.RectangularBin(edges={'x':(0,1), 'y':(5,float('inf'))})

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
        self.b.include_lower=False
        self.assertFalse({'x': 0, 'y': 10} in self.b)

    def test_include_upper(self):
        """Test inclusion of upper bounds."""
        self.b.include_upper=True
        self.assertTrue({'x': 1, 'y': 10} in self.b)
        self.b.include_upper=False
        self.assertFalse({'x': 1, 'y': 10} in self.b)

    def test_bin_centers(self):
        """Test calculation of bin centers."""
        c = self.b.get_center()
        self.assertEqual(c['x'], 0.5)
        self.assertEqual(c['y'], float('inf'))

class TestBinnings(unittest.TestCase):
    def setUp(self):
        self.b0 = binning.RectangularBin(edges={'x':(0,1), 'y':(5,float('inf'))})
        self.b1 = binning.RectangularBin(edges={'x':(1,2), 'y':(5,float('inf'))})
        self.binning = binning.Binning(phasespace=self.b0.phasespace, bins=[self.b0 ,self.b1])

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

if __name__ == '__main__':
    unittest.main()
