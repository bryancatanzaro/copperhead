from copperhead import *
import unittest
from recursive_equal import recursive_equal

@cu
def test_zip(x, y):
    return zip(x, y)

@cu
def test_unzip(x):
    y, z = unzip(x)
    return y, z

@cu
def shift_zip(x, y, z, d):
    a = zip(x, y)
    b = shift(a, d, z)
    return b

class ZipTest(unittest.TestCase):
    def setUp(self):
        self.x = [1,2,3,4,5]
        self.y = [3,4,5,6,7]
    def testZip(self):
        self.assertTrue(
            recursive_equal(test_zip(self.x, self.y),
                            [(1,3),(2,4),(3,5),(4,6),(5,7)]))
    def testShiftZip(self):
        self.assertTrue(
            recursive_equal(shift_zip(self.x, self.y, (-1, -2), 1),
                            [(2, 4), (3, 5), (4, 6), (5, 7), (-1, -2)]))
        self.assertTrue(
            recursive_equal(shift_zip(self.x, self.y, (-3, -4), -1),
                            [(-3, -4), (1, 3), (2, 4), (3, 5), (4, 6)]))
    
    def testUnzip(self):
        self.assertTrue(
            recursive_equal(test_unzip(test_zip(self.x, self.y)),
                            ([1,2,3,4,5],[3,4,5,6,7])))

if __name__ == '__main__':
    unittest.main()
