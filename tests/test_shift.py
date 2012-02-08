from copperhead import *
import numpy as np
import unittest

@cu
def test_shift(x, amount, boundary):
    return shift(x, amount, boundary)

class ShiftTest(unittest.TestCase):
    def setUp(self):
        self.source = [1,2,3,4,5]

    def run_test(self, fn, *args):
        cpuResult = fn(*args, target_place=places.here)
        gpuResult = fn(*args, target_place=places.gpu0)
        self.assertEqual(list(cpuResult), list(gpuResult))
        
    def testShiftP(self):
        self.run_test(test_shift, self.source, 2, 3)

    def testShiftN(self):
        self.run_test(test_shift, self.source, -2, 4)


if __name__ == "__main__":
    unittest.main()
