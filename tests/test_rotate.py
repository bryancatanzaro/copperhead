from copperhead import *
import numpy as np
import unittest


@cu
def test_rotate(x, amount):
    return rotate(x, amount)

class RotateTest(unittest.TestCase):
    def setUp(self):
        self.source = range(5)
        

    def run_test(self, fn, *args):
        cpuResult = fn(*args, targetPlace=places.here)
        gpuResult = fn(*args, targetPlace=places.gpu0)
        self.assertEqual(list(cpuResult), list(gpuResult))
    
    def testRotateP(self):
        self.run_test(test_rotate, self.source, 2)

    def testRotateN(self):
        self.run_test(test_rotate, self.source, -2)


if __name__ == "__main__":
    unittest.main()
