from copperhead import *
import numpy as np
import unittest

@cu
def test_shift(x, amount, boundary):
    return shift(x, amount, boundary)

class ShiftTest(unittest.TestCase):
    def setUp(self):
        self.source = np.arange(5, dtype=np.int32)
        

    def run_test(self, fn, *args):
        cpuResult = fn(list(args[0]), args[1], args[2], targetPlace=places.here)
        gpuResult = fn(*args, targetPlace=places.gpu0)
        self.assertEqual(list(cpuResult), list(gpuResult))
        
    def testShiftP(self):
        self.run_test(test_shift, self.source, np.int32(2), np.int32(3))

    def testShiftN(self):
        self.run_test(test_shift, self.source, np.int32(-2), np.int32(4))


if __name__ == "__main__":
    unittest.main()
