from copperhead import *
import numpy as np
import unittest

@cu
def test_reduce(x, p):
    return reduce(op_add, x, p)

class ReduceTest(unittest.TestCase):
    def setUp(self):
        source = [1,2,3,4,5]
        prefix = 1
        self.golden = sum(source) + prefix
        self.int32s = np.array(source, dtype=np.int32)
        self.int32c = np.int32(1)
        self.int64s = np.array(source, dtype=np.int64)
        self.int64c = np.int64(1)
        self.float32s = np.array(source, dtype=np.float32)
        self.float32c = np.float32(1)
        self.float64s = np.array(source, dtype=np.float64)
        self.float64c = np.float64(1)
    def run_test(self, x, p):
        self.assertEqual(test_reduce(x,
                                     p),
                         self.golden)

    def testInt32(self):
        self.run_test(self.int32s, self.int32c)
    def testInt64(self):
        self.run_test(self.int64s, self.int64c)
    def testFloat32(self):
        self.run_test(self.float32s, self.float32c)
    def testFloat64(self):
        self.run_test(self.float64s, self.float64c)

