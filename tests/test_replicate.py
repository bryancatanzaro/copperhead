from copperhead import *
import numpy as np
import unittest

@cu
def test_repl(x, n):
    return replicate(x, n)

class ReplicateTest(unittest.TestCase):
    def setUp(self):
        self.val = 3
        self.size = 5
        self.golden = [self.val] * self.size

    def run_test(self, x, n):
        self.assertEqual(list(test_repl(x, n)), self.golden)

    def testReplInt32(self):
        self.run_test(np.int32(self.val), self.size)
    def testReplInt64(self):
        self.run_test(np.int64(self.val), self.size)
    def testReplFloat32(self):
        self.run_test(np.float32(self.val), self.size)
    @unittest.skipIf(not runtime.float64_support, "CUDA Device does not support doubles")
    def testReplFloat64(self):
        self.run_test(np.float64(self.val), self.size)
    
if __name__ == "__main__":
    unittest.main()
