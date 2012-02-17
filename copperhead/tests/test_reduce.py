#
#   Copyright 2012      NVIDIA Corporation
# 
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
# 
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
# 
#
from copperhead import *
import numpy as np
import unittest

@cu
def test_reduce(x, p):
    return reduce(op_add, x, p)

@cu
def test_sum(x):
    return sum(x)

@cu
def test_sum_as_reduce(x):
    return reduce(op_add, x, 0)

class ReduceTest(unittest.TestCase):
    def setUp(self):
        source = range(5)
        prefix = 1
        self.golden_s = sum(source)
        self.golden_r = self.golden_s + prefix
        self.int32 = (np.array(source, dtype=np.int32), np.int32(prefix))
        self.int64 = (np.array(source, dtype=np.int64), np.int64(prefix))
        self.float32 = (np.array(source, dtype=np.float32), np.float32(prefix))
        self.float64 = (np.array(source, dtype=np.float64), np.float64(prefix))
        
    def run_test(self, f, g, *args):
        self.assertEqual(f(*args), g)

    def testReduceInt32(self):
        self.run_test(test_reduce, self.golden_r, *self.int32)
    def testReduceInt64(self):
        self.run_test(test_reduce, self.golden_r, *self.int64)
    def testReduceFloat32(self):
        self.run_test(test_reduce, self.golden_r, *self.float32)
    @unittest.skipIf(not runtime.float64_support, "CUDA device doesn't support doubles")
    def testReduceFloat64(self):
        self.run_test(test_reduce, self.golden_r, *self.float64)

    def testSumInt32(self):
        self.run_test(test_sum, self.golden_s, self.int32[0])
    def testSumInt64(self):
        self.run_test(test_sum, self.golden_s, self.int64[0])
    def testSumFloat32(self):
        self.run_test(test_sum, self.golden_s, self.float32[0])
    @unittest.skipIf(not runtime.float64_support, "CUDA device doesn't support doubles")
    def testSumFloat64(self):
        self.run_test(test_sum, self.golden_s, self.float64[0])
        
    def testSumAsReduceInt32(self):
        self.run_test(test_sum_as_reduce, self.golden_s, self.int32[0])
    def testSumAsReduceInt64(self):
        self.run_test(test_sum_as_reduce, self.golden_s, self.int64[0])
    def testSumAsReduceFloat32(self):
        self.run_test(test_sum_as_reduce, self.golden_s, self.float32[0])
    @unittest.skipIf(not runtime.float64_support, "CUDA device doesn't support doubles")
    def testSumAsReduceFloat64(self):
        self.run_test(test_sum_as_reduce, self.golden_s, self.float64[0])


if __name__ == "__main__":
    unittest.main()
