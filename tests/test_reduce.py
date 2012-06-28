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
from create_tests import create_tests

@cu
def test_reduce(x, p):
    return reduce(op_add, x, p)

@cu
def test_sum(x):
    return sum(x)

@cu
def test_reduce_as_sum(x):
    return reduce(op_add, x, 0)

class ReduceTest(unittest.TestCase):
    def setUp(self):
        source = [1,2,3,4,5]
        prefix = 1
        self.golden_s = sum(source)
        self.golden_r = self.golden_s + prefix
        self.int32 = (np.array(source, dtype=np.int32), np.int32(prefix))
        
    def run_test(self, target, f, g, *args):
        with target:
            self.assertEqual(f(*args), g)
            
    @create_tests(*runtime.backends)
    def testReduce(self, target):
        self.run_test(target, test_reduce, self.golden_r, *self.int32)

    @create_tests(*runtime.backends)
    def testSum(self, target):
        self.run_test(target, test_sum, self.golden_s, self.int32[0])

    @create_tests(*runtime.backends)
    def testSumAsReduce(self, target):
        self.run_test(target, test_reduce_as_sum, self.golden_s, self.int32[0])


if __name__ == "__main__":
    unittest.main()
