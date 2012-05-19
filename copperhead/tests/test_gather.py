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
from recursive_equal import recursive_equal


@cu
def test_gather(x, i):
    return gather(x, i)


@cu
def test_gather_fusion(x, i):
    def incr(xi):
        return xi + 1
    return map(incr, gather(x, i))

class GatherTest(unittest.TestCase):
    def setUp(self):
        self.x = cuarray([1,2,3,4,5,6,7,8,9,10])
        self.i = cuarray([2,5,7])
        self.golden = cuarray([3,6,8])
        self.fusion_golden = cuarray([4,7,9])
        
    def run_test(self, target, fn, golden):
        with target:
            self.assertTrue(recursive_equal(fn(self.x, self.i), golden))

    @create_tests(*runtime.backends)
    def testGather(self, target):
        self.run_test(target, test_gather, self.golden)

        
    @create_tests(*runtime.backends)
    def testGatherFusion(self, target):
        self.run_test(target, test_gather_fusion, self.fusion_golden)
        
if __name__ == "__main__":
    unittest.main()
