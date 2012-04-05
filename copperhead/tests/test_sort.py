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
import random
from create_tests import create_tests

@cu
def lt_sort(x):
    return sort(cmp_lt, x)

@cu
def gt_sort(x):
    return sort(cmp_gt, x)

class SortTest(unittest.TestCase):
    def setUp(self):
        self.source = np.array([random.random() for x in range(5)], dtype=np.float32)
        
        
    def run_test(self, target, fn, *args):
        python_result = fn(*args, target_place=places.here)
        copperhead_result = fn(*args, target_place=target)
        self.assertEqual(list(python_result), list(copperhead_result))
    
    @create_tests(*runtime.backends)
    def testLtSort(self, target):
        self.run_test(target, lt_sort, self.source)

    @create_tests(*runtime.backends)
    def testGtSort(self, target):
        self.run_test(target, gt_sort, self.source)


if __name__ == "__main__":
    unittest.main()

