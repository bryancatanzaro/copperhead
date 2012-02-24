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

@cu
def lt_sort(x):
    return sort(cmp_lt, x)

@cu
def gt_sort(x):
    return sort(cmp_gt, x)

class SortTest(unittest.TestCase):
    def setUp(self):
        self.source = np.array([random.random() for x in range(5)], dtype=np.float32)
        
        
    def run_test(self, fn, *args):
        cpuResult = fn(*args, target_place=places.here)
        gpuResult = fn(*args, target_place=places.gpu0)
        self.assertEqual(list(cpuResult), list(gpuResult))
    
    @unittest.skipIf(not runtime.cuda_support,'No CUDA support')
    def testLtSort(self):
        self.run_test(lt_sort, self.source)

    @unittest.skipIf(not runtime.cuda_support,'No CUDA support')
    def testGtSort(self):
        self.run_test(gt_sort, self.source)


if __name__ == "__main__":
    unittest.main()

