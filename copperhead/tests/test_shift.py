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
def test_shift(x, amount, boundary):
    return shift(x, amount, boundary)

class ShiftTest(unittest.TestCase):
    def setUp(self):
        self.source = [1,2,3,4,5]

    def run_test(self, fn, *args):
        cpuResult = fn(*args, target_place=places.here)
        gpuResult = fn(*args, target_place=places.gpu0)
        self.assertEqual(list(cpuResult), list(gpuResult))
        
    @unittest.skipIf(not runtime.cuda_support,'No CUDA support')
    def testShiftP(self):
        self.run_test(test_shift, self.source, 2, 3)

    @unittest.skipIf(not runtime.cuda_support,'No CUDA support')
    def testShiftN(self):
        self.run_test(test_shift, self.source, -2, 4)


if __name__ == "__main__":
    unittest.main()
