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
