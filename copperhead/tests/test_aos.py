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
import unittest
from recursive_equal import recursive_equal
import numpy as np

@cu
def demux(x):
    return int32(x)+1, float32(x)+0.25

@cu
def test(x):
    return map(demux, x)

class AoSTest(unittest.TestCase):
    def testAoS(self):
        three = cuarray([1,2,3])
        golden_result = [(np.int32(2), np.float32(1.25)),
                         (np.int32(3), np.float32(2.25)),
                         (np.int32(4), np.float32(3.25))]
        copperhead_result = test(three)
        self.assertTrue(recursive_equal(golden_result, copperhead_result))


if __name__ == "__main__":
    unittest.main()
