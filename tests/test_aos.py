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
    return int32(x), (float32(x)+1.25, float32(x)+2.5)

@cu
def test(x):
    return map(demux, x)

@cu
def mux((x, (y, z))):
    return float32(x) + float32(y) + float32(z)

@cu
def test2(x):
    return map(mux, x)

@cu
def test3(x):
    a = test(x)
    b = test2(a)
    return b

class AoSTest(unittest.TestCase):
    def setUp(self):
        self.three = [1,2,3]
        self.golden_result = [(1, (2.25, 3.5)),
                         (2, (3.25, 4.5)),
                         (3, (4.25, 5.5))]
        self.golden_result_2 = [6.75, 9.75, 12.75]
        
    def testAoS_1(self):
        self.assertTrue(recursive_equal(self.golden_result, test(self.three)))

    def testAoS_2(self):
        #XXX Change once cuarrays can be constructed with tuple elements
        self.assertTrue(recursive_equal(self.golden_result_2,
                                        test2(test(self.three))))
    def testAoS_3(self):
        self.assertTrue(recursive_equal(self.golden_result_2,
                                        test3(self.three)))
        
if __name__ == "__main__":
    unittest.main()
