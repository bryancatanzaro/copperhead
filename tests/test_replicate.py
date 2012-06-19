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
def test_repl(x, n):
    return replicate(x, n)

@cu
def test_internal_tuple_repl(x, n):
    return replicate((x, x), n)

@cu
def test_internal_named_tuple_repl(x, n):
    a = x, x
    return replicate(a, n)

class ReplicateTest(unittest.TestCase):
    def setUp(self):
        self.val = 3
        self.size = 5
        self.golden = [self.val] * self.size

    def run_test(self, target, x, n):
        with target:
            self.assertEqual(list(test_repl(x, n)), self.golden)

    @create_tests(*runtime.backends)
    def testRepl(self, target):
        self.run_test(target, np.int32(self.val), self.size)

    def testReplTuple(self):
        self.assertTrue(recursive_equal(test_repl((1,1), 2), [(1,1),(1,1)]))

    def testReplNestedTuple(self):
        a = test_repl(((1,2),3),2)
        self.assertTrue(recursive_equal(
                a,
                [((1,2),3),((1,2),3)]))              
        
    def testReplInternalTuple(self):
        self.assertTrue(recursive_equal(test_internal_tuple_repl(1, 2),
                                        [(1, 1), (1, 1)]))
    def testReplInternalNamedTuple(self):
        self.assertTrue(recursive_equal(test_internal_named_tuple_repl(1, 2),
                                        [(1, 1), (1, 1)]))
        
if __name__ == "__main__":
    unittest.main()
