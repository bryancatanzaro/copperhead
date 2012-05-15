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
def test_abs(x):
    return abs(x)

@cu
def test_seq_abs(x):
    return map(abs, x)

@cu
def test_exp(x):
    return exp(x)

@cu
def test_seq_exp(x):
    return map(exp, x)

@cu
def test_log(x):
    return log(x)

@cu
def test_seq_log(x):
    return map(log, x)

@cu
def test_sqrt(x):
    return sqrt(x)

@cu
def test_seq_sqrt(x):
    return map(sqrt, x)



class ScalarMathTest(unittest.TestCase):
        
    def run_test(self, target, f, g, *args):
        with target:
            self.assertTrue(recursive_equal(f(*args), g))
            
    @create_tests(*runtime.backends)
    def testAbs(self, target):
        self.run_test(target, test_abs, 1, *(1,))
        self.run_test(target, test_abs, 1, *(-1,))

    @create_tests(*runtime.backends)
    def testAbsSeq(self, target):
        self.run_test(target, test_seq_abs, [1, 1], *([1,-1],))

    @create_tests(*runtime.backends)
    def testExp(self, target):
        b = np.float32(1)
        e_b = np.exp(b)
        self.run_test(target, test_exp, e_b, *(b,))

    @create_tests(*runtime.backends)
    def testExpSeq(self, target):
        a = np.float32(0)
        e_a = np.exp(a)
        b = np.float32(1)
        e_b = np.exp(b)
        self.run_test(target, test_seq_exp, [e_a, e_b], *([a, b],))

    @create_tests(*runtime.backends)
    def testLog(self, target):
        b = np.float32(1)
        e_b = np.log(b)
        self.run_test(target, test_log, e_b, *(b,))

    @create_tests(*runtime.backends)
    def testLogSeq(self, target):
        a = np.float32(1)
        e_a = np.log(a)
        b = np.float32(1)
        e_b = np.log(b)
        self.run_test(target, test_seq_log, [e_a, e_b], *([a, b],))

    @create_tests(*runtime.backends)
    def testSqrt(self, target):
        b = np.float32(2)
        e_b = np.sqrt(b)
        self.run_test(target, test_sqrt, e_b, *(b,))

    @create_tests(*runtime.backends)
    def testSqrtSeq(self, target):
        a = np.float32(1)
        e_a = np.sqrt(a)
        b = np.float32(4)
        e_b = np.sqrt(b)
        self.run_test(target, test_seq_sqrt, [e_a, e_b], *([a, b],))



if __name__ == "__main__":
    unittest.main()
