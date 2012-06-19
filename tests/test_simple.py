#
#   Copyright 2008-2012 NVIDIA Corporation
#  Copyright 2009-2010 University of California
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
def saxpy(a, x, y):
    """Add a vector y to a scaled vector a*x"""
    return map(lambda xi, yi: a * xi + yi, x, y)

@cu
def saxpy2(a, x, y):
    return [a*xi+yi for xi,yi in zip(x,y)]

@cu
def saxpy3(a, x, y):
    def triad(xi, yi):
        return a * xi + yi
    return map(triad, x, y)

@cu
def sxpy(x, y):
    def duad(xi, yi):
        return xi + yi
    return map(duad, x, y)

@cu
def incr(x):
    return map(lambda xi: xi + 1, x)

@cu
def as_ones(x):
    return map(lambda xi: 1, x)

@cu
def idm(x):
    return map(lambda b: b, x)

@cu
def idx(x):
    def id(xi):
        return xi
    return map(id, x)

@cu
def incr_list(x):
    return [xi + 1 for xi in x]

class SimpleTests(unittest.TestCase):
    def setUp(self):
        self.hasGPU = hasattr(places, 'gpu0')
        self.ints = np.arange(7, dtype=np.int32)
        self.consts = np.array([1] * 7, dtype = np.int32)
        self.floats = np.array(self.ints, dtype=np.float32)


    def run_test(self, fn, *args):
        python_result = fn(*args, target_place=places.here)
        copperhead_result = fn(*args)
        self.assertEqual(list(python_result), list(copperhead_result))


    @create_tests(*runtime.backends)
    def testIncrInts(self, target):
        with target:
            self.run_test(incr, self.ints)
    @create_tests(*runtime.backends)
    def testIncrFloats(self, target):
        with target:
            self.run_test(incr, self.floats)
    @create_tests(*runtime.backends)
    def testIncrListInts(self, target):
        with target:
            self.run_test(incr_list, self.ints)
    @create_tests(*runtime.backends)
    def testIncrListFloats(self, target):
        with target:
            self.run_test(incr_list, self.floats)
    @create_tests(*runtime.backends)
    def testAsonesInts(self, target):
        with target:
            self.run_test(as_ones, self.ints)
    @create_tests(*runtime.backends)
    def testAsonesFloats(self, target):
        with target:
            self.run_test(as_ones, self.floats)
    @create_tests(*runtime.backends)
    def testIdmInts(self, target):
        with target:
            self.run_test(idm, self.ints)
    @create_tests(*runtime.backends)
    def testIdmFloats(self, target):
        with target:
            self.run_test(idm, self.floats)
    @create_tests(*runtime.backends)
    def testSaxpyInts(self, target):
        with target:
            self.run_test(saxpy, np.int32(2), self.ints, self.consts)
    @create_tests(*runtime.backends)
    def testSaxpyFloats(self, target):
        with target:
            self.run_test(saxpy, np.float32(2), self.floats, self.floats)
    @create_tests(*runtime.backends)
    def testSaxpy2Ints(self, target):
        with target:
            self.run_test(saxpy2, np.int32(2), self.ints, self.consts)
    @create_tests(*runtime.backends)
    def testSaxpy2Floats(self, target):
        with target:
            self.run_test(saxpy2, np.float32(2), self.floats, self.floats)
    @create_tests(*runtime.backends)
    def testSaxpy3Ints(self, target):
        with target:
            self.run_test(saxpy3, np.int32(2), self.ints, self.consts)
    @create_tests(*runtime.backends)
    def testSaxpy3Floats(self, target):
        with target:
            self.run_test(saxpy3, np.float32(2), self.floats, self.floats)    
    @create_tests(*runtime.backends)
    def testSxpyInts(self, target):
        with target:
            self.run_test(sxpy, self.ints, self.ints)
    @create_tests(*runtime.backends)
    def testSxpyFloats(self, target):
        with target:
            self.run_test(sxpy, self.ints, self.ints)    

if __name__ == "__main__":
    unittest.main()
