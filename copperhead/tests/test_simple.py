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
        cpuResult = fn(*args, target_place=places.here)
        if self.hasGPU:
            try:
                gpuResult = fn(*args, target_place=places.gpu0)
            except:
                gpuResult = []
            self.assertEqual(list(cpuResult), list(gpuResult))


    @unittest.skipIf(not runtime.cuda_support,'No CUDA support')
    def testIncrInts(self):
        self.run_test(incr, self.ints)
    @unittest.skipIf(not runtime.cuda_support,'No CUDA support')
    def testIncrFloats(self):
        self.run_test(incr, self.floats)
    @unittest.skipIf(not runtime.cuda_support,'No CUDA support')
    def testIncrListInts(self):
        self.run_test(incr_list, self.ints)
    @unittest.skipIf(not runtime.cuda_support,'No CUDA support')
    def testIncrListFloats(self):
        self.run_test(incr_list, self.floats)
    @unittest.skipIf(not runtime.cuda_support,'No CUDA support')
    def testAsonesInts(self):
        self.run_test(as_ones, self.ints)
    @unittest.skipIf(not runtime.cuda_support,'No CUDA support')
    def testAsonesFloats(self):
        self.run_test(as_ones, self.floats)
    @unittest.skipIf(not runtime.cuda_support,'No CUDA support')
    def testIdmInts(self):
        self.run_test(idm, self.ints)
    @unittest.skipIf(not runtime.cuda_support,'No CUDA support')
    def testIdmFloats(self):
        self.run_test(idm, self.floats)
    @unittest.skipIf(not runtime.cuda_support,'No CUDA support')
    def testSaxpyInts(self):
        self.run_test(saxpy, np.int32(2), self.ints, self.consts)
    @unittest.skipIf(not runtime.cuda_support,'No CUDA support')
    def testSaxpyFloats(self):
        self.run_test(saxpy, np.float32(2), self.floats, self.floats)
    @unittest.skipIf(not runtime.cuda_support,'No CUDA support')
    def testSaxpy2Ints(self):
        self.run_test(saxpy2, np.int32(2), self.ints, self.consts)
    @unittest.skipIf(not runtime.cuda_support,'No CUDA support')
    def testSaxpy2Floats(self):
        self.run_test(saxpy2, np.float32(2), self.floats, self.floats)
    @unittest.skipIf(not runtime.cuda_support,'No CUDA support')
    def testSaxpy3Ints(self):
        self.run_test(saxpy3, np.int32(2), self.ints, self.consts)
    @unittest.skipIf(not runtime.cuda_support,'No CUDA support')
    def testSaxpy3Floats(self):
        self.run_test(saxpy3, np.float32(2), self.floats, self.floats)    
    @unittest.skipIf(not runtime.cuda_support,'No CUDA support')
    def testSxpyInts(self):
        self.run_test(sxpy, self.ints, self.ints)
    @unittest.skipIf(not runtime.cuda_support,'No CUDA support')
    def testSxpyFloats(self):
        self.run_test(sxpy, self.ints, self.ints)    

if __name__ == "__main__":
    unittest.main()
