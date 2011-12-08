#
#  Copyright 2008-2010 NVIDIA Corporation
#  Copyright 2009-2010 University of California
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

from copperhead import *
import numpy as np

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
def incrList(x):
    return [xi + 1 for xi in x]

if __name__ == "__main__":
    hasGPU = hasattr(places, 'gpu0')

    def test(fn, *args):
        cpuResult = fn(*args, targetPlace=places.here)
        if hasGPU:
            try:
                gpuResult = fn(*args, targetPlace=places.gpu0)
            except:
                gpuResult = []
        print ("Procedure '%s'" % fn.__name__).ljust(50),
        if not hasGPU:
            print "... NO GPU"
            print "   python     :", list(cpuResult)
        elif list(cpuResult)==list(gpuResult.np()):
            print "... PASSED"
            print "   copperhead :", list(gpuResult)
        else:
            print "... FAILED"
            print "   python     :", list(cpuResult)
            print "   copperhead :", list(gpuResult)

        return gpuResult if hasGPU else cpuResult

    ints = range(7)
    floats = np.array(ints, dtype=np.float32)
    
    print
    print "---- Simple INTEGER tests ----"
    test(incr, ints)
    test(incrList, ints)
    test(as_ones, ints)
    test(idm, ints)
    test(idx, ints)
    test(saxpy,  2, range(7), [1]*7)
    test(saxpy2, 2, range(7), [1]*7)
    test(saxpy3, 2, range(7), [1]*7)
    test(sxpy, ints, ints)

    print
    print "---- Simple FLOAT tests ----"
    test(as_ones, floats)
    test(idm, floats)
    test(idx, floats)
    test(sxpy, floats, floats)
