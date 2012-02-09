#
#   Copyright 2008-2012 NVIDIA Corporation
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

@cu
def nrm2(x, y):
    def diff_sq(xi, yi):
        diff = xi - yi
        return diff * diff
    return sqrt(sum(map(diff_sq, x, y)))

@cu
def axpy(a, x, y):
    def triad(xi, yi):
        return a * xi + yi
    return map(triad, x, y)


def test_saxpy(length):
    x = np.arange(0, length, dtype=np.float32)
    y = np.arange(1, length + 1, dtype=np.float32)
    a = np.float32(0.5)
    with places.gpu0:
        print("Compiling and Running on GPU")
        z = axpy(a, x, y)

    #Run on Python interpreter
    with places.here:
        print("Running in Python interpreter")
        zPython = axpy(a, x, y)
        
    print("Calculating difference")
    with places.gpu0:
        error = nrm2(z, zPython)
    
    return (x, y, z, zPython, error)

if __name__ == '__main__':
    length = 1000000

    (x, y, z, zPython, error) = test_saxpy(length)

    print('Error: %s' % str(error))
   

   
