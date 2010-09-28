#
#  Copyright 2008-2009 NVIDIA Corporation
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
def axpy(a, x, y):
    "Compute a*x+y for vectors x,y and scalar a."
    return map(lambda xi, yi: a * xi + yi, x, y)

@cu
def nrm2(x):
    "Compute the squared L2 norm of the vector x."
    return sum([xi*xi for xi in x])

@cu
def norm_diff(x, y):
    "Compute the squared L2 norm of the difference between x and y"
    diff = map(op_sub, x, y)
    return nrm2(diff)

@cu
def dot(x, y):
    "Compute the dot (inner) product of vectors x and y."
    return sum([xi*yi for xi,yi in zip(x,y)])

@cu
def gemv(A, x):
    """
    Compute A*x for an m-by-n matrix A and n-vector x.

    The matrix should be represented using a nested sequence
    storing the matrix elements in row-major order.
    """

    return map(lambda Ai: dot(Ai, x), A)


if __name__ == "__main__":
    a = np.float32(2.0)
    x = np.array([1.0, 1.0, 0.5], dtype=np.float32)
    y = np.array([10.0, 10.0, 10.0], dtype=np.float32)

    print axpy(a, x, y)

    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([3.0, 1.0, 2.0], dtype=np.float32)
    
    print dot(x, y)
    print nrm2(x)

    # This stores the matrix
    # 1.0  1.0  1.0
    # 2.0  1.0  3.0
    # 1.0 -1.0  2.0
    # As a uniform nested sequence, stored in Column major order

    A_data = np.array([1.0, 2.0, 1.0, 1.0, 1.0, -1.0, 1.0, 3.0, 2.0], dtype=np.float32)
    A = CuUniform(A_data, extents=(3,3), ordering='ab')

    x = np.array([1.0, 8.0, 2.0], dtype=np.float32)
    print gemv(A, x)
