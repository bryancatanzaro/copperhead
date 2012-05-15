#
#   Copyright 2008-2009 NVIDIA Corporation
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

"""
Computing Fibonacci numbers with scan.

The Fibonacci numbers F_i obey the recurrence equation

    [ F_n+1 ]  =  [1  1] [ F_n   ]
    [ F_n   ]     [1  0] [ F_n-1 ]

or equivalently

    [ F_n+1 ]  =  [1  1]^n [ F_1 ]
    [ F_n   ]     [1  0]   [ F_0 ]

This implies that we can compute the Fibonacci sequence using parallel
prefix (a.k.a. scan) operations.  We need only initialize a sequence
of 2x2 matrices to A = replicate([[1 1] [1 0]], n) and then compute
scan(*,A) where '*' is the usual matrix multiplication operator.
"""

from copperhead import *

@cu
def vdot2(x, y):
    'Compute dot product of two 2-vectors.'
    x0, x1 = x
    y0, y1 = y
    return x0*y0 + x1*y1

@cu
def rows(A):
    'Return the 2 rows of the symmetric matrix A.'
    a0, a1, a2 = A
    return ((a0,a1), (a1,a2))

#@cutype("((a,a,a),) -> a")
@cu
def offdiag(A):
    'Return the off-diagonal element of the symmetric matrix A.'
    a0, a1, a2 = A
    return a1

#@cutype("( (a,a,a), (a,a,a) ) -> (a,a,a)")
@cu
def mul2x2(A, B):
    'Multiply two symmetric 2x2 matrices A and B represented as 3-tuples.'

    # rows of A
    a0, a1 = rows(A)

    # columns of B (which are its rows due to symmetry)
    b_0, b_1 = rows(B)

    return (vdot2(a0, b_0),
            vdot2(a0, b_1),
            vdot2(a1, b_1))

@cu
def fib(n):
    'Return a sequence containing the first n Fibonacci numbers.'

    # A is the symmetric matrix [[1 1], [1 0]] stored in a compressed
    # 3-tuple form.
    A = (1, 1, 0)

    # Calculate Fibonacci numbers by scan over the sequence [A]*n.
    F = scan(mul2x2, replicate(A, n))
    return map(offdiag, F)

print fib(92)
