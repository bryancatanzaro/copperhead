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

"""
Computing Fibonacci numbers with scan.

The Fibonacci numbers F_i obey the recurrence equation

    [ F_n+1 ]  =  [1  1] [ F_n   ]
    [ F_n   ]     [1  0] [ F_n-1 ]

or equivalently
  
    [ F_n+1 ]  =  [1  1]^n [ F_1 ]
    [ F_n   ]     [1  0]   [ F_0 ]

This implies that we can compute the Fibonacci sequence using parallel
prefix (a.k.a. scan) operations.  We need only initialize a sequence of
2x2 matrices to A = replicate([[1 1] [1 0]], n) and then compute
scan(*,A) where '*' is the usual matrix multiplication operator.
"""

from copperhead import *

import copperhead.runtime.intermediate as I

@cu
def mul2x2((a0, a1, a2), (b0, b1, b2)):
    c0 = a0 * b0 + a1 * b1
    c1 = a0 * b1 + a1 * b2
    c2 = a1 * b1 + a2 * b2
    return (c0, c1, c2)

@cu
def offdiag((a0, a1, a2)):
    return a1

@cu
def fib_imp(a, b, c):
    F = scan(mul2x2, zip3(a, b, c))
    return map(offdiag, F)

def fib(n):
    'Return a sequence containing the first n Fibonacci numbers.'
    ones = np.ones(n, dtype=np.int32)
    zeros = np.zeros(n, dtype=np.int32)
    with I.tracing(action=I.print_and_pause):
        return fib_imp(ones, ones, zeros)

print fib(15)
