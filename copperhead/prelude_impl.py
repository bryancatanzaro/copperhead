#
#   Copyright 2012 NVIDIA Corporation
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

"""
Copperhead Prelude

This module provides Copperhead implementations for non-primitive prelude
functions.
These implementations are not provided in the prelude proper, because doing so
would preclude the use of Python builtins for certain functions, like range,
which are built in to Python, but are implemented as Copperhead functions.

The compiler brings these in during the compilation process, otherwise
they will not override the Python builtins.
"""

from decorators import cu

@cu
def range(n):
    return indices(replicate(0, n))

@cu
def gather(x, i):
    def el(ii):
        return x[ii]
    return map(el, i)

@cu
def update(dst, updates):
    indices, src = unzip(updates)
    return scatter(src, indices, dst)

@cu
def any(preds):
    return reduce(op_or, preds, False)

@cu
def all(preds):
    return reduce(op_and, preds, True)

@cu
def sum(x):
    return reduce(op_add, x, cast_to_el(0, x))

@cu
def min(x):
    def min_el(a, b):
        if a < b:
            return a
        else:
            return b
    return reduce(min_el, x, x[0])

@cu
def max(x):
    def max_el(a, b):
        if a > b:
            return a
        else:
            return b
    return reduce(max_el, x, x[0])

@cu
def shift(x, a, d):
    def shift_el(i):
        i = i + a
        if i < 0 or i >= len(x):
            return d
        else:
            return x[i]
    return map(shift_el, indices(x))

@cu
def rotate(x, a):
    def torus_index(i, a, b):
        return (i + a) % b
        
    def rotate_el(i):
        return x[torus_index(i, a, len(x))]

    return map(rotate_el, indices(x))

@cu
def bounded_range(a, b):
    length = b - a
    return [xi + a for xi in range(length)]

@cu
def enumerate(x):
    return zip(range(len(x)), x)
