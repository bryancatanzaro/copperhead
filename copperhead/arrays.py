#
#  Copyright 2008-2010 NVIDIA Corporation
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
Copperhead Array Support

This module is a companion to the Copperhead Prelude.  It provides
native Python implementations for the various operations on arrays
defined in Copperhead.

NOTE: This module represents only an inital sketch for Copperhead array
support.  Its design may yet change in significant ways.  Furthermore,
the Copperhead back-end does not yet provide any support for compilation
of code using arrays.
"""

from decorators import cutype

import numpy as _numpy

# XXX It seems weird for the following to be totally disjoint from
#     the CuArrays defined in the runtime.  Should fix that disconnect
#     as things solidify.

class AnyArray(object):
    'Shared base class for all Copperhead array objects.'

    def force(self):
        return array(self.shape(), list(self.values()))

    def __str__(self):   return self.force().__str__()
    def __repr__(self):  return self.force().__repr__()


class ArrayObject(AnyArray):
    """
    The primary container class for Copperhead arrays.

    ArrayObjects largely just encapsulate NumPy arrays in a
    manner that complies with the Copperhead array interface.
    """

    def __init__(self, shape, values):
        if not isinstance(values, _numpy.ndarray):
            values = _numpy.array(values) 

        self.storage = _numpy.reshape(values, shape)

    def shape(self):  return self.storage.shape

    def values(self):
        #return _numpy.ravel(self.storage)
        for i, val in _numpy.ndenumerate(self.storage):
            yield val

    def indices(self):
        #return _numpy.ndindex(self.storage.shape)
        for i, val in _numpy.ndenumerate(self.storage):
            yield i

    def force(self): return self.storage

    def __getitem__(self, i):
        return self.storage.__getitem__(i)

    def __str__(self):  return self.storage.__str__();
    def __repr__(self): return self.storage.__repr__();


class ArrayConst(AnyArray):
    """
    Class for representing arrays all of whose values are the same.
    """

    def __init__(self, shape, value):
        self.stuple = shape
        self.value = value

    def shape(self): return self.stuple

    def indices(self):
        return _numpy.ndindex(self.shape())

    def values(self):
        for i in self.indices():
            yield self.value

    def __getitem__(self, i):  return self.value

class ArrayFunc(AnyArray):
    """
    Represents arrays whose values are defined implicitly by an
    associated function that maps indices to values.
    """

    def __init__(self, shape, fn):
        self.stuple = shape
        self.function = fn

    def shape(self): return self.stuple

    def indices(self):
        return _numpy.ndindex(self.shape())

    def values(self):
        for i in self.indices():
            yield self.function(i)

    def __getitem__(self, i):
        return self.function(i)

def _fromnumpy(A):
    return ArrayObject(A.shape, A)

@cutype("(i, [v]) -> Array(i,v)")
def array(shape, values):
    'Create an array from a linear sequence of values.'
    return ArrayObject(shape, values)

@cutype("(i, i->v) -> Array(i,v)")
def arrayfrom(shape, f):
    'Create an array whose value at index i is defined by f(i).'
    return ArrayFunc(shape, f)

@cutype("(i, v) -> Array(i,v)")
def arrayof(shape, x):
    'Create an array whose every index returns the value x.'
    return ArrayConst(shape, x)

@cutype("Array(i,v) -> i")
def shape(A):
    'Returns a tuple representing the shape of the array A.'
    return A.shape()

@cutype("Array(i,v) -> Int")
def dim(A):
    'Returns the dimensionality, sometimes known as the rank, of A.'
    return len(shape(A))

@cutype("(Array(i,v), i) -> v")
def get(A, i):
    'Return the value A[i].'
    return A.__getitem__(i)

@cutype("Array(i,v) -> [i]")
def indices(A):
    'Returns a sequence of all indices represented by A.'
    return list(A.indices())

@cutype("Array(i,v) -> [v]")
def values(A):
    'Returns a linear sequence of the values of A.'
    return list(A.values())

# XXX The maparray() function has no valid type in the Copperhead type
#     system.  Either we need to add maparray() as special syntax ala
#     map(), or we need to provide a somewhat different interface.
def _maparray(f, A, *rest):
    def each(i):
        return f(A[i], *[B[i] for B in rest])

    return array(shape(A), map(each, A.indices()))


if __name__ == "__main__":
    import doctest
    doctest.testmod()
