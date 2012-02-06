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

"""
Copperhead Prelude

This module provides native Python implementations for the set of
standard functions provided to all Copperhead programs.  When running on
a parallel device, operations like gather() or reduce() may have special
meaning.  They may, for instance, require the synchronization of various
parallel tasks.  Therefore, they are implemented in Copperhead via
architecture-specific primitive routines.  The Python implementations
here guarantee that Copperhead programs using these functions will run
correctly in the host Python interpreter as well.

Some of the functions listed here are Python built-ins, such as reduce()
and zip().  Unlike in Python, these functions are treated by Copperhead
as parallel primitives.  They are declared here so that they are visible
as Copperhead primitives.  Some, like zip(), may also have a restricted
interface in comparison to their Python counterpart.  The built-ins any,
all, sum, min, & max are treated as special cases of reduce.

Finally, the Python built-in map() is treated as a special syntactic
form by Copperhead.
"""
from __future__ import division
import __builtin__

from decorators import cutype
import copperhead.runtime.places as PL

import math
import numpy as np

# Unlike @functools.wraps, our @_wraps decorator only sets the docstring
# Thus reduce.__module__ will become 'copperhead.prelude' rather than
# '__builtin__'.  This makes it possible for the application to
# determine which reduce it's calling, in case it cares
def _wraps(wrapped):
    from functools import wraps
    return wraps(wrapped, assigned=['__doc__'])

@cutype("([a], [b]) -> [a]")
def gather(x, indices):
    """
    Return the sequence [x[i] for i in indices].

    >>> gather([8, 16, 32, 64, 128], [3, 0, 2])
    [64, 8, 32]

    >>> gather([8, 16, 32, 64, 128], [])
    []
    """
    return [x[i] for i in indices]


@cutype("([a], [b], [a]) -> [a]")
def scatter(src, indices, dst):
    """
    Create a copy of dst and update it by scattering each src[i] to
    location indices[i] of the copy.  Returns the final result.

        >>> scatter([11, 12], [3, 1], [1, 2, 3, 4])
        [1, 12, 3, 11]

    It is valid to pass empty src & indices lists to scatter, whose
    result will then be an unaltered copy of dst.

    If any indices are duplicated, one of the corresponding values
    from src will be chosen arbitrarily and placed in the result.  

        >>> scatter([], [], [1, 2, 3, 4])
        [1, 2, 3, 4]
    """
    assert len(src)==len(indices)

    result = list(dst)
    for i in xrange(len(src)):
        result[indices[i]] = src[i]
    return result

@cutype("([a], [b]) -> [a]")
def permute(x, indices):
    """
    Permute the sequence x by sending each value to the index specified
    in the corresponding array.

        >>> permute([1, 2, 3, 4], [3, 0, 1, 2])
        [2, 3, 4, 1]

    Permute requires that the lengths of its arguments match.  It will
    raise an AssertionError if they do not.

        >>> permute([1, 2, 3, 4], [3, 0, 1])
        Traceback (most recent call last):
          ...
        AssertionError

    If any indices are duplicated, one of the corresponding values
    from x will be chosen arbitrarily and placed in the result.
    """
    assert len(x)==len(indices)
    return scatter(x, indices, x)

@cutype("([a], [(b,a)]) -> [a]")
def update(dst, updates):
    """
    Compute an updated version of dst where each (i, x) pair in updates
    is used to replace the value of dst[i] with x.

        >>> update([True, False, True, False], [(1, True), (0, False)])
        [False, True, True, False]

    If the updates list is empty, dst is returned unmodified.

        >>> update(range(4), [])
        [0, 1, 2, 3]
    """
    indices, src = unzip(updates) if updates else ([],[])
    return scatter(src, indices, dst)

@cutype("(a->k, [a]) -> [(k, [a])]")
def collect(key_function, A):
    """
    Using the given function to assign keys to all elements of A, return
    a list of (key, [values]) pairs such that all elements with
    equivalent keys are gathered together in the same list.

        >>> collect(lambda x:x, [1, 1, 2, 3, 1, 3, 2, 1])
        [(1, [1, 1, 1, 1]), (2, [2, 2]), (3, [3, 3])]

    The returned pairs will be ordered by increasing key values.  The
    individual values will occur in the order in which they occur in the
    original sequence.

        >>> collect(lambda x: x<0, [1, -1, 4, 3, -5])
        [(False, [1, 4, 3]), (True, [-1, -5])]
    """
    from itertools import groupby
    B = list()

    for key,values in groupby(sorted(A, key=key_function), key_function):
        B.append((key,list(values)))

    return B

@cutype("((a,a)->a, [a], [b], [a]) -> [a]")
def scatter_reduce(fn, src, indices, dst):
    """
    Alternate version of scatter that combines -- rather than replaces
    -- values in dst with values from src.  The binary function fn is
    used to combine values, and is required to be both associative and
    commutative.
    
    If multiple values in src are sent to the same location in dst,
    those values will be combined together as in reduce.  The order in
    which values are combined is undefined.

        >>> scatter_reduce(op_add, [1,1,1], [1,2,3], [0,0,0,0,0])
        [0, 1, 1, 1, 0]

        >>> scatter_reduce(op_add, [1,1,1], [3,3,3], [0,0,0,0,0])
        [0, 0, 0, 3, 0]
    """
    assert len(src)==len(indices)

    result = list(dst)
    for i in xrange(len(src)):
        j = indices[i]
        result[j] = fn(result[j], src[i])
    return result

@cutype("([a], [b], [a]) -> [a]")
def scatter_sum(src, indices, dst):
    """
    Specialization of scatter_reduce for addition (cf. reduce and sum).
    """
    return scatter_reduce(op_add, src, indices, dst)

@cutype("([a], [b], [a]) -> [a]")
def scatter_min(src, indices, dst):
    """
    Specialization of scatter_reduce with the min operator (cf. reduce and min).
    """
    return scatter_reduce(op_min, src, indices, dst)

@cutype("([a], [b], [a]) -> [a]")
def scatter_max(src, indices, dst):
    """
    Specialization of scatter_reduce with the max operator (cf. reduce and max).
    """
    return scatter_reduce(op_max, src, indices, dst)

@cutype("([Bool], [b], [Bool]) -> [Bool]")
def scatter_any(src, indices, dst):
    """
    Specialization of scatter_reduce for logical or (cf. reduce and any).
    """
    return scatter_reduce(op_or, src, indices, dst)

@cutype("([Bool], [b], [Bool]) -> [Bool]")
def scatter_all(src, indices, dst):
    """
    Specialization of scatter_reduce for logical and (cf. reduce and all).
    """
    return scatter_reduce(op_and, src, indices, dst)

@cutype("((a,a)->a, [a]) -> [a]")
def scan(f, A):
    """
    Return the inclusive prefix scan of f over A.
    
    >>> scan(lambda x,y: x+y, [1,1,1,1,1])
    [1, 2, 3, 4, 5]

    >>> scan(lambda x,y: x, [4, 3, 1, 2, 0])
    [4, 4, 4, 4, 4]

    >>> scan(lambda x,y: x+y, [])
    []
    """
    B = list(A)

    for i in xrange(1, len(B)):
        B[i] = f(B[i-1], B[i])

    return B

@cutype("((a,a)->a, [a]) -> [a]")
def rscan(f, A):
    """
    Reverse (i.e., right-to-left) scan of f over A.

    >>> rscan(lambda x,y: x+y, [1,1,1,1,1])
    [5, 4, 3, 2, 1]

    >>> rscan(lambda x,y: x, [3, 1, 4, 1, 5])
    [5, 5, 5, 5, 5]

    >>> rscan(lambda x,y: x+y, [])
    []
    """
    return list(reversed(scan(f, reversed(A))))

@cutype("((a,a)->a, a, [a]) -> [a]")
def exclusive_scan(f, prefix, A):
    """
    Exclusive prefix scan of f over A.

    >>> exclusive_scan(lambda x,y: x+y, 0, [1, 1, 1, 1, 1])
    [0, 1, 2, 3, 4]
    """
    return scan(f, [prefix] + A[:-1])

@cutype("((a,a)->a, a, [a]) -> [a]")
def exclusive_rscan(f, suffix, A):
    """
    Reverse exclusive prefix scan of f over A.

    >>> exclusive_rscan(lambda x,y: x+y, 0, [1, 1, 1, 1, 1])
    [4, 3, 2, 1, 0]
    """
    return rscan(f, A[1:]+[suffix])



@cutype("[a] -> [Long]")
def indices(A):
    """
    Return a sequence containing all the indices for elements in A.

    >>> indices([6, 3, 2, 9, 10])
    [0, 1, 2, 3, 4]
    """
    return range(len(A))

@cutype("(a, b) -> [a]")
def replicate(x, n):
    """
    Return a sequence containing n copies of x.

        >>> replicate(True, 3)
        [True, True, True]

    If n=0, this will return the empty list.

        >>> replicate(101, 0)
        []
    """
    return [x]*n

@cutype("[[a]] -> [a]")
def join(lists):
    """
    Return a list which is the concatenation of all elements of input list.

    >>> join([[1,2], [3,4,5], [6,7]])
    [1, 2, 3, 4, 5, 6, 7]
    """
    from operator import concat
    return __builtin__.reduce(concat, lists)


@cutype("[(a,b)] -> ([a], [b])")
def unzip(seq):
    """
    Inverse of zip.  Converts a list of tuples into a tuple of lists.

    >>> unzip([(1,2), (3,4), (5,6)])
    ([1, 3, 5], [2, 4, 6])
    """
    return tuple(map(list, __builtin__.zip(*seq)))

@cutype("[a] -> [a]")
def odds(A):
    """
    Return list of all elements of A at odd-numbered indices.

        >>> odds([1, 2, 3, 4, 5])
        [2, 4]

        >>> odds([1])
        []
    """
    return A[1::2]

@cutype("[a] -> [a]")
def evens(A):
    """
    Return list of all elements of A at even-numbered indices.

        >>> evens([1, 2, 3, 4, 5])
        [1, 3, 5]

        >>> evens([1])
        [1]
    """
    return A[0::2]

@cutype("([a], [a]) -> [a]")
def interleave2(A, B):
    """
    Interleave the given lists element-wise, starting with A.

        >>> interleave2([1,2,3], [4])
        [1, 4, 2, 3]
    """
    return [x for items in map(None, A, B) for x in items if x is not None]

@cutype("([a], Int) -> [[a]]")
def split(A, tilesize):
    """
    Split the sequence A into a sequence of sub-sequences.  Every
    sub-sequence will contain tilesize elements, except for the last
    sub-sequence which may contain fewer.

        >>> split(range(8), 3)
        [[0, 1, 2], [3, 4, 5], [6, 7]]

        >>> split([1,2,3,4], 1)
        [[1], [2], [3], [4]]

    If the tilesize is larger than the size of A, only one sub-sequence
    will be returned.

        >>> split([1,2], 3)
        [[1, 2]]
    """
    tile = A[:tilesize]
    if len(A) > tilesize:
        return [tile] + split(A[tilesize:], tilesize)
    else:
        return [tile]

@cutype("([a], Int) -> [[a]]")
def splitr(A, tilesize):
    """
    Split the sequence A into a sequence of sub-sequences.  Every
    sub-sequence will contain tilesize elements, except for the first
    sub-sequence which may contain fewer.

        >>> splitr(range(8), 3)
        [[0, 1], [2, 3, 4], [5, 6, 7]]

        >>> splitr([1,2,3,4], 1)
        [[1], [2], [3], [4]]

    If the tilesize is larger than the size of A, only one sub-sequence
    will be returned.

        >>> splitr([1,2], 3)
        [[1, 2]]
    """
    tile = A[-tilesize:]
    if len(A) > tilesize:
        return splitr(A[:-tilesize], tilesize) + [tile]
    else:
        return [tile]

@cutype("([a], Int) -> ([a], [a])")
def split_at(A, k):
    """
    Return pair of sequences containing the k elements and the rest
    of A, respectively.

        >>> split_at([0,1,2,3,4,5,6,7], 3)
        ([0, 1, 2], [3, 4, 5, 6, 7])

    It is acceptable to specify values of k=0 or k=len(A).  In both
    cases, one of the returned sequences will be empty.

        >>> split_at(range(3), 0)
        ([], [0, 1, 2])

        >>> split_at(range(3), 3)
        ([0, 1, 2], [])
    """
    return A[:k], A[k:]

@cutype("([a], Int) -> [[a]]")
def split_cyclic(A, k):
    """
    Splits the sequence A into k subsequences.  Elements of A are
    distributed into subsequences in cyclic round-robin fashion.  Every
    subsequence will contain ceil(A/k) elements, except for the last
    which may contain fewer.

        >>> split_cyclic(range(10), 3)
        [[0, 3, 6, 9], [1, 4, 7], [2, 5, 8]]

    If there are fewer than k elements in A, the last n-k subsequences
    will be empty.

        >>> split_cyclic([1, 2], 4)
        [[1], [2], [], []]
    """
    return [A[i::k] for i in range(k)]

@cutype("[[a]] -> [a]")
def interleave(A):
    """
    The inverse of split_cyclic, this takes a collection of
    subsequences and interleaves them to form a single sequence.

        >>> interleave([[0, 3, 6, 9], [1, 4, 7], [2, 5, 8]])
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        >>> interleave([[1], [2], [], []])
        [1, 2]

    The input sequence may contain only empty sequences, but may not
    itself be empty.

        >>> interleave([[],[]])
        []

        >>> interleave([])
        Traceback (most recent call last):
          ...
        AssertionError
    """
    assert len(A)>0
    return [x for items in map(None, *A) for x in items if x is not None]

@cutype("([a], Int) -> [a]")
def take(a,i):
    'Return sequence containing first i elements of a'
    return a[:i]

@cutype("([a], Int) -> [a]")
def drop(a,i):
    'Return sequence containing all but the first i elements of a'
    return a[i:]

@cutype("[a] -> a")
def first(A):
    'Return the first element of the sequence A.  Equivalent to A[0].'
    return A[0]

@cutype("[a] -> a")
def second(A):
    'Return the second element of A.  Equivalent to A[1].'
    return A[1]

@cutype("[a] -> a")
def last(A):
    'Return the last element of A.  Equivalent to A[-1].'
    return A[-1]

@cutype("[Bool] -> Int")
def count(preds):
    'Count the number of True values in preds'

    # Python treats True like 1, but Copperhead does not
    return sum(preds)




@cutype("[(Bool, a)] -> [a]")
def pack(A):
    """
    Given a sequence of (flag,value) pairs, pack will produce a sequence
    containing only those values whose flag was True.  The relative
    order of values in the input is preserved in the output.

        >>> pack(zip([False, True, True, False], range(4)))
        [1, 2]
    """
    def _gen(A):
        for flag, value in A:
            if flag:
                yield value

    return list(_gen(A))



@cutype("([a], b, a) -> [a]")
def shift(src, offset, default):
    """
    Returns a sequence which is a shifted version of src.
    It is shifted by offset elements, and default will be
    shifted in to fill the empty spaces.
    """
    u, v = split_at(src, offset)
    if offset < 0:
        return join([replicate(default, -offset), u])
    else:
        return join([v, replicate(default, offset)])

@cutype("([a], b) -> [a]")
def rotate(src, offset):
    """
    Returns a sequence which is a rotated version of src.
    It is rotated by offset elements.
    """
    u, v = split_at(src, offset)
    return join([v, u])
    

@cutype("((a, a)->Bool, [a]) -> [a]")
def sort(fn, x):
    def my_cmp(xi, xj):
        if fn(xi, xj):
            return -1
        else:
            return 0
    return sorted(x, cmp=my_cmp)


@cutype("((a0)->b, [a0])->[b]")
def map1(f, a0):
    return map(f, a0)

@cutype("((a0,a1)->b, [a0], [a1])->[b]")
def map2(f, a0, a1):
    return map(f, a0, a1)

@cutype("((a0,a1,a2)->b, [a0], [a1], [a2])->[b]")
def map3(f, a0, a1, a2):
    return map(f, a0, a1, a2)

@cutype("((a0,a1,a2,a3)->b, [a0], [a1], [a2], [a3])->[b]")
def map4(f, a0, a1, a2, a3):
    return map(f, a0, a1, a2, a3)

@cutype("((a0,a1,a2,a3,a4)->b, [a0], [a1], [a2], [a3], [a4])->[b]")
def map5(f, a0, a1, a2, a3, a4):
    return map(f, a0, a1, a2, a3, a4)

@cutype("((a0,a1,a2,a3,a4,a5)->b, [a0], [a1], [a2], [a3], [a4], [a5])->[b]")
def map6(f, a0, a1, a2, a3, a4, a5):
    return map(f, a0, a1, a2, a3, a4, a5)

@cutype("((a0,a1,a2,a3,a4,a5,a6)->b, [a0], [a1], [a2], [a3], [a4], [a5], [a6])->[b]")
def map7(f, a0, a1, a2, a3, a4, a5, a6):
    return map(f, a0, a1, a2, a3, a4, a5, a6)

@cutype("((a0,a1,a2,a3,a4,a5,a6,a7)->b, [a0], [a1], [a2], [a3], [a4], [a5], [a6], [a7])->[b]")
def map8(f, a0, a1, a2, a3, a4, a5, a6, a7):
    return map(f, a0, a1, a2, a3, a4, a5, a6, a7)

@cutype("((a0,a1,a2,a3,a4,a5,a6,a7,a8)->b, [a0], [a1], [a2], [a3], [a4], [a5], [a6], [a7], [a8])->[b]")
def map8(f, a0, a1, a2, a3, a4, a5, a6, a7, a8):
    return map(f, a0, a1, a2, a3, a4, a5, a6, a7, a8)

@cutype("((a0,a1,a2,a3,a4,a5,a6,a7,a8,a9)->b, [a0], [a1], [a2], [a3], [a4], [a5], [a6], [a7], [a8], [a9])->[b]")
def map9(f, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9):
    return map(f, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9)

@cutype("a -> a")
@_wraps(math.exp)
def exp(x):
    return math.exp(x)

@cutype("() -> Float")
def inf():
    'Returns the single precision floating point value representing infinity.'
    return float('inf')

########################################################################
#
# Operators
#
# Reflect various unary/binary function names that are equivalent to
# infix operators like + and ==.
#

import operator as _op

@cutype("(a,a) -> a")
@_wraps(_op.add)
def op_add(x,y): return _op.add(x,y)

@cutype("(a,a) -> a")
@_wraps(_op.sub)
def op_sub(x,y): return _op.sub(x,y)

@cutype("(a,a) -> a")
@_wraps(_op.mul)
def op_mul(x,y): return _op.mul(x,y)

@cutype("(a,a) -> a")
@_wraps(_op.div)
def op_div(x,y): return _op.div(x,y)

@cutype("(a,a) -> a")
@_wraps(_op.mod)
def op_mod(x,y): return _op.mod(x,y)

@cutype("(a,a) -> a")
@_wraps(_op.pow)
def op_pow(x,y): return _op.pow(x,y)

@cutype("(a,a) -> a")
@_wraps(_op.lshift)
def op_lshift(x,y): return _op.lshift(x,y)

@cutype("(a,a) -> a")
@_wraps(_op.rshift)
def op_rshift(x,y): return _op.rshift(x,y)

@cutype("(a,a) -> a")
@_wraps(_op.or_)
def op_or(x,y): return _op.or_(x,y)

@cutype("(a,a) -> a")
@_wraps(_op.xor)
def op_xor(x,y): return _op.xor(x,y)

@cutype("(a,a) -> a")
@_wraps(_op.and_)
def op_and(x,y): return _op.and_(x,y)

@cutype("a -> a")
@_wraps(_op.invert)
def op_invert(x): return _op.invert(x)

@cutype("a -> a")
@_wraps(_op.pos)
def op_pos(x): return _op.pos(x)

@cutype("a -> a")
@_wraps(_op.neg)
def op_neg(x): return _op.neg(x)

@cutype("Bool -> Bool")
@_wraps(_op.not_)
def op_not(x): return _op.not_(x)

@cutype("(a, a) -> Bool")
@_wraps(_op.eq)
def cmp_eq(x,y): return _op.eq(x,y)

@cutype("(a, a) -> Bool")
@_wraps(_op.ne)
def cmp_ne(x,y): return _op.ne(x,y)

@cutype("(a, a) -> Bool")
@_wraps(_op.lt)
def cmp_lt(x,y): return _op.lt(x,y)

@cutype("(a, a) -> Bool")
@_wraps(_op.le)
def cmp_le(x,y): return _op.le(x,y)

@cutype("(a, a) -> Bool")
@_wraps(_op.gt)
def cmp_gt(x,y): return _op.gt(x,y)

@cutype("(a, a) -> Bool")
@_wraps(_op.ge)
def cmp_ge(x,y): return _op.ge(x,y)

########################################################################
#
# Python built-ins
#
# Reflect built-in Python functions that have special meaning to
# Copperhead.  These wrapper functions allow us to (a) annotate them
# with type attributes and (b) restrict arguments if necessary.
#

@cutype("( (a,a)->a, [a], a ) -> a")
def reduce(fn, x, init):
    """
    Repeatedly applies the given binary function to the elements of the
    sequence.  Using the infix notation <fn>, reduction computes the
    value: init <fn> x[0] <fn> ... <fn> x[len(x)-1].
    
    The given function is required to be both associative and
    commutative.

        >>> reduce(op_add, [1, 2, 3, 4, 5], 0)
        15

        >>> reduce(op_add, [1, 2, 3, 4, 5], 10)
        25

        >>> reduce(op_add, [], 10)
        10

    Unlike the Python built-in reduce, the Copperhead reduce function
    makes the initial value mandatory.
    """
    return __builtin__.reduce(fn, x, init)

@cutype("[Bool] -> Bool")
def any(sequence):
    """
    Returns True if any element of sequence is True.  It is equivalent
    to calling reduce(op_or, sequence, False).

        >>> any([True, False, False])
        True

        >>> any([])
        False
    """
    return __builtin__.any(sequence)

@cutype("[Bool] -> Bool")
def all(sequence):
    """
    Returns True if all elements of sequence are True.  It is equivalent
    to calling reduce(op_and, sequence, True).

        >>> all([True, False, False])
        False

        >>> all([])
        True
    """
    return __builtin__.all(sequence)

@cutype("[a] -> a")
def sum(sequence):
    """
    This is equivalent to calling reduce(op_add, sequence, 0).

        >>> sum([1, 2, 3, 4, 5])
        15

        >>> sum([])
        0
    """
    return __builtin__.sum(sequence)

@cutype("[a] -> a")
def min(sequence):
    """
    Returns the minimum value in sequence, which must be non-empty.

        >>> min([3, 1, 4, 1, 5, 9])
        1

        >>> min([])
        Traceback (most recent call last):
          ...
        ValueError: min() arg is an empty sequence
    """
    return __builtin__.min(sequence)

@cutype("[a] -> a")
def max(sequence):
    """
    Returns the maximum value in sequence, which must be non-empty.

        >>> max([3, 1, 4, 1, 5, 9])
        9

        >>> max([])
        Traceback (most recent call last):
          ...
        ValueError: max() arg is an empty sequence
    """
    return __builtin__.max(sequence)

@cutype("[a] -> Int")
def len(sequence):  return __builtin__.len(sequence)

@cutype("Int -> [Int]")
def range(n):
    """
    Returns the sequence of integers from 0 to n-1.

        >>> range(5)
        [0, 1, 2, 3, 4]

        >>> range(0)
        []
    """
    return __builtin__.range(n)

@cutype("([a], [b]) -> [(a,b)]")
def zip(*args):
    """
    Combines corresponding pairs of elements from seq1 and seq2 into a
    sequence of pairs.

        >>> zip([1, 2, 3], [4, 5, 6])
        [(1, 4), (2, 5), (3, 6)]

    Zipping empty sequences will produce the empty sequence.

        >>> zip([], [])
        []

    The given sequences must be of the same length.

        >>> zip([1, 2], [3])
        Traceback (most recent call last):
          ...
        AssertionError
    """
    # XXX We need to figure out how to make this fit Python better
    #assert len(seq1)==len(seq2)
    #return __builtin__.zip(seq1, seq2)
    return __builtin__.zip(*args)

@cutype("([a], [b], [c]) -> [(a,b,c)]")
def zip3(seq1, seq2, seq3):
    """
    Combines corresponding pairs of elements from the given sequences
    into a sequence of 3-tuples.

        >>> zip3([1, 2], [3, 4], [5, 6])
        [(1, 3, 5), (2, 4, 6)]

    Zipping empty sequences will produce the empty sequence.

        >>> zip3([], [], [])
        []

    The given sequences must be of the same length.

        >>> zip3([1, 2], [3], [4])
        Traceback (most recent call last):
          ...
        AssertionError
    """
    assert len(seq1)==len(seq2)
    assert len(seq1)==len(seq3)
    return __builtin__.zip(seq1, seq2, seq3)

@cutype("([a], [b], [c], [d]) -> [(a,b,c,d)]")
def zip4(seq1, seq2, seq3, seq4):
    """
    Combines corresponding pairs of elements from the given sequences
    into a sequence of 3-tuples.

        >>> zip4([1, 2], [3, 4], [5, 6], [7, 8])
        [(1, 3, 5, 7), (2, 4, 6, 8)]

    Zipping empty sequences will produce the empty sequence.

        >>> zip4([], [], [], [])
        []

    The given sequences must be of the same length.

        >>> zip4([1, 2], [3], [4], [])
        Traceback (most recent call last):
          ...
        AssertionError
    """
    assert len(seq1)==len(seq2)
    assert len(seq1)==len(seq3)
    assert len(seq1)==len(seq4)
    return __builtin__.zip(seq1, seq2, seq3, seq4)

@cutype("(a->Bool, [a]) -> [a]")
def filter(function, sequence):
    """
    Return a sequence containing those items of sequence for which
    function(item) is True.  The order of items in sequence is
    preserved.

        >>> filter(lambda x: x<3, [3, 1, 5, 0, 2, 4])
        [1, 0, 2]
    """
    return __builtin__.filter(function, sequence)

@cutype("[a] -> [a]")
def reversed(sequence):
    """
    Return a sequence containing the elements of the input in reverse
    order.

        >>> reversed([3, 0, 1, 2])
        [2, 1, 0, 3]
    """
    return list(__builtin__.reversed(sequence))

########################################################################
#
# Type casting functions
#

# Monomorphic casts
@cutype("a -> Int")
def int32(x):
    return np.int32(x)

@cutype("a -> Long")
def int64(x):
    return np.int64(x)

@cutype("a -> Float")
def float32(x):
    return np.float32(x)

@cutype("a -> Double")
def float64(x):
    return np.float64(x)

# Polymorphic casts
@cutype("(a, b) -> b")
def cast_to(x, y):
    return x

@cutype("(a, [b]) -> b")
def cast_to_el(x, y):
    return x


########################################################################
#
# Math functions
#


@cutype("a -> a")
def sqrt(x):
    return np.sqrt(x)

@cutype("a -> a")
def abs(x):
    return np.abs(x)

@cutype("a -> a")
def exp(x):
    return np.exp(x)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
