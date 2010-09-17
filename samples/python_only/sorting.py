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

from copperhead import *

@cu
def counting_sort(A):
    """
    Sort A using a counting sort.

    The algorithm consists of two steps.  First, it counts how many
    elements will sort to the left of each input A[i].  Second, it
    permutes the inputs according to these counts.

    For sequences of length n, this performs O(n*n) work.
    """

    def rank_index(i, A):
        'Count the number of elements sorting to the left of A[i].'
        x = A[i]
        before, after = split_at(A, i)

        lt = lambda x,y: x<y
        r1 = count([not lt(x,y) for y in before])
        r2 = count([lt(y,x)     for y in after])

        return r1 + r2

    ranks = [rank_index(i, A) for i in indices(A)]
    return permute(A, ranks)

@cutype("[Int] -> [Int]")
@cu
def shuffle(A):
    def delta(flag, ones_before, zeros_after):
        # We use arithmetic here to make up for lack of conditional
        # support in the phase analysis.
        return (flag*zeros_after) - (1-flag)*ones_before

    flags = [x&1 for x in A]
    ones  = scan(op_add, flags)
    zeros = rscan(op_add, [f^1 for f in flags])

    offsets = map(delta, flags, ones, zeros)
    return permute(A, map(op_add, indices(A), offsets))

@cutype("([Int], Int, Int) -> [Int]")
@cu
def radix_sort(A, bits, lsb):
    """
    Sort A using radix sort.
    
    Each element of A is assumed to be an integer.  The key used in
    sorting will be bits [lsb, lsb+bits).  For the general case, use
    bits=32 and lsb=0 to sort on all 32 bits.

    For sequences of length n with b-bit keys, this performs O(b*n) work.
    """

    def delta(flag, ones_before, zeros_after):
        if flag==0:  return -ones_before
        else:        return +zeros_after

    if lsb>=bits:
        return A
    else:
        flags = map(lambda x: (x>>lsb)&1, A)
        ones  = scan(op_add, flags)
        zeros = rscan(op_add, [f^1 for f in flags])

        offsets = map(delta, flags, ones, zeros)

        B = permute(A, map(op_add, indices(A), offsets))
        return radix_sort(B, bits, lsb+1)

def radix_sort8(A):   return radix_sort(A, 8, 0)
def radix_sort32(A):  return radix_sort(A, 32, 0)


@cutype("([Int], Int, Int) -> [Int]")
@cu
def quadradix_sort(A, bits, lsb):
    if lsb>=bits:
        return A
    else:
        mask = (1<<4) - 1
        keys, values = unzip(collect(lambda x: (x>>lsb)&mask, A))
        B = join(values)
        return quadradix_sort(B, bits, lsb+4)

def quadradix_sort32(A):  return quadradix_sort(A, 32, 0)


if __name__=="__main__":
    import sys
    from copperhead.runtime import intermediate

    def random_numbers(n, bits=8):
        import random
        return [int(random.getrandbits(bits)) for i in xrange(n)]

    def test_sort(S, n=277, trials=50, bits=8):
        npass, nfail = 0,0
        name = S.__name__

        for i in xrange(trials):
            data_in  = random_numbers(n, bits)
            gold     = sorted(data_in)
            data_out = S(data_in)

            if gold==data_out:
                npass = npass+1
            else:
                nfail = nfail+1

            print ("%-20s passed [%2d]\tfailed [%2d]\r" % (name, npass,nfail)),
        print

    print
    print "---- Checking Python implementations (n=277) ----"
    with places.here:
        test_sort(counting_sort,    n=277)
        test_sort(radix_sort8,      n=277, bits=8)
        test_sort(radix_sort32,     n=277, bits=32)
        test_sort(quadradix_sort32, n=277, bits=32)

    print
    print "---- Checking front-end results (n=17) ----"
    with places.frontend:
        test_sort(counting_sort,    n=17)
        test_sort(radix_sort8,      n=17, bits=8)
        test_sort(radix_sort32,     n=17, bits=32)
        test_sort(quadradix_sort32, n=17, bits=32)

    print
    print "---- Checking mid-end results (n=17) ----"
    with places.midend:
        test_sort(counting_sort,    n=17)
        test_sort(radix_sort8,      n=17, bits=8)
        test_sort(radix_sort32,     n=17, bits=32)
        test_sort(quadradix_sort32, n=17, bits=32)
