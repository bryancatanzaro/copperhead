from copperhead import *

@cu
def demux(x):
    return int32(x), (float32(x), float64(x))

@cu
def test(x):
    return map(demux, x)

a = test([1,2,3])
print repr(a)

@cu
def mux((x, (y, z))):
    return float32(x) + float32(y) + float32(z)

@cu
def test2(x):
    return map(mux, x)

b = test2(a)
print repr(b)
