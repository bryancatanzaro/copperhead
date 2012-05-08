from copperhead import *

@cu
def demux(x):
    return int32(x), (float32(x)+1.1, float64(x)+2.78)

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

@cu
def test3(x):
    a = test(x)
    b = test2(a)
    return b

c = test3([1,2,3])
print repr(c)
