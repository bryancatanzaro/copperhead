from copperhead import *

@cu
def demux(x):
    return int32(x), float32(x)


@cu
def test(x):
    return map(demux, x)

print test([1,2,3])
