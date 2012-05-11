from copperhead import *

@cu
def test_zip(x, y):
    return zip(x, y)

@cu
def test_unzip(x):
    y, z = unzip(x)
    return y, z

x = [1,2]
y = [3,4]


@cu
def demux(x):
    return [(xi, xi) for xi in x]

z = test_zip(x, y)
q = demux([1,2,3])
a, b = test_unzip(q)
