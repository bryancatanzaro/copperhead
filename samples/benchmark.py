from copperhead import *
import numpy as np
import timeit

@cu
def ident(x):
    def ident_e(xi):
        return xi
    return map(ident_e, x)

iters = 1000
s = 10000000
t = np.float32
a = np.ndarray(shape=(s,), dtype=t)
b = cuarray(a)
p = runtime.places.gpu0

#Optional: Send data to execution place
b = force(b, p)


def test_ident():
    for x in xrange(iters):
        r = ident(b)
    # Materialize result
    # If you don't do this, you won't time the actual execution
    # But rather the asynchronous function calls
    force(r, p)

 
with p:
    time = timeit.timeit('test_ident()', setup='from __main__ import test_ident', number=1)

bandwidth = (2.0 * 4.0 * s * float(iters))/time/1.0e9
print('Sustained bandwidth: %s GB/s' % bandwidth)
