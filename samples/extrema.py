from copperhead import *
import numpy as np

@cu
def extrema_op(a, b):
    a_min_idx, a_min_val, a_max_idx, a_max_val = a
    b_min_idx, b_min_val, b_max_idx, b_max_val = b
    if a_min_val < b_min_val:
        if a_max_val > b_max_val:
            return a
        else:
            return a_min_idx, a_min_val, b_max_idx, b_max_val
    else:
        if a_max_val > b_max_val:
            return b_min_idx, b_min_val, a_max_idx, a_max_val
        else:
            return b

@cu
def extrema_id(x):
    return -1, max_bound(x), -1, min_bound(x)

@cu
def extrema(x, id):
    return reduce(extrema_op, zip(indices(x), x, indices(x), x), id)

n = 1e7
a = np.array(np.random.ranf(n), dtype=np.float32)
b = cuarray(a)
id = extrema_id(np.float32(0))


if __name__ == '__main__':
    from timeit import Timer
    def test():
        x = extrema(b, id)
    t = Timer("test()", "from __main__ import test, b, id")
    iter = 100
    time = t.timeit(number=iter)
    print("Time for extrema on array of %s elements: %s s" % (n, time/float(iter)))
    print("Net achieved bandwidth: %s GB/s" % (float(n) * float(4) * float(iter)/(time * 1.0e9)))
