from copperhead import *
import numpy as np

# @cu
# def extrema_op(a, b):
#     a_min_idx, a_min_val, a_max_idx, a_max_val = a
#     b_min_idx, b_min_val, b_max_idx, b_max_val = b
#     if a_min_val < b_min_val:
#         if a_max_val > b_max_val:
#             return a
#         else:
#             return a_min_idx, a_min_val, b_max_idx, b_max_val
#     else:
#         if a_max_val > b_max_val:
#             return b_min_idx, b_min_val, a_max_idx, a_max_val
#         else:
#             return b

@cu
def argmin_op(a, b):
    a_idx, a_val = a
    b_idx, b_val = b
    if a_val < b_val:
        return a
    else:
        return b

@cu
def argmax_op(a, b):
    a_idx, a_val = a
    b_idx, b_val = b
    if a_val > b_val:
        return a
    else:
        return b

@cu
def argmin_id(x):
    return -1, max_bound(x)
@cu
def argmax_id(x):
    return -1, min_bound(x)

@cu
def extrema(x, min_id, max_id):
    min_idx, min_val = reduce(argmin_op, zip(indices(x), x), min_id)
    max_idx, max_val = reduce(argmax_op, zip(indices(x), x), max_id)
    return min_idx, min_val, max_idx, max_val

n = 1e7
a = np.array(np.random.ranf(n), dtype=np.float32)
b = cuarray(a)

min_id = argmin_id(np.float32(0))
max_id = argmax_id(np.float32(0))
with places.openmp:
    x = extrema(b, min_id, max_id)
print(x)
if __name__ == '__main__':
    from timeit import Timer
    def test():
        with places.openmp:
            x = extrema(b, min_id, max_id)
    t = Timer("test()", "from __main__ import test, b, min_id, max_id")
    iter = 1000
    time = t.timeit(number=iter)
    print("Time for extrema on array of %s elements: %s s" % (n, time/float(iter)))
    print("Net achieved bandwidth: %s GB/s" % (float(n) * float(4) * float(iter)/(time * 1.0e9)))
