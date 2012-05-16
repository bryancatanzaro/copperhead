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
    return -1, max_bound(x), 1, min_bound(x)

@cu
def extrema(x, x_id):
    return reduce(extrema_op, zip(indices(x), x, indices(x), x), x_id)

n = 1e7
a = np.array(np.random.ranf(n), dtype=np.float32)
b = cuarray(a)

b_id = extrema_id(np.float32(0))
x = extrema(b, b_id)
print(x)
