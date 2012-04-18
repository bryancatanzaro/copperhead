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

a = (1, 2, 3, 4)
b = (0, 1, 2, 3)

print(extrema_op(a,b))


@cu
def extrema_id(x):
    return -1, min_bound(x), -1, max_bound(x)



print(extrema_id(np.float32(1)))
