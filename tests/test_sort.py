from copperhead import *
import numpy as np

@cu
def lt_sort(x):
    return sort(cmp_lt, x)

print(lt_sort(CuArray(np.array([3,1,5], dtype=np.float32))))
