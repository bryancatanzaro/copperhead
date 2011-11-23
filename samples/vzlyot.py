from copperhead import *
import numpy as np

@cu
def xpy(x, y):
    return map(op_add, x, y)

a = CuArray(np.array([1,2,3], dtype=np.int32))
b = CuArray(np.array([3,2,1], dtype=np.int32))

c = xpy(a, b)
print(a)
print(b)
print(c)
