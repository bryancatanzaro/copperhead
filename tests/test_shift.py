from copperhead import *
import numpy as np

@cu
def test_shift(x, amount, boundary):
    return shift(x, amount, boundary)

a = np.array([5,4,3,2,1], dtype=np.float32)
b = test_shift(a, np.int32(-2), np.float32(2.78))
print(b)
c = test_shift(a, np.int32(2), np.float32(3.14))
print(c)
