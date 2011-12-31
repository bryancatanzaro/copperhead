from copperhead import *
import numpy as np

@cu
def test_rotate(x, amount):
    return rotate(x, amount)

a = np.array([5,4,3,2,1], dtype=np.float32)
b = test_rotate(a, np.int32(-2))
print(b)
c = test_rotate(a, np.int32(2))
print(c)
