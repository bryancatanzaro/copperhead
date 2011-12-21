from copperhead import *
import numpy as np

@cu
def scalar_fn(x, y):
    return x + y

a = scalar_fn(np.float32(1.0), np.float32(2.0))
print a
