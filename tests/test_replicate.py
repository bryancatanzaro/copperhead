from copperhead import *
import numpy as np

@cu
def test_repl(x, n):
    return replicate(x, n)

a = test_repl(np.float32(2.78), np.int32(100))
print(a)
