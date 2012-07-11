from copperhead import *
import unittest

@cu
def closure_test(x):
    x = map(lambda xi: xi * 3, x)
    def stencil(i):
        if i == 0 or i == len(x)-1:
            return x[i]
        else:
            return (x[i-1] + x[i] + x[i+1])/3
    return map(stencil, indices(x))

class ClosureTest(unittest.TestCase):
    def testClosure(self):
        self.assertEqual(list(closure_test(
                np.array([0, .25, .5, .75, 1.0], dtype=np.float32))),
                         [0, 0.75, 1.5, 2.25, 3])

if __name__ == '__main__':
    unittest.main()
        
