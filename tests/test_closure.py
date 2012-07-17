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

@cu
def inline_closure_test(x, it):
    def inc():
        return map(lambda xi: xi + 1, x)
    if it > 0:
        incremented = inc()
        return inline_closure_test(incremented, it-1)
    else:
        return x

@cu
def cond_closure_test(a, x, it):
    def inc(xi):
        return a + xi
    if it > 0:
        incremented = map(inc, x)
        return cond_closure_test(a, incremented, it-1)
    else:
        return x

@cu
def nested_closure_test(a, x, it):
    def inc(xi):
        return a + xi
    def work():
        return map(inc, x)
    if it > 0:
        x = work()
        return nested_closure_test(a, x, it-1)
    else:
        return x
    
class ClosureTest(unittest.TestCase):
    def testClosure(self):
        self.assertEqual(list(closure_test(
                np.array([0, .25, .5, .75, 1.0], dtype=np.float32))),
                         [0, 0.75, 1.5, 2.25, 3])
    def testClosureInline(self):
        self.assertEqual(list(inline_closure_test(np.array([1,3,2], dtype=np.int32),
                                                  np.int32(2))),
                         [3,5,4])
    def testClosureCond(self):
        self.assertEqual(list(cond_closure_test(np.int32(2),
                                                np.array([5,5,5], dtype=np.int32),
                                                np.int32(2))),
                         [9,9,9])
    def testClosureNested(self):
        self.assertEqual(list(nested_closure_test(np.int32(2),
                                                  np.array([5,5,5], dtype=np.int32),
                                                  np.int32(2))),
                         [9,9,9])
if __name__ == '__main__':
    unittest.main()
