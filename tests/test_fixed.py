from copperhead import *

import unittest

@cu
def dismantle((x, (y, z))):
    return x

@cu
def tuple_inline_test(x):
    return dismantle((x,(x,x)))

class TupleInlineTest(unittest.TestCase):
    def testTupleInline(self):
        self.assertEqual(tuple_inline_test(2), 2)


@cu
def rebind_test(x):
    y = [xi + 1 for xi in x]
    z = y
    return z

class RebindTest(unittest.TestCase):
    def testRebind(self):
        self.assertEqual(list(rebind_test([0,1,2])),
                         [1,2,3])
        
if __name__ == '__main__':
    unittest.main()

