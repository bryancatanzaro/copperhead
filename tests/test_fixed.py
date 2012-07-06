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

if __name__ == '__main__':
    unittest.main()

