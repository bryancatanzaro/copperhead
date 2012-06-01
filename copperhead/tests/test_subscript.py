import unittest
from copperhead import *
from recursive_equal import recursive_equal

@cu
def test_sub(a):
    def el(i):
        return a[len(a)-1-i]
    return map(el, indices(a))

@cu
def test_deref(a):
    return a[0]

@cu
def test_deref_zip(a):
    b = zip(a, a)
    return b[1]

class Subscripting(unittest.TestCase):
    def testSub(self):
        source = [1,2,3,4,5]
        result = test_sub(source)
        self.assertTrue(recursive_equal([5,4,3,2,1],result))
    def testDeref(self):
        source = [4,3,2,1]
        result = test_deref(source)
        self.assertTrue(4, result)
    def testDerefZip(self):
        source = [5,6,7]
        result = test_deref_zip(source)
        self.assertEqual((6,6), result)

if __name__ == '__main__':
    unittest.main()
