import unittest
from copperhead import *
from recursive_equal import recursive_equal

class TupleData(unittest.TestCase):
    def testTypeTupleScalars(self):
        source = (1, 2.0)
        result_type, result_value = runtime.driver.induct(source)
        self.assertEqual(repr(result_type), "Tuple(Long, Double)")
    def testTypeTupleSequences(self):
        source = ([1, 2], [3.0, 4.0])
        result_type, result_value = runtime.driver.induct(source)
        self.assertEqual(repr(result_type), "Tuple(Seq(Long), Seq(Double))")
    def testTypeNestedTuple(self):
        source = (1, (2, 3.0, (4.0, 5)), 6.0)
        result_type, result_value = runtime.driver.induct(source)
        self.assertEqual(repr(result_type), "Tuple(Long, Tuple(Long, Double, Tuple(Double, Long)), Double)")

@cu
def test_tuple((m, n), b):
    """Test tuple assembly/disassembly.
    Tuples disassembled in arguments!
    Tuples disassembled in statements!
    Tuples assigned to other tuples!
    Tuples assigned to identifiers!
    Tuples returned!"""

    #tuple = tuple bind
    q, r = m, n
    #tuple pack
    s = q, r
    #tuple unpack
    t, u = s
    o, p = b
    #return tuple
    return t + o, u + p

@cu
def test_tuple_return():
    """Test returning a tuple by identifier"""
    a = 1, 2
    return a

@cu
def test_nested_tuple_return():
    return (1, (2, 3))

@cu
def test_tuple_seq(x, y):
    return x, y

@cu
def test_containerize(x):
    def sub(xi):
        return -xi
    y = map(sub, x)
    z = x, y
    return z

@cu
def test_tuple_seq_args(x):
    y, z = x
    return y, z


class TupleExtract(unittest.TestCase):
    def testTuple(self):
        source_a = (1, 2)
        source_b = (5, 6)
        golden = (6, 8)
        self.assertEqual(test_tuple(source_a, source_b), golden)
    def testNestedTupleReturn(self):
        self.assertTrue(
            recursive_equal(test_nested_tuple_return(), (1, (2, 3))))
    def testTupleReturn(self):
        self.assertEqual(test_tuple_return(), (1, 2))
    def testTupleSeqSeq(self):
        self.assertTrue(recursive_equal(test_tuple_seq([1,2], [3,4]), ([1,2],[3,4])))
    def testTupleSeqScalar(self):
        self.assertTrue(recursive_equal(test_tuple_seq([1,2], 3), ([1,2],3)))
    def testTupleSeqTuple(self):
        self.assertTrue(recursive_equal(test_tuple_seq([1,2], (3,4)), ([1,2],(3,4))))
    def testTupleSeqArgs(self):
        self.assertTrue(recursive_equal(test_tuple_seq_args(([1,2],[3,4])),
                                        ([1,2], [3,4])))
    def testContainerize(self):
        self.assertTrue(recursive_equal(test_containerize([1,2]), ([1,2], [-1,-2])))

if __name__ == "__main__":
    unittest.main()
