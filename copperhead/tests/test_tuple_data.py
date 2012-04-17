import unittest
from copperhead import *

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
def tuple_double(a, b):
    m, n = a
    o, p = b
    r = m + o, n + p
    return r
        
        
class TupleExtract(unittest.TestCase):
    def testTupleDouble(self):
        source = (1, 2)
        golden = (2, 4)
        self.assertEqual(tuple_double(source, source), golden)
        
if __name__ == "__main__":
    unittest.main()
