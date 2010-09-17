from copperhead import *
import numpy as np
import unittest
import itertools

class DataTests(unittest.TestCase):
    def setUp(self):
        py_doubles = [1.0, 2.0, 3.0]
        np_floats = [np.float32(x) for x in py_doubles]
        np_doubles = [np.float64(x) for x in py_doubles]
        py_ints = [1, 2, 3]
        np_ints = [np.int32(x) for x in py_ints]
        np_longs = [np.int64(x) for x in py_ints]
        py_bools = [True, False]
        np_bools = [np.bool_(x) for x in py_bools]
        self.cu_floats = [CuFloat(x) for x in np_floats]
        self.cu_doubles = [CuDouble(x) for x in np_doubles]
        self.cu_ints = [CuInt(x) for x in np_ints]
        self.cu_longs = [CuLong(x) for x in np_longs]
        self.cu_bools = [CuBool(x) for x in np_bools]
        self.doubles = py_doubles + np_doubles + self.cu_doubles
        self.floats = np_floats + self.cu_floats
        self.ints = py_ints + np_ints + self.cu_ints
        self.longs = np_longs + self.cu_longs
        self.bools = py_bools + np_bools + self.cu_bools
        comparators = ['<', '>', '<=', '>=', '==', '!=']
        real_operators = comparators + ['+', '-', '*', '/', '%']
        int_operators = comparators + real_operators + ['>>', '<<']
        self.bools_operators = ['and', 'or']
        self.doubles_operators = real_operators
        self.floats_operators = real_operators
        self.ints_operators = int_operators
        self.longs_operators = int_operators

    def tearDown(self): pass
        
    def do_data_test(self, name):
        operators = getattr(self, name + '_operators')
        data = getattr(self, name)
        cu_data = getattr(self, 'cu_' + name)
        for operator in operators:
            test_expr = 'x ' + operator + ' y'
            true_expr = 'x ' + operator + ' y.value'
            for x, y in itertools.product(data, cu_data):
                self.assertEqual(eval(test_expr), eval(true_expr))

    def test_doubles(self):
        self.do_data_test('doubles')
    def test_floats(self):
        self.do_data_test('floats')
    def test_ints(self):
        self.do_data_test('ints')
    def test_longs(self):
        self.do_data_test('longs')
    def test_bools(self):
        self.do_data_test('bools')

if __name__ == "__main__":
    unittest.main()
