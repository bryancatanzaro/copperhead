from copperhead import *
import unittest

# @cu
# def infinite_loop(val):
#     return infinite_loop(val+1)

# @cu
# def nested_non_tail(val, limit_a, limit_b):
#     if (val < limit_a):
#         if (val < limit_b):
#             return nested_non_tail(val+1, limit_a, limit_b)
#         else:
#             return val
#     else:
#         if (val > limit_b):
#             return val
#         else:
#             return nested_non_tail(val+1, limit_a, limit_b)

# @cu
# def non_tail_recursive(val, limit):
#     if (val < limit):
#         return non_tail_recursive(val, limit) + 1
#     else:
#         return val

#These should all fail
#print(nested_non_tail(0, 1, 2))
#print(infinite_loop(0))
#print(non_tail_recursive(0))


@cu 
def thencount(val, limit):
    if (val < limit):
        return thencount(val+1, limit)
    else:
        return val

@cu
def elsecount(val, limit):
    if (val == limit):
        return val
    else:
        return elsecount(val+1, limit)

@cu
def prethencount(val, limit):
    incval = val + 1
    if (val < limit):
        return prethencount(incval, limit)
    else:
        return val

@cu
def preelsecount(val, limit):
    incval = val + 1
    if (val == limit):
        return val
    else:
        return preelsecount(incval, limit)
    
@cu
def vinc(x, val, limit):
    if (val == limit):
        return x
    else:
        return vinc(map(lambda xi: xi+1, x), val+1, limit)

@cu
def previnc(x, val, limit):
    incval = map(lambda xi:xi+1, x)
    if (val == limit-1):
        return incval
    else:
        return previnc(incval, val+1, limit)
    
@cu
def divergent(limit):
    def divergent_sub(val):
        if (val == limit):
            return val
        else:
            return divergent_sub(val+1)
    return map(divergent_sub, range(limit))

class TailRecursionTest(unittest.TestCase):
    def test_thencount(self):
        self.assertEqual(thencount(0, 10), 10)
    def test_elsecountself(self):
        self.assertEqual(elsecount(0, 10), 10)
    def test_prethencount(self):
        self.assertEqual(prethencount(0, 10), 10)
    def test_preelsecount(self):
        self.assertEqual(preelsecount(0, 10), 10)
    def test_vinc(self):
        self.assertEqual(list(vinc([0,0,0], 0, 5)), [5,5,5])
    def test_previnc(self):
        self.assertEqual(list(previnc([0,0,0], 0, 5)), [5,5,5])
    def test_divergent(self):
        self.assertEqual(list(divergent(5)), [5,5,5,5,5])

if __name__ == '__main__':
    unittest.main()

