#
#  Copyright 2008-2010 NVIDIA Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import unittest

from copperhead import prelude
from copperhead.compiler import passes, coresyntax as S


class RewriteTest(unittest.TestCase):

    def setUp(self):  pass
    def tearDown(self):  pass

    def expr(self, source, match=None):
        M = passes.Compilation(source=source, globals=prelude.__dict__)
        e = passes.parse(source, mode='eval')
        f = self.run_pass(e, M)
        result = str(f)
        self.assertEqual(result, match or source)

    def stmt(self, source, match=None):
        M = passes.Compilation(source=source, globals=prelude.__dict__)
        e = passes.parse(source, mode='exec')
        f = self.run_pass(e, M)
        result = '\n'.join(map(str, f))
        self.assertEqual(result, match or source)
        

class SingleAssignment(RewriteTest):

    def setUp(self):
        self.run_pass = passes.single_assignment_conversion

    # XXX This fails because the single assignment conversion
    # doesn't always return the same results - the answer
    # depends on what other single assignment conversions
    # have been previously performed.  If you run
    # test_rewrites on its own, you get one answer, if
    # you run it as part of test_all, you get a different
    # answer.  Python 2.7 has an expectedFailure() tag
    # for these kinds of cases, but it doesn't work on
    # Python 2.6, so I'm just commenting it out.
    
    #@unittest.expectedFailure()
    #def testSimple(self):
    #    self.stmt('x=3; y=x', '_x_1 = 3\n_y_2 = _x_1')

class ConditionalProtection(RewriteTest):

    def setUp(self):
        self.run_pass = passes.protect_conditionals

    def testSimple(self):
        self.expr("x if True else y",
                  "((lambda : x) if True else (lambda : y))()")


_T1_before = """\
def f(a, x, Y):
    return map(lambda y: a*x+y, Y)
"""

_T1_after = """\
def f(a, x, Y):
    return map(closure([a, x], lambda y, _K0, _K1: op_add(op_mul(_K0, _K1), y)), Y)"""

_T2_before = """\
def f(A, extern):
    def g(B):
        return extern(B)
    return g(A)"""

_T2_after = """\
def f(A, extern):
    def g(B, _K0):
        return _K0(B)
    return (closure([extern], g))(A)"""

_T3_before = """\
def f(A):
    extern = lambda x: x
    def g(B):
        return extern(B)
    return g(A)"""

_T3_after = """\
def f(A):
    extern = lambda x: x
    def g(B, _K0):
        return _K0(B)
    return (closure([extern], g))(A)"""

_T4_before = """\
def f(A):
    x, y = 5, lambda x: x
    return map(lambda ai: y(ai), A)"""

_T4_after = """\
def f(A):
    (x, y) = (5, lambda x: x)
    return map(closure([y], lambda ai, _K0: _K0(ai)), A)"""

_T5_before = """\
def f(x):
    def g(y):
        if y<x:
            return y
        else:
            return g(y-1)
    return g(100)"""

_T5_after = """\
def f(x):
    def g(y, _K0):
        if cmp_lt(y, _K0):
            return y
        else:
            return (closure([_K0], g))(op_sub(y, 1))
    return (closure([x], g))(100)"""



class ClosureConversion(RewriteTest):

    def setUp(self):
        self.run_pass = passes.closure_conversion

    def testLambda(self):
        self.expr("lambda x: x")
        self.expr("lambda x: lambda x: x")
        self.expr("lambda x: lambda y: y")
        self.expr("lambda x: lambda y: x",
                  "lambda x: closure([x], lambda y, _K0: _K0)")
        self.expr("lambda x: lambda y: (x, y)",
                  "lambda x: closure([x], lambda y, _K0: (_K0, y))")

        # XXX z is assumed to be globally defined, and hence is not
        #     closed over.  is this the semantics we want?  it seems to
        #     be the best fit with Python.
        self.expr("lambda x: lambda y: z")

        self.expr("lambda x: lambda y: lambda z: x",
                  "lambda x: closure([x], lambda y, _K0: closure([_K0], lambda z, _K0: _K0))")

    def _testMap(self):
        self.expr("map(f, A)")
        self.expr("map(lambda x, y: (x, y), A, B)")
        self.expr("map(lambda x: a*x, A)",
                  "map(closure([a], lambda x, _K0: op_mul(_K0, x)), A)")

        #t_closure("map(lambda x: lambda y: a*x + y, A)")
        #t_closure("map(lambda x, y: a*x + y, A)")
        #t_closure("map(lambda x, y: a*x + b*y, A)")


    def testProcedures(self):
        self.stmt("def f(x):\n    return x")
        self.stmt("def f(x):\n    return scan(x)")
        self.stmt(_T1_before, _T1_after)
        self.stmt(_T2_before, _T2_after)
        self.stmt(_T3_before, _T3_after)
        self.stmt(_T4_before, _T4_after)
        self.stmt(_T5_before, _T5_after)

    def _testShadows(self):
        self.expr("lambda f, A: scan(f, A)")
        self.expr("lambda xxx: lambda f, A: xxx(f, A)",
                  "lambda xxx: closure([xxx], lambda f, A, _K0: _K0(f, A)")
        self.expr("lambda scan: lambda f, A: scan(f, A)",
                  "lambda scan: closure([scan], lambda f, A, _K0: _K0(f, A)")

class LambdaLifting(RewriteTest):

    def setUp(self):
        self.run_pass = passes.lambda_lift

    def testBasic(self):
        # t_lift("def f(x,y,z):  x=x+y; return z-x")
        # t_lift("Z = map(f, A, X, Y)")
        # t_lift("Z = map(lambda a,x,y: a*x+y, A, X, Y)")
        # t_lift("Z = lambda x,y: x/y")
        # t_lift("Z = map(lambda x:x+1, map(lambda x:x/2, A))")
        pass

if __name__ == "__main__":
    unittest.main()
