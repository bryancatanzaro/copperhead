#
#   Copyright 2008-2012 NVIDIA Corporation
# 
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
# 
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
# 
#

"""
Test Copperhead abstract syntax tree parsing and manipulation.
"""

import unittest
import __builtin__

from copperhead import prelude
from copperhead.compiler import pltools, pyast, coresyntax as S, rewrites as F, passes as P

def expr(text):  return pyast.expression_from_text(text)
def stmt(text):  return pyast.statement_from_text(text)

class ExprParsingTests(unittest.TestCase):

    def setUp(self): pass
    def tearDown(self): pass

    def testLiterals(self):
        self.assertEqual(str(expr('1')), '1')
        self.assertEqual(str(expr('x')), 'x')

        self.assertEqual(repr(expr('1')), "Number(1)")
        self.assertEqual(repr(expr('x')), "Name('x')")

    def testOperators(self):
        self.assertEqual( str(expr('1+2')), 'op_add(1, 2)' )
        self.assertEqual( str(expr('1>>2')), 'op_rshift(1, 2)' )
        self.assertEqual( str(expr('1<2')), 'cmp_lt(1, 2)' )
        
        self.assertEqual( str(expr('x and y and (32*2)')),
                          #XXX Issue 3: Short circuit operators
                          # When this issue is fixed, replace with:
                          #'x and y and op_mul(32, 2)'
                          'op_band(x, op_band(y, op_mul(32, 2)))' )

    def testMap(self):
        self.assertEqual(str(expr('map(fmad, a, x, y)')), 'map(fmad, a, x, y)')

    def testIf(self):
        self.assertEqual(str(expr('x if True else y')), 'x if True else y')

    def testLambda(self):
        self.assertEqual(str(expr('lambda x: lambda y: x+y')),
                         'lambda x: lambda y: op_add(x, y)')

    def testComprehensions(self):
        self.assertEqual(str(expr('[x+3 for x in range(10)]')),
                'map(lambda x: op_add(x, 3), range(10))')
        self.assertEqual(str(expr('[x+y for x,y in zip(range(10),range(10))]')),
                'map(lambda x, y: op_add(x, y), range(10), range(10))')

    def testSubscripts(self):
        self.assertEqual(str(expr('A[i]')), 'A[i]')
        self.assertEqual(repr(expr('A[10]')),
                         "Subscript(Name('A'), Number(10))")
        self.assertEqual(repr(expr('A[i]')),
                         "Subscript(Name('A'), Name('i'))")

class StmtParsingTests(unittest.TestCase):
    
    def setUp(self): pass
    def tearDown(self): pass

    def match(self, src, ref):
        code = stmt(src)
        text = '\n'.join(map(str, code))
        self.assertEqual(text, ref)

    def testBindings(self):
        self.match('x = 4', 'x = 4')

    def testReturn(self):
        self.match('return x+y', 'return op_add(x, y)')

    def testProcedure(self):
        self.match('def f(a,b): return a * b - 4',
                   'def f(a, b):\n    return op_sub(op_mul(a, b), 4)')

    def testCond(self):
        self.match('if x<1: return x\nelse:  return 1/x',
            'if cmp_lt(x, 1):\n    return x\nelse:\n    return op_div(1, x)')


class FreeVariableTests(unittest.TestCase):
    
    def setUp(self):
        # The proper "global" environment for these tests must include
        # __builtins__ so that things like True are considered defined.
        self.env = pltools.Environment(prelude.__dict__, __builtin__.__dict__)

    def tearDown(self): pass

    def check(self, src, vars):
        f = S.free_variables(src, self.env)
        self.assertEqual(sorted(f), sorted(list(vars)))

    def testLiterals(self):
        self.check(expr('32'), [])
        self.check(expr('True and False'), [])
        self.check(expr('None'), [])
        self.check(expr('True and x'), 'x')

    def testSimple(self):
        self.check(expr('x+y'), 'xy')
        self.check(expr('f(g(x, y))'), 'fgxy')

    def testLambda(self):
        self.check(expr('lambda x: y*x'), 'y')
        self.check(expr('lambda y: lambda x: y*x'), [])

    def testClosure(self):
        # closure are not supported by our front-end parser, so we have
        # to construct the AST manually.
        self.check(S.Closure([S.Name('a'), S.Name('b')],
                             S.Lambda(map(S.Name, ['x','y','z']),
                                      S.And(map(S.Name, ['w', 'x','y','z'])))),
                   'abw')

    def testBindings(self):
        self.check(stmt('x=3+y'), 'y')
        self.check(stmt('x=3; z=x+y'), 'y')
        self.check(stmt('x=3; z=x+y; return z'), 'y')

        # XXX This is currently the correct behavior according to
        #     Copperhead, but represents a semantic deviation from
        #     Python.  See the Copperhead wiki for details.
        self.check(stmt('f = lambda x: (x+y, f)'), 'fy')


class SubstitutionTests(unittest.TestCase):

    def setUp(self): pass
    def tearDown(self): pass

    def check(self, src, subst, ref):
        e = expr(src)
        code = S.substituted_expression(e, subst)
        self.assertEqual(str(code), ref)

    def testExpressions(self):
        self.check('x+y', {'x': 'x_1'}, 'op_add(x_1, y)')

    def testLambda(self):
        self.check('lambda x: y*x', {'x': 'x_1', 'y': 'y_1'},
                   'lambda x: op_mul(y_1, x)')

        self.check('lambda y: lambda x: y*x', {'x': 'x_1', 'y': 'y_1'},
                   'lambda y: lambda x: op_mul(y, x)')

class SyntaxErrorTests(unittest.TestCase):
    def check(self, fn, *args):
        self.assertRaises(SyntaxError, fn, *args)
    def testCond(self):
        self.check(stmt, """
if True:
  return False
""")
    def testArity(self):
        self.check(F.arity_check, stmt("""
a = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)"""))
        self.check(F.arity_check, stmt("""
def foo(a, b, c, d, e, f, g, h, i, j, k):
  return 0
"""))
        self.check(P.run_compilation, P.frontend, stmt("""
def foo(x):
  def sub(a, b, c, d, e, f, g, h, i, j):
    return a + b + c + d + e + f + g + h + i + j + x 
  return sub(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
"""), P.Compilation(silence=True))
    def testReturn(self):
        self.check(F.return_check, stmt("""
def foo(x):
  y = x
"""))
        self.check(F.return_check, stmt("""
def foo(x):
  if True:
    return x
  else:
    y = x
"""))
        self.check(F.return_check, stmt("""
def foo(x):
  if True:
    y = x
  else:
    return x
"""))
        self.check(F.return_check, stmt("""
def foo(x):
  if True:
    y = x
  else:
    y = x
"""))
    def testBuiltin(self):
        self.check(F.builtin_check, stmt("""
def map(x):
  return x
"""))
        self.check(F.builtin_check, stmt("""
def zip(x):
  return x
"""))
        self.check(F.builtin_check, stmt("""
def op_add(x):
  return x
"""))


        #If this raises an exception, the test will fail
        stmt("""
def foo(x):
  return x
""")
if __name__ == "__main__":
    unittest.main()
