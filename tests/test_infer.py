#! /usr/bin/env python
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
from copperhead.compiler.pyast import expression_from_text as parseE, \
                                      statement_from_text as parseS

from copperhead.compiler.parsetypes import type_from_text



from copperhead.compiler.typeinference import InferenceError, infer, TypingContext


def thunk(type):
    'Create type-carrying thunk'
    fn = lambda x:x

    if isinstance(type, str):
        type = type_from_text(type)
    elif not isinstance(type, T.Type):
        raise TypeError, \
                "type (%s) must be string or Copperhead Type object" % type

    fn.cu_type = type

    return fn

# Fake globals to be used in place of the Prelude for now
G1 = {
        'op_add'   : thunk('(a,a) -> a'),
        'op_sub'   : thunk('(a,a) -> a'),
        'op_mul'   : thunk('(a,a) -> a'),
        'op_neg'   : thunk('a -> a'),
        'range'    : thunk('Long -> [Long]'),
        'cmp_lt'   : thunk('(a,a) -> Bool'),
        'cmp_eq'   : thunk('(a,a) -> Bool'),

        'sum'      : thunk('[a] -> a'),
        'any'      : thunk('[Bool] -> Bool'),

        'ZZ'       : thunk('[Long]'),
        'RR'       : thunk('[Double]'),
        'BB'       : thunk('[Bool]'),
     }


class ExpressionTypeTests(unittest.TestCase):

    def setUp(self): pass

    def tearDown(self): pass

    def typing(self, source, t):
        P = parseE(source)
        self.assertEqual(str(infer(P, globals=G1)), t)

    def illegal(self, source):
        P = parseE(source)
        self.assertRaises(InferenceError, lambda:infer(P, globals=G1))

    def testLiterals(self):
        self.typing("5", "Long")
        self.typing("1.33", "Double")
        self.typing("True", "Bool")
        self.typing("False", "Bool")
        self.typing("None", "Void")

    def testBuiltins(self):
        self.typing("op_add", "ForAll a: (a, a) -> a")
        self.typing("range", "Long -> [Long]")
        self.illegal("undefined_variable")
        self.typing("op_add(2,3)", "Long")
        self.typing("range(9)", "[Long]")

    def testTuples(self):
        self.typing("(1, True, 3.0, 4, 5)", "(Long, Bool, Double, Long, Long)")

    def testConditionals(self):
        self.typing("12 if True else 24", "Long")
        self.illegal("12 if -100 else 24")
        self.illegal("12 if False else 24.0")

    def testArithmetic(self):
        self.typing("2 + 3", "Long")
        self.typing("2.0 + 3.0", "Double")
        self.illegal("2 + 3.0")

    def testBoolean(self):
        self.typing("True and False", "Bool")
        self.typing("True or  False", "Bool")
        self.illegal("True or  0")
        self.illegal("1 and False")
        self.illegal("1 and 0")

    def testLambdas(self):
        self.typing("lambda: 1", "Void -> Long")
        self.typing("lambda x: 1", "ForAll a: a -> Long")
        self.typing("lambda x: x", "ForAll a: a -> a")
        self.typing("lambda x: x+1", "Long -> Long")
        self.typing("lambda x,y: x+y", "ForAll a: (a, a) -> a")
        self.typing("lambda x,y: x+y*2.0", "(Double, Double) -> Double")
        self.typing("lambda x: 12 if x else 24", "Bool -> Long")
        self.typing("True and (lambda x:True)(3)", "Bool")

    def testMaps(self):
        self.typing("map(lambda x: x, ZZ)", "[Long]")
        self.typing("map(lambda x: x, range(9))", "[Long]")
        self.typing("map(lambda x: x+1, range(9))", "[Long]")
        self.illegal("map(lambda x: x+2.0, range(9))")
        self.illegal("map(lambda x: x and True, range(9))")
        self.typing("map(lambda x: x<42, range(9))", "[Bool]")

    def testReduction(self):
        self.typing("any(BB)", "Bool")
        self.typing("any(map(lambda x: x<42, ZZ))", "Bool")
        self.typing("sum(ZZ)", "Long")
        self.typing("sum(RR)", "Double")
        self.typing("sum(range(9))", "Long")

    def testIdentity(self):
        self.typing("lambda x: x", "ForAll a: a -> a")
        self.typing("(lambda x: x)(lambda x:x)", "ForAll a: a -> a")
        self.typing("(lambda x: x)(lambda x:x)(7)", "Long")
        self.illegal("lambda x: x(x)")
        self.illegal("(lambda i: i(i))(lambda x:x)")

    def testWrapping(self):
        self.typing("lambda A: map(lambda x:x, A)", "ForAll a: [a] -> [a]")
        self.typing("lambda A: map(lambda x:x+1, A)", "[Long] -> [Long]")
        self.typing("lambda A: any(A)", "[Bool] -> Bool")
        self.typing("lambda A: sum(A)", "ForAll a: [a] -> a")

    def testVecAdd(self):
        self.typing("lambda x,y:  map(lambda a, b: a + b, x, y)",
                    "ForAll a: ([a], [a]) -> [a]")

    def testSaxpy(self):
        self.typing("lambda Z: map(lambda xi, yi: 2*xi + 3*yi, Z, Z)",
                    "[Long] -> [Long]")

        self.typing("lambda a: map(lambda xi, yi: a*xi + yi, ZZ, ZZ)",
                    "Long -> [Long]")

        self.typing("lambda a: map(lambda xi, yi: a*xi + yi, RR, RR)",
                    "Double -> [Double]")

        self.illegal("lambda a: map(lambda xi, yi: a*xi + yi, RR, ZZ)")

        self.typing("lambda a: lambda x,y: map(lambda xi, yi: a*xi + yi, x, y)",
                    "ForAll a: a -> ([a], [a]) -> [a]")

        self.typing("lambda a,x,y: map(lambda xi, yi: a*xi + yi, x, y)",
                    "ForAll a: (a, [a], [a]) -> [a]")

    def testSlicing(self):
        self.typing("ZZ[10]", "Long")
        self.typing("RR[10]", "Double")
        self.typing("lambda i: RR[i]", "Long -> Double")
        self.illegal("ZZ[1.0]")
        self.illegal("ZZ[undef]")


class StatementTypeTests(unittest.TestCase):

    def setUp(self):
        self.tycon = TypingContext(globals=G1)

    def tearDown(self): pass

    def illegal(self, source):
        ast = parseS(source)
        self.assertRaises(InferenceError,
                lambda: infer(ast, globals=G1))

    def typing(self, source, t="Void"):
        ast = parseS(source)
        result = infer(ast, context=self.tycon)
        self.assertEqual(str(result), t)

    def testReturnLiteral(self):
        self.typing("return 1", "Long")
        self.typing("return True", "Bool")
        self.typing("return (1,1)", "(Long, Long)")
        self.typing("return (1.0, False)", "(Double, Bool)")

    def testReturnSimple(self):
        self.typing("return 1+3*4", "Long")
        self.illegal("return x+3")
        self.typing("x = 5", "Void")
        self.typing("x = 4; return 1", "Long")
        self.typing("x = 4; return x", "Long")
        self.typing("x=4; y=3; return x*y", "Long")
        self.typing("x=7; x=False; return x", "Bool")

    def testTupleBinding(self):
        self.typing("p0 = (0.0, 0.0); return p0", "(Double, Double)")
        self.typing("x0, y0 = (5, -5)")
        self.typing("return x0", "Long")
        self.typing("return y0", "Long")
        self.illegal("x0, y0, z0 = (5, -5)")
        self.illegal("x0, y0 = (5, -5, 55)")

    def testSimpleProcedures(self):
        self.typing("def f1(x): return x")
        self.typing("def f2(x): y=3; return x+3")
        self.typing("def f3(x): return f2(x)")
        self.typing("def f4():  return 5")
        self.typing("def f5(x): return lambda y: x+y")

        self.typing("return f1", "ForAll a: a -> a")
        self.typing("return f2", "Long -> Long")
        self.typing("return f3", "Long -> Long")
        self.typing("return f4", "() -> Long")
        self.typing("return f5", "ForAll a: a -> a -> a")

    def testIdentity(self):
        self.typing("def id(x): return x")
        self.typing("return id(True) and id(False)", "Bool")
        self.illegal("return id(True) and id(3)")
        self.typing("return id(True) and 1<id(3)", "Bool")
        self.typing("return id(id)(True) and id(id)(False)", "Bool")
        self.typing("return id(id)(True) and 1<id(id)(3)", "Bool")
        self.typing("g=id(id); return g(True) and g(False)", "Bool")
        self.typing("g=id(id); return g(True) and 1<g(3)", "Bool")
        self.illegal("def f6(f): return f(True) and 1<f(3)")

    def testReduction(self):
        self.typing("return sum(range(9))", "Long")

        self.typing("def red1(A): return sum(A)")
        self.typing("return red1", "ForAll a: [a] -> a")
        self.typing("return red1(range(9))", "Long")

        self.typing("red2 = lambda(A): sum(A)")
        self.typing("return red2", "ForAll a: [a] -> a")

    def testIncr(self):
        self.typing("def incr(a): return a+1")
        self.typing("def add1(x): return map(incr, x)")
        self.typing("return incr", "Long -> Long")
        self.typing("return add1", "[Long] -> [Long]")

    def testSaxpy(self):
        self.typing(saxpy)
        t = "ForAll a: (a, [a], [a]) -> [a]"
        self.typing("return saxpy1", t)
        self.typing("return saxpy2", t)
        self.typing("return saxpy3", t)
        self.typing("return saxpy4", t)

    def testIfThenElse(self):
        self.typing(ifelse1, "Long")
        self.illegal(ifelse2)
        self.typing(conditionals)
        self.typing("return abs", "Long -> Long")

    def testEmbeddedFunctions(self):
        self.typing(idseq)
        self.typing("return idseq0", "ForAll a: [a] -> [a]")
        self.typing("return idseq1", "ForAll a: [a] -> [a]")
        self.typing("return idseq2", "ForAll a: [a] -> [a]")

    def testRecursive(self):
        self.typing(recursive)
        self.typing("return fun1", "ForAll a, b: a -> b")
        self.typing("return fun2", "ForAll a: a -> Long")
        self.typing("return fun3", "Long -> Long")


class ClosureTypeTests(unittest.TestCase):

    def setUp(self):
        self.tycon = TypingContext(globals=G1)

    def tearDown(self): pass

    def illegal(self, source):
        from copperhead.compiler.rewrites import closure_conversion
        ast = parseS(source)
        ast = closure_conversion(ast, G1)
        self.assertRaises(InferenceError,
                lambda: infer(ast, globals=G1))

    def typing(self, source, t="Void"):
        from copperhead.compiler.rewrites import closure_conversion
        ast = parseS(source)
        ast = closure_conversion(ast, G1)
        result = infer(ast, context=self.tycon)
        self.assertEqual(str(result), t)

    def testLambdaClosures(self):
        self.typing("a=1; f=lambda x: x+a; return f(2)", "Long")
        self.illegal("a=1; f=lambda x: x+a; return f(True)")
        self.illegal("a=True; f=lambda x: x+a; return f(2)")

        self.typing("a,b=1,2; f=lambda x: a*x+b; return f(2)", "Long")
        self.illegal("a,b=1,2.0; f=lambda x: a*x+b; return f(2)")

    def testProcedureClosures(self):
        self.typing(saxpy)
        self.typing("return saxpy1", "ForAll a: (a, [a], [a]) -> [a]")
        self.typing("return saxpy2", "ForAll a: (a, [a], [a]) -> [a]")
        self.typing("return saxpy3", "ForAll a: (a, [a], [a]) -> [a]")
        self.typing("return saxpy4", "ForAll a: (a, [a], [a]) -> [a]")

    def testEmbeddedFunctions(self):
        self.typing(idseq)
        self.typing("return idseq0", "ForAll a: [a] -> [a]")
        self.typing("return idseq1", "ForAll a: [a] -> [a]")
        self.typing("return idseq2", "ForAll a: [a] -> [a]")

class FrontendTypeTests(unittest.TestCase):
    def setUp(self):
        self.tycon = TypingContext(globals=G1)

    def tearDown(self): pass

    def typing(self, source, t="Void"):
        from copperhead.compiler import rewrites as Front

        ast = parseS(source)
        ast = Front.closure_conversion(ast, G1)
        ast = Front.single_assignment_conversion(ast)
        ast = Front.lambda_lift(ast)
        result = infer(ast, context=self.tycon)
        self.assertEqual(str(result), t)

    def testSaxpy(self):
        self.typing(saxpy)
        self.typing("return saxpy1", "ForAll a: (a, [a], [a]) -> [a]")
        self.typing("return saxpy2", "ForAll a: (a, [a], [a]) -> [a]")
        self.typing("return saxpy3", "ForAll a: (a, [a], [a]) -> [a]")
        self.typing("return saxpy4", "ForAll a: (a, [a], [a]) -> [a]")



class TypingWithPrelude(unittest.TestCase):

    def setUp(self):
        self.tycon = TypingContext(globals=prelude.__dict__)

    def tearDown(self): pass

    def illegal(self, source):
        ast = parseS(source)
        self.assertRaises(InferenceError,
                lambda: infer(ast, globals=prelude.__dict__))

    def typing(self, source, t="Void"):
        ast = parseS(source)
        result = infer(ast, context=self.tycon)
        self.assertEqual(str(result), t)

    def testSpvv(self):
        self.typing(spvv)
        self.typing("return spvv1", "ForAll a: ([Long], [a], [Long]) -> Long")
        self.typing("return spvv2", "ForAll a, b: ([a], [b], [a]) -> a")

    def testZipping(self):
        self.typing("def zippy1(x,y): return zip(x, y)")
        self.typing("def zippy2(x,y): return map(lambda xi,yi: (xi,yi), x, y)")
        self.typing("def zippy3(x,y): f=(lambda x,y: (x,y)); return map(f, x, y)")
        self.typing("def zippy4(x,y,z): return zip(x, y)")
        self.typing("def zippy5(x,y,z): z=x; return zip(x, y)")

        self.typing("return zippy1", "ForAll a, b: ([a], [b]) -> [(a, b)]")
        self.typing("return zippy2", "ForAll a, b: ([a], [b]) -> [(a, b)]")
        self.typing("return zippy3", "ForAll a, b: ([a], [b]) -> [(a, b)]")
        self.typing("return zippy4", "ForAll a, b, c: ([a], [b], c) -> [(a, b)]")
        self.typing("return zippy5", "ForAll a, b, c: ([a], [b], c) -> [(a, b)]")

    def testDot(self):
        self.typing("def dot1(x,y): return sum(map(lambda a, b: a * b, x, y))")
        self.typing("def dot2(x,y): return reduce(op_add, map(lambda a, b: a * b, x, y), 0)")
        self.typing("def dot3(x,y): y=x; return sum(map(lambda a, b: a * b, x, y))")

        self.typing("return dot1", "ForAll a: ([a], [a]) -> a")
        self.typing("return dot2", "([Long], [Long]) -> Long")
        self.typing("return dot3", "ForAll a, b: ([a], b) -> a")

        self.typing(dots)
        self.typing("return dot4", "ForAll a: ([a], [a]) -> a")
        self.typing("return dot5", "ForAll a: ([a], [a]) -> a")


spvv = """
def spvv1(x, cols, y):
    z = gather(y, cols)
    return reduce(lambda a, b: a + b, map(lambda a, b: a * b, x, z), 0)

def spvv2(x, cols, y):
    z = gather(y, cols)
    return sum(map(op_mul, x, z))
"""


saxpy = """
def saxpy1(a, x ,y):
    return map(lambda xi, yi: a*xi + yi, x, y)

def saxpy2(a, x, y):
    return [a*xi + yi for xi,yi in zip(x,y)]

def saxpy3(a, x, y):
    triad = lambda xi, yi: a*xi + yi
    return map(triad, x, y)

def saxpy4(a, x, y):
    def triad(xi, yi):
        return a * xi + yi
    return map(triad, x, y)
"""

ifelse1 = """
if 1<0: x=1; y=0; return x+y
else:   x=2; y=3; return y-x
"""

ifelse2 = """
if 1<0: x=1; y=0; return x+y
else:   x=2; y=3; return 3.0*y-x
"""

conditionals = """
def abs(x):
    if x==0:  return x
    elif x<0: return -x
    else:     return x
"""

dots = """
def dot4(x, y):
    return sum(map(op_mul, x, y))

def dot5(x, y):
    _e0 = map(op_mul, x, y)
    _returnValue = sum(_e0)
    return _returnValue
"""

idseq = """
def ident(x): return x

def idseq0(A):
    return map(ident, A)

def idseq1(A):
    return map(lambda x: ident(x), A)

def idseq2(A):
    def _ident(x): return x
    return map(lambda x: _ident(x), A)
"""

recursive = """\
def fun1(x):  return fun1(x)

def fun2(x):  return 1 + fun2(x)

def fun3(x):  return 1 + fun3(x-1)
"""

if __name__ == "__main__":
    unittest.main()
    exit()
