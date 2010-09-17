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

from copperhead import *
from copperhead.compiler import passes, shapeinference
from copperhead.runtime import intermediate


# Corpus of tests

@cu
def id(x):
    return x

@cu
def incr(x):
    return map(lambda xi: xi + 1, x)

@cu
def plus1(A):
    one = 1
    return [ai+one for ai in A]

@cu
def plus_scan(A):
    return scan(op_add, A)

@cu
def axpy(a, x, y):
    return map(lambda xi, yi: a * xi + yi, x, y)

@cu
def flat1(A):
    return join(A)

@cu
def flat2(A):
    return join(join(A))

@cu
def dot1(x, y):
    return sum(map(op_mul, x, y))

@cu
def dot2(x, y):
    return sum([xi*yi for xi,yi in zip(x,y)])

@cu
def eo(A, pred):
    if pred: return evens(A)
    else:    return odds(A)

@cu
def ones(n):
    return replicate(1, n)

@cu
def eventually_permute(A, indices, i):
    if i<=0:
        return permute(A, indices)
    else:
        return eventually_permute(A, indices, i-1)

@cu
def eventually_id(x):
    A = x
    B = A
    C = B
    D = zip(A,B)
    return C

@cu
def partial_sums(A):
    parts = split(A, 256)
    return map(sum, parts)

@cu
def total(A):
    parts = split(A, 256)
    partials = map(sum, parts)
    return sum(partials)

@cu
def first_tile(A):
    parts = split(A, 256)
    return parts[0]

@cu
def nested_map(A):
    return map(lambda row: map(lambda el: el * 2.0, row), A)

@cu
def transform_plus(f, a, b):
    return f(a) + f(b)

@cu
def apply2(f, a, b):
    return f(a, b)

@cu
def spmv2(Ax, Aj, x):
    def spvv(x, cols, y):
        z = gather(y, cols)
        return sum(map(lambda a, b: a * b, x, z))

    return map(lambda y, cols: spvv(y, cols, x), Ax, Aj)

def new_shaping(ast, M):
    shapeinference.infer(ast, context=M.shape_context)
    return ast

def verbose_shaping(ast, M):
    shapeinference.infer(ast, verbose=True, context=M.shape_context)
    return ast

compiler = passes.Pipeline('shape_pipeline',
                           [passes.frontend,
                            new_shaping])

debug = passes.Pipeline('shape_debugging',
                           [passes.frontend,
                            verbose_shaping])

def trial(fn):
    code = passes.compile([fn.get_ast()], target=debug, globals=globals())


from copperhead.compiler.shapeinference import Unit, Shape, Unknown,\
                                               shapeof, elementof, \
                                               extentof, eq

with intermediate.tracing(parts=[compiler], including=['frontend']):
    trial(incr)
    trial(plus1)
    trial(plus_scan)
    trial(dot2)
    trial(ones)
    trial(axpy)
    trial(eventually_permute)
    trial(partial_sums)
    trial(total)
    trial(eo)
    trial(flat1)
    trial(transform_plus)
    trial(apply2)
    trial(spmv2)
    pass

class ShapeTests(unittest.TestCase):

    def setUp(self):
        self.M = passes.Compilation(globals=globals())

    def tearDown(self):  pass

    def shape(self, fn, args, shape):
        suite = [fn.get_ast()]
        code = passes.run_compilation(compiler, suite, self.M)
        shapefn = self.M.shape_context.locals[fn.__name__]
        inferred, constraints = shapefn(*map(shapeinference.ShapeOf, args))
        self.assertEqual(str(inferred), shape)

    def testDirect(self):
        self.shape(id, ['x'], 'ShapeOf(x)')
        self.shape(incr, ['x'], 'Shape(ExtentOf(x), Unit)')
        self.shape(plus1, ['x'], 'Shape(ExtentOf(x), Unit)')
        self.shape(axpy, ['a','x','y'], 'Shape(ExtentOf(x), Unit)')
        self.shape(eventually_permute, ['A','n','i'], "ShapeOf(A)")
        self.shape(eventually_id, ['A'], "ShapeOf(A)")
        self.shape(plus_scan, ['A'], "ShapeOf(A)")

    def testDataDependent(self):
        self.shape(ones, ['n'], "Shape(['Unknown'], Unit)")

    def testReduction(self):
        self.shape(dot1, ['x','y'], 'Unit')
        self.shape(dot2, ['x','y'], 'Unit')

    def testSplitting(self):
        self.shape(eo, ['x','p'], "Shape(['Unknown'], ElementOf(x))")
        self.shape(partial_sums, ['A'], "Shape(['Unknown'], Unit)")
        self.shape(total, ['A'], "Unit")
        self.shape(first_tile, ['A'], "Shape(['Unknown'], ElementOf(A))")


    def testJoining(self):
        self.shape(flat1, ['A'], "Shape(['Unknown'], ElementOf(ElementOf(A)))")
        self.shape(flat2, ['A'], "Shape(['Unknown'], ElementOf(ElementOf(ElementOf(A))))")

    def testNested(self):
        self.shape(nested_map, ['A'], "Shape(ExtentOf(A), Shape(ExtentOf(ElementOf(A)), Unit))")
    
    def testFunctions(self):
        self.shape(transform_plus, [lambda x: (Unit,[]), 'a', 'b'], "Unit")
        self.shape(transform_plus, ['f', 'a', 'b'], "Unit")
        self.shape(apply2, ['f', 'a', 'b'], "Any")
        self.shape(apply2, [lambda a,b: (Unit,[]), 'a', 'b'], "Unit")
        self.shape(apply2, [lambda a,b: (shapeof(a),[]),'a','b'], "ShapeOf(a)")

    def testSpmv(self):
        self.shape(spmv2, ['Ax', 'Aj', 'x'], "Shape(ExtentOf(Ax), Unit)")

if __name__ == "__main__":
    unittest.main()
