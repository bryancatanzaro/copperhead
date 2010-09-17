#
#  Copyright 2008-2010 NVIDIA Corporation
#  Copyright 2009-2010 University of California
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

"""
Automatic inference of sequence shapes.

This module assumes that the programs it is given are:

    (1) fully typed

    (2) in single assignment form
"""

from __future__ import absolute_import

from . import coresyntax as AST, coretypes as T
from .shapetypes import *
from .pltools import Environment
from itertools import ifilter
from . import utility as U

class ShapeError(TypeError):
    "An error occuring during shape inference"
    pass

def unitary_type(t):
    return t in (T.Bool, T.Int, T.Float)

def match_formal(x, actual):
    return (actual is Unit) or \
           (isinstance(actual, ShapeOf) and actual.value==x)

def combine_length(l1, l2):
    return l1 if l1==l2 else Unknown

def combine_extents(x1, x2):
    if isinstance(x1, list) and isinstance(x2, list):
        return map(combine_length, x1, x2)

    elif isinstance(x1, ExtentOf) and isinstance(x2, ExtentOf):
        if x1.value==x2.value:
            return x1
        else:
            raise ValueError("cannot combine abstract extents %r, %r" % (x1,x2))

    elif isinstance(x2, ExtentOf):
        return [Unknown] * len(x1)

    elif isinstance(x1, ExtentOf):
        return [Unknown] * len(x2)

    else:
        raise ValueError("%r, %r not valid shape extents" % (x1,x2))


def combine(s1, s2):
    if s1 is Identity:  return s2
    if s2 is Identity:  return s1

    if s1 is Unit or s2 is Unit:  return Unit

    if isinstance(s1, ShapeOf) and isinstance(s2, ShapeOf):
        if (s1.value != s2.value):
            import pdb
            pdb.set_trace()
        assert s1.value==s2.value
        return s1

    if isinstance(s1, ElementOf) and isinstance(s2, ElementOf):
        assert s1.value==s2.value
        return s1

    # XXX Include some special cases to prevent unnecessary
    #     complication from this generic rule?
    return Shape(extents=combine_extents(extentof(s1), extentof(s2)),
                 element=combine(elementof(s1), elementof(s2)))


def lookup(id, env):
    return env[id] if (id in env) else Any

def eval_extents(s, env):
    if isinstance(s, ExtentOf):
        if isinstance(s.value, str):
            x = lookup(s.value, env)
        else:
            x = instantiate(s.value, env)
        result = extentof(x)
        return result
    else:
        return s

def instantiate(s, env):
    if isinstance(s, Shape):
        return Shape(extents=eval_extents(s.extents, env),
                     element=instantiate(s.element, env))

    elif isinstance(s, ShapeOf):
        if isinstance(s.value, str):
            return lookup(s.value, env)
        else:
            return instantiate(s.value, env)

    elif isinstance(s, ElementOf):
        if isinstance(s.value, str):
            x = lookup(s.value, env)
        else:
            x = instantiate(s.value, env)

        return elementof(x)

    elif isinstance(s, ResultOf):
        f = lookup(s.function, env)
        if isinstance(f, ShapeOf) and callable(f.value):
            f = f.value

        if (f is Any) or (not callable(f)):
            return Any
        else:
            args = [instantiate(p,env) for p in s.parameters]
            result, constraints = f(*args)
            return result

class ShapingContext:
    def __init__(self, globals=None, locals=None):
        self.globals = globals or dict()
        self.locals = locals or Environment()

class ShapeInference(AST.SyntaxVisitor):

    def __init__(self, context=None):
        self.context = context or ShapingContext()
        self.verbose = False

        # These members are re-initialized within each procedure to
        # capture information within that procedure.
        self.return_shapes = None
        self.constraints = None

    def _unit(self, ast):
        ast.shape = Unit
        return ast.shape

    def _default(self, ast):
        t = getattr(ast, 'type', None)

        if unitary_type(t):
            return self._unit(ast)
        else:
            return self.visit_children(ast)

    def _Tuple(self, ast):
        ast.shape = tuple(self.visit_children(ast))
        return ast.shape

    def _Name(self, ast):
        # Handle special values
        if ast.id in ('True', 'False', 'None'):
            return self._unit(ast)

        elif ast.id in self.context.locals:
            return self.context.locals[ast.id]

        elif ast.id in self.context.globals:
            obj = self.context.globals[ast.id]
            s = getattr(obj, 'cu_shape', None)

            if s is None:
                raise ShapeError("global name without a shape (%s)" % ast.id)
            else:
                return s

        else:
            raise ShapeError("name is undefined (%s)" % ast.id)

    def _Lambda(self, ast):
        raise AssertionError("shape inference must follow lambda lifting")

    def _Closure(self, ast):
        closed = self.visit(ast.closed_over())
        # XXX At this point in the compiler pipeline, the body of a
        # closure should just be a reference to an external procedure.
        # So we shouldn't need to bind _K0, etc. when processing the
        # body.  If that assumption changes in the future, bind the
        # closed variables appropriately here.
        s = self.visit(ast.body())

        return lambda *args: s(*(list(args)+closed))

    def _Subscript(self, ast):
        A = self.visit(ast.value())
        assert isinstance(ast.slice(), (AST.Index, AST.Name))
        return elementof(A)

    def _Apply(self, ast):
        fn = self.visit(ast.function())
        args = self.visit(ast.arguments())

        if callable(fn):
            result, constraints = fn(*args)
        else:
            # this should be the only other possiblity
            assert isinstance(fn, ShapeOf)

            constraints = []

            # determine the shape of the result from its type where possible
            if unitary_type(ast.function().type.result_type()):
                result = Unit

            # if the shape is not determined by the known type, then we
            # will have to defer further analysis to instantiation
            else:
                result = ResultOf(ast.function().id, args)

            if self.verbose:
                print "Applying unknown function:", repr(ast.function())
                print "   ::", ast.function().type

        self.constraints.extend(constraints)
        return result

    def _Map(self, ast):
        fn = self.visit(ast.function())
        args = self.visit(ast.inputs())
       
        # The shape of each element is controlled by the output of the
        # function fn
        input_elements = [elementof(x) for x in args]
        
        element, constraints = fn(*input_elements)

        self.constraints.extend(constraints)
        for x in args[1:]:
            self.constraints.append(eq(args[0], x))

        # The extent of the output is controlled by the inputs
        ast.shape = Shape(extents=extentof(args[0]), element=element)
        return ast.shape

    def _Procedure(self, ast):
        formals = [v.id for v in U.flatten(ast.formals())]
        fntype = ast.name().type
        if isinstance(fntype, T.Polytype):
            fntype = fntype.monotype()

        if self.verbose:
            print "PROCEDURE", ast.name(), "::", fntype

        # We collect the returned shapes from all return sites.  They
        # may or may not match in a correct program, so we may or may
        # not be able to say much about the return shape of this
        # procedure.
        self.return_shapes = []

        # During inference we will collect a (possibly empty) set of
        # shape contraints.  For each procedure, we will collect all
        # constraints implied by its body on its arguments, and export
        # them for the use of calling procedures.
        self.constraints = []

        # Special shape function to handle recursive calls.  We can only
        # successfully compute the returned shape when the shapes of the
        # parameters to the recursive invocation match the originals.
        def recursive(*actuals):
            if all(map(match_formal, formals, actuals)):
                return (Identity, [])
            else:
                raise NotImplementedError("recursion must preserve sizes")

        self.context.locals.begin_scope()
        for x in formals:
            self.context.locals[x] = ShapeOf(x)
        self.context.locals[ast.name().id] = recursive

        for stmt in ast.body():
            self.visit(stmt)
        self.context.locals.end_scope()

        

        if self.verbose:
            for s in self.return_shapes:
                print "... returns", s

            for c in self.constraints:
                print "... with constraint", c

        def find_shapes(x):
            if isinstance(x, BaseShape) or not hasattr(x, '__iter__'):
                yield x
            else:
                for xi in x:
                    if isinstance(xi, BaseShape) or not hasattr(x, '__iter__'):
                        yield xi
                    else:
                        for i in find_shapes(iter(xi)):
                            yield i
        
        for x in self.return_shapes[1:]:
            for xi, yi in zip(find_shapes(x), find_shapes(self.return_shapes[0])):
                self.constraints.append(eq(xi, yi))
        
        # The extent of the output is controlled by the inputs
        return_shape = self.return_shapes[0]
        constraints = self.constraints

        if self.verbose:
            print "... final return shape =", return_shape
            print "... final constraints  =", constraints

        def S(*actuals):
            if self.verbose:
                print "  instancing %s for procedure %s" % \
                         (return_shape, ast.name())
                print "  formals = ", formals
                print "  actuals = ", actuals
            shape = instantiate(return_shape, dict(zip(formals, actuals)))
            if self.verbose:
                print "  instanced to: ", shape

            return (shape, constraints)

        self.context.locals[ast.name().id] = S

    def _Bind(self, ast):
        s = self.visit(ast.value())
        ast.binder().shape = s
        # For structured binding, the shape is unaffected by structure
        # So we just unpack all the destinations and assign them the same
        # shape.
        for name in U.flatten(ast.binder()):
            self.context.locals[name.id] = s

    def _Return(self, ast):
        s = self.visit(ast.value())
        self.return_shapes.append(s)
        ast.shape = s
        return s

    def _While(self, ast):
        raise NotImplementedError("while statements not yet supported")


def infer(ast, verbose=False, globals=None, context=None):
    ctx = context or ShapingContext(globals=globals)
    S = ShapeInference(context=ctx)
    S.verbose = verbose
    return S.visit(ast)

def collect_local_shapes(suite, M):
    def select(A, kind): return ifilter(lambda x: isinstance(x, kind), A)
    for fn in select(suite, AST.Procedure):
        if fn.name().id in M.toplevel:
            shapes = {}
            for bind in select(fn.parameters, AST.Bind):
                destination = bind.binder()
                for name in U.flatten(destination):                    
                    shapes[name.id] = destination.shape
            fn.shapes = shapes
            M.shapes = shapes
