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

"""
Simple Hindley-Milner style type inference
"""

import itertools
import coresyntax as AST
import coretypes as T

from pltools import Environment, resolve, name_supply, resolution_map
from unifier import unify

class InferenceError(TypeError):
    "An error occuring during type inference"
    def __init__(self, msg): TypeError.__init__(self, msg)

class TypingContext:
    """
    Encapsulate information passed around during type inference.
    Binding everything up in a structure like this makes it possible to
    add optional annotations without rewriting the parameter lists of
    all procedures involved in inference.
    """

    def __init__(self, 
                 typings=None,       # typings of bound identifiers
                 environment=None,   # mapping of type names to types
                 tvsupply=None,
                 globals=None):

        # Initialize environment with empty mappings if none provided
        self.typings = typings or Environment()
        self.environment = environment or Environment()

        # Since Copperhead is embedded within Python, we need
        # visibility into a Python procedures namespace to resolve what
        # it's identifiers are referring to.  This is where we record
        # that global namespace (if provided).  It should never be modified.
        self.globals = globals or dict()

        # Supply of unique type variable names: #tv0, #tv1, #tv2, ...
        # They're known to be unique because they are illegal identifiers.
        self._tvsupply = tvsupply or name_supply(['#tv'], drop_zero=False)


    ######################################################################
    #
    # The following methods provide convenient ways of working with the
    # state encapsulated in the TypingContext
    #

    def fresh_typevar(self):
        return T.Typevar(self._tvsupply.next())

    def fresh_typevars(self, n):
        return [T.Typevar(n) for n in itertools.islice(self._tvsupply, n)]

    def fresh_typelist(self, keys, varmap=None):
        tvs = [self.fresh_typevar() for k in keys]
        if varmap:
            for k, v in zip(keys, tvs):
                varmap[k] = v
        return tvs

    def begin_scope(self):
        self.typings.begin_scope()
       
    def end_scope(self):
        self.typings.end_scope()

    ######################################################################
    #
    # Following are the methods required by the unification interface.
    #
       
    def instantiate(tcon, t):
        'Instantiate Polytypes as Monotypes with fresh type variables'
        if isinstance(t, T.Polytype):
            vars = tcon.fresh_typelist(t.variables)
            return T.substituted_type(t.monotype(),
                                      dict(zip(t.variables, vars)))
        else:
            return t

    def resolve_variable(tcon, t):
        'Resolve current mapping (if any) of typevars'
        if isinstance(t, T.Typevar):  return resolve(t, tcon.environment)
        else:                         return t

    def occurs_check(tcon, t1, t2):
        if T.occurs(t1, t2):
            raise InferenceError, "%s occurs in %s" % (t1,t2)

    def is_variable(tcon, t):  return isinstance(t, T.Typevar)

    def error(tcon, msg):  raise InferenceError, msg


def resolve_type(t, tcon):
    """
    Return a new type where all type variables occurring free in T and
    bound in the given context are resolved to their bound values.
    """

    free = list(T.free_in_type(t))
    if not free:  return t

    R = resolution_map(free, tcon.environment)
    for v in free:
        if R[v] != v:
            R[v] = resolve_type(R[v], tcon)

    return T.substituted_type(t, R)



def normalize_type(t, tcon):
    """
    Return a normalized type that eliminates all free variables from 't'.
    Any type variables bound in the given context will be resolved, and
    all remaining variables will be quantified with a Polytype.
    """
    return T.quantify_type(resolve_type(t, tcon))


class ExpressionTyper(AST.SyntaxVisitor):

    def __init__(self, context=None):
        self.tcon = context or TypingContext()

    def __call__(self, tree): return self.visit(tree)

    def _Number(self, ast):
        if isinstance(ast.val, int):     return T.Int
        elif isinstance(ast.val, float): return T.Float
        else:                            return T.Number

    def _Name(self, ast):
        # First, look in the local typings environment
        if ast.id in self.tcon.typings:
            return self.tcon.typings[ast.id]

        # If not there and not in globals, then it's unknown
        elif ast.id not in self.tcon.globals:
            raise InferenceError, "%s has no known type" % ast.id

        # For global objects, try both 'cu_type' slot.
        obj = self.tcon.globals[ast.id] 
        t = getattr(obj, 'cu_type', None)

        # XXX TODO: recursively invoke type inference on CuFunctions if
        # their cu_type has not yet been defined.
        if isinstance(t, T.Type):         return t
        else:
            raise InferenceError, "%s has no known type" % ast.id

    def _If(self, ast):
        tb = self.visit(ast.test())
        t1 = self.visit(ast.body())
        t2 = self.visit(ast.orelse())
        try:
            unify(tb, T.Bool, self.tcon)
        except InferenceError, exn:
            raise InferenceError, "conditional must have boolean test %s" % ast
        unify(t1, t2, self.tcon)

        return t1

    def _Index(self, ast):  return self.visit(ast.value())

    def _Subscript(self, ast):
        valtype = self.visit(ast.value())
        idxtype = self.visit(ast.slice())
        restype = self.tcon.fresh_typevar()

        unify(idxtype, T.Int, self.tcon)
        unify(valtype, T.Seq(restype), self.tcon)

        return restype

    def _Apply(self, ast):
        fntype = self.visit(ast.function())
        argtypes = self.visit(ast.arguments())
        restype = self.tcon.fresh_typevar()
        inst = T.Fn(T.Tuple(*argtypes), restype)


        unify(fntype, inst, self.tcon)

        outputType = resolve_type(restype, self.tcon)

        return outputType

    def _Closure(self, ast):
        fnType = self.visit(ast.parameters[0])
        if isinstance(fnType, T.Polytype):
            fnTypeM = fnType.monotype()
        else:
            fnTypeM = fnType
        closedArgTypes = self.visit(ast.variables)
        fnArgTypes = fnTypeM.parameters[0].parameters
        closedFnArgTypes = fnArgTypes[-len(closedArgTypes):]
        openFnArgTypes = fnArgTypes[0:-len(closedArgTypes)]
        items = [self.tcon.fresh_typevar() for x in fnArgTypes]
        for t, item in zip(items[-len(closedArgTypes):], closedArgTypes):
            unify(t, item, self.tcon)
        restype = self.tcon.fresh_typevar()
        unify(fnType, T.Fn(T.Tuple(*items), restype), self.tcon)
        outputType = T.Fn(T.Tuple(*items[0:-len(closedArgTypes)]), restype)
        return resolve_type(outputType, self.tcon)
    
    def _Map(self, ast):
        fn, args = ast.parameters[0], ast.parameters[1:]

        # Type the function we're applying
        fntype = self.visit(fn)
        argtypes = self.visit(args)
        items = [self.tcon.fresh_typevar() for x in argtypes]
        for t, seq in zip(items, argtypes):
            unify(T.Seq(t), seq, self.tcon)
            
        # Unify function type with element types
        restype = self.tcon.fresh_typevar()
        unify(fntype, T.Fn(T.Tuple(*items), restype), self.tcon)
        return T.Seq(resolve(restype, self.tcon.environment))

   
    
    
    def _Lambda(self, ast):
        argnames = [arg.id for arg in ast.variables]
        body = ast.parameters[0]

        self.tcon.begin_scope()
        argtypes = self.tcon.fresh_typelist(argnames, self.tcon.typings)
        restype = self.visit(body)
        self.tcon.end_scope()

        return resolve_type(T.Fn(T.Tuple(*argtypes), restype), self.tcon)

    def _And(self, ast):
        argtypes = self.visit(ast.parameters)
        try:
            for t in argtypes:
                unify(t, T.Bool, self.tcon)
        except InferenceError, exn:
            raise InferenceError, "logic expression requires boolean arguments %s" % ast
        return T.Bool

    def _Or(self, ast): return self._And(ast)

    def _Tuple(self, ast):
        return T.Tuple(*self.visit(ast.parameters))


def infer_expression_type(e, tenv=None):
    tcon = TypingContext(typings=tenv)
    typer = ExpressionTyper(tcon)
    return normalize_type(typer(e), tcon)

def collect_types(values):
    return itertools.ifilter(lambda x: isinstance(x,T.Type), values)


class TypeEngine(AST.SyntaxVisitor):

    def __init__(self,
                 idtypes=None,          # mapping identifiers -> types
                 visible_types=None,    # visible type names
                 entrytypes = {},       # XXX HACK for type instantiation
                 globals=None):

        self.entrytypes = entrytypes
        self.tcon = TypingContext(typings=idtypes,
                                  environment=visible_types,
                                  globals=globals)

        self.typer = ExpressionTyper(self.tcon)

    def __call__(self, tree): return self.visit(tree)

    def expression_type(self, e):
        # NOTE: performing resolution here is optional, but should
        #       reduce the proliferation of temporary type variables in
        #       intermediate stages
        #return self.typer(e)
        return resolve_type( self.typer(e), self.tcon )

    def _Return(self, ast):
        return self.expression_type(ast.value())

    def _Bind(self, ast):
        type = self.expression_type(ast.value())

        # Convert functional types to polytypes as necessary
        if isinstance(type, T.Fn):
            free = [v for v in AST.free_variables(ast.value(), {})]
            bound = [self.tcon.typings[id] for id in free]
            bound = [resolve(t, self.tcon.environment) for t in bound]
            type = T.quantify_type(type, bound)

        ##This code allows us to unpack tuples returned from functions
        def match(binder, type):
            if isinstance(binder, AST.Name):
                self.tcon.typings[binder.id] = type
            elif isinstance(binder, AST.Tuple):
                if len(binder.parameters)!=len(type.parameters):
                    raise InferenceError, \
                            "binder and value don't match (%s)" % ast
                for b, t in zip(binder.parameters, type.parameters):
                    match(b, t)
            else:
                raise InferenceError, "illegal binder (%s)" % ast

        match(ast.binder(), type)

        return None

    def typeBlock(self, block):
        restype = self.tcon.fresh_typevar()
        for stmt in block:
            t = self.visit(stmt)
            if isinstance(t, T.Type):
                unify(restype, t, self.tcon)
        def buildType(binder):
            if isinstance(binder, AST.Name):
                id = binder.id
                type = self.tcon.typings[id]
                return type
            subtypes = [buildType(x) for x in binder.parameters]
            tupleType = T.Tuple(*subtypes)
            return tupleType
        for stmt in block:
            if isinstance(stmt, AST.Bind):
                indicatedType = buildType(stmt.binder())
                normalizedType = normalize_type(indicatedType, self.tcon)
                stmt.type = normalizedType
        return restype
    
    def _While(self, ast):
        conditionType = self.expression_type(ast.parameters[0])
        unify(conditionType, T.Bool, self.tcon)
        bodyType = self.typeBlock(ast.parameters[1])
        return bodyType
                
                
    def _Cond(self, ast):
        tb = self.expression_type(ast.test())
        try:
            unify(tb, T.Bool, self.tcon)
        except InferenceError, exn:
            raise InferenceError, "conditional must have boolean test %s" % ast

        # both branches must end in a return and must agree in type
        t1 = self.visit(ast.body())[-1]
        t2 = self.visit(ast.orelse())[-1]
        unify(t1, t2, self.tcon)

        return t1

    def _Procedure(self, ast):

        vars = [v.id for v in ast.variables]
        id, args = vars[0], vars[1:]
        body = ast.parameters
        self.tcon.begin_scope()
        if id in self.entrytypes:
            #print('%s id found' % id,)
            argtypes = self.entrytypes[id]
            for argName, argType in zip(args, argtypes):
                self.tcon.typings[argName] = argType
        else:
        # (1) Create type variables for each argument and the result
            argtypes = self.tcon.fresh_typelist(args, self.tcon.typings)


        restype = self.typeBlock(ast.parameters)

        procvariables = list(self.tcon.typings.iterkeys())        
        proctypes = [normalize_type(self.tcon.typings[x], self.tcon) for x in procvariables]
        proctypes = dict(zip(procvariables, proctypes))
        ast.typings = proctypes
        self.tcon.end_scope()

        # (3) Build the type of the function
        fntype = normalize_type(T.Fn(argtypes, restype), self.tcon)
        # (4) Augment the environment
        self.tcon.typings[id] = fntype
        ast.type = fntype
        #if id in self.entrytypes:
        #    print(', type found: %s' % fntype.__str__())
        return None
