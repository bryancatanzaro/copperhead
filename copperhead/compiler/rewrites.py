#
#   Copyright 2008-2012 NVIDIA Corporation
#  Copyright 2009-2010 University of California
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

"""Basic syntactic rewrites for Copperhead compiler.

This module implements the rewrite passes used by the Copperhead
compiler to transform the input program into a more easily analyzed
form.  The routines in this module assume that the syntax trees they are
given are well-formed, but they do not generally make any assumptions
about type soundness.

The rewrites provided by this module are fairly standard and operate on
the source program without consideration for any parallelism.

Supported rewrites include:

    o Closure conversion

    o Lambda lifting

    o Single assignment conversion
"""

import coresyntax as S
import pltools
from utility import flatten
import copy
import coretypes as T

class SourceGatherer(S.SyntaxRewrite):
    def __init__(self, globals):
        self.globals = globals
        self.env = pltools.Environment()
        self.clean = []
    def gather(self, suite):
        self.sources = []
        self.gathered = set()
        for stmt in suite:
            self.clean.append(self.rewrite(stmt))
        while self.sources:
            stmt = self.sources.pop(0)
            self.clean.insert(0, self.rewrite(stmt))
        return list(flatten(self.clean))
    def _Procedure(self, proc):
        proc_id = proc.name().id
        self.env[proc_id] = proc_id
        self.env.begin_scope()
        for param in flatten(proc.formals()):
            id = param.id
            self.env[id] = id
        self.rewrite_children(proc)
        self.env.end_scope()
        return proc
    def _Bind(self, bind):
        destination = bind.binder()
        if isinstance(destination, S.Tuple):
            for dest in flatten(destination):
                self.env[dest.id] = dest.id
        else:
            id = destination.id
            self.env[id] = id
        self.rewrite_children(bind)
        return bind
    def _Name(self, name):
        if not name.id in self.env:
            if name.id in self.globals:
                fn = self.globals[name.id]
                if hasattr(fn, 'syntax_tree') and \
                    fn.__name__ not in self.gathered:
                    self.sources.append(fn.syntax_tree)
                    self.gathered.add(fn.__name__)
        return name

def gather_source(stmt, M):
    gatherer = SourceGatherer(M.globals)
    gathered = gatherer.gather(stmt)
    return gathered

class IdentifierMarker(S.SyntaxRewrite):
    def __init__(self, globals):
        self.globals = globals
    def _Name(self, name):
        if name.id in self.globals:
            if hasattr(self.globals[name.id], 'syntax_tree'):
                #A user wrote this identifier
                return S.mark_user(name)
            else:
                return name
        else:
            return S.mark_user(name)
    def _Procedure(self, proc):
        self.rewrite_children(proc)
        proc.variables = map(S.mark_user, proc.variables)
        return proc
    def _Lambda(self, lamb):
        return self._Procedure(lamb)
    def _Bind(self, bind):
        self.rewrite_children(bind)
        bind.id = self.rewrite(bind.id)
        return bind
def mark_identifiers(stmt, M):
    marker = IdentifierMarker(M.globals)
    marked = marker.rewrite(stmt)
    #Rather than make core syntax deal sordidly with strings
    #Wrap them up here.
    def mark_user(x):
        return S.mark_user(S.Name(x)).id
    M.entry_points = map(mark_user, M.entry_points)
    for x in M.input_types.keys():
        M.input_types[mark_user(x)] = M.input_types[x]
        del M.input_types[x]
    return marked


class VariadicLowerer(S.SyntaxRewrite):
    def __init__(self):
        self.applies = set(['zip'])
        # XXX Do this for unzip as well
        self.binders = set(['unzip'])
    def _Map(self, ast):
        args = ast.parameters
        arity = len(args) - 1
        assert(arity > 0)
        return S.Apply(S.Name('map' + str(arity)),
                       args)
    def _Apply(self, ast):
        fn_id = ast.function().id
        if fn_id in self.applies:
            args = ast.arguments()
            arity = len(args)
            return S.Apply(S.Name(fn_id + str(arity)),
                         args)
        else:
            return ast
                        

def lower_variadics(stmt):
    rewriter = VariadicLowerer()
    lowered = rewriter.rewrite(stmt)
    return lowered

class SingleAssignmentRewrite(S.SyntaxRewrite):
    import itertools
    serial = itertools.count(1)

    def __init__(self, env, exceptions):
        self.env = pltools.Environment(env)
        self.exceptions = exceptions
        self.freeze = False
    def _Return(self, stmt):
        result = S.Return(S.substituted_expression(stmt.value(), self.env))
        return result
    def _Cond(self, cond):
        condition = S.substituted_expression(cond.parameters[0], self.env)
        self.rewrite_children(cond)
        return S.Cond(condition, cond.parameters[1], cond.parameters[2])
    def _While(self, cond):
        condition = S.substituted_expression(cond.parameters[0], self.env)
        self.freeze = True
        self.rewrite_children(cond)
        cond.parameters[0] = condition
        self.freeze = False
        return cond
    def _Bind(self, stmt):
        var = stmt.binder()
        varNames = [x.id for x in flatten(var)]
        operation = S.substituted_expression(stmt.value(), self.env)
        for name in varNames:
            if self.freeze:
                if name in self.env:
                    rename = self.env[name]
                elif name not in self.exceptions:
                    rename = '%s_%s' % (name, SingleAssignmentRewrite.serial.next())
                else:
                    rename = name
            elif name not in self.exceptions:
                rename = '%s_%s' % (name, SingleAssignmentRewrite.serial.next())
            else:
                rename = name
           
            self.env[name] = S.Name(rename)
        result = S.Bind(S.substituted_expression(var, self.env), operation)
        return result

    def _Procedure(self, stmt):
        self.env.begin_scope()
        for var in flatten(stmt.variables):
            self.env[var.id] = var

        result = self.rewrite_children(stmt)
        self.env.end_scope()

        return result


def single_assignment_conversion(stmt, env={}, exceptions=set()):
    'Rename locally declared variables so that each is bound exactly once'

    rewrite = SingleAssignmentRewrite(env, exceptions)
    return rewrite.rewrite(stmt)


class LambdaLifter(S.SyntaxRewrite):
    """
    Convert every expression of the form:
    
        lambda x1,...,xn: E 

    into a reference to a proceduce __lambdaN and add

        def __lambdaN(x1,...,xn): return E

    to the procedure list.

    This rewriter assumes that closure conversion has already been
    performed.  In other words, there are no freely occurring
    local variables in the body of the lambda expression.
    """

    def __init__(self):
        # Collect lifted Lambdas as Procedures 
        self.proclist = []
        self.names = pltools.name_supply(stems=['_lambda'], drop_zero=False)

    def _Lambda(self, e):
        fn = S.Name(self.names.next())

        self.rewrite_children(e)
        body = S.Return(e.parameters[0])
        self.proclist.append(S.Procedure(fn, e.variables, [body]))

        return fn

    def _Procedure(self, ast):
        # We explicitly interleave lifted lambda procedures with the
        # statements from which they come.  This guarantees correct
        # ordering of existing nested procedures with new
        # lambda-generated procedures.
        body = []
        for stmt in ast.parameters:
            stmt = self.rewrite(stmt)
            body = body + self.proclist + [stmt]
            self.proclist = []
        ast.parameters = body
        return ast

def lambda_lift(e):
    lift = LambdaLifter()
    eL = lift.rewrite(e)
    return lift.proclist + eL



class ProcedureFlattener(S.SyntaxRewrite):
    """
    Flatten the list of defined procedures so that no definition is
    nested within another procedure.  This should only be applied after
    closure conversion and lambda lifting are complete.
    """

    def __init__(self):
        self.toplevel = list()

    def _Procedure(self, e):
        self.rewrite_children(e)
        e.parameters = filter(lambda x: x is not None, e.parameters)
        self.toplevel.append(e)
        return None

    # XXX If things other than procedures become allowed as top-level
    #     forms, make sure that they are handled here.

def procedure_flatten(e):
    flattener = ProcedureFlattener()
    eF = flattener.rewrite(e)
    return flattener.toplevel

class _ClosureRecursion(S.SyntaxRewrite):
    # XXX Most of the code in this rewriter simply serves to track
    #     variables defined in the current scope.  That should be
    #     abstracted into a more generic base class that could be used
    #     elsewhere.
    def __init__(self, env):
        self.env = env

    def locally_bound(self, B):
        for v in flatten(B):
            self.env[v.id] = v.id

    def _Bind(self, ast):
        self.rewrite_children(ast)
        binders = [v for v in S.walk(ast.binder()) if isinstance(v, S.Name)]
        self.locally_bound(binders)
        return ast

    def _Lambda(self, ast):
        self.env.begin_scope()
        self.locally_bound(ast.formals())
        self.rewrite_children(ast)
        self.env.end_scope()
        return ast

    def _Procedure(self, ast):
        self.env.begin_scope()
        self.locally_bound(ast.variables)
        self.rewrite_children(ast)
        self.env.end_scope()
        return ast

    # XXX This rewrite rule -- coupled with the rule for _Procedure in
    #     _ClosureConverter -- is an ugly hack for rewriting calls to
    #     procedures.  We should find a more elegant solution!
    def _Name(self, ast):
        x = getattr(self.env, ast.id, None)
        if ast.id in self.env and isinstance(self.env[ast.id], S.Closure):
            return S.Closure(self.env[ast.id].variables, ast)
        else:
            return ast


class _ClosureConverter(_ClosureRecursion):

    def __init__(self, globals=None):
        self.globals = globals or dict()
        self.env = pltools.Environment()

    def _Lambda(self, e):
        _ClosureRecursion._Lambda(self, e)
        
        formals = [v.id for v in flatten(e.formals())]
        # Take the free variable list, stick it in a set to make sure we don't
        # duplicate a variable, and then put it back in a list to make sure
        # it's got a defined ordering, which sets don't have
        free = list(set([v for v in S.free_variables(e.body(), formals)
                        if v in self.env]))

        if free:
            bound = [S.Name("_K%d" % i) for i in range(len(free))]
            body = S.substituted_expression(e.body(), dict(zip(free, bound)))

            e.parameters = [body]
            e.variables = e.variables + bound

            return S.Closure([S.Name(x) for x in free], e)
        else:
            return e

    def _Procedure(self, ast):
        binders = [v.id for v in flatten(ast.variables)] # NOTE: this includes name

        _ClosureRecursion._Procedure(self, ast)

        # Take the free variable list, stick it in a set to make sure we don't
        # duplicate a variable, and then put it back in a list to make sure
        # it's got a defined ordering, which sets don't have
        free = list(set([v for v in S.free_variables(ast.body(), binders)
                        if v in self.env]))

        if free:
            bound = [S.Name("_K%d" % i) for i in range(len(free))]
            ast.variables = ast.variables + bound
            ast.parameters = S.substituted_expression(ast.parameters,
                                                      dict(zip(free, bound)))


            # Transform recursive calls of this procedure within its own body.
            recursive = _ClosureRecursion(self.env)
            self.env[ast.name().id] = S.Closure(bound,
                                                ast.name())
            ast.parameters = recursive.rewrite(ast.parameters)

            # Register rewrite for calls to this procedure in later
            # parts of the defining scope
            self.env[ast.name().id] = S.Closure([S.Name(x) for x in free],
                                                ast.name())
        # else:
#             self.locally_bound([ast.name()])

        return ast

def closure_conversion(ast, globals=None):
    """
    Detect and explicitly tag all variables in the given syntax tree
    which are lexically closed over by lambdas or nested procedure
    definitions.

    A variable occurring within a lambda/procedure is considered to form
    a closure if:

        - it is not bound as a formal parameter of the lambda/procedure

        - it is bound in the containing scope of the lambda/procedure

    Such variables are lifted into arguments to explicit "closure"
    forms, and are passed as explicit arguments to the nested
    lambda/procedure.

        e.g., lambda x: lambda y: x =>
              lambda x: closure([x], lambda y, _K0: _K0)

    Global variables (if any) defined in the globals parameter are never
    closed over, since they are  globally visible.

    The copperhead.interlude module provide a native Python
    implementation of the Copperhead closure() expression.
    """
    converter = _ClosureConverter(globals=globals)
    converted = converter.rewrite(ast)
    
    return converted


class ExpressionFlattener(S.SyntaxRewrite):
    def __init__(self):
        self.stmts = [list()]
        self.names = pltools.name_supply(stems=['e'], drop_zero=False)


    def top(self): return self.stmts[-1]
    def emit(self, ast): self.top().append(ast)
    def push(self):  self.stmts.append(list())
    def pop(self):
        x = self.top()
        self.stmts.pop()
        return x

    def _Lambda(self, ast):
        raise ValueError, "lambda's cannot be flattened (%s)" % e

    def _Name(self, ast): return ast
    def _Number(self, ast): return ast
    def _Closure(self, ast): return ast

    def _Expression(self, e):
        subexpressions = e.parameters
        e.parameters = []
        for sub in subexpressions:
            sub = self.rewrite(sub)
            # XXX It doesn't seem right to include Closure on this list
            #     of "atomic" values.  But phase_assignment breaks if I
            #     don't do this.
            if not isinstance(sub, (S.Name, S.Literal, S.Closure)):
                tn = S.Name(self.names.next())
                self.emit(S.Bind(tn, sub))
            else:
                tn = sub
            e.parameters.append(tn)
        return e

    def _Bind(self, stmt):
        e = self.rewrite(stmt.value())
        stmt.parameters = [e]
        self.emit(stmt)
        return stmt

    def _Return(self, stmt):
        e = self.rewrite(stmt.value())
        if isinstance(e, S.Name):
            # If we're returning one of the procedure formals unchanged,
            # we need to copy its value into a return variable.
            # Here is where we check:
            if e.id not in self.formals:
                #No need to copy value into a return variable
                stmt.parameters = [e]
                self.emit(stmt)
                return
        # If we're returning a tuple, we always copy the value into a return
        # variable.  We may undo this later on, for entry-point procedures.
        ret = S.Name("result")
        self.emit(S.Bind(ret, e))
        stmt.parameters = [ret]
        self.emit(stmt)

    def _Cond(self, stmt):
        test = self.rewrite(stmt.test())

        self.push()
        self.rewrite(stmt.body())
        body = self.pop()

        self.push()
        self.rewrite(stmt.orelse())
        orelse = self.pop()

        stmt.parameters = [test, body, orelse]
        self.emit(stmt)

    def _Procedure(self, stmt):
        self.push()
        self.formals = set((x.id for x in flatten(stmt.formals())))
        self.rewrite_children(stmt)
        self.formals = None
        body = self.pop()
        stmt.parameters = body
        self.emit(stmt)

    def _default(self, ast):
        if isinstance(ast, S.Expression):
            return self._Expression(ast)
        else:
            raise ValueError, "can't flatten syntax (%s)" % ast

def expression_flatten(s):
    flattener = ExpressionFlattener()
    flattener.rewrite(s)
    return flattener.top()

class LiteralCaster(S.SyntaxRewrite):
    def __init__(self, globals):
        self.globals = globals
    def _Procedure(self, proc):
        self.literal_names = set()
        self.rewrite_children(proc)
        return proc
    def _Bind(self, bind):
        if isinstance(bind.value(), S.Number):
            self.literal_names.add(bind.binder().id)
        self.rewrite_children(bind)
        return bind
    def _Apply(self, appl):
        #Rewrite children
        self.rewrite_children(appl)
        #Insert typecasts for arguments
        #First, retrieve type of function, if we can't find it, pass
        fn_obj = self.globals.get(appl.function().id, None)
        if not fn_obj:
            return appl
        #If the function doesn't have a recorded Copperhead type, pass
        if not hasattr(fn_obj, 'cu_type'):
            return appl
        fn_type = fn_obj.cu_type
        if isinstance(fn_type, T.Polytype):
            fn_input_types = fn_type.monotype().input_types()
        else:
            fn_input_types = fn_type.input_types()
        def build_cast(cast_name, args):
            "Helper function to build cast expressions"
            return S.Apply(S.Name(cast_name),
                           args)
        def insert_cast(arg_type, arg):
            "Returns either the argument or a casted argument"
            if hasattr(arg, 'literal_expr'):
                if arg_type is T.Int:
                    return build_cast("int32", [arg])
                elif arg_type is T.Long:
                    return build_cast("int64", [arg])
                elif arg_type is T.Float:
                    return build_cast("float32", [arg])
                elif arg_type is T.Double:
                    return build_cast("float64", [arg])
                elif isinstance(arg_type, str):
                    #We have a polymorphic function
                    #We must insert a polymorphic cast
                    #This means we search through the inputs
                    #To find an input with a related type
                    for in_type, in_arg in \
                        zip(fn_input_types, appl.arguments()):
                        if not hasattr(in_arg, 'literal_expr'):
                            if in_type == arg_type:
                                return build_cast("cast_to", [arg, in_arg])
                            elif isinstance(in_type, T.Seq) and \
                                in_type.unbox() == arg_type:
                                return build_cast("cast_to_el", [arg, in_arg])
            #No cast was found, just return the argument
            return arg
        casted_arguments = map(insert_cast, fn_input_types, appl.arguments())
        appl.parameters[1:] = casted_arguments
        #Record if this expression is a literal expression
        if all(map(lambda x: hasattr(x, 'literal_expr'), appl.arguments())):
            appl.literal_expr = True
        return appl
    def _Number(self, ast):
        ast.literal_expr = True
        return ast
    def _Name(self, ast):
        if ast.id in self.literal_names:
            ast.literal_expr = True
        return ast
    
def cast_literals(s, M):
    caster = LiteralCaster(M.globals)
    casted = caster.rewrite(s)
    #Inserting casts may nest expressions
    return expression_flatten(casted)

class ReturnFinder(S.SyntaxVisitor):
    def __init__(self, binding, env):
        self.binding = list(flatten(binding))
        self.env = env
    def _Return(self, node):
        val = list(flatten(node.value()))
        assert(len(val) == len(self.binding))
        for b, v in zip(self.binding, val):
            self.env[v.id] = b

class FunctionInliner(S.SyntaxRewrite):
    def __init__(self):
        self.activeBinding = None
        self.statements = []
        self.procedures = {}
    def _Bind(self, binding):
        self.activeBinding = binding.binder()
        self.rewrite_children(binding)
        self.activeBinding = None
        statements = self.statements
        self.statements = []
        if statements == []:
            return binding
        return statements
    def _Apply(self, apply):
        functionName = apply.parameters[0].id
        if functionName in self.procedures:
            instantiatedFunction = self.procedures[functionName]
            functionArguments = instantiatedFunction.variables[1:]
            instantiatedArguments = apply.parameters[1:]
            env = pltools.Environment()
            for (internal, external) in zip(functionArguments, instantiatedArguments):
                env[internal.id] = external
            return_finder = ReturnFinder(self.activeBinding, env)
            return_finder.visit(instantiatedFunction)
            statements = [S.substituted_expression(x, env) for x in \
                              instantiatedFunction.body() \
                              if not isinstance(x, S.Return)]

            singleAssignmentInstantiation = single_assignment_conversion(statements, exceptions=set((x.id for x in flatten(self.activeBinding))))
            self.statements = singleAssignmentInstantiation
            return None
        return apply
    
    def _Procedure(self, proc):
        self.rewrite_children(proc)
        proc.parameters = list(flatten(proc.parameters))
        
        procedureName = proc.variables[0].id
        self.procedures[procedureName] = proc
        return proc
    
def inline(s):
    inliner = FunctionInliner()
    return list(flatten(inliner.rewrite(s)))

def procedure_prune(ast, entries):
    needed = set(entries)

    # First, figure out which procedures we actually need by determining
    # the free variables in each of the entry points
    for p in ast:
        needed.update(S.free_variables(p.body()))

    # Now, only keep top-level procedures that have been referenced
    return [p for p in ast if p.name().id in needed]


class ConditionalProtector(S.SyntaxRewrite):
    """
    Convert every expression of the form:

        E1 if P else E2

    into the equivalent form:

        ((lambda: E1) if P else (lambda: E2))()

    The purpose of this rewriter is to protect the branches of the
    conditional during later phases of the compiler.  It guarantees that
    exactly one of E1/E2 will ever be evaluated.
    """

    def __init__(self):
        pass


    def _If(self, e):
        self.rewrite_children(e)

        test   = e.test()
        body   = S.Lambda([], e.body())
        orelse = S.Lambda([], e.orelse())

        e.parameters = [test, body, orelse]

        return S.Apply(e, [])
