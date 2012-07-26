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
import itertools
import inspect

class SourceGatherer(S.SyntaxRewrite):
    def __init__(self, globals):
        self.globals = globals
        import copperhead.prelude_impl as PI
        self.prelude_impl = set(dir(PI))
        self.PI = PI
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
                #Is this a prelude function implemented in Copperhead?
                if (fn.__name__ in self.prelude_impl):
                    #If it's a function or a builtin, override
                    #If it's not either, then the user has redefined
                    # a prelude function and we'll respect their wishes
                    if inspect.isbuiltin(fn) or \
                            inspect.isfunction(fn):
                        fn = getattr(self.PI, fn.__name__)
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
        self.locals = pltools.Environment()
        import copperhead.prelude_impl as PI
        self.prelude_impl = set(dir(PI))
        self.PI = PI
        self.bools = ["True", "False"]
    def _Name(self, name):
        if name.id in self.bools:
            return name
        if name.id in self.globals and name.id not in self.locals:
            if hasattr(self.globals[name.id], 'syntax_tree') \
                    or name.id in self.prelude_impl:
                #A user wrote this identifier or it's a non-primitive
                #part of the prelude - mark it
                return S.mark_user(name)
            else:
                return name
        else:
            return S.mark_user(name)
    def _Procedure(self, proc):
        self.locals.begin_scope()
        for x in proc.formals():
            #Tuples may be arguments to procedures
            #Mark all ids found in each formal argument
            for y in flatten(x):
                self.locals[y.id] = True
        self.rewrite_children(proc)
        self.locals.end_scope()
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
        self.arity = -1
    def _Bind(self, bind):
        if isinstance(bind.binder(), S.Tuple):
            self.arity = bind.binder().arity()
        self.rewrite_children(bind)
        return bind
    def _Map(self, ast):
        args = ast.parameters
        arity = len(args) - 1
        assert(arity > 0)
        return S.Apply(S.Name('map' + str(arity)),
                       args)
    def _Apply(self, ast):
        fn_id = ast.function().id
        arity = -1
        if fn_id in self.applies:
            args = ast.arguments()
            arity = len(args)
        elif fn_id in self.binders:
            arity = self.arity
            assert(arity > 0)
            
        if arity > 0:
            return S.Apply(S.Name(fn_id + str(arity)), ast.arguments())
        else:
            return ast
                        

def lower_variadics(stmt):
    rewriter = VariadicLowerer()
    lowered = rewriter.rewrite(stmt)
    return lowered

class SingleAssignmentRewrite(S.SyntaxRewrite):
    def __init__(self, env, exceptions, state):
        self.env = pltools.Environment(env)
        self.exceptions = exceptions
        if state:
            self.serial = state
        else:
            self.serial = itertools.count(1)

    def _Return(self, stmt):
        result = S.Return(S.substituted_expression(stmt.value(), self.env))
        return result
    def _Cond(self, cond):
        condition = S.substituted_expression(cond.parameters[0], self.env)
        self.env.begin_scope()
        body = self.rewrite(cond.body())
        self.env.end_scope()
        self.env.begin_scope()
        orelse = self.rewrite(cond.orelse())
        self.env.end_scope()
        return S.Cond(condition, body, orelse)
    def _Bind(self, stmt):
        var = stmt.binder()
        varNames = [x.id for x in flatten(var)]
        operation = S.substituted_expression(stmt.value(), self.env)
        for name in varNames:
            if name not in self.exceptions:
                rename = '%s_%s' % (name, self.serial.next())
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


def single_assignment_conversion(stmt, env={}, exceptions=set(), M=None):
    'Rename locally declared variables so that each is bound exactly once'
    state = None
    if M:
        state = getattr(M, 'single_conv_state', None)
    rewrite = SingleAssignmentRewrite(env, exceptions, state)
    rewritten = rewrite.rewrite(stmt)
    if M:
        M.single_conv_state = rewrite.serial
    return rewritten

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

    def _Apply(self, ast):
        #This catches the case where a procedure that is being
        #converted to a closure is recursive. In this case,
        #we don't make a new closure, we simply call the one
        #we've already got
        if not isinstance(ast.function(), S.Name):
            import pdb
            pdb.set_trace()
        proc_name = ast.function().id
        if proc_name in self.env and isinstance(self.env[proc_name], list):
            return S.Apply(ast.function(),
                           ast.arguments() + self.env[proc_name])
        return self.rewrite_children(ast)
    
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
        # self.proc_name = []
        
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
            self.env[ast.name().id] = bound
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

def expression_flatten(s, M):
    #Expression flattening may be called multiple times
    #Keep around the flattener name supply
    flattener = ExpressionFlattener()

    if hasattr(M, 'flattener_names'):
        flattener.names = M.flattener_names
    else:
        flattener = ExpressionFlattener()
        M.flattener_names = flattener.names
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
        #If function is a closure, pass
        fn = appl.function()
        if isinstance(fn, S.Closure):
            return appl
        fn_obj = self.globals.get(fn.id, None)
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
    return expression_flatten(casted, M)

class TupleNamer(S.SyntaxRewrite):
    def _Procedure(self, proc):
        names = pltools.name_supply(stems=['tuple'], drop_zero=False)
        disassembly = []
        def make_name(arg):
            if not isinstance(arg, S.Tuple):
                return arg
            else:
                made_name = S.Name(names.next())
                assembled = S.Bind(S.Tuple(*[make_name(x) for x in arg]),
                                   made_name)
                disassembly.insert(0, assembled)
                return made_name
        new_variables = map(make_name, proc.formals())
        return S.Procedure(proc.name(), new_variables, disassembly + proc.parameters)

def name_tuples(s):
    namer = TupleNamer()
    named = namer.rewrite(s)
    return named
    
class ReturnFinder(S.SyntaxVisitor):
    def __init__(self):
        #XXX HACK. Need to perform conditional statement->expression flattening
        #In order to inline properly. This dodges the issue.
        self.in_conditional = False
        self.return_in_conditional = False
    def _Cond(self, cond):
        self.in_conditional = True
        self.visit_children(cond)
        self.in_conditional = False
    def _Return(self, node):
        if self.in_conditional:
            self.return_in_conditional = True
            return
        self.return_value = node.value()
                                       
class FunctionInliner(S.SyntaxRewrite):
    def __init__(self, M):
        self.activeBinding = None
        self.statements = []
        self.procedures = {}
        self.M = M
    def _Bind(self, binding):
        self.activeBinding = binding.binder()
        self.rewrite_children(binding)
        self.activeBinding = None
        statements = self.statements
        self.statements = []
        if statements == []:
            return binding
        return statements
    def _Apply(self, appl):
        fn = appl.function()
        if isinstance(fn, S.Closure):
            fn_name = fn.body().id
        else:
            fn_name = fn.id
        if fn_name in self.procedures:
            instantiatedFunction = self.procedures[fn_name]
            functionArguments = instantiatedFunction.variables[1:]
            instantiatedArguments = appl.parameters[1:]
            if isinstance(fn, S.Closure):
                instantiatedArguments.extend(fn.variables)
            env = pltools.Environment()
            for (internal, external) in zip(functionArguments, instantiatedArguments):
                env[internal.id] = external
            return_finder = ReturnFinder()
            return_finder.visit(instantiatedFunction)
            #XXX HACK. Need to do conditional statement->expression conversion
            # In order to make inlining possible
            if return_finder.return_in_conditional:
                return appl
            env[return_finder.return_value.id] = self.activeBinding
            statements = filter(lambda x: not isinstance(x, S.Return),
                                instantiatedFunction.body())
            statements = [S.substituted_expression(x, env) for x in \
                              statements]
            singleAssignmentInstantiation = single_assignment_conversion(statements, exceptions=set((x.id for x in flatten(self.activeBinding))), M=self.M)
            self.statements = singleAssignmentInstantiation
            return None
        return appl
    def _Cond(self, cond):
        body = list(flatten(self.rewrite(cond.body())))
        orelse = list(flatten(self.rewrite(cond.orelse())))
        return S.Cond(cond.test(), body, orelse)
    
    def _Procedure(self, proc):
        self.rewrite_children(proc)
        proc.parameters = list(flatten(proc.parameters))
        
        procedureName = proc.variables[0].id
        self.procedures[procedureName] = proc
        return proc

class LiteralOpener(S.SyntaxRewrite):
    """
    It is possible that inlining has produced closures over literals.
    These closures can be pruned down by propagating literal values,
    and in some cases the closures can be eliminated completely.

    This pass performs this transformation to ensure that we never
    close over literals.  Removing this pass will cause assertion
    failures in the backend, which assumes closures are performed
    only over names."""
    def __init__(self):
        self.procedures = {}
        self.name_supply = pltools.name_supply(stems=['_'], drop_zero=False)
    def _Procedure(self, proc):
        self.propagated = []
        self.rewrite_children(proc)
        self.procedures[proc.name().id] = proc
        if self.propagated:
            return self.propagated + [proc]
        else:
            return proc
    def _Closure(self, c):
        closed_over_literal = any(map(lambda x: not isinstance(x, S.Name),
                                      c.closed_over()))
        if not closed_over_literal:
            return c
        #Find procedure being closed over
        proc_name = c.body().id
        proc = self.procedures[proc_name]
        proc_args = proc.formals()
        closed_args = c.closed_over()
        #Construct new set of arguments, with literals closed over removed
        replaced_args = proc_args[:-len(closed_args)]
        replaced_closed_over = []
        #Also record what replacements to make
        replacement = {}
        for orig_arg, closed_arg in zip(proc_args[-len(closed_args):],
                                        closed_args):
            if isinstance(closed_arg, S.Name):
                replaced_args.append(orig_arg)
                replaced_closed_over.append(closed_arg)
            else:
                replacement[orig_arg.id] = closed_arg
        #If we are only closing over literals, we will return a name
        #rather than a reduced closure. Check.
        fully_opened = len(replacement) == len(closed_args)
        replaced_stmts = [
            S.substituted_expression(si, replacement) \
                for si in proc.body()]
        replaced_name = S.Name(proc_name + self.name_supply.next())
        self.propagated.append(
            S.Procedure(
                replaced_name,
                replaced_args,
                replaced_stmts))
        if fully_opened:
            return replaced_name
        else:
            return S.Closure(replaced_closed_over,
                             replaced_name)

def procedure_prune(ast, entries):
    needed = set(entries)

    # First, figure out which procedures we actually need by determining
    # the free variables in each of the entry points
    for p in ast:
        needed.update(S.free_variables(p.body()))

    # Now, only keep top-level procedures that have been referenced
    return [p for p in ast if p.name().id in needed]

        
def inline(s, M):
    inliner = FunctionInliner(M)
    inlined = list(flatten(inliner.rewrite(s)))
    literal_opener = LiteralOpener()
    opened = list(flatten(literal_opener.rewrite(inlined)))
    return opened

class Unrebinder(S.SyntaxRewrite):
    """Rebindings like
    y = x
    or
    y0, y1 = x0, x1
    Can be eliminated completely.
    This is important for the backend, as phase inference and
    containerization assume that there is a unique identifier
    for every use of a variable.
    It's also inefficient to rebind things unnecessarily.
    This pass removes extraneous rebindings.
    """
    def __init__(self):
        self.env = pltools.Environment()

    def recursive_record(self, lhs, rhs):
        if isinstance(lhs, S.Name) and isinstance(rhs, S.Name):
            #Simple rebind
            self.env[lhs.id] = rhs.id
            return True
        elif isinstance(lhs, S.Tuple) and isinstance(rhs, S.Tuple):
            #Compound rebind:
            #Do not mark as extraneous unless all components are
            recorded = True
            for x, y in zip(lhs, rhs):
                recorded = self.recursive_record(x, y) and recorded
            return recorded
        else:
            return False
                
    def _Bind(self, b):
        self.rewrite_children(b)
        lhs = b.binder()
        rhs = b.value()

        extraneous = self.recursive_record(lhs, rhs)
        if extraneous:
            return None
        else:
            return b

    def _Name(self, n):
        if n.id in self.env:
            n.id = self.env[n.id]
        return n
        
    def rewrite_suite(self, suite):
        rewritten = map(self.rewrite, suite)
        return filter(lambda xi: xi is not None, rewritten)
            
    def _Procedure(self, p):
        stmts = self.rewrite_suite(p.body())
        p.parameters = stmts
        return p

    def _Cond(self, cond):
        body = self.rewrite_suite(cond.body())
        orelse = self.rewrite_suite(cond.orelse())
        return S.Cond(cond.test(), body, orelse)

def unrebind(ast):
    rewritten = Unrebinder().rewrite(ast)
    return rewritten
    
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

class ArityChecker(S.SyntaxVisitor):
    def _Tuple(self, tup):
        self.visit_children(tup)
        if tup.arity() > 10:
            raise SyntaxError, 'Tuples cannot have more than 10 elements'
    def _Procedure(self, proc):
        self.visit_children(proc)
        if len(proc.formals()) > 10:
            raise SyntaxError, 'Procedures cannot have more than 10 arguments'
        
    
def arity_check(ast):
    ArityChecker().visit(ast)
    
class ReturnChecker(S.SyntaxVisitor):
    def suite_must_return(self, suite, error):
        if not isinstance(suite[-1], S.Return):
            raise SyntaxError, error
    def _Cond(self, cond):
        cond_error = 'Both branches of a conditional must end in a return'
        def check_cond_suite(suite):
            if isinstance(suite[0], S.Cond):
                self.visit_children(suite[0])
                if len(suite) != 1:
                    raise SyntaxError, cond_error
            else:
                self.suite_must_return(suite, cond_error)
        check_cond_suite(cond.body())
        check_cond_suite(cond.orelse())
    def _Procedure(self, proc):
        proc_error = 'A procedure must end in a return'
        last = proc.body()[-1]
        if isinstance(last, S.Cond):
            self.visit_children(proc)
        else:
            self.suite_must_return(proc.body(), proc_error)

def return_check(ast):
    ReturnChecker().visit(ast)

class BuiltinChecker(S.SyntaxVisitor):
    def __init__(self):
        import copperhead.prelude as P
        self.builtins = set(
            filter(lambda n: n[0] != '_', dir(P)))
    def _Procedure(self, proc):
        name = proc.name().id
        if name in self.builtins:
            raise SyntaxError, '%s is a builtin to Copperhead and cannot be redefined' % name

def builtin_check(ast):
    BuiltinChecker().visit(ast)
