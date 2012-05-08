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

"""
Second-generation type inference module

This module implements a new approach to type inference.  It separates
the inference process into separate (I) constraint generation and (II)
constraint solution phases.
"""

from itertools import ifilter, chain, islice
import coresyntax as AST
import coretypes as T
from pltools import Environment, resolve, name_supply, resolution_map
from utility import flatten

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
                 globals=None,
                 tvsupply=None):

        # Record the global Python namespace (if any) in which code is
        # defined.  This maps identifiers to values; by convention
        # Copperhead objects have a 'cu_type' slot that we will
        # use for typing information.
        #
        # This dictionary should never be modified.
        self.globals = globals or dict()

        # Supply of unique type variable names: #tv0, #tv1, #tv2, ...
        # They're known to be unique because they are illegal identifiers.
        self._tvsupply = tvsupply or name_supply(['#tv'], drop_zero=False)

        # The typings environment maps local identifiers to their
        # corresponding types.
        self.typings = Environment()

        # Type variables associated with formal parameters, both in
        # lambdas and procedures, are required to be monomorphic.  This
        # set records all introduced type variables that are known to be
        # monomorphic.  Since we make all internal variables unique, we
        # only need a single set for this, rather than a hierarchical
        # environment structure as with self.typings.
        self.monomorphs = set()

        # During inference, we accumulate a set of freely occurring
        # identifiers.  This set contains AST nodes, rather than names.
        # Thus, multiple occurrences of a given name (e.g., 'x') may be
        # found in this set if they occurred at separate source
        # locations.  For example, the expression 'x+x' will introduce
        # two free occurrences, not one.
        self.free_occurrences = set()

        # The inference system builds up a set of assumptions about the
        # types of identifiers occurring outside the current compilation
        # unit and which have no known 'cu_type' attribute.
        # This table provides a mapping from type variables created for
        # free occurrences to the AST node at which they occur.
        self.assumptions = dict()


    ######################################################################
    #
    # The following methods provide convenient ways of working with the
    # state encapsulated in the TypingContext
    #

    def fresh_typevar(self):
        return T.Typevar(self._tvsupply.next())

    def fresh_typevars(self, n):
        return [T.Typevar(n) for n in islice(self._tvsupply, n)]

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

    def assuming(self, t, ast):
        assert t not in self.assumptions.keys()
        self.assumptions[t] = ast
        self.free_occurrences.add(ast)
        # XXX because monomorphs is scoped within functions, we treat
        # assumptions specially when generalizing bindings rather than
        # accumulating them here
        #self.monomorphs.add(t)

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
        if isinstance(t, T.Typevar): return resolve(t, tcon.typings)
        else: return t
        
    def occurs_check(tcon, t1, t2):
        if T.occurs(t1, t2):
            raise InferenceError, "%s occurs in %s" % (t1,t2)

    def is_variable(tcon, t):  return isinstance(t, T.Typevar)

    def error(tcon, msg):  raise InferenceError, msg


def resolve_type(t, S):
    """
    Return a new type where all type variables occurring free in t
    are replaced with their values under the substitution S.  If t has
    no free variables, it is returned unchanged.
    """
    
    free = list(T.free_in_type(t))
    if not free:  return t

    R = resolution_map(free, S)
    for v in free:
        if R[v] != v:
            R[v] = resolve_type(R[v], S)

    return T.substituted_type(t, R)

class TypeLabeling(AST.SyntaxVisitor):
    """
    Perform a traversal of an AST, labeling each with a new type variable.
    """

    def __init__(self, context):
        self.context = context
        self.verbose = False

    def _Lambda(self, node):
        if self.verbose:  print "Labeling: ", node
        self.visit(node.formals())
        self._default(node)

    def _Procedure(self, node):
        if self.verbose:  print "Labeling procedure:", node.name()
        self.visit(node.name())
        self.visit(node.formals())
        self.visit(node.body())

    def _Bind(self, node):
        if self.verbose:  print "Labeling:", node
        self.visit(node.binder())
        self.visit(node.value())


    def _default(self, node):
        t = self.context.fresh_typevar()
        t.source = node
        node.type = t
        if self.verbose:
            print "...", t, ":>", node
        self.visit_children(node)

class TypeResolution(AST.SyntaxVisitor):
    """
    Traverse the AST and resolve the types bound to the type variables
    associated with every node.
    """

    def __init__(self, solution, quantified):
        self.solution = solution
        self.quantified = quantified

    def _Lambda(self, node):
        self._default(node)
        self.visit(node.formals())

    def _Procedure(self, node):
        self.visit(node.body())
        self.visit(node.name())
        self.visit(node.formals())

    def _Bind(self, node):
        self.visit(node.value())
        self.visit(node.binder())

    def _default(self, node):
        if getattr(node, 'type', None):
            node.type = resolve_type(node.type, self.solution)
            #
            # XXX all quantification should normally have happened
            # already by this point (i.e., in the solver) solver by this
            # point
            #node.type = T.quantify_type(node.type, self.solution, self.quantified)
        self.visit_children(node)

class Constraint(object): pass

class Equality(Constraint):

    def __init__(self, lhs, rhs, source=None):
        self.parameters = (lhs, rhs)
        self.source = source

    def __str__(self):
        lhs, rhs = self.parameters
        return str(lhs)+" == "+str(rhs)

class Generalizing(Constraint):

    def __init__(self, t1, t2, monomorphs, source=None):
        self.parameters = (t1, t2, monomorphs)
        self.source = source

    def __str__(self):
        return "Generalizing(%s, %s, %s)" % self.parameters

class ClosedOver(Constraint):

    def __init__(self, t, closed, body, source=None):
        self.parameters = (t, closed, body)
        self.source = source

    def __str__(self):
        return "ClosedOver(%s, %s, %s)" % self.parameters



class ConstraintGenerator(AST.SyntaxFlattener):

    def __init__(self, context=None):

        self.context = context or TypingContext()

    def _Number(self, ast):
        if isinstance(ast.val, int):
            yield Equality(ast.type, T.Long, ast)
        elif isinstance(ast.val, float):
            yield Equality(ast.type, T.Double, ast)
        
    def _Name(self, ast):
        if ast.id is 'True' or ast.id is 'False':
            yield Equality(ast.type, T.Bool, ast)
        elif ast.id is 'None':
            yield Equality(ast.type, T.Void, ast)

        elif ast.id in self.context.typings:
            yield Equality(ast.type,
                       self.context.instantiate(self.context.typings[ast.id]),
                       ast)

        elif ast.id in self.context.globals:
            obj = self.context.globals[ast.id]
            t = getattr(obj, 'cu_type', None)

            if isinstance(t, T.Type):
                yield Equality(ast.type, self.context.instantiate(t), ast)
            else:
                # This name has no known type at present.  Therefore, we
                # treat it as freely occurring (as below).
                self.context.assuming(ast.type, ast)

        else:
            # This was a freely occurring variable reference
            self.context.assuming(ast.type, ast)

    def _Tuple(self, ast):
        for c in self.visit_children(ast): yield c
        yield Equality(ast.type,
                       T.Tuple(*[x.type for x in ast.children()]),
                       ast)

    def _Index(self, ast):
        for c in self.visit_children(ast): yield c
        yield Equality(ast.type, ast.value().type, ast)

    def _Subscript(self, ast):
        for c in self.visit_children(ast): yield c
        yield Equality(ast.slice().type, T.Long, ast)
        yield Equality(T.Seq(ast.type), ast.value().type, ast)


    def _If(self, ast):
        for c in self.visit_children(ast): yield c

        tb, t1, t2 = ast.test().type, ast.body().type, ast.orelse().type
        yield Equality(tb, T.Bool, ast)
        yield Equality(t1, t2, ast)
        yield Equality(ast.type, t1, ast)

    def _And(self, ast):
        for x in ast.children():
            for c in self.visit(x): yield c
            yield Equality(x.type, T.Bool, ast)

        yield Equality(ast.type, T.Bool, ast)

    def _Or(self, ast): return self._And(ast)

    def _Apply(self, ast):
        for c in self.visit_children(ast): yield c
        fntype = ast.function().type
        argtypes = [x.type for x in ast.arguments()]

        yield Equality(fntype,
                       T.Fn(argtypes, ast.type),
                       ast)

    def _Map(self, ast):
        for c in self.visit_children(ast): yield c

        fn, args = ast.parameters[0], ast.parameters[1:]
        argtypes = [x.type for x in args]

        # Type variables that name the element types for each of the
        # argument sequences
        items = self.context.fresh_typelist(args)

        for itemtype, seqtype in zip(items, argtypes):
            itemtype.source = None
            yield Equality(seqtype, T.Seq(itemtype), ast)

        restype = self.context.fresh_typevar()
        restype.source = None

        yield Equality(fn.type, T.Fn(items, restype), ast)
        yield Equality(ast.type, T.Seq(restype), ast)

    def _Lambda(self, ast):
        restype = ast.parameters[0].type
        argnames = [arg.id   for arg in ast.variables]
        argtypes = [arg.type for arg in ast.variables]

        con = self.context
        con.begin_scope()
        con.typings.update(dict(zip(argnames, argtypes)))
        con.monomorphs.update(argtypes)

        for c in self.visit_children(ast): yield c

        self.context.end_scope()

        yield Equality(ast.type, T.Fn(argtypes, restype), ast)

    def _Closure(self, ast):
        for c in self.visit_children(ast): yield c

        yield ClosedOver(ast.type, ast.closed_over(), ast.body().type, ast)

    # ... statements ...

    def _Return(self, ast):
        for c in self.visit_children(ast): yield c
        yield Equality(ast.type, ast.value().type, ast)

    def _Bind(self, ast):
        # Constraints produced by the RHS
        for c in self.visit_children(ast): yield c

        # Make binders in the LHS visible in the typing environment
        bindings = [(node.id, node.type) for node in AST.walk(ast.binder())
                                         if isinstance(node, AST.Name)]

        self.context.typings.update(dict(bindings))

        # Generate destructuring constraints (if any) in the LHS
        for c in self.visit(ast.binder()): yield c

        # XXX We only allow polymorphic bindings when the LHS is a
        #     single identifier.  Generalizing this would be nice but is
        #     fraught with peril if the RHS is not required to have a
        #     syntactically equivalent structure to the LHS.
        if isinstance(ast.binder(), AST.Name):
            M = self.context.monomorphs
            yield Generalizing(ast.binder().type, ast.value().type, M, ast)
        else:
            yield Equality(ast.binder().type, ast.value().type, ast)



    def visit_block(self, ast, restype, block):
        for stmt in block:
            # Generate constraints from each statement in turn
            for c in self.visit(stmt): yield c

            # Any statement claiming to have a return type must return
            # the same type as all the others
            t_i = getattr(stmt, 'type', None)
            if t_i:
                yield Equality(restype, t_i, ast)

    def _Cond(self, ast):
        for c in self.visit(ast.test()): yield c
        yield Equality(ast.test().type, T.Bool, ast)

        for c in self.visit_block(ast, ast.type, ast.body()): yield c
        for c in self.visit_block(ast, ast.type, ast.orelse()): yield c

    def _Procedure(self, ast):
        con = self.context

        # Create a new type variable for the return type of the procedure
        restype = con.fresh_typevar()
        restype.source = None

        # Get the names and type variables for the formal parameters
        argnames = [arg.id   for arg in flatten(ast.formals())]
        argtypes = [arg.type for arg in flatten(ast.formals())]

        
        
        # Make the definition of this function visible within its body
        # to allow for recursive calls
        con.typings[ast.name().id] = ast.name().type

        con.begin_scope()

        # Make the formals visible
        con.typings.update(dict(zip(argnames, argtypes)))
        prior_monomorphs = con.monomorphs
        con.monomorphs = prior_monomorphs | set(argtypes)

        argtypeit = iter(argtypes)
        
        # Construct the formals types
        def make_type(formal):
            if hasattr(formal, '__iter__'):
                return T.Tuple(*[make_type(x) for x in iter(formal)])
            else:
                return argtypeit.next()

        formals_types = make_type(ast.formals())

        # XXX This makes the restriction that recursive invocations of F
        # in the body of F must have the same type signature.  Probably
        # not strictly necessary, but allowing the more general case
        # might prove problematic, especially in tail recursive cases.
        # For now, we will forbid this.
        con.monomorphs.add(ast.name().type)

        # Produce all constraints for arguments
        # Tuple arguments, for example, will produce constraints
        for a in ast.formals():
            for c in self.visit(a): yield c
        
        # Produce all the constraints for the body
        for c in self.visit_block(ast, restype, ast.body()): yield c

        con.monomorphs = prior_monomorphs

        M = set(self.context.monomorphs)
        yield Generalizing(ast.name().type,
                           T.Fn(formals_types, restype),
                           M,
                           ast)

        con.end_scope()


    # XXX Should probably segregate this elsewhere, since while blocks are
    #     only allowed internally
    def _While(self, ast):
        for c in self.visit(ast.test()): yield c
        yield Equality(ast.test().type, T.Bool, ast)

        for c in self.visit_block(ast, ast.type, ast.body()): yield c

class ConstrainInputTypes(AST.SyntaxFlattener):
    def __init__(self, input_types):
        self.input_types = input_types
        if input_types:
            self.entry_points = set(input_types.keys())
        else:
            self.entry_points = set()
    def _Procedure(self, ast):
        ident = ast.name().id
        if ident in self.entry_points:
            for formal, entry_type in zip(ast.formals(), self.input_types[ident]):
                yield Equality(formal.type, entry_type, formal)
    


class Solver1(object):

    def __init__(self, constraints, context):
        self.constraints = constraints
        self.context     = context

        # The solution that we're building is a substitution mapping
        # type variables into types.  Any type variable not contained in
        # the solution is assumed to map to itself
        self.solution = dict()

        # Keep track of type variables we generalize into quantifiers
        self.quantified = dict()

        # A single pass through the input constraints may produce a
        # solution, but some constraints may remain.  Here we collect
        # the remaining constraints for later solution.
        from collections import deque
        self.pending = deque()

        self.verbose = False

    def unify_variable(self, tvar, t):
        assert isinstance(tvar, T.Typevar)
        assert isinstance(t, T.Type)

        t = resolve_type(t, self.solution)
        self.context.occurs_check(tvar, t)

        #if len(list(T.free_in_type(t))) > 0:
        if False:
            self.pending.append(Equality(tvar, t, tvar.source))
        else:
            self.solution[tvar] = t


    def unify_monotypes(self, t1, t2):
        "Unify the given types, which must be either Monotypes or Variables"

        assert isinstance(t1, (T.Monotype, T.Typevar)) 
        assert isinstance(t2, (T.Monotype, T.Typevar)) 

        con = self.context

        # Resolve w.r.t. the current solution before proceeding
        t1 = resolve(t1, self.solution)
        t2 = resolve(t2, self.solution)
        if self.verbose: print "\tunifying", t1, "and", t2

        if t1==t2:                  pass
        elif con.is_variable(t1):   self.unify_variable(t1, t2)
        elif con.is_variable(t2):   self.unify_variable(t2, t1)

        else:
            t1 = self.context.instantiate(t1)
            t2 = self.context.instantiate(t2)
            if t1.name != t2.name or len(t1.parameters) != len(t2.parameters):
                con.error('type mismatch %s and %s' % (t1,t2))
           
            for (u,v) in zip(t1.parameters, t2.parameters):
                self.unify_monotypes(u, v)

    def generalize_binding(self, t1, t2, M):
        assert isinstance(t1, T.Typevar)
        assert isinstance(t2, (T.Typevar, T.Monotype))

        # Generalization occurs for identifiers introduced in
        # declaration statements.  This may, for instance, occur as the
        # result of a declaration:
        #     x = E
        # or a procedure definition
        #     def f(...): S
        #
        # The type variable t1 associated with the binder (e.g., 'x') is
        # allowed to become a Polytype.  We generate this polytype by
        # quantifying over the free variables of t2 that do not occur in
        # M.

        # NOTE: In general, we must be careful about when we can solve
        # Generalizing constraints.  In the current solver, where
        # constraints are generated in post-order, they can be solved
        # immediately.  In other orderings, they may need to be deferred
        # if they contain "active" type variables.

        r1 = resolve(t1, self.solution)
        r2 = resolve_type(t2, self.solution)

        if self.verbose:
            print "\tt1 =", t1
            print "\tt2 =", t2
            print "\tresolve(t1) =", r1
            print "\tresolve(t2) =", r2

        if r1 != t1 and isinstance(r2, T.Monotype):
            self.unify_monotypes(r1, r2)
            r2 = resolve_type(t2, self.solution)

        # The set of type variables associated with assumptions should
        # also be treated as monomorphs.  While we may have made
        # assumptions about a polymorphic function, we will have
        # generated a fresh type variable for each occurrence.  These
        # type variables are thus properly monomorphic.
        #
        R = set()
        for x in chain(M, self.context.assumptions.keys()):
            R |= set(T.names_in_type(resolve_type(x, self.solution)))

        if self.verbose:
            print "\tR =", R

        assert self.context.is_variable(t1)
        self.solution[t1] = T.quantify_type(r2, R, self.quantified)
        if self.verbose:
            print "\t--> Quantifying", t2, "to", self.solution[t1]


    def closing_over(self, t, closed, body):
        # NOTE: Like the Generalizing constraint, we're implicitly
        #       assuming that due to constraint ordering the body
        #       referenced here has already been solved.

        fntype = self.context.instantiate(resolve_type(body, self.solution))
        if not isinstance(fntype, T.Fn):
            raise InferenceError, "closure must be over functional types"

        # Get the list of argument types from the Fn type
        innertypes = fntype.input_types()

        # Partition the inner argument types into the outer types
        # exported to the world and those corresponding to the closed
        # arguments
        outertypes, closedtypes = innertypes[:-len(closed)], \
                                  innertypes[-len(closed):]

        for v, t2 in zip(closed, closedtypes):
            if self.verbose:
                print "\tt1 =", v.type
                print "\tt2 =", t2
            t1 = resolve(v.type, self.solution)
            t1 = self.context.instantiate(t1)
            if self.verbose:
                print "\tresolve(t1) =", t1
                print "\tresolve(t2) =", resolve_type(t2, self.solution)
            self.unify_monotypes(t1, t2)

        self.unify_monotypes(t, T.Fn(outertypes, fntype.result_type()))

    def compact_solution(self):
        for v, t in self.solution.items():
            self.solution[v] = resolve_type(t, self.solution)

        # note: could delete any self-mappings for additional cleansing

    def solve1(self, c):
        if self.verbose: print ".. solving", c, "\tfrom", c.source

        if isinstance(c, Equality):
            lhs, rhs = c.parameters
            self.unify_monotypes(lhs, rhs)

        elif isinstance(c, Generalizing):
            t1, t2, monomorphs = c.parameters
            self.generalize_binding(t1, t2, monomorphs)

        elif isinstance(c, ClosedOver):
            t, closed, body = c.parameters
            self.closing_over(t, closed, body)

        else:
            self.context.error("encountered unknown constraint type") 

    def solve(self):
        # (1) Process the initial constraint system
        for c in self.constraints:
            self.solve1(c)

        # (2) Process all remaining constraints
        while self.pending:
            self.solve1(self.pending.popleft())

        # (3) Clean-up the solution
        self.solution.update(self.quantified)
        self.compact_solution()

        if self.verbose:
            print
            print
            print "The solution is:"
            for v, t in self.solution.items():
                print "  %s == %s \t\t {%s}" % (v, t, getattr(v,'source',''))

def infer(P, verbose=False, globals=None, context=None, input_types=None):
    'Run type inference on the given AST.  Returns the inferred type.'
    tcon = context or TypingContext(globals=globals)
    # Label every AST node with a temporary type variable
    L = TypeLabeling(tcon)
    L.verbose = verbose
    L.visit(P)

    # Generate constraints from AST
    # And chain them to constraints from input_types
    C = chain(ConstrainInputTypes(input_types).visit(P),
              ConstraintGenerator(tcon).visit(P))


    S = Solver1(C,tcon)
    S.verbose = verbose

    S.solve()

    if verbose:
        print "\nThe free occurrence set is:"
        print "    ", tcon.free_occurrences
        print "\nThe assumption set is:"
        print "    ", tcon.assumptions

    if len(tcon.free_occurrences) > 0:
        undef = set([x.id for x in tcon.free_occurrences])
        raise InferenceError, 'undefined variables: %s' % list(undef)

    if len(tcon.assumptions) > 0:
        assumed = set(assumptions.values())
        raise InferenceError, 'unexplored assumptions: %s' % list(assumed)

    # Resolve the AST type slots to their solved types
    TypeResolution(S.solution, S.quantified).visit(P)

    # Resolve any outstanding variables in the typing context
    # XXX it's inefficient to go through the whole typings
    # environment when we could just record which ones were
    # introduced by the program P
    for id in tcon.typings:
        t = tcon.typings[id]
        if t in S.solution:
            tcon.typings[id] = resolve_type(t, S.solution)

    if isinstance(P, list):
        result = getattr(P[-1], 'type', T.Void)
    else:
        result = getattr(P, 'type', T.Void)
        
    # We quantify here to normalize the result of this procedure.
    # For instance, running the solver on a polymorphic expression like
    # "op_plus" will instantiate the polytype for op_plus with fresh
    # type variables.  While what we want internally, this is not what
    # external clients expect.
    return T.quantify_type(result)
