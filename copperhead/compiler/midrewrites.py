#
#  Copyright 2008-2009 NVIDIA Corporation
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

"""Mid-end rewrites for the Copperhead compiler.

This module implements the rewrite passes used by the Copperhead
compiler to transform the AST created by the front-end of the compiler
to an AST which represents the mid-end IR for the Copperhead
compiler.  The distinction between the two is that rewrites in the
front-end only transform the program to make it easier to compile, but
don't change the semantics.  The mid-end is allowed to change the
semantics of the program, doing things which are not legal in
Copperhead programs, but which are required for compilation.

For example:
  - Adding side effects to enable inplace operations (eliminating
  copies)
  - Phase analysis - to find synchronization points
  - Shape analysis - to find shapes of data
  - Data parallelization: How should the data be distributed and operated on?
     - D: Distributed (think of CTAs)
     - P: Parallel (think of a single CTA)
     - S: Sequential (think of a for loop)
  - Kernel Fission - for procedure calls which take multiple phases
  - Kernel Fusion - fusing together procedure calls with respect to
  data dependences
  - Other scheduling
  """

import coresyntax as S
import coretypes as T
from utility import flatten
import copy
import visitor as V
import intrinsics as I
import pltools as P
import midendsyntax as M
import codesnippets as C
import midtypes as MT
import pdb
from ..runtime import places as PL, cubox as CB, cufunction as CF
import phasetypes as PT
import cintrinsics as CI

class Iterizer(S.SyntaxRewrite):
    def __init__(self):
        pass
    def _Procedure(self, proc):
        self.currentProcedureName = proc.variables[0].id
        self.body = []
        self.currentBin = self.body
        self.tailRecursive = False
        self.rewrite_children(proc)
        if not self.tailRecursive:
            return proc
        def convertRecursion(stmt):
            if not isinstance(stmt, S.Bind):
                return stmt
            if not isinstance(stmt.id, S.Name):
                return stmt
            if stmt.id.id == C.anonymousReturnValue.id:
                arguments = proc.variables[1:]
                apply = stmt.parameters[0]
                assert apply.parameters[0].id == self.currentProcedureName
                newArguments = apply.parameters[1:]
                return [S.Bind(x, y) for x, y in zip(arguments, newArguments)] 
            else:
                return stmt
        recursiveBranch = list(flatten([convertRecursion(x) for x in self.recursiveBranch]))
        recursiveBranch = filter(lambda x: isinstance(x, S.Bind), recursiveBranch)
        whileLoop = M.While(self.condition, [recursiveBranch + self.body])
        cleanup = self.nonrecursiveBranch
        proc.parameters = self.body + [whileLoop] + cleanup
        return proc
        
    def _Bind(self, stmt):
        self.currentBin.append(stmt)
        self.binder = stmt.id
        self.rewrite_children(stmt)
        return stmt

    def _Return(self, stmt):
        self.currentBin.append(stmt)
        return stmt
    
    def _Apply(self, apply):
        applyFnName = apply.parameters[0].id
        if applyFnName == self.currentProcedureName:
            self.recursionDetected = True
            if not self.binder.id == C.anonymousReturnValue.id:
                raise SyntaxError('Recursive function detected, but recursion not in tail position')
        return apply
    def _Cond(self, cond):
        self.recursionDetected = False
        self.currentBranch = []
        self.currentBin = self.currentBranch
        thenBranch = self.rewrite(cond.parameters[1])
        if self.recursionDetected:
            self.recursiveBranch = self.currentBranch
            self.tailRecursive = True
            self.condition = cond.parameters[0]
        else:
            self.nonrecursiveBranch = self.currentBranch
        self.recursionDetected = False
        self.currentBranch = []
        self.currentBin = self.currentBranch
        elseBranch = self.rewrite(cond.parameters[2])
        if self.recursionDetected:
            self.recursiveBranch = self.currentBranch
            self.tailRecursive = True
            self.condition = S.Apply(S.Name('op_not'), [cond.parameters[0]])
        else:
            self.nonrecursiveBranch = self.currentBranch
        return cond

    
    
def remove_recursion(stmt):
    
    iterizer = Iterizer()
    return iterizer.rewrite(stmt)


class Sequentializer(S.SyntaxRewrite):
    def __init__(self, entry_points, p_hier):
        self.entry_points = entry_points
        self.context = I.unknown
        self.p_hier = p_hier
    def _Apply(self, apply):
        self.rewrite_children(apply)
        fn = apply.function()
        if hasattr(fn, 'context'):
            apply.context = fn.context()
        else:
            apply.context = self.context
        return apply
    def _Map(self, map):
        return self._Apply(map)
        
    def _Procedure(self, proc):
        name = proc.variables[0]
        if name.id in self.entry_points:
            proc.context = self.p_hier[0]
        else:
            proc.context = self.p_hier[1]
        self.context = proc.context
        self.rewrite_children(proc)
        
        return proc
        
def sequentialize(stmt, entry_points, p_hier):
    sequentializer = Sequentializer(entry_points, p_hier)
    sequentialized = sequentializer.rewrite(stmt)
    return sequentialized

class Unzipper(S.SyntaxRewrite):
    """This rewrite pass finds calls to zip and removes them,
    replacing all references to the result of the zip with a tuple
    of inputs."""
    def __init__(self, entry_points):
        self.entry_points = entry_points
    def _Bind(self, binder):
        self.binder = binder
        self.rewrite_children(binder)
        return self.binder
    def _Apply(self, ast):
        fn = ast.function()
        
        # XXX we should probably not key on function name here
        if fn.id[0:3] != 'zip':
            return ast
        
        binder = self.binder
        self.binder = S.Null()
        dest = binder.binder().id
        self.env[dest] = M.Zip(*(x for x in ast.arguments()))
        return S.Null()    
            
    def _Procedure(self, proc):
        if proc.name().id not in self.entry_points:
            return proc
        self.env = P.Environment()
        self.rewrite_children(proc)
        proc.parameters = S.stripNull(proc.parameters)
        proc = S.substituted_expression(proc, self.env)
        return proc

def unzip_transform(stmt, entry_points):
    unzipper = Unzipper(entry_points)
    unzipped = unzipper.rewrite(stmt)
    return unzipped

class VariantSelector(S.SyntaxRewrite):
    def __init__(self, globals):
        self.globals = globals
    def _Apply(self, apply):
        if apply.context is I.distributed:
            fn = apply.function()
            fn_name = str(fn)
            if fn_name in self.globals:
                global_fn = self.globals[fn_name]
                variants = global_fn.variants
                placed_variant = variants.get(PL.default_place, variants[PL.here])
                phase_fn = placed_variant.cu_phase
                fn.cu_phase = phase_fn
                if hasattr(CI, '_' + fn_name):
                    intrinsic = getattr(CI, '_' + fn_name)
                    scalar = getattr(intrinsic, 'scalar', False)
                    fn.scalar = scalar
                    return apply
                if isinstance(placed_variant, CB.CuBox):
                    fn.box = True
                elif not isinstance(placed_variant, CF.CuFunction):
                    fn.box = True
                    
                   
        return apply

def select_variant(stmt, globals):
    selector = VariantSelector(globals)
    return selector.rewrite(stmt)

class ScalarPlacer(S.SyntaxRewrite):
    """This rewrite rule decides whether scalar operations will be executed
    on the host, or at the place.  Since they can be executed at either place,
    we can choose."""
    def __init__(self):
        self.box_results = set()
        self.scalar_results = {}
    def _Bind(self, bind):
        destination = bind.binder()
        self.box = False
        self.scalar = False
        self.rewrite_children(bind)
        if self.box:
            self.box_results.update((str(x) for x in flatten(destination)))
        if self.scalar:
            for dest in flatten(destination):
                self.scalar_results[str(dest)] = bind.value()
        return bind
    def _Apply(self, apply):
        # Scalar intrinsics can be executed on the host or at the place
        # This rule will execute them on the host if any of the inputs to the
        # scalar intrinsic are coming from a boxed function
        # Justification for this rule is that boxed functions return their
        # results to the host, so having the host compute scalar functions
        # on the results of boxed functions makes sense.  On the other hand,
        # scalar intrinsics should be executed on the place if their inputs
        # are available on the place.
        def place_host(app):
            """This function places a scalar operation on the host.
            It also places any scalar operations which feed into this function
            on the host as well, recursively."""
            app.function().box = True
            app.function().host = True
            for x in app.arguments():
                if str(x) in self.scalar_results:
                    place_host(self.scalar_results[str(x)])
        
        if hasattr(apply.function(), 'box'):
            self.box = True
        scalar = hasattr(apply.function(), 'scalar') and apply.function().scalar
        if scalar:
            for x in apply.arguments():
                if str(x) in self.box_results:
                    place_host(apply)
                    self.box = True
                    break
                
        self.scalar = scalar
        return apply
    def _Procedure(self, proc):
        # XXX Phase analysis only works for parallel procedures - fix!
        if not hasattr(proc, 'context'):
            return proc
        if proc.context is not I.distributed:
            return proc
        self.box_results = set()
        self.rewrite_children(proc)
        return proc

class PhaseAnalyzer(S.SyntaxRewrite):
    def __init__(self, globals):
        self.globals = globals
        self.pre_box = None
    def _Closure(self, closure):
        closed = closure.closed_over()
        if self.sync:
            sync = self.sync
        else:
            sync = []
        for var in closed:
            id = var.id
            if id not in self.declarations:
                self.declarations[id] = PT.total
            elif self.declarations[id] is PT.unknown:
                self.declarations[id] = PT.total
            elif self.declarations[id] < PT.total:
                sync.append(id)
        if sync:
            self.sync = sync
        return closure
    def _Bind(self, bind):
        self.destination = bind.binder()
        self.sync = None
        self.box = False
        self.rewrite_children(bind)

        def record_sync():
            for variable in self.declarations.keys():
                if not variable in [x.id for x in flatten(self.destination)]:
                    self.declarations[variable] = PT.total
        

        if not self.sync and self.pre_box:
            record_sync()                    
            result = [M.PhaseBoundary(list(flatten(self.pre_box))), bind]
        elif self.sync:
            record_sync()
            result = [self.sync, bind]
        else:
            result = bind

        if self.box:
            self.pre_box = self.destination
        else:    
            self.pre_box = None
        return result
    def _Map(self, map):
        self.rewrite_children(map)
        if self.sync:
            sync = self.sync
        else:
            sync = []
        for input in map.inputs():
            id = input.id

            if self.declarations[id] is PT.unknown:
                self.declarations[id] = PT.local
            elif self.declarations[id] < PT.local:
                sync.append(id)
        if sync:
            self.sync = M.PhaseBoundary(sync)
        self.declarations[self.destination.id] = PT.local
        return map
    def _Apply(self, apply):
        for arg in apply.arguments():
            if isinstance(arg, S.Name):
                if arg.id not in self.declarations:
                    self.declarations[arg.id] = PT.none
        def name_filter(x):
            if isinstance(x, S.Name):
                return self.declarations[x.id]
            else:
                return PT.total
        completions = [name_filter(x) for x in apply.arguments()]
        fn_name = apply.function().id
        fn_phase = apply.function().cu_phase
        input_phases, output_completion = \
            fn_phase(*completions)
        for name in flatten(self.destination):
            self.declarations[name.id] = output_completion
        sync = []

       
                    
        
        if hasattr(apply.function(), 'box'):
            self.box = True
            for x in apply.arguments():
                # XXX Special case for dealing with zip -
                # The unzip transform pushes tuple args around the AST
                # This needs to be rethought
                # Right now it will only work for zip as argument to Box
                # functions.
                # The right thing to do is generate host code for this
                # But we need to transition the entire entry-point procedure
                # to C++ on the host in order to do this.
                
                if hasattr(x, '__iter__'):
                    for xi in x:
                        sync.append(xi.id)
                else:
                    sync.append(x.id)
                      
        else:
            for x, y in zip(apply.arguments(), input_phases):
                if isinstance(x, S.Name):
                    x_phase = self.declarations[x.id]
                    if x_phase is PT.unknown:
                        self.declarations[x.id] = y
                    elif x_phase < y:
                        sync.append(x.id)
        if sync:
            self.sync = M.PhaseBoundary(sync)
        return apply
    def _Procedure(self, proc):
        # XXX Phase analysis only works for parallel procedures - fix!
        if not hasattr(proc, 'context'):
            return proc
        if proc.context is not I.distributed:
            return proc

        self.declarations = dict(((x.id, PT.total) for x in proc.formals()))
        self.rewrite_children(proc)
        proc.parameters = list(flatten(proc.parameters))
        self.declarations = None
        #Construct phase for this procedure?
        return proc

def phase_analysis(stmt, globals):
    placer = ScalarPlacer()
    placed = placer.rewrite(stmt)
    analyzer = PhaseAnalyzer(globals)
    rewritten = analyzer.rewrite(placed)
    return rewritten


class PhasePartition(S.SyntaxRewrite):
    def __init__(self, entry_points):
        self.entry_points = entry_points
    def _Procedure(self, proc):
        if proc.name().id not in self.entry_points:
            return proc
        phases = []
        current_phase = 0
        current_body = []
        for stmt in proc.body():
            if isinstance(stmt, S.Return):
                proc.return_node = stmt
            elif not isinstance(stmt, M.PhaseBoundary):
                current_body.append(stmt)
            else:
                if current_body:
                    name = S.Name(proc.name().id + "_phase" + str(current_phase))
                    phase = S.Procedure(name, [], current_body)
                    phase.context = proc.context
                    phases.append(phase)
                    current_phase += 1
                current_body = []
        name = S.Name(proc.name().id + "_phase" + str(current_phase))
        phase = S.Procedure(name, [], current_body)
        phase.context = proc.context
        phases.append(phase)
        proc.parameters = phases
        
        return proc

class PhaseArguments(S.SyntaxRewrite):
    def __init__(self, entry_points):
        self.entry_points = entry_points
    def _Procedure(self, proc):
        if proc.name().id not in self.entry_points:
            return proc
        calls = []
        for phase in proc.body():
            free = S.free_variables(phase)
            variables = set(free).intersection(proc.typings.keys())
            phase.variables = phase.variables + [S.Name(x) for x in variables]
        return proc

class PhaseReturns(S.SyntaxRewrite):
    def __init__(self, entry_points):
        self.entry_points = entry_points
        self.in_process = False
        self.computed = set()
    def _Procedure(self, proc):
        if proc.name().id not in self.entry_points:
            if not self.in_process:
                proc.master = False
                proc.entry_point = False
                return proc
            self.escaping = set()
            self.rewrite_children(proc)
            returns = S.Tuple(*[S.Name(x) for x in self.escaping if x not in self.computed])
            proc.parameters.append(S.Return(returns))
            for name in returns.parameters:
                self.computed.add(name.id)
            proc.returns = returns
            proc.master = False
            proc.entry_point = True
            proc.typings = self.typings
            return proc
        phase_args = set()
        for phase in proc.body():
            phase_args = phase_args.union([x.id for x in phase.formals()])
        phase_args = phase_args.union((x.id for x in flatten(proc.return_node.parameters)))

        self.phase_args = phase_args
        self.in_process = True
        self.typings = proc.typings
        self.rewrite_children(proc)
        self.in_process = False
        calls = [S.Bind(x.returns, S.Apply(x.name(), x.formals())) \
                     for x in proc.body()]
        proc.parameters = proc.parameters + calls + [proc.return_node]
        proc.master = True
        return proc
    def _Bind(self, bind):
        destinations = flatten(bind.binder())
        for destination in destinations:
            if destination.id in self.phase_args:
                self.escaping.add(destination.id)
        return bind

class UnBoxer(S.SyntaxRewrite):
    'Eliminates phases which call CuBox functions'
    def __init__(self, entry_points):
        self.entry_points = entry_points
    def _Procedure(self, proc):
        if proc.name().id not in self.entry_points:
            return proc
        new_phases = []
        replacements = {}
        host = {}
        for phase in filter(lambda x: isinstance(x, S.Procedure), proc.body()):
            first_stmt = phase.body()[0]
            if not isinstance(first_stmt, S.Bind):
                new_phases.append(phase)
                continue
            if not isinstance(first_stmt.value(), S.Apply):
                new_phases.append(phase)
                continue
            function = first_stmt.value().function()
            box = getattr(function, 'box', False)
            host = getattr(function, 'host', False)
            if not box:
                new_phases.append(phase)
                continue
            replacements[phase.name().id] = (first_stmt, host)
        for bind in filter(lambda x: isinstance(x, S.Bind), proc.body()):
            if not isinstance(bind.value(), S.Apply):
                continue
            fn_name = bind.value().function().id
            fn_host = getattr(bind.value().function(), 'host', False)
            if not fn_name in replacements:
                continue
            if fn_host:
                continue
            rep_bind, host = replacements[fn_name]
            apply = rep_bind.value()
            id = apply.function().id
            if not host:
                id += '.variants[execution_place]'
            
            apply.parameters[0] = S.Name(id)
            
            apply.parameters[0].box = True
            bind.parameters[0] = apply
            bind.id = rep_bind.id
        proc.parameters = [x for x in proc.body() if not isinstance(x, S.Procedure)]
        proc.parameters = new_phases + proc.parameters
        return proc
    
def phase_rewrite(stmt, entry_points):
    partitioner = PhasePartition(entry_points)
    partitioned = partitioner.rewrite(stmt)
    argumenter = PhaseArguments(entry_points)
    argumented = argumenter.rewrite(partitioned)
    returner = PhaseReturns(entry_points)
    returned = returner.rewrite(argumented)
    unboxer = UnBoxer(entry_points)
    unboxed = unboxer.rewrite(returned)
    return unboxed
    

