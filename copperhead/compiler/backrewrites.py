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

"""Back-end rewrites for the Copperhead compiler.

This module implements the rewrite passes used by the Copperhead
compiler to transform the mid-end IR into CUDA.  The back-end can do
things which are not valid in Python at all, such as converting return
statements into pass-by-reference function arguments.  After all the
back-end rewrites are complete, code generation should proceed by
pretty-printing the back-end.  In the future, there will be multiple
backends for Copperhead, to target multiple parallel platforms.  We'll
build that machinery when the time comes.
"""
from __future__ import division
import coresyntax as S
import backendsyntax as B
import codesnippets as C
import pltools as P
import coretypes as T
import backtypes as BT
import midtypes as MT
import phasetypes as PT
import shapetypes as ST
from utility import flatten
import visitor as V
import copy
import cintrinsics as CI
import hintrinsics as HI
import intrinsics as I
from itertools import imap
import math
import shapetypes as ST
import typeinference as TI
from itertools import ifilter
from ..runtime.cudata import cu_to_c_types
import pdb

class ReferenceConverter(S.SyntaxRewrite):
    def __init__(self, env):
        self.declarations = env
        self.newTypes = {}
    def _Bind(self, binder):
        type = binder.id.type
        if isinstance(type, T.Polytype):
            type = type.monotype()
        result = binder.binder()

        def recordTypes(result, type):
            if not isinstance(result, S.Tuple):
                self.declarations[result.id] = type
                return
            for subresult, subtype in zip(result.parameters, type.parameters):
                recordTypes(subresult, subtype)
            return
        recordTypes(result, type)
        return binder
    def _While(self, whileStmt):
        self.rewrite(whileStmt.parameters[1])
        return whileStmt
    def _Cond(self, cond):
        new_parameters = self.rewrite(cond.parameters[1:])
        stripped_parameters = [S.stripNull(x) for x in new_parameters]
        cond.parameters[1:] = stripped_parameters
        return cond
    def _Procedure(self, proc):
        if proc.master:
            proc.typings.update(self.newTypes)
            return proc
        proc = copy.deepcopy(proc)
        self.returnValue = None
        self.declarations.begin_scope()
        self.rewrite_children(proc)
        filteredBody = S.stripNull(proc.parameters)
        proc.parameters = filteredBody
        originalVariables = proc.variables[1:]
        argumentTypes = [proc.typings[x.id] for x in flatten(originalVariables)]
        typelist = [x.monotype() if isinstance(x, T.Polytype) else x for x in argumentTypes]

        returnValues = self.returnValue.parameters #Unpack return tuple
       
        returnVariableTypes = [proc.typings[x.id] for x in returnValues]
        returnVariableTypes = [x.monotype() if isinstance(x, T.Polytype) else x for x in returnVariableTypes]
        def scalar_reference(type, name):
            if not isinstance(type, T.Seq):
                return B.Reference(name)
            else:
                return name
        if not proc.entry_point: #Return everything by reference for non entry point functions
            returnValues = [scalar_reference(x, y) for x, y in zip(returnVariableTypes, returnValues)]
        proc.variables.extend(returnValues)
        if proc.entry_point:
            #Promote scalar return values to sequences (of length one)
            #Since CUDA requires pointers for outputs
            for index, type in enumerate(returnVariableTypes):
                if not isinstance(type, T.Seq):
                    returnVariableTypes[index] = T.Seq(type)
        
        typelist.extend(returnVariableTypes)
        
        procedureMonotype = MT.FnSide(typelist)
        proc.type = T.quantify_type(procedureMonotype)

        # XXX We need to be using globals here instead of typings
        proc.typings.update(self.newTypes)

        self.newTypes[proc.variables[0].id] = proc.type
        self.declarations.end_scope()
        if proc.variables[0].id not in self.declarations:
            self.declarations[proc.variables[0].id] = proc.variables[0].id
        return proc
    def _Return(self, stmt):
        returnValue = stmt.value()
        if not isinstance(returnValue, S.Tuple):
            self.returnValue = S.Tuple(returnValue)
        else:
            self.returnValue = returnValue
        return S.Null()




def reference_conversion(ast, env):
    rewrite = ReferenceConverter(env)
    referenceConverted = rewrite.rewrite(ast)
    return referenceConverted


def name_closure(closure):
    body_id = closure.body().id
    return S.Name(C.markGenerated(body_id + '_closure' +
                                  str(len(closure.closed_over()))))

class ClosureLifter(S.SyntaxRewrite):
    def __init__(self):
        self.lifted_closures = {}
        self.prev_closure_names = set()
        self.closure_declarations = []
        self.closure_instantiations = set()
        self.typings = None
        self.functors = set()
        
    def _Bind(self, bind):
        self.rewrite_children(bind)
        if self.closure_declarations:
            result = self.closure_declarations + [bind]
            self.closure_declarations = []
            return result
        return bind
    def _Closure(self, closure):
        closure.fn_type = self.typings[closure.body().id]
        closure_name = name_closure(closure)
        closure_instance = C.instance_name(closure_name)
        if closure_instance.id not in self.closure_instantiations:
            self.closure_instantiations.add(closure_instance.id)
            declare = True
        else:
            declare = False

        closed_over = closure.closed_over()
        closure_types = [self.typings[x.id] for x in closed_over]
        def normalize_type(type, name):
            if isinstance(type, T.Polytype):
                type = type.monotype()
            if isinstance(type, MT.FnSide):
                #Functions become their own types, once we structify them
                return T.Monotype(str(name))
            return type
        filtered_types = [normalize_type(x, y) for x, y in zip(closure_types,
                                                               closed_over)]
        closure_type = BT.DependentType(closure_name, filtered_types)
        closure_type_name = C.type_from_name(closure_instance)
        closure_typedef = B.CTypedef(closure_type, closure_type_name)
        closure_type_inst = B.CType(closure_type_name)
        def normalize_name(type, name):
            if isinstance(type, T.Polytype):
                type = type.monotype()
            if isinstance(type, MT.FnSide):
                #Instantiate function structs
                self.functors.add(str(name))
                return S.Apply(name, [])
            return name
        filtered_names = [normalize_name(x, y) for x, y in zip(closure_types,
                                                               closed_over)]
        closure_construct = B.CConstructor(closure_type_inst, filtered_names)
        closure_type_decl = B.CTypeDecl(closure_type_name, closure_instance)

        if declare:
            closure_instantiation = B.CBind(closure_type_decl, closure_construct)        
            self.closure_declarations = [closure_typedef, closure_instantiation]
        else:
            closure_instantiation = B.CBind(closure_instance, closure_construct)
            self.closure_declarations = [closure_instantiation]
        if closure_name.id not in self.lifted_closures:
            if closure_name.id not in self.prev_closure_names:
                self.lifted_closures[closure_name.id] = closure
                self.prev_closure_names.add(closure_name.id)
        return closure_instance
    def _Procedure(self, proc):
        self.typings = proc.typings
        self.closure_instantiations = set()
        self.rewrite_children(proc)
        if self.closure_instantiations:
            proc.instantiated_functors = self.closure_instantiations
        self.typings = None
        proc.parameters = list(flatten(proc.parameters))
        if self.lifted_closures:
            result = self.lifted_closures.values() + [proc]
            self.lifted_closures = {}
            return result
        return proc
               
    
def closure_lift(ast, M):
    rewrite = ClosureLifter()
    lifted = list(flatten(rewrite.rewrite(ast)))
    M.functors = M.functors.union(rewrite.functors)
    
    return lifted

import sequential as SQ

class SequentialFusor(S.SyntaxRewrite):
    def __init__(self):
        self.functors = set()
    def _Procedure(self, proc):
        context = proc.context
        if not (context is I.sequential or context is I.block):
            return proc
        
        type = proc.type
        if isinstance(type, T.Polytype):
            type = type.monotype()
        assert isinstance(type, MT.FnSide)
        #Strip Tuple type out
        input_types = type.parameters[0].parameters 
        
        #Check to see if we need to modify this procedure
        #For now, we know there is no need for fusion if the procedure
        #operates only on scalars
        rewrite = any([isinstance(x, T.Seq) for x in input_types])
        if not rewrite:
            return proc

        self.rewrite_children(proc)
        proc.parameters = list(flatten(proc.parameters))
        for formal in proc.formals():
            if isinstance(formal, S.Name):
                name = formal.id
            else:
                name = formal.parameters[0].id
            type_name = C.type_from_name(name)
            type = proc.typings[name]
            typedef = B.CTypedef(type, type_name)
            proc.parameters.insert(0, typedef)
        
        return proc
    def _CType(self, type):
        return type
    def _Bind(self, bind):
        result = SQ.seq_convert(bind, self.functors)
        return result
    def _CTypedef(self, typedef):
        return typedef
        
        
def sequential_fusion(ast, M):
    rewrite = SequentialFusor()
    fused = rewrite.rewrite(ast)
    M.functors = M.functors.union(rewrite.functors)    
    return fused

class Tuplifier(S.SyntaxRewrite):
    def __init__(self):
        self.name_supply = P.name_supply()
    def _Bind(self, bind):
        self.tup = False
        self.rewrite_children(bind)
        if self.tup:
            bind.no_return_convert = True
        return bind
    def _Tuple(self, tup):
        self.tup = True
        return C.make_tuple(tup.parameters)
    def _CTypedef(self, typedef):
        return typedef
    def _Procedure(self, proc):
        if proc.master:
            return proc
        tuple_unboxing = []
        def box(arg):
            if isinstance(arg, S.Tuple):
                arg_name = C.generateName('tuple_' + self.name_supply.next())
                arg_type = T.Tuple(*[proc.typings[x.id] for x in arg])
                proc.typings[arg_name.id] = arg_type
                for idx, idf in enumerate(arg):
                    tuple_unboxing.append(C.tuple_get(idx, idf, arg_name))
                return arg_name
            else:
                return arg
        new_args = map(box, proc.formals())
        if tuple_unboxing:
            proc.variables[1:] = new_args
            proc.parameters = tuple_unboxing + proc.parameters
        self.rewrite_children(proc)
        return proc
def tuplify(ast, M):
    tuplifier = Tuplifier()
    return tuplifier.rewrite(ast)



   
class StructFinder(S.SyntaxFlattener):
    def __init__(self):
        self.structFunctions = set()
    def _Procedure(self, proc):
        self.master = proc.master
        self.declarations = set()
        for parameter in proc.variables[1:]:
            self.declarations.add(parameter.id)
        self.visit_children(proc)
    def _Closure(self, closure):
        fn = closure.body()
        self.structFunctions.add(fn.id)
    def _Apply(self, appl):
        # Box functions appear in the master function
        # But they need their arguments structified as well
        # So if we're in the master function
        # And this is not a box function being applied,
        # Don't structify
        # Otherwise, do structify.
        if self.master:
            if not hasattr(appl.function(), 'box'):
                return
        for argument in appl.parameters:
            if hasattr(argument, 'id'):
                if argument.id not in self.declarations:
                    self.structFunctions.add(argument.id)
    def _Map(self, map):
        self.structFunctions.add(map.function().id)
    
    
class Structifier(S.SyntaxRewrite):
    def __init__(self, structFunctions):
        self.structFunctions = structFunctions
    def _Procedure(self, proc):
        name = proc.variables[0].id
        if name in self.structFunctions:
            #Change name of function to operator()
            proc.variables[0].id = C.applyOperator
            #Make a struct that encloses this procedure, with the correct name
            struct = B.CStruct(S.Name(name), [proc])
            return struct
        else:
            return proc

class ValueConverter(S.SyntaxRewrite):
    def __init__(self):
        self.in_struct = False
    def _CStruct(self, struct):
        self.in_struct = True
        self.rewrite_children(struct)
        struct.parameters = list(flatten(struct.parameters))
        self.in_struct = False
        return struct
    def _Procedure(self, proc):
        if not self.in_struct:
            return proc
        
        import copy
        if isinstance(proc.type, T.Polytype):
            proc_type = proc.type.monotype()
            vars = proc.type.variables
        else:
            proc_type = proc.type
            vars = []
        argument_type = copy.deepcopy(proc_type.parameters[0])
        return_type = argument_type.parameters[-1]
        argument_type.parameters = argument_type.parameters[:-1]
        new_type = T.Fn(argument_type, return_type)
        if vars:
            new_type = T.Polytype(copy.deepcopy(vars), new_type)
        new_version = copy.deepcopy(proc)
        new_version.type = new_type
        #Strip reference, since we know the return is by reference
        return_name = new_version.variables[-1].parameters[0]
        new_version.variables = new_version.variables[:-1]
        self.return_name = str(return_name)
        new_version.parameters = self.rewrite(new_version.parameters)
        return [proc, new_version]
    def _Bind(self, bind):
        if not self.in_struct:
            return bind
        destination = bind.binder()
        if str(destination) == self.return_name:
            return S.Return(bind.value())
        return bind
    def _CBind(self, cbind):
        return self._Bind(cbind)
    def _CTypedef(self, n):
        return n
    def _CNode(self, n):
        return n
    def _CType(self, n):
        return n
    def _CTypeDecl(self, n):
        return n
    def _CNamespace(self, n):
        return n
    def _CTypename(self, n):
        return n

def structify(stmt, M):
    structFinder = StructFinder()
    structFinder.visit(stmt)
    structifier = Structifier(structFinder.structFunctions.union(M.functors))
    result = structifier.rewrite(stmt)
    value_converter = ValueConverter()
    valued = value_converter.rewrite(result)
    
    return valued




class CTypeRewriter(S.SyntaxRewrite):
    import itertools
    serial = itertools.count(1)
    def __init__(self):
        self.declarations = P.Environment()
    def updateTypes(self, declarations):
        newDeclarations = {}
       
        for declaration in declarations:
            if isinstance(declaration.name, B.Reference):
                #Strip reference info from name
                decl_name = declaration.name.parameters[0].id
            else:
                decl_name = declaration.name.id
                
            newDeclarations[decl_name] = None
        self.declarations.update(newDeclarations)
    
    def _Procedure(self, proc):
        if proc.master:
            return proc
        self.typings = proc.typings
        procedureName = proc.variables[0].id
        procedure_type = proc.type
        argument_types = [proc.typings[str(x.id)] for x in proc.formals()]
        templateTypes = []
        if isinstance(procedure_type, T.Polytype):
            templateTypes = procedure_type.variables
        # def convertFn(type):
        #     if isinstance(type, T.Fn):
        #         newTypeName = 'OP%s' % CTypeRewriter.serial.next()
        #         newType = BT.FnArg(newTypeName)
        #         templateTypes.append(newType)
        #         return newType
        #     return type
        # procedureArgumentTypes = procedureType.parameters[0].parameters
        # procedureArgumentTypes = [convertFn(x) for x in procedureArgumentTypes]
        # procedureType.parameters[0].parameters = procedureArgumentTypes
        # procedureArgumentTypes = procedureType.parameters[0].parameters
        

        
        convertedArgs = [B.CTypeDecl(x, y) for (x, y) in zip(argument_types, proc.formals())]
        
            
        self.declarations.begin_scope()
        self.updateTypes(convertedArgs)
        
        self.rewrite_children(proc)
      
        self.declarations.end_scope()
        if templateTypes:
            newProcedure = S.Procedure(proc.variables[0], convertedArgs, proc.parameters, proc)
            if hasattr(proc, 'instantiated_functors'):
                newProcedure.instantiated_functors = proc.instantiated_functors
            return B.Template([S.Name(x) for x in templateTypes], [newProcedure])
        else:
            newProcedure = S.Procedure(proc.variables[0], convertedArgs, proc.parameters, proc)
            if hasattr(proc, 'instantiated_functors'):
                newProcedure.instantiated_functors = proc.instantiated_functors
            return newProcedure

    def _CBind(self, binding):
        return binding
    def _CTypedef(self, typedef):
        return typedef
    
    def _Bind(self, binding):
        self.rewrite_children(binding)
        binding.id = self.rewrite(binding.id)
        binder = binding.binder()
        binderType = self.typings[binder.id]
        def declareResult(binder, type):
            if isinstance(binder, S.Name):
                if binder.id not in self.declarations:
                    self.declarations[binder.id] = None
                    if isinstance(type, T.Polytype):
                        type = type.monotype()
                    return B.CTypeDecl(type, binder)
                else:
                    return binder
            return S.Tuple(*[declareResult(subResult, subType) for subResult, subType in zip(binder.parameters, type.parameters)])
        if hasattr(binding, 'no_return_convert'):
            result = B.CBind(declareResult(binder, binderType), binding.parameters[0])
        else:
            result = S.Bind(declareResult(binder, binderType), binding.parameters[0])
        return result

    def _TemplateInst(self, template_inst):
        def convert_to_ctype_name(cu_type_name):
            cu_type_str = str(cu_type_name)
            if cu_to_c_types.has_key(cu_type_str):
                return S.Name(cu_to_c_types[cu_type_str])
            return cu_type_name
        template_inst.parameters = [convert_to_ctype_name(x) for x in template_inst.parameters]
        return template_inst
    def _Return(self, stmt):
        return stmt
def ctype_conversion(stmt):
    'Change all types to C types'
   
    rewrite = CTypeRewriter()
    rewritten = rewrite.rewrite(stmt)
    
    return rewritten



class CNodeRewriter(S.SyntaxRewrite):
    def __init__(self, p_hier):
       self.namesupply = P.name_supply("ijk")
       self.types = P.Environment()
       self.structified = set()
       self.return_convert = False
       self.p_hier = p_hier
      
    def index(self, input, offset):
        if isinstance(input, S.Closure):
            operand = input.variables[0]
        else:
            operand = input
        
        possibleSequenceName = operand.id
        possibleSequenceType = self.types[possibleSequenceName]
        if isinstance(possibleSequenceType, S.Seq):
            return S.Subscript(input, S.Index(offset))
        else:
            return operand
    def _TemplateInst(self, inst):
        return inst
    def _Closure(self, closure):
        closure_type = name_closure(closure)
        size = len(closure.closed_over())
        types = [T.Monotype('T' + str(x)) for x in range(size)]
        names = [S.Name('k' + str(x)) for x in range(size)]
        state = [B.CTypeDecl(x, y) for x, y in zip(types, names)]

        constructor_names = [S.Name('_k' + str(x)) for x in range(size)]
        constructor_decl_names = [B.CTypeDecl(x, y) for x, y in zip(types, constructor_names)]
        constructor_body = [S.Bind(x, y) for x, y in zip(names, constructor_names)]
        constructor = S.Procedure(closure_type, constructor_decl_names, constructor_body)
        constructor.entry_point = False
        constructor.type = None
        constructor_c = B.CFunction(constructor, return_type = None)

        closure_fn_type = closure.fn_type
        if isinstance(closure_fn_type, T.Polytype):
            operator_type_variables = [S.Name(x) for x in closure_fn_type.variables]
            closure_fn_type = closure_fn_type.monotype()
        else:
            operator_type_variables = None
        closure_fn_arg_types = closure_fn_type.parameters[0].parameters
        input_argument_size = len(closure_fn_arg_types)
        input_argument_size -= 1 + size #lop off return value and closure parameters
        argument_names = [S.Name('arg' + str(x)) for x in range(input_argument_size)]
        argument_types = []
        
        for i in range(input_argument_size):
            argument_types.append(closure_fn_arg_types[i])
        instantiated_names = copy.copy(argument_names)
        instantiated_names.extend(names)
        instantiated_names.append(C.anonymousReturnValue)
        argument_names.append(B.Reference(C.anonymousReturnValue))
        argument_types.append(closure_fn_arg_types[-1])
        operator_arguments = [B.CTypeDecl(x, y) for x, y in zip(argument_types, argument_names)]
        # XXX The extra S.Name in here could be eliminated if we had a CApply Node
        operator_body = [S.Apply(S.Name(B.CConstructor(closure.body(), [])), instantiated_names)]
        operator = S.Procedure(C.applyOperator, operator_arguments, operator_body)
        operator.entry_point = False
        operator.type = None
        operator_c = B.CFunction(operator)

        value_arguments = operator_arguments[:-1]
        value_names = instantiated_names[:-1]
        
        # XXX The extra S.Name in here could be eliminated if we had a CApply Node
        value_body = [S.Return(S.Apply(S.Name(B.CConstructor(closure.body(), [])), value_names))]
        value_operator = S.Procedure(C.applyOperator, value_arguments, value_body)
        value_operator.entry_point = False
        value_operator.type = None
        value_operator_c = B.CFunction(value_operator, return_type=B.CType(argument_types[-1]))
        
        
        if (operator_type_variables):
            operator_type_variables = [x for x in operator_type_variables \
                                           if str(x) in argument_types]
            operator_c = B.Template(operator_type_variables,
                                    [operator_c])
            value_operator_c = B.Template(operator_type_variables,
                                          [value_operator_c])
        
        
        
        result = B.Template([S.Name(str(x)) for x in types], [B.CStruct(closure_type, [B.CLines(state), constructor_c, operator_c, value_operator_c])])
        return result
    def _Cond(self, cond):
        self.rewrite_children(cond)
        return B.CIf(cond.parameters[0], cond.parameters[1], cond.parameters[2])
    def _CStruct(self, struct):
        struct_id = struct.variables[0].id
        self.structified.add(struct_id)
        self.rewrite_children(struct)
        
        return struct
    def _CTypeDecl(self, typedecl):
        typename = str(typedecl.ctype)
        identifier = typedecl.name
        self.types[identifier] = typename
        return typedecl
    def _CTypedef(self, typedef):
        return typedef
    def _CBind(self, bind):
        return bind
    def _CNamespace(self, ns):
        return ns
    def _Return(self, stmt):
        # XXX This needs to be fixed for functions with named returns
        self.return_type = B.CType(self.typings[str(C.anonymousReturnValue)])
        self.return_convert = False
        self.rewrite_children(stmt)
        return stmt
    def _Procedure(self, proc):
        if proc.master:
            return proc
        self.types.begin_scope()

        if isinstance(proc.type, T.Polytype):
            inputTypes = proc.type.parameters[0].parameters[0].parameters
        else:
            inputTypes = proc.type.parameters[0].parameters
        inputNames = [x.name for x in proc.variables[1:]]
        filteredNames = [x.id for x in inputNames]
        
        for inputType, inputName in zip(inputTypes, filteredNames):
            self.types[inputName] = inputType
        
        self.procName = proc.variables[0].id
        self.return_type = 'void'
        self.typings = proc.typings
        if hasattr(proc, 'instantiated_functors'):
            self.instantiated_functors = proc.instantiated_functors
        else:
            self.instantiated_functors = set()
        self.rewrite_children(proc)
        proc.parameters = list(flatten(proc.parameters))
        if proc.entry_point:
            proc.parameters = CI.boilerplate + proc.parameters
        
            
        self.types.end_scope()

        return B.CFunction(proc, self.return_type)
    def _CFunction(self, cfunc):
        self.rewrite_children(cfunc)
        cfunc.parameters = list(flatten(cfunc.parameters))
        return cfunc
    def _CIf(self, cif):
        self.rewrite_children(cif)
        cif.parameters = [cif.parameters[0]] + [list(flatten(cif.parameters[1]))] + [list(flatten(cif.parameters[2]))]
        return cif

    def _While(self, whileStmt):
        self.rewrite_children(whileStmt)
        body = list(flatten(whileStmt.parameters[1]))
        test = whileStmt.parameters[0]
        return B.CWhile(test, body)
    def _CLines(self, lines):
        return lines
    def _Apply(self, proc):
        self.rewrite_children(proc)
        fn = proc.parameters[0]
        if not isinstance(fn, S.Name):
            return proc
        procedureName = fn.id
        
        if self.return_convert:
            arguments = proc.parameters[1:]
            arguments.extend(self.destinations)
            proc.parameters[1:] = arguments
            
        return proc
    
    def _Bind(self, binding):
        destinations = [binding.binder()]
        self.return_convert = False
        if hasattr(binding, 'no_return_convert'):
            return binding
        else:
            self.return_convert = True
        self.destinations = destinations
        source = binding.value()
        self.rewrite_children(binding)
        binding.id = self.rewrite(binding.id)

        self.destinations = None
        if self.return_convert:
            self.return_convert = False
            if isinstance(source, S.Apply) or isinstance(source, S.Map):
                return binding.value()
        self.return_convert = False
        return binding
        
    def _Null(self, null):
        return null
    def _Map(self, map):
        if map.context is not I.distributed:
            return map
        parameters = map.parameters[1:]
        
        # Map needs to index differently if the function being mapped is
        # Sequential or Block-wise parallel.
        # Here we check to see what the parallelization context of the function will be
        sub_context = self.p_hier[1]
       
        if sub_context is I.sequential:
            declarations = []
            if self.return_convert:
                dest = self.destinations[0]
                if isinstance(dest, B.CTypeDecl):
                    parameters.append(dest.name)
                    dest.lower()
                    len_set = B.CBind(B.CMember(dest.name, S.Name('length')),
                                      B.CMember(parameters[0], S.Apply(S.Name('size'), [])))
                    declarations = [dest, len_set]
                else:
                    parameters.extend(self.destinations)

            indexed_parameters = [S.Subscript(x, C.globalIndex) for x in parameters]



            fn_name = map.function()
            if fn_name.id not in self.instantiated_functors:
                body = S.Apply(S.Name(B.CConstructor(fn_name, [])), indexed_parameters)
            else:
                body = S.Apply(fn_name, indexed_parameters)
            guard = B.CIf(C.threadGuard(map.inputs()[0]), body)
            if declarations:
                return declarations + [guard]
            return guard
        elif sub_context is I.block:
            dest = self.destinations[0]
            assert(not isinstance(dest, B.CTypeDecl))
            result = S.Name(dest.id + '_el')
            code = [B.CTypeDecl(self.types[str(dest)].unbox(), result)]
            assert(self.return_convert)
            indexed_parameters = [S.Subscript(x, C.blockIdx) for x in parameters]
            indexed_parameters.append(result)
            fn_name = map.function()
            if fn_name.id not in self.instantiated_functors:
                body = S.Apply(S.Name(B.CConstructor(fn_name, [])), indexed_parameters)
            else:
                body = S.Apply(fn_name, indexed_parameters)
            code.append(body)
            store = S.Bind(S.Subscript(dest, C.blockIdx), result)
            guarded_store = B.CIf(C.thread_zero(store), store)
            code.append(guarded_store)
            guard = B.CIf(C.blockGuard(map.inputs()[0]), code)
            return guard
    
def cnode_conversion(stmt, p_hier):
    'Change all nodes to C nodes'
    rewrite = CNodeRewriter(p_hier)
    cnodes = rewrite.rewrite(stmt)
    
    return cnodes


class TemplateUniquifier(S.SyntaxRewrite):
    def __init__(self):
        self.template_map = None
    def _Template(self, template):
        template_vars = template.variables
        template_var_names = [str(x.id) for x in template_vars]
        
        unique_template_names = [C.markGenerated(x) for x in template_var_names]
        template.variables = [S.Name(x) for x in unique_template_names]
        self.template_map = dict(zip(template_var_names, unique_template_names))
        self.rewrite_children(template)
        self.template_map = None
        return template
    def _CFunction(self, cfn):
        if not self.template_map:
            return cfn
        cfn.return_type = self.rewrite(cfn.return_type)
        new_args = [self.rewrite(x) for x in cfn.arguments]
        cfn.arguments = new_args
        self.rewrite_children(cfn)
        return cfn
    def _CTypedef(self, typedef):
        typedef.decl = self.rewrite(typedef.decl)
        return typedef
    def _CNamespace(self, namespace):
        return namespace
    def _CType(self, type):
        if not self.template_map:
            return type
        if isinstance(type.cu_type, B.CType):
            type.cu_type = self.rewrite(type.cu_type)
        if isinstance(type.cu_type, BT.DependentType):
            new_parameters = []
            for parameter in type.cu_type.parameters[1:]:
                parameter_name = str(parameter)
                if parameter_name in self.template_map:
                    new_parameters.append(S.Name(self.template_map[parameter_name]))
                else:
                    new_parameters.append(self.rewrite(parameter))
            new_type = B.CType(BT.DependentType(type.cu_type.parameters[0], new_parameters))
            return new_type
            
        else:
            typename = type.atomic_type
            if typename in self.template_map:
                type.atomic_type = self.template_map[typename]
        return type
    def _CTypeDecl(self, decl):
        if not self.template_map:
            return decl
        decl.update_ctype(self.rewrite(decl.ctype))
        return decl
    def _Bind(self, bind):
        self.rewrite_children(bind)
        bind.id = self.rewrite(bind.id)
        return bind
    def _CBind(self, bind):
        return self._Bind(bind)
    def _Null(self, ast):
        return ast
    def _TemplateInst(self, inst):
        if not self.template_map:
            return inst
        template_names = inst.parameters
        result = []
        for name in template_names:
            typename = str(name)
            if typename in self.template_map:
                result.append(self.template_map[typename])
            else:
                result.append(name)
        inst.parameters = [S.Name(x) for x in result]
        return inst
    def _DependentType(self, dtype):
        dtype.parameters = [self.rewrite(x) for x in dtype.parameters]
        return dtype
    def _Name(self, name):
        return name
                
    def _default(self, x):
        if not hasattr(x, 'children'):
            return x
        else:
            return self.rewrite_children(x)


def rename_templates(suite):
    """This compiler pass is used to rename template types to unique names that
    will not clash with variable names."""
    uniquifier = TemplateUniquifier()
    result = uniquifier.rewrite(suite)
    return result

def final_type(proc, global_input_types):
    """Instantiate the type for a procedure with the input types discovered
    by the runtime."""
    typings = proc.typings
    tcon = TI.TypingContext()
    result_type = tcon.fresh_typevar()
    input_types = global_input_types[proc.name().id]
    input_tuple_type = T.Tuple(*input_types)
    fn_input_type = T.Fn(input_tuple_type,
                         result_type)
    substitution_map = {}
    fn_inferred_type = proc.name().type
    if isinstance(fn_inferred_type, T.Polytype):
        poly_vars = fn_inferred_type.variables
        tcon_vars = tcon.fresh_typelist(poly_vars)
        for pol, var in zip(poly_vars, tcon_vars):
            substitution_map[pol] = var
        fn_type = T.substituted_type(fn_inferred_type.monotype(),
                                     substitution_map)
    else:
        fn_type = fn_inferred_type
    constraint = [TI.Equality(fn_type, fn_input_type)]
    solver = TI.Solver1(constraint, tcon)
    solver.solve()
    concrete_map = {}
    for key, val in substitution_map.iteritems():
        concrete_map[key] = solver.solution[val]
    final_typings = {}
    for key, val in typings.iteritems():
        final_type = TI.resolve_type(val, concrete_map)
        final_typings[key] = final_type
    return final_typings


def make_c_interface(proc):
    """Wrap a master procedure in a C function, including data conversion."""
    hosted_names = [C.markGenerated(x.id + '_host') \
                    for x in proc.formals()]
    typings = proc.final_typings
    def host_type(name):
        final_type = typings[name.id]
        if isinstance(final_type, T.Seq):
            return T.Monotype('CuSequence')
        elif isinstance(final_type, T.Tuple):
            return T.Monotype('CuTuple')
        else:
            return T.Monotype('CuScalar')
    hosted_types = [host_type(x) for x in proc.formals()]
    hosted_decls = [B.CTypeDecl(x, y) \
                    for x, y in zip(hosted_types, hosted_names)]
    def extract(name, hosted_name):
        final_type = typings[name.id]
        ctype = B.CType(final_type)
        atomic_type = ctype.atomic_type
        if ctype.depth < 0:
            if isinstance(final_type, T.Tuple):
                extraction = B.TemplateInst(S.Name('extract_tuple'),
                                            [B.CType(x) for x in final_type])
            else:
                extraction = B.TemplateInst(S.Name('extract_scalar'),
                                            [ctype])
        elif ctype.depth == 0:
            extraction = B.TemplateInst(S.Name('extract_stored_sequence'),
                                        [atomic_type])
        else:
            extraction = B.TemplateInst(S.Name('extract_nested_sequence'),
                                        [atomic_type, ctype.depth])
        return B.CApply(extraction, [S.Name(hosted_name)])
    extractions = [extract(x, y) for x, y in zip(proc.formals(), hosted_names)]
    cuda_types = [B.CType(typings[x.id]) for x in proc.formals()]
    def reference(x):
        typ = typings[x.id]
        if isinstance(typ, T.Seq) or isinstance(typ, T.Tuple):
            return x
        else:
            return B.Reference(x)
    referenced = [reference(x) for x in proc.formals()]
    
        
    cuda_decls = [B.CTypeDecl(x, y) \
                  for x, y in zip(cuda_types, referenced)]
    
    extract_decls = [B.CBind(x, y) for x, y in zip(cuda_decls, extractions)]
   
    proc.parameters = extract_decls + proc.parameters
    proc.parameters = filter(lambda x: not isinstance(x, S.Null), proc.parameters)
    proc.variables[1:] = hosted_decls
    proc.type = proc.name().type
    proc.entry_point = None
    cfunction = B.CFunction(proc)
    cfunction.typings = typings
    cfunction.shapes = proc.shapes
    return cfunction
                       
class HostDriver(S.SyntaxRewrite):
    def __init__(self, input_types):
        self.input_types = input_types
        self.binder = None
        self.return_convert = False
        self.result = None
    def _CInclude(self, inc):
        return inc
    def _CFunction(self, func):
        return func
    def _CType(self, ctype):
        return ctype
    def _Procedure(self, proc):
        self.rewrite_children(proc)
        # Unify the type of this procedure with its inputs
        # Introspected by the runtime
        proc.final_typings = final_type(proc, self.input_types)
        allocations = []
        cside = copy.deepcopy(proc)
        # Return convert this procedure itself
        # Allocate data for return values
        for name in self.result:
            cside.variables.append(name)
            shape = proc.shapes[name.id]
            result_type = proc.final_typings[name.id]
            if isinstance(shape, ST.ShapeOf):
                shape_constructor = S.Name(str(shape))
                type_constructor = S.Name(repr(result_type))
                arguments = [S.Bind(S.Name('type'), type_constructor),
                             S.Bind(S.Name('shape'), shape_constructor)]
                allocation = S.Apply(S.Name('CuArray'), arguments)
            elif shape.extents == [] and shape.element is None:
                if isinstance(result_type, T.Tuple):
                    result_type_name = 'CuTuple'
                    arguments = [S.Name('Cu' + str(x) + '()') \
                                 for x in result_type]
                    allocation = S.Apply(S.Name(result_type_name),
                                         [S.Name(P.strlist(arguments, sep=', ',\
                                                           bracket='[]',\
                                                           form=str))])
                else:
                    result_type_name = 'Cu' + str(result_type)
                    allocation = S.Apply(S.Name(result_type_name), [])
            else:
                
                source_extents = S.Name(str(shape.extents))
                source_element = S.Name(str(shape.element))
                shape_constructor = S.Apply(S.Name('Shape'), [source_extents,
                                                              source_element])
                type_constructor = S.Name(repr(result_type))
                arguments = [S.Bind(S.Name('type'), type_constructor),
                             S.Bind(S.Name('shape'), shape_constructor)]
                allocation = S.Apply(S.Name('CuArray'), arguments)
            binding = S.Bind(name, allocation)
            allocations.append(binding)
        call_variables = copy.copy(cside.variables[1:])
        cside = make_c_interface(cside)
        call = S.Apply(S.Name('entry_point'), call_variables)
        ret = S.Return(S.Tuple(*self.result))
        proc.parameters = allocations + [call, ret]
        return [cside, proc]
    def _Bind(self, binder):
        'Return convert non-box function calls'
        self.binder = binder.binder()
        self.return_convert = None
        self.rewrite_children(binder)
        self.binder = None
        if self.return_convert:
            return self.return_convert
        return binder
    def _Apply(self, appl):
        "Return convert non-box function calls"
        if hasattr(appl.function(), 'box'):
            return appl
        appl.parameters.extend(self.binder.parameters)
        self.return_convert = appl
        return None
    def _Return(self, ret):
        'Record result of return as an iterable'
        result = ret.value()
        if hasattr(result, '__iter__'):
            self.result = result
        else:
            self.result = (result,)
        return S.Null()

class KernelLauncher(S.SyntaxRewrite):
    """Compute grid sizes, instantiate templated phases, launch CUDA kernels"""
    def __init__(self):
        self.identifier = Identifier()
        self.declared = set()
    def _CFunction(self, cfn):
        if not cfn.cuda_kind is None:
            return cfn
        block_decl = [B.CBind(B.CTypeDecl(T.Int, S.Name('block_size')), S.Number(256)),
                      B.CBind(B.CTypeDecl(T.Int, S.Name('grid_size')), S.Number(0))]
        self.typings = cfn.typings
        self.shapes = cfn.shapes
        self.declared = set()
        self.rewrite_children(cfn)
        cfn.parameters = block_decl + list(flatten(cfn.parameters))
        return cfn
    def _Procedure(self, proc):
        return proc
    def _CInclude(self, cinc):
        return cinc
    def _CType(self, ctype):
        return ctype
    def _CBind(self, cbind):
        # Record the identifiers of things which have been declared
        binder = cbind.binder()
        if isinstance(binder, B.CTypeDecl):
            name = str(self.identifier.visit(binder.name))
            self.declared.add(name)
        return cbind
    def _Bind(self, bind):
        if not isinstance(bind.value(), S.Apply):
            return bind
        appl = bind.value()
        if not hasattr(appl.function(), 'box'):
            return bind
        binder = bind.binder()
        name = str(self.identifier.visit(binder))
        self.declared.add(name)
        return bind
    def _Apply(self, appl):
        """Compute grid size, instantiate phase for non-box,
        allocate data for temporary results for non-box,
        Instantiate functors for box."""
 
        if hasattr(appl.function(), 'box'):
            fn_type = self.typings[appl.function().id]
            input_types = fn_type.input_types()
            instantiated_inputs = []
            for typ, arg in zip(input_types, appl.arguments()):
                if isinstance(typ, T.Fn):
                    instantiated_inputs.append(B.CConstructor(arg, []))
                else:
                    instantiated_inputs.append(arg)
            instantiated_apply = S.Apply(appl.function(), instantiated_inputs)
            return instantiated_apply
        allocations = []
        # Allocate data for temporary results
        # XXX Only works for basic sequences
        for arg in appl.arguments():
            argname = str(arg)
            if argname not in self.declared:
                self.declared.add(argname)
                typ = self.typings[argname]
                ctyp = B.CType(typ)
                atomic_type = ctyp.atomic_type
                assert(isinstance(typ, T.Seq))
                assert(ctyp.depth == 0)
                shape = self.shapes[argname]
                if isinstance(shape, ST.ShapeOf):
                    source = shape.value
                    extent = B.CApply(B.CMember(S.Name(source), S.Name('size')),
                                      [])
                elif isinstance(shape.extents, ST.ExtentOf):
                    source = shape.extents.value
                    extent = B.CApply(B.CMember(S.Name(source), S.Name('size')),
                                      [])
                else:
                    extent = S.Name(str(shape.extents))
                tv_name = S.Name(C.markGenerated(argname + '_tv'))
                tv_type = S.Name('thrust::device_vector<' + atomic_type + ' >')
                allocation = B.CTypeDecl(tv_type, B.CConstructor(tv_name,
                                                                 [extent]))
                raw_pointer = B.CApply(S.Name('thrust::raw_pointer_cast'),
                                       [S.Subscript(
                                           B.Reference(tv_name),
                                           S.Number(0))
                                           ])
                sequence = B.CTypeDecl(ctyp, B.CConstructor(arg, [raw_pointer,
                                                                  extent]))
                allocations.extend([allocation, sequence])
        
        # Compute grid size
        def update_grid_size(name):
            typ = self.typings[name.id]
            if isinstance(typ, T.Seq):
                arguments = [S.Name('block_size'), name, S.Name('grid_size')]
                return B.CApply(S.Name('update_grid_size'), arguments)
        updates = (update_grid_size(x) for x in appl.arguments())
        updates = filter(lambda x: x, updates)

        # Compute template arguments, if any
        fn_type = self.typings[appl.function().id]
        if isinstance(fn_type, T.Polytype):
            final_parameter_types = [self.typings[x.id] for x in appl.arguments()]
            fn_final_type = MT.FnSide(T.Tuple(*final_parameter_types))
            tcon = TI.TypingContext()
            substitution_map = {}
            poly_vars = fn_type.variables
            tcon_vars = tcon.fresh_typelist(poly_vars)
            for pol, var in zip(poly_vars, tcon_vars):
                substitution_map[pol] = var
            inst_input = T.substituted_type(fn_type.monotype(),
                                            substitution_map)
            
            constraint = [TI.Equality(inst_input, fn_final_type)]
            solver = TI.Solver1(constraint, tcon)
            solver.solve()
            concrete_map = {}
            for key, val in substitution_map.iteritems():
                concrete_map[key] = solver.solution[val]
            template_map = sorted(concrete_map.iteritems())
            template_parameters = [B.CType(y) for x, y in template_map]
            fn_inst = B.TemplateInst(appl.function(), template_parameters)
        else:
            # No template arguments, just use the function
            fn_inst = appl.function()
            
        new_fn = B.CudaChevrons(fn_inst, S.Name('grid_size'),
                                S.Name('block_size'))
        capply = B.CApply(new_fn, appl.arguments())
        return allocations + [updates, capply]

class HostIntrinsic(S.SyntaxRewrite):
    def __init__(self):
        self.identifier = Identifier()
        self.declared = set()
    def _CStruct(self, cstruct):
        return cstruct
    def _Procedure(self, proc):
        return proc
    def _CType(self, ctype):
        return ctype
    def _CTypedef(self, ctypedef):
        return ctypedef
    def _CFunction(self, cfn):
        # Only rewrite host C functions
        if cfn.cuda_kind:
            return cfn
        self.declared = set()
        self.rewrite_children(cfn)
        cfn.parameters = list(flatten(cfn.parameters))
        return cfn
    def _Bind(self, bind):
        value = bind.value()
        binder = bind.binder()
        if str(binder) not in self.declared:
            bind.allocate = True
        else:
            bind.allocate = False
        if isinstance(value, S.Apply):
            fn = value.function()
            intrinsic = getattr(HI, '_' + str(fn), None)
            if intrinsic:
                return intrinsic(bind)
        return bind
    def _CBind(self, cbind):
     
        binder = cbind.binder()
        if isinstance(binder, B.CTypeDecl):
            # Record the identifiers of things which have been declared

            name = str(self.identifier.visit(binder.name))
            self.declared.add(name)
     
            # Insert typedefs for every declared variable
            # These will be used for some of the host intrinsics
            # E.g. stored_sequence<float > x = ...
            # typedef stored_sequence<float > _T_x
            ctype = binder.ctype
            name = self.identifier.visit(binder.name)
            constructed_typename = C.type_from_name(name)
            typedef = B.CTypedef(ctype, constructed_typename)
            return [cbind, typedef]
        return cbind

class HostReturner(S.SyntaxRewrite):
    """Makes sure work done by Copperhead is visible to Python.
    Since thrust tuples don't work by reference, returning tuples
    needs to be done separately."""
    def _Procedure(self, proc):
        return proc
    def _CFunction(self, cfn):
        # Only rewrite host C functions
        if cfn.cuda_kind:
            return cfn
        self.typings = cfn.typings
        self.input_names = set((str(x.name) for x in cfn.arguments))
        self.input_tuples = {}
        self.returned_tuples = {}
        self.rewrite_children(cfn)
        def store_tuple(x, y):
            return B.CApply(S.Name('store_tuple'), [S.Name(x), S.Name(y)])
        tuple_storage = [store_tuple(x, y) for x, y in self.returned_tuples.iteritems()]
        cfn.parameters.extend(tuple_storage)
        return cfn
    def _CBind(self, cbind):
        binder = cbind.binder()
        if not isinstance(binder, B.CTypeDecl):
            return cbind
        name = str(binder.name)
        if name not in self.typings:
            return cbind
        value = cbind.value()
        if not isinstance(value, B.CApply):
            return cbind
        arguments = value.parameters[1:]
        if not arguments:
            return cbind
        input_name = str(arguments[0])
        if input_name in self.input_names:
            typ = self.typings[name]
            if isinstance(typ, T.Tuple):
                self.input_tuples[name] = input_name
        return cbind
    def _Bind(self, bind):
        binder = bind.binder()
        if isinstance(binder, S.Name):
            name = str(binder)
            if name in self.input_tuples:
                self.returned_tuples[name] = self.input_tuples[name]
        return bind
    def _CTypedef(self, ctypedef):
        return ctypedef
    def _CType(self, ctype):
        return ctype

class InstantiatedTyper(S.SyntaxVisitor):
    def __init__(self):
        self.declarations = {}
        self.identifier = Identifier()
    def _CFunction(self, cfn):
        if cfn.cuda_kind:
            return
        self.visit_children(cfn)
    def _CBind(self, cbind):
        binder = cbind.binder()
        if not isinstance(binder, B.CTypeDecl):
            return cbind
        name = str(self.identifier.visit(binder.name))
        self.declarations[name] = str(binder.ctype)
        return
    def _CTypeDecl(self, ctypedecl):
        name = str(self.identifier.visit(ctypedecl.name))
        self.declarations[name] = str(ctypedecl.ctype)
        return

class PhaseTyper(S.SyntaxRewrite):
    def __init__(self, declarations):
        self.declarations = declarations
    def _CFunction(self, cfn):
        if cfn.cuda_kind != '__global__':
            return cfn
        for ctypedecl in cfn.arguments:
            name = str(ctypedecl.name)
            declared_type = self.declarations.get(name, None)
            if declared_type:
                ctypedecl.update_ctype(T.Monotype(declared_type))
        return cfn
    def _CType(self, ctype):
        return ctype

    
def host_driver(stmt, input_types, preamble):
    'Create host C++ code and host Python code'
    driver = HostDriver(input_types)
    driven = list(flatten(driver.rewrite(stmt)))
    launcher = KernelLauncher()
    launched = launcher.rewrite(driven)
    hoster = HostIntrinsic()
    hosted = hoster.rewrite(launched)
    returner = HostReturner()
    returned = returner.rewrite(hosted)
    # Fix types to global functions
    # For example, if data comes from an index_sequence
    # Make the global function use index_sequences instead of
    # stored_sequence<int > (as dictated by the type)
    inst_typer = InstantiatedTyper()
    inst_typer.visit(returned)
    phase_typer = PhaseTyper(inst_typer.declarations)
    phase_typed = phase_typer.rewrite(returned)
    return phase_typed

class IntrinsicConverter(S.SyntaxRewrite):
    def _Procedure(self, fn):
        if fn.master:
            return fn
        self.typings = fn.typings
        self.formals = set((x.id for x in fn.formals()))
        self.rewrite_children(fn)
        fn.parameters = list(flatten(fn.parameters))
        return fn
    def _Bind(self, bind):
        self.intrinsic = None
        self.rewrite_children(bind)
        if self.intrinsic:
            write_out = bind.binder().id in self.formals
            bind.no_return_convert = True
            return self.intrinsic(bind, self.typings, write_out)
        return bind
    def _CBind(self, bind):
        return bind
    def _Apply(self, apply):
        if not hasattr(apply.parameters[0], 'id'):
            return apply
        fnName = apply.parameters[0].id
        if not isinstance(fnName, str):
            return apply
        if hasattr(CI, '_' + fnName):
            self.intrinsic = getattr(CI, '_' + fnName)
        return apply
    def _CLines(self, ast):
        return ast
    def _CIf(self, ast):
        return ast
    def _CTypedef(self, typedef):
        return typedef
    def _Return(self, ret):
        self.intrinsic = None
        self.rewrite_children(ret)
        if self.intrinsic:
            bind = S.Bind(S.Null, ret.value())
            bind = self.intrinsic(bind, self.typings, True)
            ret.parameters[0] = bind.value()
        return ret
    def _Cond(self, cond):
        self.intrinsic = None
        test = self.rewrite(cond.parameters[0])
        if self.intrinsic:
            bind = S.Bind(S.Null, test)
            bind = self.intrinsic(bind, self.typings, False)
            cond.parameters[0] = bind.value()
        self.rewrite_children(cond)
        return cond
    def _And(self, and_node):
        self.intrinsic = CI._op_band
        return and_node
    def _Or(self, or_node):
        self.intrinsic = CI._op_bor
        return or_node
        
def intrinsic_conversion(stmt):
    rewriter = IntrinsicConverter()
    converted = rewriter.rewrite(stmt)
    return converted

class Identifier(S.SyntaxVisitor):
    def _Name(self, name):
        if isinstance(name.id, str):
            return name.id
        else:
            return self.visit(name.id)
    def _Reference(self, ref):
        return self.visit(ref.parameters[0])
    def _CConstructor(self, const):
        return self.visit(const.id)
    def _Subscript(self, sub):
        return self.visit(sub.value())
    def _CType(self, ctype):
        return str(ctype)
    def _TemplateInst(self, inst):
        return self.visit(inst.id)
    def _CNamespace(self, inst):
        return str(inst)

class UniformConverter(S.SyntaxRewrite):
    def __init__(self, uniforms):
        self.uniforms = uniforms
        self.identifier = Identifier()
        self.live_uni = set()
        self.instances = {}
    def _CStruct(self, struct):
        name = struct.variables[0].id
        self.live_uni = set()
        self.instances = {}
        if name not in self.uniforms:
            return struct
        if name + '_constructor' in self.uniforms:
            closure_data_decls = struct.parameters[0].parameters
            closure_data_names = [x.name.id for x in closure_data_decls]
            for marked_arg in self.uniforms[name + '_constructor']:
                self.live_uni.add(closure_data_names[marked_arg])

        self.uniforms['operator()'] = self.uniforms[name]
        self.rewrite_children(struct)
        del self.uniforms['operator()']
        self.live_uni = set()
        return struct
    def _CTypedef(self, ctypedef):
        self.typedefs[ctypedef.decl.name] = ctypedef
        name = ctypedef.decl.name
        if len(name) < 3:
            return ctypedef
        if name[0:2] == '_T':
            if name[2:] in self.live_uni:
                ctypedef.decl.uniform()
            if name[3:] in self.live_uni:
                ctypedef.decl.uniform()
        return ctypedef
    def _CBind(self, cbind):
        self.rewrite_children(cbind)
        destination = self.identifier.visit(cbind.binder())
        source = self.identifier.visit(cbind.value())
        if str(source) in self.typedefs:
            typedef = self.typedefs[source]
            ctype = typedef.decl.type
            if not isinstance(ctype, BT.DependentType):
                return cbind
            struct_name = ctype.parameters[0].id
            inst_name = destination[1]
            self.instances[inst_name] = struct_name
        return cbind
    def _CType(self, ctype):
        return ctype
    def _CFunction(self, cfun):
        if str(cfun.id) not in self.uniforms:
            return cfun
        self.typedefs = {}
        uniforms = self.uniforms[str(cfun.id)]
      
        for n, x in enumerate(cfun.arguments):
            if n in uniforms:
                x.uniform()
                self.live_uni.add(x.name.id)
        self.rewrite_children(cfun)
        # Store the uniform sequences for this function
        # This will be used by the Pycuda wrapper
        cfun.uniforms = uniforms
        live_uni = set()
        return cfun
    def _Apply(self, app):
        app_uni = set()
        for n, arg in enumerate(app.arguments()):
            name = self.identifier.visit(arg)
            if isinstance(name, str) and name in self.live_uni:
                app_uni.add(n)
                # Mark the argument so that Pycuda wrapper can wrap it
                arg.uniform = True
        fn_id = self.identifier.visit(app.function())
        if fn_id in self.instances:
            fn_id = self.instances[fn_id]
        if app_uni:
            self.uniforms[fn_id] = app_uni
        return app
    def _CConstructor(self, const):
        app_uni = set()
        for n, arg in enumerate(const.arguments()):
            name = self.identifier.visit(arg)
            if isinstance(name, str) and name in self.live_uni:
                app_uni.add(n)
        fn_id = self.identifier.visit(const.id)
        if app_uni:
            
            self.uniforms[fn_id] = app_uni
            if fn_id in self.typedefs:
                # If we've got a typedef for this constructor,
                # We need to find its definition of these arguments
                # And make sure they're marked as well
                typedef = self.typedefs[fn_id]
                ctype = typedef.decl.type
                if isinstance(ctype, BT.DependentType):
                    dependent_name = ctype.parameters[0]
                    for n in range(len(ctype.parameters)-1):
                        if n in app_uni:
                            ctype.parameters[n+1].mark_uniform()
                    self.uniforms[str(dependent_name) + '_constructor'] = app_uni
                typedef.decl.update_ctype(ctype)
        return const
    def _Null(self, null):
        return null
    def _Procedure(self, proc):
        named_uniforms = set()
        for n, x in enumerate(proc.formals()):
            if n in self.uniforms:
                named_uniforms.add(x.id)
        self.uniforms = {}
        self.live_uni = named_uniforms
        self.rewrite_children(proc)
        self.live_uni = set()
        return proc
    
def uniform_conversion(suite, uniforms):
    # If we don't have any nested uniform sequences, just return
    if not uniforms:
        return suite
    rewriter = UniformConverter(uniforms)
    
   
    converted = []
    for stmt in reversed(suite):
        converted.insert(0, rewriter.rewrite(stmt))

    return converted

