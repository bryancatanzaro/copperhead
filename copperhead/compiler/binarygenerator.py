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


from __future__ import division
import coresyntax as S
import coretypes as T
import inspect
import pltools
import copy
from ..runtime.cudata import CuArray
import numpy as np
import parsetypes
import pdb
from itertools import ifilter
import codepy.bpl
import codepy.cuda
import codepy.cgen as CG

    
def ast_to_string(input):
    return pltools.strlist(input, sep='\n', form=str)

def ast_to_python_string(input):
    python_ast = filter(lambda x: inspect.getmodule(x) == S, input)
    return ast_to_string(python_ast)

def find_master(input, M):
    python_ast = filter(lambda x: inspect.getmodule(x) == S, input)
    assert(len(python_ast) == 1)
    M.procedure_name = python_ast[0].variables[0].id
    return input

def final_python_code(ast, M):
    M.python_code = ast_to_python_string(ast)
    return ast


def prepare_compilation(ast, M):
    host_module = codepy.bpl.BoostPythonModule(max_arity=len(host_fn.arguments))
    host_module.add_function(
        CG.FunctionBody(
            CG.FunctionDeclaration(CG.Value('void', 'entry_point'),
                                   cgvalue_args),
            CG.Block([CG.Statement(host_call)])))
    
    device_module = codepy.cuda.CudaModule(host_module)
    for x in M.preamble:
        if x:
            dir, name = x
            device_module.add_to_preamble([CG.Include(name)])
    wrapped_cuda_code = [CG.Line(str(x)) for x in cuda_code]
    device_module.add_to_module(wrapped_cuda_code)
    device_module.add_function(
        CG.FunctionBody(
            CG.FunctionDeclaration(CG.Value('void', host_fn_name),
                                   cgvalue_args),
            CG.Block([CG.Statement(str(x)) for x in host_fn.parameters])))
    M.host_module = host_module
    M.device_module = device_module
    return ast
    
def make_binary(ast, M):
    python_code = M.python_code
    procedure_name = M.procedure_name
    host_code = str(M.host_module.generate())
    device_code = str(M.device_module.generate())
    # XXX This import can't happen at the file scope because of import
    # dependency issues.  We should refactor things to avoid this workaround.
    from ..runtime import nvcc_toolchain, host_toolchain
    module = M.device_module.compile(host_toolchain, nvcc_toolchain)
    
    scope = copy.copy(M.globals)

    # XXX Is this list of things a program might need sufficient?
    # Revisit this, also perhaps refactor it to make it less awkward
    scope['Int'] = T.Int
    scope['Float'] = T.Float
    scope['Bool'] = T.Bool
    scope['Long'] = T.Long
    scope['Double'] = T.Double
    scope['Seq'] = T.Seq
    scope['Unit'] = ST.Unit
    scope['Shape'] = ST.Shape
    scope['CuInt'] = CD.CuInt
    scope['CuFloat'] = CD.CuFloat
    scope['CuBool'] = CD.CuBool
    scope['CuLong'] = CD.CuLong
    scope['CuDouble'] = CD.CuDouble
    scope['entry_point'] = module.entry_point

    #Compile the python code
    exec python_code in scope
    
    return (python_code, host_code, device_code), scope[procedure_name]

