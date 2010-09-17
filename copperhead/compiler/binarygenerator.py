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
import coresyntax as S, backendsyntax as B
import inspect
import pltools
import copy
from ..runtime.cuarray import CuArray
from ..options import DB
import numpy as np
import parsetypes
import pdb
import shapeinference as SI
from itertools import ifilter

try:
    import pycuda.compiler as pycudacomp
    pycudaExists = True
except ImportError:
    pycudaExists = False
    
def ast_to_string(input):
    return pltools.strlist(input, sep='\n', form=str)

def ast_to_cuda_string(input):
    cuda_ast = filter(lambda x: inspect.getmodule(x) == B, input)
    return ast_to_string(cuda_ast)

def ast_to_python_string(input):
    python_ast = filter(lambda x: inspect.getmodule(x) == S, input)
    return ast_to_string(python_ast)

def find_entry_points(input, M):
    cuda_ast = filter(lambda x: isinstance(x, B.CExtern), input)
    cuda_ast = map(lambda x: x.parameters[0], cuda_ast)
    M.entry_points = map(lambda x: x.id, cuda_ast)
    return input

def find_master(input, M):
    python_ast = filter(lambda x: inspect.getmodule(x) == S, input)
    assert(len(python_ast) == 1)
    M.procedure_name = python_ast[0].variables[0].id
    return input

def final_python_code(ast, M):
    M.python_code = ast_to_python_string(ast)
    return ast

def final_cuda_code(ast, M):
    'Extract Python and CUDA code from AST.'
    M.cuda_code = ast_to_cuda_string(ast)
    return ast

def nvccOptions():
    import os.path, os
    runtimePath = os.path.dirname(os.path.realpath(__file__))
    copperheadPath = os.path.join(runtimePath, os.path.pardir)
    includePath = os.path.join(copperheadPath, 'include')
    includeOption = '-I%s' %includePath
    thrust_path = os.getenv('THRUST_PATH')
    thrust_include = '-I%s' % thrust_path
    
    ##For Snow Leopard compatibility
    noBlocks = '-U__BLOCKS__'
    return [includeOption, noBlocks, thrust_include]

def block_size(p_hier):
    return (256, 1, 1)

def grid_size(block_size, p_hier, *args):
    global_size = [1, 1]
    def update_size(extents):
        for index in xrange(0, 2):
            if extents[index] > global_size[index]:
                global_size[index] = extents[index]
    
    for arg in args:
        extents = arg.shape.extents
        if len(extents) < 2:
            extents = extents + [1]
        update_size(extents)
    if p_hier is DB:
        grid_size = tuple(global_size)
    else:
        grid_size = tuple([(x-1)//y + 1 for x,y in zip(global_size, block_size)])
    return grid_size

import os
import errno

def mkdir_path(path):
    try:
        os.makedirs(path)
    except os.error as e:
        if e.errno != errno.EEXIST:
            raise

def write_code(name, code, ext):
    f = open('%s.%s' % (name, ext), 'w')
    f.write(code)
    f.close()

def compute_shape(name, env):
    new_extents = SI.instantiate(env[name], env)
    return new_extents
    
def make_binary(ast, M):
    if not pycudaExists:
        return None
    options = nvccOptions()
    python_code = M.python_code
    cuda_code = M.cuda_code
    procedure_name = M.procedure_name
    if M.save_code:
        code_dir = procedure_name + '_code'
        mkdir_path(code_dir)
        prefix = os.path.join(code_dir, procedure_name)
        write_code(prefix, python_code, 'py')
        write_code(prefix, cuda_code, 'cu')
    
    
    entry_points = M.entry_points
    #Compile the CUDA code
    cudaModule = pycudacomp.SourceModule(cuda_code, no_extern_c=True, options=options)
    #Inject the CUDA functions into a local scope
    python_entry_points = {}
    for entry_point in entry_points:
        python_entry_points[entry_point] = cudaModule.get_function(entry_point)
    scope = copy.copy(M.globals)
    scope.update(python_entry_points) #Add entry point functions

    scope['np'] = np #Add numpy
    scope['p_hier'] = M.p_hier
    #Assumes inputs are all from the same place
    scope['_block_size'] = block_size
    scope['_grid_size'] = grid_size
    scope['execution_place'] = M.inputPlaces.values()[0][0]
    scope['type_from_text'] = parsetypes.type_from_text
    scope['shapes'] = M.shapes
    scope['compute_shape'] = compute_shape
    scope['pdb'] = pdb
    #Compile the python code
    exec python_code in scope
    
    
    return scope[procedure_name]

