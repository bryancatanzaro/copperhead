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


from __future__ import division
import coresyntax as S
import coretypes as T
import inspect
import pltools
import copy
import numpy as np
import parsetypes
import pdb
from itertools import ifilter
import codepy.bpl
import codepy.cuda
import codepy.cgen as CG
import pickle
import os

def prepare_compilation(M):
    from ..runtime import cuda_support
    if cuda_support:
        from ..runtime import cuda_tag
        if M.tag == cuda_tag:
            return prepare_cuda_compilation(M)
    return prepare_host_compilation(M)

def prepare_cuda_compilation(M):
    assert(len(M.entry_points) == 1)
    procedure_name = M.entry_points[0]
    hash, (wrap_type, wrap_name), wrap_args = M.wrap_info
    wrap_decl = CG.FunctionDeclaration(CG.Value(wrap_type, wrap_name),
                                       [CG.Value(x, y) for x, y in wrap_args])
    host_module = codepy.bpl.BoostPythonModule(max_arity=max(10,M.arity),
                                               use_private_namespace=False)
    host_module.add_to_preamble([
        CG.Include("prelude/runtime/cunp.hpp"),
        CG.Include("prelude/runtime/cuarray.hpp"),
        CG.Line('using namespace copperhead;')])
    device_module = codepy.cuda.CudaModule(host_module)
    hash_namespace_open = CG.Line('namespace %s {' % hash)
    hash_namespace_close = CG.Line('}')
    using_declaration = CG.Line('using namespace %s;' % hash)
    host_module.add_to_preamble([hash_namespace_open, wrap_decl,
                                 hash_namespace_close, using_declaration])

    host_module.add_to_init([CG.Statement(
                "boost::python::def(\"%s\", &%s)" % (
                    procedure_name, wrap_name))])

    device_module.add_to_preamble(
        [CG.Include("prelude/prelude.h"),
         CG.Include("prelude/runtime/cunp.hpp"),
         CG.Include("prelude/runtime/make_cuarray.hpp"),
         CG.Include("prelude/runtime/make_sequence.hpp"),
         CG.Include("prelude/runtime/tuple_utilities.hpp"),
         CG.Line('using namespace copperhead;')])
    wrapped_cuda_code = [CG.Line(M.compiler_output)]
    device_module.add_to_module(wrapped_cuda_code)
    M.device_module = device_module
    if M.compile:
        M.current_toolchains = (M.toolchains.host_toolchain,
                                M.toolchains.nvcc_toolchain)
    else:
        M.current_toolchains = (M.toolchains.null_host_toolchain,
                                M.toolchains.null_nvcc_toolchain)
    M.codepy_module = device_module
    M.code = (str(host_module.generate()), str(device_module.generate()))
    M.kwargs = dict(host_kwargs=dict(cache_dir=M.code_dir),
                    nvcc_kwargs=dict(cache_dir=M.code_dir),
                    debug=M.verbose)
    return []

def prepare_host_compilation(M):
    assert(len(M.entry_points) == 1)
    procedure_name = M.entry_points[0]
    hash, (wrap_type, wrap_name), wrap_args = M.wrap_info
    host_module = codepy.bpl.BoostPythonModule(max_arity=max(10,M.arity),
                                               use_private_namespace=False)
    host_module.add_to_preamble([CG.Include("prelude/prelude.h"),
                                 CG.Include("prelude/runtime/cunp.hpp"),
                                 CG.Include("prelude/runtime/make_cuarray.hpp"),
                                 CG.Include("prelude/runtime/make_sequence.hpp"),
                                 CG.Include("prelude/runtime/tuple_utilities.hpp"),
                                 CG.Line('using namespace copperhead;')])

    host_module.add_to_init([CG.Statement(
                "boost::python::def(\"%s\", &%s)" % (
                    procedure_name, wrap_name))])
    wrapped_code = [CG.Line(M.compiler_output),
                    CG.Line('using namespace %s;' % hash)]
    host_module.add_to_module(wrapped_code)
    M.codepy_module = host_module
    if M.compile:
        M.current_toolchains = (M.toolchains.host_toolchain,)
    else:
        M.current_toolchains = (M.toolchains.null_host_toolchain,)
    M.code = (str(host_module.generate()),)
    M.kwargs = dict(cache_dir=M.code_dir,
                    debug=M.verbose)
    return []

def make_binary(M):
    assert(len(M.entry_points) == 1)
    procedure_name = M.entry_points[0]

    code = M.code
    codepy_module = M.codepy_module
    toolchains = M.current_toolchains
    kwargs = M.kwargs
    try:
        module = codepy_module.compile(*toolchains, **kwargs)
    except Exception as e:
        if isinstance(e, NotImplementedError):
            raise e
        for m in code:
            print m
        print e
        raise e

    name = M.input_types.keys()[0]
    input_type = M.input_types[name]
    #Unmark name
    input_name = name[1:]
    
    copperhead_info = (input_name, input_type, M.tag)
    module_dir, module_file = os.path.split(module.__file__)
    info_file = open(os.path.join(module_dir, 'cuinfo'), 'w')
    pickle.dump(copperhead_info, info_file)
    info_file.close()
    return code, getattr(module, procedure_name)
