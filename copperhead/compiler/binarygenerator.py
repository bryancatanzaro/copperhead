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

def prepare_compilation(M):
    from ..runtime import omp_tag
    if M.tag == omp_tag:
        return prepare_omp_compilation(M)
    else:
        return prepare_cuda_compilation(M)

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
         CG.Line('using namespace copperhead;')])
    wrapped_cuda_code = [CG.Line(M.device_code)]
    device_module.add_to_module(wrapped_cuda_code)
    M.host_module = host_module
    M.device_module = device_module
    return []
    
def make_binary(M):
    assert(len(M.entry_points) == 1)
    procedure_name = M.entry_points[0]

    # XXX This import can't happen at the file scope because of import
    # dependency issues.  We should refactor things to avoid this workaround.
    from ..runtime import nvcc_toolchain, host_toolchain, omp_toolchain, omp_tag
    if M.tag == omp_tag:
        host_code = str(M.host_module.generate())
        try:
            module = M.host_module.compile(omp_toolchain, debug=M.verbose)
        except Exception as e:
            print(host_code)
            print e
            raise e
        return (host_code, device_code), getattr(module, procedure_name)
    else:
        host_code = str(M.host_module.generate())
        device_code = str(M.device_module.generate())   
        try:
            module = M.device_module.compile(host_toolchain, nvcc_toolchain, debug=M.verbose)
        except Exception as e:
            print(host_code)
            print(device_code)
            print e
            raise e
        return (host_code, device_code), getattr(module, procedure_name)

