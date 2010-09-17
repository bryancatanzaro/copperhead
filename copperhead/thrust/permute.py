#
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

from copperhead.decorators import cutype, cubox
from copperhead.runtime import nvcc_toolchain, host_toolchain, cudata, cuarray
import copperhead.compiler.passes as p
import codepy.bpl
import codepy.cuda
from codepy.cgen import *
import copy


@cubox
@cutype("([a], [Int]) -> [a]")
def permute(source, indices, cache):
    source.using_remote()
    indices.using_remote()
    
    result_type = cudata.cu_to_c_types[str(source.type.unbox())]
    if result_type in cache:
        compiled_function = cache[result_type]
        return compiled_function(source, indices)
    
    host_module = codepy.bpl.BoostPythonModule()
    host_module.add_function(
        FunctionBody(
            FunctionDeclaration(Value("void", "entryPoint"),
                                [Pointer(Value("PyObject", "x")),
                                 Pointer(Value("PyObject", "indices")),
                                 Pointer(Value("PyObject", "outputArray"))]),
            Block([Statement(x) for x in [
                        '//%s' % result_type,  # This is a hack due to a
                        # bug in the way codepy invokes the cuda compiler
                        'CuArray cuX = extractArray(x)',
                        'CuArray cuI = extractArray(indices)',
                        'CuArray output = extractArray(outputArray)',
                        'permuteInstance(cuX.devicePointer, cuI.devicePointer, cuI.length, output.devicePointer)',
                        'return',
                        ]])))
    host_module.add_to_preamble([
            Include('cuarray.h'),
            Include('boost/python/extract.hpp')])

    device_module = codepy.cuda.CudaModule(host_module)
    device_module.add_to_preamble([Include('wrappers/permute.h')])
    permuteInstance = FunctionBody(
        FunctionDeclaration(Value('void', 'permuteInstance'),
                            [Value('CUdeviceptr', 'x'),
                             Value('CUdeviceptr', 'i'),
                             Value('int', 'length'),
                             Value('CUdeviceptr', 'o')]),
        Block([Statement(x) for x in [
                    'permute<%s>((%s*)x, (int*)i, length, (%s*)o)' % (result_type, result_type, result_type),
                    'return']]))
    device_module.add_function(permuteInstance)

    
    module = device_module.compile(host_toolchain, nvcc_toolchain, debug=True)
    def fn(source, indices):
        output_shape = copy.deepcopy(indices.shape)
        output_type = copy.copy(source.type)
        output = cuarray.CuArray(shape=output_shape, type=output_type)
        module.entryPoint(source, indices, output)
        return output
    cache[result_type] = fn
    return fn(source, indices)

    
