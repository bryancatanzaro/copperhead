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
import codepy.bpl
import codepy.cuda
from codepy.cgen import *
import copy

@cubox
@cutype("([a]) -> [a]")
def sort(array, cache):
    array.using_remote()
    result_type = cudata.cu_to_c_types[str(array.type.unbox())]
    if result_type in cache:
        compiled_function = cache[result_type]
        return compiled_function(array)
    
    host_module = codepy.bpl.BoostPythonModule()
    host_module.add_function(
        FunctionBody(
            FunctionDeclaration(Value("void", "entryPoint"),
                                [Pointer(Value("PyObject", "cuArray")),
                                 Pointer(Value("PyObject", "outputArray"))]),
            Block([Statement(x) for x in [
                        '//%s' % result_type,  # This is a hack due to a
                        # bug in the way codepy invokes the cuda compiler
                        'CuArray input = extractArray(cuArray)',
                        'CuArray output = extractArray(outputArray)',
                        'sortInstance(input.devicePointer, input.length, output.devicePointer)',
                        'return']])))
    host_module.add_to_preamble([Include('cuarray.h'),
                                 Include('boost/python/extract.hpp')])

    device_module = codepy.cuda.CudaModule(host_module)
    device_module.add_to_preamble([Include('wrappers/sort.h')])
    sumInstance = FunctionBody(
        FunctionDeclaration(Value('void', 'sortInstance'),
                            [Value('CUdeviceptr', 'inputPtr'),
                             Value('int', 'length'),
                             Value('CUdeviceptr', 'outputPtr')]),
        Block([Statement(x) for x in [
                    'sort<%s>((%s*)inputPtr, length, (%s*)outputPtr)' % (result_type, result_type, result_type),
                    'return']]))
    device_module.add_function(sumInstance)
    print(device_module.generate())
    print(host_module.generate())
    module = device_module.compile(host_toolchain, nvcc_toolchain)
    def fn(array):
        output_shape = copy.deepcopy(array.shape)
        output_type = copy.copy(array.type)
        output = cuarray.CuArray(shape=output_shape, type=output_type)
        module.entryPoint(array, output)
        return output
    cache[result_type] = fn
    return fn(array)
                        
                   
                        
    
    
    
