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
import pycuda
import pycuda.gpuarray as gp
import numpy as np

import cudata as CD
from copperhead.compiler import passes

import places

class Cuda(places.Place):
    pass
   
class DefaultCuda(Cuda):
    
    def __init__(self):
        self.allocator = pycuda.tools.DeviceMemoryPool()
    def cleanup(self):
        self.allocator.stop_holding()
    def allocate(self, x):
        gpudata = self.allocator.allocate(x.nbytes)
        return self.from_data(x.shape, x.dtype, gpudata)
        
    def new_copy(self, x):
        assert isinstance(x, np.ndarray)
        return gp.to_gpu(x, allocator=self.allocator.allocate)

    def from_data(self, shape, dtype, gpudata):
        return gp.GPUArray(shape=shape, dtype=dtype, gpudata=gpudata)

    def update(self, x, idx, val):
        for i in range(len(idx)):
            dev_pointer = x[idx[i]:].gpudata
            host_pointer = val[i:i+1]
            pycuda.driver.memcpy_htod(dev_pointer, host_pointer)
            
    def extract(self, x, idx):
        result = np.ndarray(shape = len(idx), dtype=x.dtype)
        for i in range(len(idx)):
            host_pointer = result[i:i+1]
            dev_pointer = x[idx[i]:].gpudata
            pycuda.driver.memcpy_dtoh(host_pointer, dev_pointer)
        return result
    def execute(self, cufn, args, kwargs):
        return execute(cufn, *args, **kwargs)


def execute(cufn, *v, **k):
    # This doesn't save the induced variables for later use
    # People may be upset if they end up accidentally repeatedly
    # inducing large arrays to Copperhead types (it's expensive).
    cu_inputs = [CD.induct(x) for x in v]
    cu_types = [x.type for x in cu_inputs]
    signature = ','.join([str(x) for x in cu_types])
    if signature in cufn.cache:
        return cufn.cache[signature](*cu_inputs)
    cu_shapes = [x.shape for x in cu_inputs]
    cu_places = [x.place for x in cu_inputs if hasattr(x, 'place')]
    cu_uniforms = [n for n, x in enumerate(cu_inputs) if \
                   isinstance(x, CD.CA.CuUniform)] 
    
    ast = cufn.get_ast()
    name = ast[0].name().id
    code, compiled_fn = \
                 passes.compile(cufn.get_ast(),
                                globals=cufn.get_globals(),
                                input_types={name : cu_types},
                                input_shapes={name : cu_shapes},
                                input_places={name : cu_places},
                                uniforms=cu_uniforms,
                                **k)
    cufn.cache[signature] = compiled_fn
    cufn.code[signature] = code
    return_value = compiled_fn(*cu_inputs)

    return return_value
