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
import numpy as np
from copperhead.compiler import passes, conversions, coretypes

import places

class Cuda(places.Place):
    pass
   
class DefaultCuda(Cuda):
    def execute(self, cufn, args, kwargs):
        return execute(cufn, *args, **kwargs)


def induct(x):
    from . import cudata
    """Compute Copperhead type of an input, also convert data structure"""
    if isinstance(x, cudata.CuArray):
        return (conversions.back_to_front_type(x.type), x)
    if isinstance(x, np.ndarray):
        induced = cudata.CuArray(x)
        return (conversions.back_to_front_type(induced.type), induced)
    if isinstance(x, np.float32):
        return (coretypes.Float, x)
    if isinstance(x, np.float64):
        return (coretypes.Double, x)
    if isinstance(x, np.int32):
        return (coretypes.Int, x)
    if isinstance(x, np.int64):
        return (coretypes.Long, x)
    if isinstance(x, np.bool):
        return (coretypes.Bool, x)
    if isinstance(x, list):
        induced = cudata.CuArray(np.array(x))
        return (conversions.back_to_front_type(induced.type), induced)
    if isinstance(x, float):
        #Treat Python floats as double precision
        return (coretypes.Double, np.float64(x))
    if isinstance(x, int):
        #Treat Python ints as 64-bit ints (following numpy)
        return (coretypes.Long, np.int64(x))
    
def execute(cufn, *v, **k):
    cu_types, cu_inputs = zip(*map(induct, v))
    signature = ','.join([str(x) for x in cu_types])
    if signature in cufn.cache:
        return cufn.cache[signature](*cu_inputs)

    ast = cufn.get_ast()
    name = ast[0].name().id
    code, compiled_fn = \
                 passes.compile(ast,
                                globals=cufn.get_globals(),
                                input_types={name : cu_types},
                                **k)
    cufn.cache[signature] = compiled_fn
    cufn.code[signature] = code
    return_value = compiled_fn(*cu_inputs)

    return return_value
