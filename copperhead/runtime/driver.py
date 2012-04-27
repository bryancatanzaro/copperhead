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
import tags

from . import cuda_support, omp_support, tbb_support

class Sequential(places.Place):
    def __str__(self):
        return "Sequential"
    def __repr__(self):
        return str(self)
    def tag(self):
        return tags.cpp
    def execute(self, cufn, args, kwargs):
        return execute(self.tag(), cufn, *args, **kwargs)

if cuda_support:
    class Cuda(places.Place):
        def __str__(self):
            return "Cuda"
        def __repr__(self):
            return str(self)
        def tag(self):
            return tags.cuda

    class DefaultCuda(Cuda):
        def execute(self, cufn, args, kwargs):
            return execute(self.tag(), cufn, *args, **kwargs)
if omp_support:
    class OpenMP(places.Place):
        def __str__(self):
            return "OpenMP"
        def __repr__(self):
            return str(self)
        def tag(self):
            return tags.omp
        def execute(self, cufn, args, kwargs):
            return execute(self.tag(), cufn, *args, **kwargs)

if tbb_support:
    class TBB(places.Place):
        def __str__(self):
            return "TBB"
        def __repr__(self):
            return str(self)
        def tag(self):
            return tags.tbb
        def execute(self, cufn, args, kwargs):
            return execute(self.tag(), cufn, *args, **kwargs)
    
def induct(x):
    from . import cudata
    """Compute Copperhead type of an input, also convert data structure"""
    if isinstance(x, cudata.cuarray):
        return (conversions.back_to_front_type(x.type), x)
    if isinstance(x, np.ndarray):
        induced = cudata.cuarray(x)
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
        induced = cudata.cuarray(np.array(x))
        return (conversions.back_to_front_type(induced.type), induced)
    if isinstance(x, float):
        #Treat Python floats as double precision
        return (coretypes.Double, np.float64(x))
    if isinstance(x, int):
        #Treat Python ints as 64-bit ints (following numpy)
        return (coretypes.Long, np.int64(x))
    if isinstance(x, bool):
        return (coretypes.Bool, np.bool(x))
    if isinstance(x, tuple):
        sub_types, sub_elements = zip(*(induct(y) for y in x))
        return (coretypes.Tuple(*sub_types), tuple(sub_elements))
    #Can't digest this input
    raise ValueError("This input is not convertible to a Copperhead data structure: %r" % x)
    
def execute(tag, cufn, *v, **k):
    """Call Copperhead function. Invokes compilation if necessary"""

    if len(v) == 0:
        #Functions which take no arguments
        cu_types, cu_inputs = ((),())
    else:
        cu_types, cu_inputs = zip(*map(induct, v))
    #Derive unique hash for function based on inputs and target place
    signature = ','.join([str(tag)]+[str(x) for x in cu_types])
    #Have we executed this function before, in which case it is loaded in cache?
    if signature in cufn.cache:
        return cufn.cache[signature](*cu_inputs)

    #XXX can't we get rid of this circular dependency?
    from . import toolchains
    #Compile function
    ast = cufn.get_ast()
    name = ast[0].name().id
    code, compiled_fn = \
                 passes.compile(ast,
                                globals=cufn.get_globals(),
                                input_types={name : cu_types},
                                tag=tag,
                                code_dir=cufn.code_dir,
                                toolchains=toolchains,
                                **k)
    #Store the binary and the compilation result
    cufn.cache[signature] = compiled_fn
    #Call the function
    return compiled_fn(*cu_inputs)
