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
import numpy as np
from copperhead.compiler import passes

import places

class Cuda(places.Place):
    pass
   
class DefaultCuda(Cuda):
    def execute(self, cufn, args, kwargs):
        return execute(cufn, *args, **kwargs)


def execute(cufn, *v, **k):
    #XXX need to provide induction here
    cu_inputs = [x for x in v]
    cu_types = [x.type for x in cu_inputs]
    signature = ','.join([str(x) for x in cu_types])
    if signature in cufn.cache:
        return cufn.cache[signature](*cu_inputs)
    
    ast = cufn.get_ast()
    name = ast[0].name().id
    code, compiled_fn = \
                 passes.compile(cufn.get_ast(),
                                globals=cufn.get_globals(),
                                input_types={name : cu_types},
                                **k)
    cufn.cache[signature] = compiled_fn
    cufn.code[signature] = code
    return_value = compiled_fn(*cu_inputs)

    return return_value
