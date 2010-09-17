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

import copperhead.compiler.phasetypes as P
import copperhead.compiler.coretypes as T

class CuBox(object):
    def __init__(self, fn):
        self.fn = fn
        if hasattr(fn, 'cu_type'):
            self.cu_type = fn.cu_type
        self.compiled = {}
        self.__name__ = fn.__name__
    def __call__(self, *args):
        scope = self.fn.func_globals
        args_cache = args + (self.compiled,)
        return self.fn(*args_cache)
    def cu_phase(self, *args):
        type = self.cu_type
        if isinstance(type, T.Polytype):
            type = type.monotype()
        input_types = type.input_types()
        input_phases = []
        for type, arg in zip(input_types, args):
            if not isinstance(type, T.Fn):
                input_phases.append(P.total)
            else:
                input_phases.append(P.none)
        return input_phases, P.none
        
