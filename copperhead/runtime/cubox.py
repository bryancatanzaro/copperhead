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
import copperhead.compiler.backendsyntax as B
import copperhead.compiler.coresyntax as S
import copperhead.compiler.pltools as PL
import cufunction as cuf
import copy

class CuBox(cuf.CuFunction):
    def __init__(self, fn, *args):
        self.fn = fn
        self.__doc__ = fn.__doc__
        self.__name__ = fn.__name__
        self.cu_type  = getattr(fn, 'cu_type', None)
        self.cu_shape = getattr(fn, 'cu_shape', None)

        # Type inference is deferred until the first __call__
        # invocation.  This avoids the need for procedures to be defined
        # textually before they are called.
        self.inferred_type = None
        self.inferred_shape = None
        self.cache = {}
        self.code = {}
        self.syntax_tree = self.make_wrapper()
        self.preamble = set(args)
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
    def make_wrapper(self):
        """Generate a wrapper for this box function that will call itself"""
        fn_monotype = self.cu_type
        if isinstance(self.cu_type, T.Polytype):
            fn_monotype = fn_monotype.monotype()
        input_types = fn_monotype.input_types()
        wrapped_name = '_' + self.__name__ + '_wrap'
        name_supply = PL.name_supply()
        input_arguments = [S.Name(name_supply.next()) for x in input_types]
        call_arguments = copy.copy(input_arguments)
        wrapper = [S.Procedure(S.Name(wrapped_name),
                              input_arguments,
                              [S.Return(S.Apply(S.Name(self.__name__),
                                                call_arguments))])]
        return wrapper
