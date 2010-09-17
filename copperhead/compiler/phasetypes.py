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

class Phase(object):
    def __init__(self, name, val=0):
        self.name = name
        self.val = val
    def __str__(self):
        return self.name
    def __repr__(self):
        return "<Phase: %s>" % self
    def __copy__(self):
        #Enforce singleton
        return self
    def __deepcopy__(self, memo):
        #Enforce singleton
        return self
    def __cmp__(self, other):
        if not isinstance(other, Phase):
            raise TypeError
        return self.val - other.val
    
Phase.total = Phase('total', 2)
Phase.local = Phase('local', 1)
Phase.none =  Phase('none', 0)
Phase.unknown = Phase('unknown')

total = Phase.total
local = Phase.local
none = Phase.none
unknown = Phase.unknown


def cuphase(*args):
    input_completion = args[0]
    output_completion = args[1]
    def constructed(*input_args):
        input_phases = []
        for decl, given in zip(input_completion, input_args):
            if given is unknown:
                input_phases.append(decl)
            else:
                if decl > given:
                    input_phases.append(total)
                else:
                    input_phases.append(given)
        return input_phases, output_completion
    return constructed

