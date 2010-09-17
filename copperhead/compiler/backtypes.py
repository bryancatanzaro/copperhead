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

from coretypes import *


class FnArg(Monotype):
    """The struct type of a function being passed as an argument"""
    def __init__(self, name):
        Monotype.__init__(self, "FnArg", name)
    def __str__(self):
        return str(self.parameters[0])

class DependentType(Monotype):
    def __init__(self, name, parameters=None):
        Monotype.__init__(self, "DependentType", name, *parameters)
    def __str__(self):
        return str(self.parameters[0]) + '<' + ', '.join([str(x) for x in self.parameters[1:]]) + ' >'
    def __repr__(self):
        return 'DependentType[' + repr(self.parameters[0]) + ']<' + ', '.join([repr(x) for x in self.parameters[1:]]) + ' >'

    
