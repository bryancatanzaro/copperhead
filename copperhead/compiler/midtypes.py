#
#  Copyright 2008-2010 NVIDIA Corporation
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



class FnSide(Monotype):
    """A Function which returns information by stateful modification of its inputs."""
    def __init__(self, args):
        if not args:
            args = Void
        elif isinstance(args, list) or isinstance(args, tuple):
            args = Tuple(*args)
        Monotype.__init__(self, "FnSide", args)

    def __str__(self):
        arg_type = self.parameters
        return str(arg_type) 
    
