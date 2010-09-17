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

from coresyntax import *
from coresyntax import _indent
from pltools import strlist

class While(Statement):
    'While loop'
    def __init__(self, condition, body):
        self.parameters = [condition] + body

    def __repr__(self):
        return 'While(%r, %r)' % (self.parameters[0], self.parameters[1])

    def __str__(self):
        body = _indent(strlist(self.parameters[1], sep='\n', form=str))
        return 'while %s:\n%s' % (self.parameters[0], body)
    
    def test(self):   return self.parameters[0]
    def body(self):   return self.parameters[1]

class PhaseBoundary(Statement):
    'Phase Boundary'
    def __init__(self, parameters):
        self.parameters = parameters

    def __repr__(self):
        return 'PhaseBoundary(%s)' % strlist(self.parameters)
    def __str__(self):
        return self.__repr__()

class Zip(Expression):
    def __init__(self, *args):
        self.parameters = args
    def __repr__(self):
        return 'Zip(%s)' % strlist(self.parameters, sep=', ', form=repr)
    def __str__(self):
        return 'zip(%s)' % strlist(self.parameters, sep=', ', form=str) 
    def __iter__(self):
        return iter(self.parameters)
