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

"""
Support execution of intermediate forms generated in the compiler.
"""

from __future__ import absolute_import

import sys

from . import places
from ..compiler import passes, coretypes as T
from .. import prelude, interlude


def _typeof(x):
    """
    Compute the appropriate Copperhead type for a given Python object.
    This is a quick little procedure to suit the needs of these
    intermediate execution places.  It should really be replaced by
    suitable functionality elsewhere in the compiler/runtime.
    """
    if isinstance(x, int):
        return T.Int
    elif isinstance(x, float):
        return T.Float
    elif isinstance(x, tuple):
        return T.Tuple( *[_typeof(y) for y in x] )
    elif isinstance(x, list):
        return T.Seq(_typeof(x[0]))
    else:
        return ValueError, "%s has no Copperhead type" % x

class Intermediate(places.PythonInterpreter):
    """
    Intermediate places compile code through a portion of the Copperhead
    compilation pipeline and then execute the results in the native
    Python interpreter.
    """

    def __init__(self, target):
        self.compilation_target = target

    def execute(self, cufn, args, kwargs):

        name = cufn.__name__
        text = passes.compile(cufn.get_source(),
                              globals=cufn.get_globals(),
                              target=self.compilation_target,
                              inputTypes={name : map(_typeof, args)})

        bindings = dict(cufn.get_globals())
        bindings.update(interlude.__dict__)

        try:
            exec text in bindings

            fn2 = bindings[name]
            bindings['__entry__'] = lambda: fn2(*args)
            result = eval('__entry__()',  bindings)
        except:
            print
            print "ERROR IN INTERMEDIATE CODE:", sys.exc_value
            print text
            print
            raise

        return result

places.frontend = Intermediate(passes.frontend)

import pdb
def print_and_pause(name, ast, M):
    print
    print "after", name, ":"
    code = passes.ast_to_string(ast)
    print passes.S._indent(code)
    pdb.set_trace()

def print_repr(name, ast, M):
    print "after", name, ":"
    print repr(ast)

class tracing(object):
    """
    Tracing objects are context managers for Python 'with' statements.
    They are a debugging tool that allows top-level code to capture a
    program in various stages of the compiler pipeline.  The default
    action taken is to simply print the source code.

    This facility is only meant for use by those familiar with the
    compiler internals.  It is unlikely to ever be useful for others.

    For example, to print the result of every Copperhead function at the
    end of the front-end compiler, simply do this:

        with tracing(parts=[passes.frontend], including=['END frontend']):
            # invoked Copperhead functions here
    """

    def __init__(self, action=None,
                       parts=[passes.frontend],
                       including=None,
                       excluding=[]):

        self.action = action
        self.parts = parts
        self.including = including
        self.excluding = excluding

        self._saved = []

    def send(self, data):
        name, ast, M = data
        if (name not in self.excluding) and \
                (not self.including or name in self.including):

            if self.action is None:
                print
                print "after", name, ":"

                code = passes.ast_to_string(ast)
                print passes.S._indent(code)
            else:
                self.action(name, ast, M)

    def __enter__(self):
        self._saved = [part.capture for part in self.parts]
        for part in self.parts:
            part.capture = self

    def __exit__(self, *args):
        for part, old in zip(self.parts, self._saved):
            part.capture = old
