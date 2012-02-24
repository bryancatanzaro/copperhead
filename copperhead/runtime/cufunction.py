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

from __future__ import with_statement     # make with visible in Python 2.5
from __future__ import absolute_import

import inspect
from ..compiler import pyast, typeinference
from . import places

class CuFunction:

    def __init__(self, fn):
        self.fn = fn
        self.__doc__ = fn.__doc__
        self.__name__ = fn.__name__

        # Copy attributes that may have been set by Copperhead decorators.
        self.cu_type  = getattr(fn, 'cu_type', None)
        self.cu_shape = getattr(fn, 'cu_shape', None)
        self.cu_phase = getattr(fn, 'cu_phase', None)

        # Type inference is deferred until the first __call__
        # invocation.  This avoids the need for procedures to be defined
        # textually before they are called.
        self.inferred_type = None
        self.inferred_shape = None

        # Parse and cache the Copperhead AST for this function
        stmts = pyast.statement_from_text(self.get_source())
        self.syntax_tree = stmts
        self.cache = {}
        self.code = {}
        
    def __call__(self, *args, **kwargs):
        # XXX This type check should be done
        # XXX It is commented out due to a small bug in type inference
        # XXX regarding gathering multiple @cu functions before inferring
        
        #if self.inferred_type is None:
        #    self.infer_type()
        P = kwargs.pop('target_place', places.default_place)
        return P.execute(self, args, kwargs)

    def get_source(self):
        """
        Return a string containing the source code for the wrapped function.

        NOTE: This will only work if the function was defined in a file.
        We have no access to the source of functions defined at the
        interpreter prompt.
        """
        return inspect.getsource(self.fn)

    def get_globals(self):
        """
        Return the global namespace in which the function was defined.
        """
        return self.fn.func_globals

    def get_ast(self):
        """
        Return the cached Copperhead AST.
        """
        return self.syntax_tree

    def python_function(self):
        """
        Return the underlying Python function object for this procedure.
        """
        return self.fn

    def infer_type(self):
        """
        Every Copperhead function must have a valid static type.  This
        method infers the most general type for the wrapped function.
        It will raise an exception if the function is not well-typed.
        """
        typer = typeinference.TypingContext(globals=self.get_globals())

        typeinference.infer(self.syntax_tree, context=typer)
        self.inferred_type = self.syntax_tree[0].name().type

        # XXX TODO: Should unify inferred_type with cu_type, if any
        if not self.cu_type:
            self.cu_type = self.inferred_type

        return self.inferred_type

    def get_code(self):
        return self.code

