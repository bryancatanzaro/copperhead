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
import tempfile
import os.path
from .. import compiler
from . import places
import fnmatch
import os
import pickle

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

        # Parse and cache the Copperhead AST for this function
        stmts = compiler.pyast.statement_from_text(self.get_source())
        self.syntax_tree = stmts
        # Establish code directory
        self.code_dir = self.get_code_dir()
        self.cache = self.get_cache()
        self.code = {}
        
    def __call__(self, *args, **kwargs):
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
        typer = compiler.typeinference.TypingContext(globals=self.get_globals())

        compiler.typeinference.infer(self.syntax_tree, context=typer)
        self.inferred_type = self.syntax_tree[0].name().type

        # XXX TODO: Should unify inferred_type with cu_type, if any
        if not self.cu_type:
            self.cu_type = self.inferred_type

        return self.inferred_type

    def get_code(self):
        return self.code

    def get_code_dir(self):
        #Rationale for the default code directory location:
        # PEP 3147
        # http://www.python.org/dev/peps/pep-3147/
        #
        # Which standardizes the __pycache__ directory as a place to put
        # compilation artifacts for python programs
        source_dir, source_file = os.path.split(inspect.getsourcefile(self.fn))
        candidate = os.path.join(source_dir, '__pycache__', source_file, self.__name__)
        
        if os.path.exists(candidate):
            return candidate
        try:
            os.makedirs(candidate)
            return candidate
        except OSError:
            #Fallback!
            #Can't create a directory where the source file lives
            #(Maybe the source file is in a system directory)
            #Let's put it in a tempdir which we know will be writable
            candidate = os.path.join(tempfile.gettempdir(),
                                     'copperhead-cache-uid%s' % os.getuid(),
                                     source_file, self.__name__)
            if os.path.exists(candidate):
                return candidate
            #No check here to ensure this succeeds - fatal error if it fails
            os.makedirs(candidate)    
            return candidate

    def get_cache(self):
        #XXX Can't we get rid of this circular dependency?
        from . import toolchains
        cache = {}
        cuinfos = []
        for r, d, f in os.walk(self.code_dir):
            for filename in fnmatch.filter(f, 'cuinfo'):
                cuinfos.append(os.path.join(r, filename))
            
        for cuinfo in cuinfos:
            cuinfo_file = open(cuinfo, 'r')
            input_name, input_type, tag = pickle.load(cuinfo_file)
            cuinfo_file.close()

            if input_name == self.__name__:
                try:
                    input_types = {}
                    input_types[input_name] = input_type
                    
                    code, compiled_fn = \
                        compiler.passes.compile(self.get_ast(),
                                                globals = self.get_globals(),
                                                input_types=input_types,
                                                tag=tag,
                                                code_dir=self.code_dir,
                                                toolchains=toolchains,
                                                compile=False)
                    signature = ','.join([str(tag)]+[str(x) for x in input_type])
                    cache[signature] = compiled_fn

                except:
                    # We don't process exceptions at this point
                    pass
            
        return cache
