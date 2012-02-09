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

"""
Decorators for Copperhead procedure declarations
"""

def cu(fn):
    """
    Decorator for declaring that a procedure is visible in Copperhead.

    For example:
        @cu
        def plus1(x):
            return [xi + 1 for xi in x]
    """
    from runtime import CuFunction

    # Wrap Python procedure in CuFunction object that will intercept
    # calls (e.g., for JIT compilation).
    cufn = CuFunction(fn)

    return cufn

def cutype(type):
    """
    Decorator for declaring the type of a Copperhead procedure.

    For example:
        @cutype("[Int] -> Int")
        @cu
        def plus1(x):
            return [xi + 1 for xi in x]
    """
    from compiler.parsetypes import T, type_from_text

    if isinstance(type, str):
        type = type_from_text(type)
    elif not isinstance(type, T.Type):
        raise TypeError, \
                "type (%s) must be string or Copperhead Type object" % type

    def setter(fn):
        # Don't use CuFunction methods here, as we may be decorating a
        # raw Python procedure object.
        fn.cu_type = type
        return fn

    return setter

def cushape(shape):
    """
    Decorator for declaring the shape of a Copperhead procedure.

    This decorator expects a python function which, given input shapes,
    will compute a tuple: (output_shapes, output_constraints).
    
    For example:
        @cushape(lambda *x:  (Unit, []))
        @cushape(lambda a,b: (a, [sq(a,b)])

    This is largely meant for internal use.  User programs should
    generally never have declared shapes.

    """


    if not callable(shape):
        raise TypeError("%s is not a legal procedure shape" % shape)


    def setter(fn):
        fn.cu_shape = shape
        return fn
    return setter


def cuphase(*args):
    """
    Decorator for declaring the phase completion of a Copperhead procedure.

    This decorator expects a tuple consisting of input completion requirements
    and output completion declaration, which it then fashions into a phase
    procedure.
    
    For example:
        @cuphase((P.local, P.total), P.local)

    This is largely meant for internal use.  User programs should
    generally never have declared phases.

    """
    import compiler.phasetypes as P
    cu_phase = P.cuphase(*args)
    def setter(fn):
        fn.cu_phase = cu_phase
        return fn
    return setter

def cubox(*args):
    """
    Decorator for black box Copperhead procedures.

    This decorator accepts parameters: each parameter is a file to be included
    in compilation when this black box is compiled.

    For example:
        @cubox('wrappers/reduce.h')
    """
    from runtime import CuBox
    def setter(fn):
        return CuBox(fn, *args) 
    return setter
