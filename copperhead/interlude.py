#
#   Copyright 2008-2012 NVIDIA Corporation
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
Copperhead Interlude

The Copperhead compiler transforms input programs through a number of
discrete phases before generating C code in the back-end.  These
intermediate forms of the program are all valid Python programs,
although they may or may not be valid Copperhead programs.  The
Copperhead Interlude provides implementations of various special
constructions used by the compiler.
"""

class mutable:
    """
    The Copperhead mid-end uses mutable blocks to explicitly declare
    what identifiers may be rebound (e.g., in loop bodies).  This
    context manager object makes such constructions syntactically valid.

    In native Python, a mutable block essentially does nothing, since
    all Python identifiers may be re-assigned.

        >>> x = 1
        >>> with mutable(x):
        ...    x = x + 1
        >>> print x
        2

    The only real effect in Python is simply that the variables named as
    arguments to mutable must already be defined.

    The typical use of mutable blocks within the Copperhead compiler is
    in conjunction with variables being modified in loop bodies.

        count = 0
        i = 0
        with mutable(i, count):
            while i<len(A):
                if A[i]:
                    count = count + 1
                i = i + 1
    """

    def __init__(self, *vars): pass
    def __enter__(self): pass
    def __exit__(self, *args): pass

class closure:
    """
    The Copperhead compiler transforms procedures and lambdas that close
    over external values into explicit closure objects.
    
    
    Consider a procedure like the following example:

        >>> def original_scale(a, X):
        ...    def mul(x):
        ...        return a*x
        ...    return map(mul, X)
        >>> original_scale(2, [1, 2, 3, 4])
        [2, 4, 6, 8]

    The inner procedure mul() closes over the value of a in the body of
    scale.  The Copperhead compiler will transform this procedure into
    one where the closed values are explicitly captured with a closure()
    object and explicitly passed as arguments to mul().

        >>> def transformed_scale(a, X):
        ...    def mul(x, _K0):
        ...        return _K0*x
        ...    return map(closure([a], mul), X)
        >>> transformed_scale(2, [1, 2, 3, 4])
        [2, 4, 6, 8]
    """

    def __init__(self, K, fn):
        self.K = K
        self.fn = fn

    def __call__(self, *args):
        args = list(args) + list(self.K)
        return self.fn(*args)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
