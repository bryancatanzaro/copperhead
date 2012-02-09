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

"""Type system for Copperhead.

This module defines the core type system intended for use in Copperhead
and related projects.  It is designed to be suitable for type inference
using the Hindley-Milner algorithm.

The type system is described by the following grammar:

    Type = Monotype | Polytype | Typevar

    Monotype = Con(Type1, ..., TypeN)  for all type constructors <Con>
    Polytype = [a1, ..., an] Type      for all type variables <ai>
    Typevar  = name                    for all strings <name>
"""

from pltools import strlist, name_supply

class Type:
    pass

        
class Monotype(Type):
    def __init__(self, name, *parameters):
        self.name = name
        self.parameters = parameters

    def __repr__(self):
        if not self.parameters:
            return self.name
        else:
            args = strlist(self.parameters, bracket='()', form=repr)
            return "%s%s" % (self.name, args)

    def __str__(self):
        if not self.parameters:
            return self.name
        else:
            args = strlist(self.parameters, bracket='()', form=str)
            return "%s%s" % (self.name, args)

    def __eq__(self, other):
        if not isinstance(other, Monotype):
            return False
        if self.name != other.name:
            return False
        if self.parameters != other.parameters:
            return False
        return True
    def __ne__(self, other):
        return not self.__eq__(other)
    def __hash__(self):
        return id(self)
    
class Polytype(Type):
    def __init__(self, variables, monotype):
        self.variables = variables
        self.parameters = (monotype,)

    def __repr__(self):
        return "Polytype(%r, %r)" % (self.variables, self.monotype())

    def __str__(self):
        vars = strlist(self.variables, form=str)
        return "ForAll %s: %s" % (vars, self.monotype())

    def monotype(self): return self.parameters[0]

    def __eq__(self, other):
        if not isinstance(other, Polytype):
            return False
        if self.variables != other.variables:
            return False
        return self.parameters == other.parameters
    def __ne__(self, other):
        return not self.__eq__(other)
    def __hash__(self):
        return id(self)
  
Int    = Monotype("Int")
Long   = Monotype("Long")
Float  = Monotype("Float")
Double = Monotype("Double")
Number = Monotype("Number")
Bool   = Monotype("Bool")
Void   = Monotype("Void")

class Fn(Monotype):
    def __init__(self, args, result):
        if not args:
            args = Void
        elif isinstance(args, list) or isinstance(args, tuple):
            args = Tuple(*args)
        elif not isinstance(args, Tuple):
            args = Tuple(args)

        Monotype.__init__(self, "Fn", args, result)

    def __str__(self):
        arg_type, result_type = self.parameters
        return str(arg_type) + " -> " + str(result_type)

    def input_types(self):
        arg = self.parameters[0]
        if isinstance(arg, Tuple):
            return arg.parameters
        else:
            return []

    def result_type(self):
        return self.parameters[1]
    
class Seq(Monotype):
    def __init__(self, eltype):  Monotype.__init__(self, "Seq", eltype)
    def __str__(self):  return "[" + str(self.parameters[0]) + "]"
    def unbox(self): return self.parameters[0]

class Tuple(Monotype):
    def __init__(self, *types):  Monotype.__init__(self, "Tuple", *types)
    def __str__(self):
        if len(self.parameters) > 1:
            return strlist(self.parameters, "()", form=str)
        else:
            if not(self.parameters):
                return '()'
            else:
                return str(self.parameters[0])

    def __iter__(self):
        return iter(self.parameters)
class Array(Monotype):
    def __init__(self, idxtype, eltype):
        Monotype.__init__(self, "Array", idxtype, eltype)


class Typevar(Type, str): pass

from itertools import ifilter, chain
import copy

def quantifiers(t):
    'Produce list of immediately bound quantifiers in given type.'
    return t.variables if isinstance(t, Polytype) else []


def names_in_type(t):
    'Yields the sequence of names occurring in the given type t'
    if isinstance(t, Typevar) or isinstance(t, str):
        yield t
    else:
        for n in chain(*[names_in_type(s) for s in t.parameters]):
            yield n

def free_in_type(t):
    'Yields the sequence of names occurring free in the given type t'
    if isinstance(t, Typevar) or isinstance(t, str):
        yield t
    else:
        bound = quantifiers(t)
        for n in chain(*[free_in_type(s) for s in t.parameters]):
            if n not in bound:
                yield n

def substituted_type(t, subst):
    """
    Return copy of T with all variables mapped to their values in SUBST,
    if any.  All names in SUBST must be unbound in T.
    """
    if isinstance(t, Typevar) or isinstance(t, str):
        return subst[t] if t in subst else t
    else:
        if isinstance(t, Polytype):
            for v in t.variables:
                assert v not in subst
        u = copy.copy(t)
        u.parameters = [substituted_type(p, subst) for p in t.parameters]
        return u

def occurs(id, t):
    'Returns true if the given identifier occurs as a name in type t'
    return id in names_in_type(t)

def quantify_type(t, bound=None, quantified=None):
    """
    If t is a type containing free type variables not occuring in bound,
    then this function will return a Polytype quantified over those
    variables.  Otherwise, it returns t itself.

    If an optional quantified dictionary is provided, that dictionary
    will be used to map free variables to quantifiers.  Any free
    variables not found in that dictionary will be mapped to fresh
    quantifiers, and the dictionary will be augmented with these new
    mappings.
    """

    if bound is None:
        free = list(free_in_type(t))
    else:
        free = list(ifilter(lambda t: t not in bound, free_in_type(t)))

    if not free:
        return t
    else:
        supply = name_supply()
        if quantified is None: quantified = dict()

        for v in free:
            if v not in quantified:
                quantified[v] = supply.next()

        quantifiers = sorted([quantified[v] for v in set(free)])
        return Polytype(quantifiers, substituted_type(t, quantified))
