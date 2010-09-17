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

from __future__ import absolute_import

from collections import namedtuple
from .pltools import strlist

class BaseShape(object):

    # Make conversion to string formats less verbose than the default
    # namedtuple formatting.
    def __repr__(self):
        return self.__class__.__name__ + strlist(list(self), '()', form=repr)
    def __str__(self):
        return self.__class__.__name__ + strlist(list(self), '()', form=str)

    def _shapeof(self): return self


class Shape(namedtuple('Shape', 'extents element'), BaseShape):

    def __str__(self):
        if self.extents==[] and self.element is None:
            return "Unit"
        elif self.extents is None and self.element is None:
            return "Any"
        else:
            return BaseShape.__str__(self)

    def _extentof(self):  return self.extents
    def _elementof(self): return self.element



Unit = Shape([], None)    # indivisible values
Any  = Shape(None, None)  # any shape at all; normally this is never used

Unknown = str('Unknown')  # for extents that are unknown

class _IdentityShape(BaseShape):
    def __repr__(self): return "Identity"
    def __str__(self):  return "Identity"

Identity = _IdentityShape()

class ShapeOf(namedtuple('ShapeOf', 'value'), BaseShape):
    def _elementof(self): return elementof(self.value)
    def _extentof(self):  return extentof(self.value)

class ElementOf(namedtuple('ElementOf', 'value'), BaseShape):
    pass

class ResultOf(namedtuple('ResultOf', 'function parameters'), BaseShape):
    pass

def shapeof(s):
    fn = getattr(s, '_shapeof', None)
    return fn() if (fn is not None) else ShapeOf(s)

def elementof(s):
    fn = getattr(s, '_elementof', None)
    return fn() if (fn is not None) else ElementOf(s)

class ExtentOf(namedtuple('ExtentOf', 'value')):
    def __repr__(self):  return "ExtentOf(%r)" % (self.value,)
    def __str__(self):   return "ExtentOf(%s)" % (self.value,)
    

def extentof(s):
    fn = getattr(s, '_extentof', None)
    return fn() if (fn is not None) else ExtentOf(s)

class Equality(object):
    def __init__(self, values):
        self.values = values

    def __repr__(self):  return "Equality(%r)" % self.values
    def __str__(self):   return "Equality(%s)" % self.values

def eq(s1, s2):
    if isinstance(s1, ShapeOf):
        return Equality([s1,s2])
    elif isinstance(s2, ShapeOf):
        return Equality([s2,s1])
    else:
        return all([a==b for a,b in zip(s1.extents, s2.extents)])

def depthof(s, d=0):
    if extentof(s) is None:
        return None
    if not extentof(s) and not elementof(s):
        return d
    return depthof(elementof(s), d+1)

def flat_extentsof(s, i=[]):
    if elementof(s) is None:
        return i
    return flat_extentsof(elementof(s), i + [extentof(s)])

def shape_from_extents(i):
    if i:
        return Shape([i[0]], shape_from_extents(i[1:]))
    return Unit
