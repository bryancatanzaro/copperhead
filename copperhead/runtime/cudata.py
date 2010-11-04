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


import copperhead.compiler.coretypes as T
import copperhead.compiler.shapetypes as ST
import numpy as np
import cuarray as CA
import itertools

np_to_cu_types = {'int32'   : T.Int,
                  'int64'   : T.Long,
                  'float32' : T.Float,
                  'float64' : T.Double,
                  'bool'    : T.Bool}

cu_to_np_types = {'Int' : 'i4',
                  'Long' : 'i8',
                  'Float' : 'f4',
                  'Double' : 'f8',
                  'Bool' : 'b'}

py_to_cu_types = {'int' : T.Int,
                  'float': T.Double}

cu_to_c_types = {'Int' : 'int',
                 'Long' : 'long',
                 'Float' : 'float',
                 'Double' : 'double',
                 'Bool' : 'bool'}

cu_to_py_types = {'Int' : 'i',
                  'Long' : 'i',
                  'Float' : 'f',
                  'Double' : 'f'}

index_type = 'i4'

def find_base_type(cutype):
    if isinstance(cutype, T.Seq):
        return find_base_type(cutype.unbox())
    if not isinstance(cutype, T.Tuple):
        return cu_to_np_types[str(cutype)]
    return ','.join((cu_to_np_types[str(x)] for x in cutype.parameters))


class DataException(Exception):
    pass

def induct(data, typ=None):
    if isinstance(data, CuData):
        return data
    if isinstance(data, tuple):
        return CuTuple(data, typ)
    if hasattr(data, '__iter__'):
        return CA.CuArray(data)
    if isinstance(data, np.float32) or typ == T.Float:
        return CuFloat(data)
    if isinstance(data, np.float64) or isinstance(data, float):
        return CuDouble(data)
    if isinstance(data, int) or isinstance(data, np.int32):
        return CuInt(data)
    if isinstance(data, np.int64):
        return CuLong(data)
    if isinstance(data, np.bool_) or isinstance(data, bool):
        return CuBool(data)
    raise DataException("Unknown data type")


class Operator(object):
    def __init__(self, instance, op):
        self.instance = instance
        self.op = op
        self.__doc__ = self.op.__doc__
    def __call__(self, *args):
        def conv(x):
            if isinstance(x, CuData):
                return x.value
            else:
                if isinstance(x, self.instance.value_class):
                    return x
                else:
                    # XXX This rule converts all arguments to the same class
                    # as the instance being operated on.
                    # It's a workaround for cases such as the following:
                    # np.float32(1.0) * CuFloat(1.0)
                    # Because np.float32.__mul__() doesn't know what to do
                    # with a CuFloat object, Python tries calling
                    # CuFloat.__rmul__(), but somehow the np.float32
                    # presents itself as a Python float object instead of
                    # an np.float32.  Then, the result is a Python float object
                    # and Copperhead perceives it as a CuDouble.

                    # This introduces screwiness:
                    # CuFloat(1.0) + CuDouble(1.0) yields CuDouble(2.0)
                    # and
                    # CuDouble(1.0) + CuFloat(1.0) yields CuDouble(2.0)
                    # as you might expect

                    # However,
                    # CuFloat(1.0) + np.float64(1.0) yields CuFloat(2.0)
                    # and
                    # np.float64(1.0) + CuFloat(1.0) yields CuFloat(2.0)
                    # Similarly with Python floats:
                    # CuFloat(1.0) + 1.0 = CuFloat(2.0)
                    # and
                    # 1.0 + CuFloat(1.0) + CuFloat(2.0)

                    # I'm ok with this for now, I think Copperhead has a
                    # strictly more restrictive typesystem, so this behavior
                    # doesn't particularly bother me.  But it needs to be
                    # documented (or else this whole scheme needs to be
                    # reworked.)
                    
                    return self.instance.value_class(x)
        converted_args = map(conv, (self.instance,) + args)
        result = self.op(*converted_args)
        return induct(result)
        

class OperatorDescriptor(object):
    def __init__(self, op):
        self.op = op
    def __get__(self, obj, type=None):
        return Operator(obj, self.op)
    def __set__(self, obj, value):
        raise TypeError
    def __delete__(self, obj):
        raise TypeError

class DelegateMetaclass(type):
    forward = ['add', 'div', 'floordiv', 'mul', 'mod', 'pow', 'sub', 'truediv', 'lshift', 'rshift']
    backwards = ['r' + x for x in forward]
    neither = ['invert', 'eq', 'ge', 'gt', 'le', 'lt', 'ne', 'neg', 'pos']
    uni = ['neg', 'pos']
    operators = ['__' + x + '__' for x in forward + backwards + neither + uni]
    
    def __init__(cls, name, bases, dct):
        for attr in DelegateMetaclass.operators:
            setattr(cls, attr, OperatorDescriptor(getattr(cls.value_class, attr)))

            
class CuData(object):
    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, self.value_class.__repr__(self.value))
    def __str__(self):
        return self.value_class.__str__(self.value)

class CuScalar(CuData):
    value_class = None
    shape = ST.Unit
    def __nonzero__(self):
        return self.value.__nonzero__()

class CuDouble(CuScalar):
    value_class = np.float64
    __metaclass__ = DelegateMetaclass
    
    def __init__(self, value=0.0):
        self.value = self.value_class(value)
        self.type = T.Double

class CuFloat(CuScalar):
    value_class = np.float32
    __metaclass__ = DelegateMetaclass
    
    def __init__(self, value=0.0):
        self.value = self.value_class(value)
        self.type = T.Float

class CuInt(CuScalar):
    value_class = np.int32
    __metaclass__ = DelegateMetaclass

    def __init__(self, value=0):
        self.value = self.value_class(value)
        self.type = T.Int

class CuLong(CuScalar):
    value_class = np.int64
    __metaclass__ = DelegateMetaclass

    def __init__(self, value=0):
        self.value = self.value_class(value)
        self.type = T.Long
        
class CuBool(CuScalar):
    value_class = np.bool_
    __metaclass__ = DelegateMetaclass
    
    def __init__(self, value=False):
        self.value = self.value_class(value)
        self.type = T.Bool


class CuTuple(CuData):
    def __init__(self, data, typ=None):
        if typ is None:
            typ = itertools.repeat(None)
        self.value = tuple((induct(x, t) for x, t in zip(data, typ)))
        self.shape = ST.Unit
        self.type = T.Tuple(*[x.type for x in self.value])
    def __repr__(self):
        return 'CuTuple' + repr(self.value)
    def __str__(self):
        return '(' + ', '.join((str(x) for x in self.value)) + ')'
    def __iter__(self):
        return self.value.__iter__()


        
