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

import numpy as np
import places
import copperhead.compiler.shapetypes as S
import copperhead.compiler.utility as utility
from cudata import *
import itertools

def op_mul(a, b):
    return a * b
def op_add(a, b):
    return a + b
def scan(f, A):
    B = list(A)

    for i in xrange(1, len(B)):
        B[i] = f(B[i-1], B[i])

    return B
def exclusive_scan(f, prefix, A):
    return scan(f, [prefix] + A[:-1])


def _make_type(depth, base):
    if depth == 0:
        return base
    return T.Seq(_make_type(depth-1, base))

def _prod(x):
    return reduce(op_mul, x)

def iterate(arr):
    for idx in xrange(arr.shape.extents[0]):
        yield arr[idx]

def strip(arr):
    def pull(x):
        if isinstance(x, CuData):
            return x.value
        else:
            return x
    return [pull(x) for x in arr]

class CuArray(CuData):
    def __init__(self, input=None, shape=None, type=None, place=None,
                 remote=None):
        

        self.place = place or places.default_place
        self.base = None
        if input is None and remote is None:
            #We are making a blank array
            #This constructor can only make homogeneous nested 1-D arrays
            assert shape is not None
            assert type is not None
            self.shape = shape
            self.type = type
            depth = S.depthof(shape)
            self.depth = depth
            self.local = []
            flat_shape = S.flat_extentsof(shape)
            for extent in flat_shape:
                if len(extent) > 1:
                    raise DataException("Copperhead doesn't yet support multidimensional arrays")
            flat_shape = list(utility.flatten(flat_shape))
            
            for rank in xrange(depth - 1):
                inc = _prod(flat_shape[rank + 1:])
                stop = flat_shape[rank] * inc + 1
                self.local.append(np.arange(0, stop, inc, dtype=index_type))
            
            nitems = _prod(flat_shape)
            dtype = find_base_type(type)
            self.local.append(np.empty(nitems, dtype=dtype))
            self.remote = [self.place.allocate(x) for x in self.local]
            self.local_valid = False
            self.remote_valid = True
            return
        if remote is None:
            #We are making an array from a Python datatype

            #Is it directly a Numpy Array?
            if isinstance(input, np.ndarray):
                self.depth = 1
                self.shape = S.shape_from_extents(input.shape)
                np_scalar_type = input.dtype.name
                cu_scalar_type = np_to_cu_types[np_scalar_type]
                self.type = T.Seq(cu_scalar_type)
                self.local = [input]
                self.remote = [self.place.new_copy(x) for x in self.local]
                self.local_valid = True
                self.remote_valid = True
                return

            #Infer CuArray from Python data
            if type is not None:
                self.type = type
                self.depth = 0
                walk_type = type
                while isinstance(walk_type, T.Seq):
                    walk_type = walk_type.unbox()
                    self.depth = self.depth + 1
                base_type = walk_type
            
            else:
                self.depth = 0
                walk_seq = input
                self.type = None

                # Perversely, numpy scalars have a __getitem__ method 
                while hasattr(walk_seq, '__getitem__') and \
                          not isinstance(walk_seq, np.number) and \
                          not isinstance(walk_seq, np.bool_):
                    walk_seq = walk_seq[0]
                    self.depth = self.depth + 1
                   
                # CURRENT LIMITATION
                # Can only infer types of Python structures with atomic leaves
                # IE, not lists of tuples.
                # If you need a list of tuples, specify the type manually!
                try:
                    base_type = py_to_cu_types[walk_seq.__class__.__name__]
                except KeyError:
                    try:
                        base_type = np_to_cu_types[str(walk_seq.dtype)]
                    except:
                        raise DataException("Complex Python data type given, " +
                                            "CuType can't be inferred.  " +
                                            "Please specify type")
                self.type = _make_type(self.depth, base_type)
            depth = self.depth
            if isinstance(base_type, T.Tuple):
                np_type = ','.join((cu_to_np_types[str(x)] for x in base_type.parameters))
            else:
                np_type = cu_to_np_types[str(base_type)]
            shape = [(True, None) for x in range(depth)]
            local_structures = [[] for x in range(depth)]

            try:
                def _to_csr(inp, dep):
                    (consistent, child_size) = shape[dep]
                    if not child_size:
                        child_size = len(inp)
                    consistent = consistent and child_size == len(inp)
                    if not consistent:
                        shape[dep] = (False, None)
                    else:
                        shape[dep] = (True, child_size)
                    if dep == depth - 1:
                        #Reached a leaf node
                        local_structures[dep].extend(inp)
                    else:
                        for node in inp:
                            local_structures[dep].append(_to_csr(node, dep+1))
                    return len(inp)
                def to_csr(inp):
                    _to_csr(inp, 0)
                    for i in xrange(0, depth - 1):
                        local_structures[i].append(0)
                        local_structures[i] = exclusive_scan(op_add, 0, local_structures[i])
                to_csr(input)
                extents = map(lambda x: x[1] if x[0] else S.Unknown, shape)
                self.shape = S.shape_from_extents(extents)
            except IndexError:
                raise DataException("Input Python data type is not " +
                                    "homogeneously nested. " +
                                    "CuType can't be constructed.")
            self.local = [np.array(x, dtype=index_type) for x in local_structures[:-1]]
            self.local.append(np.array(local_structures[-1], dtype=np_type))
            self.remote = [self.place.new_copy(x) for x in self.local]
            self.local_valid = True
            self.remote_valid = True
            return
        elif remote is not None:
            if type is None:
                raise DataException("Can't infer type from remote array " +
                                    "Please provide type.")
            self.type = type
            if shape is None:
                raise DataException("Can't infer shape from remote array " +
                                    "Please provide shape.")
            if not isinstance(shape, S.Shape):
                self.shape = S.shape_from_extents(shape)
            else:
                self.shape = shape
            if not isinstance(remote, list):
                #We're creating a non-nested sequence directly from a pointer
                
                dtype = find_base_type(self.type)
                extents = (self.shape.extents[0])
                self.remote = [self.place.from_data(extents, dtype, remote)]
            else:
                self.remote = remote
            if input is not None:
                self.local = input
                self.local_valid = True
            else:
                self.local = [None for x in self.remote]
                self.local_valid = False
            self.remote_valid = True
                                    
                
        else:
            raise Exception("CuArray cannot be constructed with the " +
                            "given parameters")

    
    def _localize_reading(self):
        if not self.local_valid:
            for index, remote in enumerate(self.remote):
                local = self.local[index]
                if local is not None:
                    remote.get(local)
                else:
                    self.local[index] = remote.get()
            self.local_valid = True

    def _localize_writing(self):
        self._localize_reading()
        self.remote_valid = False

    def using_remote(self):
        if not self.remote_valid:
            for remote, local in zip(self.remote, self.local):
                remote.set(local)
            self.remote_valid = True
            self.local_valid  = False
        return self.remote


    
            
    def __iter__(self):
        self._localize_reading()
        return iterate(self)

    def __eq__(self, other):
        if len(self) != len(other):
            return False
        for x, y in zip(self, other):
            if x != y:
                return False
        return True

    def __getitem__(self, index):
        self._localize_reading()
        if isinstance(index, CuScalar):
            index = int(index.value)
        if self.depth == 1:
            return self.local[-1].__getitem__(index)
        # XXX The following code allows limits indexing doubly-nested sequences
        # But not any sequences which are more deeply nested.
        # We will fix this when we move to a C++ based CuArray class
        # (keeping a Python version and a C++ version to handle this is
        # more complication than it's worth)
        elif self.depth == 2:
            begin = self.local[0][index]
            end = self.local[0][index + 1]
            length = int(end - begin)
            sub_local = [self.local[1][begin:end]]
            sub_remote = [self.remote[1][begin:end]]
            sub_shape = S.shape_from_extents([length])
            sub_type = self.type.unbox()
            result = CuArray(input=sub_local, type=sub_type, shape=sub_shape,
                           remote=sub_remote)
            result.base = self
            return result
        else:
            raise DataException("Unimplemented: indexing a CuArray which has a nesting depth greater than 2.")

    def __str__(self):
        self._localize_reading()
        return self.local[-1].__str__()

    def __repr__(self):
        self._localize_reading()
        return self.local[-1].__repr__()
    
    def __len__(self):
        return self.local[-1].__len__()

    def __del__(self):
        # XXX Why does this need to be done explicitly?
        # If I don't do this explicitly, the data is not freed
        # And I end up quickly running out of memory...

        # If I'm a view of another array, just return
        if self.base:
            return
        for x in self.remote:
            x.gpudata.free()

    def numpy(self):
        self._localize_reading()
        return self.local[-1]
    
    def update(self, indices, values):
        """Destructively update the CuArray by scattering the values into the
        array at positions indicated by indices."""
        # XXX Implement for nested sequences
        assert(self.depth == 1)
        indices = [int(x) for x in strip(indices)]
        values = np.array(strip(values))
        
        if self.local_valid and not self.remote_valid:
            for i, v in zip(indices, values):
                self.local[-1][i] = v
        else:
            self.place.update(self.remote[-1], indices, values)
            self.local_valid = False

    def extract(self, indices):
        """Fetch the values specified by indices."""
        # XXX Implement for nested sequences
        assert(self.depth == 1)
        indices = [int(x) for x in strip(indices)]
        if self.local_valid:
            return [self.local[-1][x] for x in indices]
        else:
            return self.place.extract(self.remote[-1], indices)
            
                
def _comp_strides(perm, ext):
    x = zip(list(perm), ext, itertools.count())
    x.sort()
    strides = [0] * len(ext)
    stride = 1
    for c, e, i in x:
        strides[i] = stride
        stride = stride * e
    return tuple(strides)

class CuUniform(CuData):
    def __init__(self, local_blob, extents, strides=None, ordering=None, typ=None, place=None, offset=0, remote=None):
        self.place = place or places.default_place
        self.extents = extents
        assert(isinstance(local_blob, np.ndarray))
        self.local = local_blob
        if remote is None:
            self.remote = self.place.new_copy(self.local)
        else:
            self.remote = remote
        if strides is None:
            assert(ordering is not None)
            self.strides = _comp_strides(ordering, extents)
        else:
            self.strides = strides
        self.offset = offset
        if typ is None:
            base_type = np_to_cu_types[str(self.local.dtype)]
            self.type = _make_type(len(extents), base_type)
        else:
            self.type = typ
        self.shape = S.shape_from_extents(extents)
        self.local_valid = True
        self.remote_valid = True

    def _localize_reading(self):
        if not self.local_valid:
            self.remote.get(self.local)
            self.local_valid = True

    def _localize_writing(self):
        self._localize_reading()
        self.remote_valid = False

    def using_remote(self):
        if not self.remote_valid:
            self.remote.set(self.local)
            self.remote_valid = True
            self.local_valid  = False
        return self.remote

    def __iter__(self):
        self._localize_reading()
        return iterate(self)

    def __getitem__(self, index):
        if len(self.strides) > 1:
            extents = self.extents[1:]
            if isinstance(index, CuScalar):
                index = index.value
            advance = index * self.strides[0]
            offset = self.offset + advance
            strides = self.strides[1:]
            typ = self.type.unbox()
            place = self.place
            slic = CuUniform(self.local, extents, strides=strides,
                             typ=typ, place=place, offset=offset,
                             remote=self.remote)
            slic.local_valid = self.local_valid
            slic.remote_valid = self.remote_valid
            return slic
        else:
            self._localize_reading()
            return induct(self.local[index * self.strides[0] + self.offset])
    
