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

# Tools for programming language implementation

import string, itertools

def strlist(items, bracket=None, sep=', ', form=repr):
    'Convert a list to strings and join with optional separator and brackets'
    body = sep.join(map(form, items))
    if bracket:
        open,close = bracket
        return open+body+close
    else:
        return body


class Environment:
    "Associative map with chained maps for fallback"

    def __init__(self, *chained):
        self.maps = [dict()] + list(chained)

    def lookup(self, key):
        for M in self.maps:
            if key in M: return M[key]

        raise KeyError, "no value defined for %s" % key

    def has_key(self, key):
        for M in self.maps:
            if key in M: return True
        return False

    def __len__(self):  return sum([M.__len__() for M in self.maps])

    def __getitem__(self, key):  return self.lookup(key)
    def __setitem__(self, key, value):  self.maps[0][key] = value
    def __contains__(self, key): return self.has_key(key)

    def __iter__(self):
        return itertools.chain(*[M.__iter__() for M in self.maps])
    def __repr__(self):
        return 'Environment(' + repr(self.maps) + ')'

    def iterkeys(self):
        return self.__iter__()

    def begin_scope(self):  self.maps = [dict()] + self.maps
    def end_scope(self):  self.maps.pop(0)
    def update(self, E, **F): self.maps[0].update(E, **F)

def resolve(name, env):
    """
    Resolve the value assigned to NAME by ENV, possibly via chained bindings
    """
    t = name
    while t in env:
        t = env[t]
    return t

def resolution_map(names, env):
    """
    Return a dictionary that maps each NAME to its resolution in ENV.
    """
    return dict(zip(names, [resolve(n, env) for n in names]))

def name_supply(stems=string.ascii_lowercase, drop_zero=True):
    """
    Produce an infinite stream of unique names from stems.
    Defaults to a, b, ..., z, a1, ..., z1, a2, ..., z2, ...
    """
    k = 0
    while 1:
        for a in stems:
            yield a+str(k) if (k or not drop_zero) else a
        k = k+1

def name_list(length, **kwargs):
    """
    Produce a list of unique names of the given length.
    """
    return list(itertools.islice(name_supply(**kwargs), length))
