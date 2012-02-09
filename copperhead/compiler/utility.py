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

def flatten(x):
    if hasattr(x, '__iter__'):
        for xi in x:
            if hasattr(xi, '__iter__'):
                for i in flatten(iter(xi)):
                    yield i
            else:
                yield xi
    else:
        yield x

def interleave(*args):
    iterators = [iter(x) for x in args]
    while iterators:
        for it in iterators:
            try:
                yield it.next()
            except:
                iterators.remove(it)

import copy

class ExtendingList(list):
    def __init__(self, default=0):
        list.__init__(self)
        self.default=default
        
    def enlarge(self, length):
        if len(self) < length:
            extension = length - len(self)
            super(ExtendingList, self).extend([copy.copy(self.default) \
                                               for x in range(extension)])
    def __getitem__(self, index):
        self.enlarge(index + 1)
        return super(ExtendingList, self).__getitem__(index)
    def __setitem__(self, index, value):
        self.enlarge(index + 1)
        return super(ExtendingList, self).__setitem__(index, value)
