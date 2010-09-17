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

class Place(object):

    def __init__(self):
        self._previous = None

    def __enter__(self):
        global default_place
        self._previous = default_place
        default_place = self

    def __exit__(self, *args):
        global default_place
        if self._previous:
            default_place = self._previous
            self._previous = None


class PythonInterpreter(Place):

    def __init__(self): pass

    def new_copy(self, x):
        assert isinstance(x, np.ndarray)
        return np.array(x)

    def execute(self, cufn, args, kwargs):
        fn = cufn.python_function()
        return fn(*args, **kwargs)

here = PythonInterpreter()

default_place = here
