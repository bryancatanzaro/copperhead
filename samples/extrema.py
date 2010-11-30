#
#  Copyright 2010 University of California
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

from copperhead import *

@cu
def comp((a_low_idx, a_low_val, a_high_idx, a_high_val),
            (b_low_idx, b_low_val, b_high_idx, b_high_val)):
    if a_low_val < b_low_val:
        if a_high_val > b_high_val:
            return (a_low_idx, a_low_val, a_high_idx, a_high_val)
        else:
            return (a_low_idx, a_low_val, b_high_idx, b_high_val)
    else:
        if a_high_val > b_high_val:
            return (b_low_idx, b_low_val, a_high_idx, a_high_val)
        else:
            return (b_low_idx, b_low_val, b_high_idx, b_high_val)

@cu
def extrema(a, id):
    idx = indices(a)
    return reduce(comp, zip4(idx, a, idx, a), id)

import numpy
a = numpy.array([1.0, 2.0, 3.0, 4.0, -1.0, 6.0, 2.0], dtype=numpy.float32)
inf = numpy.float32(float('inf'))

print extrema(a, (0, inf, 0, -inf))
