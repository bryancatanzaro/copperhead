#
#   Copyright 2012      NVIDIA Corporation
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

import collections
import itertools

def recursive_equal(a, b):
    if isinstance(a, collections.Iterable):
        elwise_equal = all(itertools.imap(recursive_equal, a, b))
        length_check = sum(1 for x in a) == sum(1 for x in b)
        return elwise_equal and length_check
    else:
        return a == b
