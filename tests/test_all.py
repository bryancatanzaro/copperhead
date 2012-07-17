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

from test_syntax import *
from test_types import *
from test_unify import *
from test_infer import *
from test_indices import *
from test_simple import *
from test_reduce import *
from test_replicate import *
from test_rotate import *
from test_sort import *
from test_shift import *
from test_aos import *
from test_scalar_math import *
from test_gather import *
from test_subscript import *
from test_tuple_data import *
from test_scatter import *
from test_fixed import *
from test_closure import *
from test_tail_recursion import *
from test_zip import *
from test_update import *
from test_scan import *
from test_filter import *

if __name__ == "__main__":
    unittest.main()
