#
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

import os, sys
import exceptions

try:
    from copperhead.runtime import nvcc_toolchain
   
    current_path = os.path.dirname(os.path.abspath(__file__))
    

    thrust_path = os.getenv('THRUST_PATH')
    if thrust_path is None:
        raise exceptions.ImportError("""Cannot import Thrust library.
  Please define the THRUST_PATH environment variable to point to your
  Thrust installation""")
    tuple_header = os.path.join(thrust_path, 'thrust', 'tuple.h')
    if not os.path.exists(tuple_header):
        raise exceptions.ImportError("""Cannot import Thrust library
  Your THRUST_PATH environment variable is set to:
  """ + str(thrust_path) + """
  However, that path does not contain a valid Thrust distribution.""")
        
    nvcc_toolchain.add_library('thrust',
                               [current_path, thrust_path], [], []) 

    #Expose functions as thrust.fn for those who want to call them directly
    from reduce import sum, reduce
    from scan import sum_scan, scan
    from sort import sort
    from gather import gather
    from scatter import scatter
    from permute import permute
    from rscan import rscan

    import copperhead.runtime.places as P
    #Register functions with Copperhead Prelude
    import copperhead.prelude as prelude

    _thrust_functions = [sum, reduce, scan, gather, scatter,
                         permute, rscan]        
    for fn in _thrust_functions:
        name = fn.__name__
        prelude_fn = getattr(prelude, name)
        prelude_fn.variants[P.gpu0] = fn

except ImportError as inst:
    raise inst
