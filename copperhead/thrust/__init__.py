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
import copperhead.runtime.places as P

try:
    from copperhead.runtime import nvcc_toolchain, cubox
   
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



    #Register functions with Copperhead Prelude
    import copperhead.prelude as prelude
    
    _thrust_functions = ['sum', 'reduce', 'zip', 'zip3', 'zip4', 'indices', 'scan', 'rscan'] #reduce, scan, gather, scatter,
                         #permute, rscan]
    _thrust_wrapper = ('.', 'thrust_wrappers.h')
    _no_wrapper = None
    _thrust_wrappers = [_thrust_wrapper, _thrust_wrapper, _no_wrapper, _no_wrapper, _no_wrapper, _no_wrapper, _thrust_wrapper, _thrust_wrapper]


    
    for name, wrap in zip(_thrust_functions, _thrust_wrappers):
        prelude_fn = getattr(prelude, name)
        prelude_fn.variants[P.gpu0] = cubox.CuBox(prelude_fn, wrap)
        

except ImportError as inst:
    # If the GPU place exists, we should report the error
    if hasattr(P, 'gpu0'):
        raise inst
