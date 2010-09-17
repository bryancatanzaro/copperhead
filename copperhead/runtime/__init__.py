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

from cuarray import CuArray, CuUniform
from cufunction import CuFunction
from cubox import CuBox
from cudata import CuScalar, CuData, CuTuple, induct

import places



try:
    import pycuda.driver as cuda
    cuda.init()
    from pycuda.tools import make_default_context
    context = make_default_context()
    device = context.get_device()
    import atexit
    atexit.register(context.detach)
    
    import driver
    
    places.gpu0 = driver.DefaultCuda()
    places.default_place = places.gpu0
    atexit.register(places.gpu0.cleanup)

except ImportError:
    print "PyCUDA not available.  Falling back to native Python execution."

try:
    import codepy.toolchain
    host_toolchain = codepy.toolchain.guess_toolchain()
    import os.path
    include_path = os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__))),
            'include')
    host_toolchain.add_library('copperhead', [include_path], [], [])
    nvcc_toolchain = codepy.toolchain.guess_nvcc_toolchain()
    nvcc_toolchain.add_library('copperhead', [include_path], [], [])
except ImportError:
    pass
