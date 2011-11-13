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

import cufunction
from cufunction import CuFunction
import places
import utility

import driver
places.gpu0 = driver.DefaultCuda()
places.default_place = places.gpu0


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

import imp as _imp
import os as _os
import glob as _glob
_cur_dir, _cur_file = _os.path.split(__file__)

def _find_module(name):
    _ext_poss = _glob.glob(_os.path.join(_cur_dir, name+'*'))
    if len(_ext_poss) != 1:
        raise ImportError(name)
    return _imp.load_dynamic(name, _ext_poss[0])

cudata = _find_module('cudata')
