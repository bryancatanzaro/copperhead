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


import imp as _imp
import os as _os
import glob as _glob
_cur_dir, _cur_file = _os.path.split(__file__)

def _find_lib(name):
    _ext_poss = _glob.glob(_os.path.join(_cur_dir, name+'*'))
    if len(_ext_poss) != 1:
        import pdb
        pdb.set_trace()
        raise ImportError(name)
    return _ext_poss[0]
                           

def _find_module(name):
    return _imp.load_dynamic(name, _find_lib(name))

_load = _find_module('load')

_load.load_library(_find_lib('libcopperhead'))
_load.load_library_init(_find_lib('libcunp'), 'initialize_cunp')

cudata = _find_module('cudata')


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
                    'library')
    
    host_toolchain.add_library('copperhead', [include_path], [], [])
    

    #Load configuration from siteconf
    import siteconf as _siteconf
    def _listize(x):
        if x:
            return [x]
        else:
            return []
    #Override codepy's guesses as to where boost is
    host_toolchain.add_library('boost-python',
                               _listize(_siteconf.BOOST_INC_DIR),
                               _listize(_siteconf.BOOST_LIB_DIR),
                               _listize(_siteconf.BOOST_PYTHON_LIBNAME))

    
    nvcc_toolchain = codepy.toolchain.guess_nvcc_toolchain()
    
    #BEGIN XXX WAR codepy misinterpretations of Python Makefile
    #The cflags it guesses are just awful for nvcc - remove them all
    nvcc_toolchain.cflags = []

    #Work around NVCC weirdness on OS X
    import sys
    if sys.platform == 'darwin':
        nvcc_toolchain.cflags.append('-m64')
    nvcc_toolchain.cflags.extend(['-Xcompiler', '-fPIC'])

    #If you see a '-framework' in the libraries, skip it
    # and its successor
    if '-framework' in host_toolchain.libraries:
        new_libraries = []
        shadow = False
        for x in host_toolchain.libraries:
            if shadow:
                shadow = False
            elif x == '-framework':
                shadow = True
            else:
                new_libraries.append(x)
        host_toolchain.libraries = new_libraries
                
    #END XXX
    if _siteconf.BOOST_INC_DIR:
        nvcc_includes = [include_path, _siteconf.BOOST_INC_DIR]
    else:
        nvcc_includes = [include_path]
    nvcc_toolchain.add_library('copperhead', nvcc_includes, [], [])
    nvcc_toolchain.cflags.append('-arch=sm_10')
    import numpy
    (np_path, np_file) = os.path.split(numpy.__file__)
    numpy_include_dir = os.path.join(np_path, 'core', 'include')
    nvcc_toolchain.add_library('numpy', [numpy_include_dir], [], [])
except ImportError:
    pass

