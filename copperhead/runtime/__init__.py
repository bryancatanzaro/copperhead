#
#   Copyright 2008-2012 NVIDIA Corporation
#  Copyright 2009-2010 University of California
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


import imp
import os
import glob
cur_dir, cur_file = os.path.split(__file__)

def find_lib(name):
    ext_poss = glob.glob(os.path.join(cur_dir, name+'*'))
    if len(ext_poss) != 1:
        raise ImportError(name)
    return ext_poss[0]
                           

def find_module(name):
    return imp.load_dynamic(name, find_lib(name))


load = find_module('load')

#Load configuration from siteconf
import siteconf as siteconf

load.load_library(find_lib('libcopperhead'))
load.load_library_init(find_lib('libcunp'), 'initialize_cunp')
try:
    cudata = find_module('cudata')
    cuda_info = find_module('cuda_info')
    cuda_support = True
except:
    #CUDA support not found
    cuda_support = False

import cufunction
from cufunction import CuFunction
import places
import utility

import driver
places.gpu0 = driver.DefaultCuda()
if cuda_support:
    places.default_place = places.gpu0
else:
    places.default_place = places.here

import codepy.toolchain
host_toolchain = codepy.toolchain.guess_toolchain()
import os.path
include_path = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__))),
                'library')

host_toolchain.add_library('copperhead', [include_path], [], [])


def listize(x):
    if x:
        return [x]
    else:
        return []
#Override codepy's guesses as to where boost is
host_toolchain.add_library('boost-python',
                           listize(siteconf.BOOST_INC_DIR),
                           listize(siteconf.BOOST_LIB_DIR),
                           listize(siteconf.BOOST_PYTHON_LIBNAME))
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

#Sanitize some poor cflags guesses on OS X
unwanted = set(['-Wshorten-64-to-32', '-Wstrict-prototypes'])
host_toolchain.cflags = filter(lambda x: x not in unwanted, host_toolchain.cflags)


if cuda_support:
    nvcc_toolchain = codepy.toolchain.guess_nvcc_toolchain()

    #BEGIN XXX WAR codepy misinterpretations of Python Makefile
    #The cflags it guesses are just awful for nvcc - remove them all
    nvcc_toolchain.cflags = []

    #Work around NVCC weirdness on OS X
    import sys
    if sys.platform == 'darwin':
        nvcc_toolchain.cflags.append('-m64')
    nvcc_toolchain.cflags.extend(['-Xcompiler', '-fPIC'])


    if siteconf.BOOST_INC_DIR:
        nvcc_includes = [include_path, siteconf.BOOST_INC_DIR]
    else:
        nvcc_includes = [include_path]
    nvcc_toolchain.add_library('copperhead', nvcc_includes, [], [])

    #find architecture of GPU #0
    major, minor = cuda_info.get_cuda_info()[0]
    nvcc_toolchain.cflags.append('-arch=sm_%s%s' % (major, minor))
    #does GPU #0 support doubles?
    float64_support = major >=2 or (major == 1 and minor >= 3)


    import numpy
    (np_path, np_file) = os.path.split(numpy.__file__)
    numpy_include_dir = os.path.join(np_path, 'core', 'include')
    nvcc_toolchain.add_library('numpy', [numpy_include_dir], [], [])
else:
    float64_support = True

__all__ = ['load', 'siteconf', 'cudata', 'cuda_info', 'host_toolchain', 'nvcc_toolchain', 'float64_support', 'cuda_support']
