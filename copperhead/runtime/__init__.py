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

def find_lib(dir, name):
    possible_extensions = ['.so', '.dll', '.dylib']
    ext_poss = []
    for ext in possible_extensions:
        candidate = os.path.join(dir, name + ext)
        if os.path.exists(candidate):
            ext_poss.append(candidate)
    if len(ext_poss) != 1:
        print("Trying to load: %s from dir: %s" %(name, dir))
        raise ImportError(name)
    return ext_poss[0]
                           

def find_module(dir, name):
    return imp.load_dynamic(name, find_lib(dir, name))


load = find_module(cur_dir, 'load')

#Load configuration from siteconf
import siteconf as siteconf

load.load_library(find_lib(cur_dir, 'libcopperhead'))
load.load_library_init(find_lib(cur_dir, 'libcunp'), 'initialize_cunp')

cudata = find_module(cur_dir, 'cudata')
#Register libcopperhead destructor
import atexit
atexit.register(cudata.take_down)

try:
    cuda_utils = find_module(cur_dir, 'cuda_utils')
except:
    pass

import tags as tags

cuda_support = hasattr(tags, 'cuda')
omp_support = hasattr(tags, 'omp')
tbb_support = hasattr(tags, 'tbb')

import places
import utility
import null_toolchain

import driver
places.sequential = driver.Sequential()

if cuda_support:
    places.gpu0 = driver.DefaultCuda()
    places.default_place = places.gpu0
    cuda_tag = tags.cuda
else:
    places.default_place = places.sequential

if omp_support:
    places.openmp = driver.OpenMP()
    omp_tag = tags.omp
    if places.default_place == places.sequential:
        places.default_place = places.openmp
    
if tbb_support:
    places.tbb = driver.TBB()
    tbb_tag = tags.tbb
    if places.default_place == places.sequential:
        places.default_place = places.tbb

import codepy.toolchain
host_toolchain = codepy.toolchain.guess_toolchain()

def add_defines(toolchain):
    if cuda_support:
        toolchain.defines.append('CUDA_SUPPORT')
    if omp_support:
        toolchain.defines.append('OMP_SUPPORT')
    if tbb_support:
        toolchain.defines.append('TBB_SUPPORT')

add_defines(host_toolchain)

#Don't use the version of g++/gcc used for compiling Python
#That version is too old.  Use the standard g++/gcc
host_toolchain.cc = siteconf.CXX
host_toolchain.ld = siteconf.CC

#enable C++11 features in g++
host_toolchain.cflags.append('-std=c++0x')

import os.path
include_path = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__))),
    'inc')

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
#Add Thrust
host_toolchain.add_library('thrust',
                           listize(siteconf.THRUST_DIR),
                           [],
                           [])

#Sanitize some poor flag choices on OS X
def sanitize_flags(flag_list, objectionables):
    new_flags = []
    shadow = False
    for x in flag_list:
        if shadow:
            shadow = False
        elif x in objectionables:
            shadow = objectionables[x]
        else:
            new_flags.append(x)
    return new_flags

host_toolchain.cflags = sanitize_flags(host_toolchain.cflags, {'-Wshorten-64-to-32' : False,
                                                               '-Wstrict-prototypes' : False})
host_toolchain.libraries = sanitize_flags(host_toolchain.libraries, {'-framework' : True})
host_toolchain.ldflags = sanitize_flags(host_toolchain.ldflags, {'-u' : True})


if cuda_support:
    #Add cuda support to host toolchain
    host_toolchain.add_library('cuda',
                           listize(siteconf.CUDA_INC_DIR),
                           listize(siteconf.CUDA_LIB_DIR),
                           ['cudart'])

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
    nvcc_toolchain.add_library('thrust',
                           listize(siteconf.THRUST_DIR),
                           [],
                           [])

    nvcc_toolchain.add_library('cuda', [], [], [])
    #find architecture of GPU #0
    major, minor = cuda_utils.get_cuda_info()[0]
    nvcc_toolchain.cflags.append('-arch=sm_%s%s' % (major, minor))
    #does GPU #0 support doubles?
    float64_support = major >=2 or (major == 1 and minor >= 3)
    
    nvcc_toolchain.add_library('numpy', [siteconf.NP_INC_DIR], [], [])
    add_defines(nvcc_toolchain)
    #Null toolchain can't compile, but it can do everything else
    #This is used to detect whether a binary has already been compiled
    null_nvcc_toolchain = null_toolchain.make_null_toolchain(nvcc_toolchain)

else:
    float64_support = True

if omp_support:
    host_toolchain.cflags.append('-fopenmp')
if tbb_support:
    host_toolchain.add_library('tbb',
                              [siteconf.TBB_INC_DIR],
                              [siteconf.TBB_LIB_DIR], ['tbb'])
#Null toolchain can't compile, but it can do everything else
#This is used to detect whether a binary has already been compiled
null_host_toolchain = null_toolchain.make_null_toolchain(host_toolchain)

backends = [places.sequential]
if cuda_support:
    backends.append(places.gpu0)
if omp_support:
    backends.append(places.openmp)
if tbb_support:
    backends.append(places.tbb)

import collections

if cuda_support:
    ToolchainCollection = collections.namedtuple("ToolchainCollection",
                                                 ['host_toolchain',
                                                  'null_host_toolchain',
                                                  'nvcc_toolchain',
                                                  'null_nvcc_toolchain'])
    toolchains = ToolchainCollection(host_toolchain, null_host_toolchain,
                                     nvcc_toolchain, null_nvcc_toolchain)
else:
    ToolchainCollection = collections.namedtuple("ToolchainCollection",
                                                 ['host_toolchain',
                                                  'null_host_toolchain'])

    toolchains = ToolchainCollection(host_toolchain, null_host_toolchain)
    
import cufunction
from cufunction import CuFunction
import np_interop
from np_interop import to_numpy
__all__ = ['load', 'siteconf', 'cudata', 'cuda_utils', 'toolchains', 'float64_support', 'cuda_support', 'omp_support', 'tbb_support', 'backends', 'to_numpy']
