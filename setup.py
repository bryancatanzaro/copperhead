#!/usr/bin/env python
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

from distribute_setup import use_setuptools
use_setuptools()

from setuptools import setup

from distutils.errors import CompileError

from distutils.command.build_py import build_py as BuildPyCommand
from distutils.command.build_ext import build_ext as BuildExtCommand
from distutils.command.clean import clean as CleanCommand
from distutils.cmd import Command
import distutils.extension
import subprocess
import os.path
import os    
import fnmatch

try:
    subprocess.check_call(['scons'], shell=True)
except subprocess.CalledProcessError:
    raise CompileError("Error while building Python Extensions")

def remove_head_directories(path, heads=1):
    def explode_path(path):
        head, tail = os.path.split(path)
        return explode_path(head) + [tail] \
            if head and head != path else [head or tail]
    exploded_path = explode_path(path)
    if len(exploded_path) < (heads+1):
        return ''
    else:
        return os.path.join(*exploded_path[heads:])

build_product_patterns = ['*.h', '*.hpp', '*.so', '*.dll', '*.dylib']
build_products = []
build_path = 'stage'

for pattern in build_product_patterns:  
    for root, dirs, files in os.walk(build_path):
        dir_path = remove_head_directories(root, 2)
        for filename in fnmatch.filter(files, pattern):
            build_products.append(os.path.join(dir_path, filename))

setup(name="copperhead",
        version="0.2a2",
        description="Data Parallel Python",
        long_description="Copperhead is a Data Parallel Python dialect, with runtime code generation and execution for CUDA, OpenMP, and TBB.",
        classifiers=[
            'Development Status :: 2 - Pre-Alpha',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: Apache Software License',
            'Natural Language :: English',
            'Programming Language :: Python',
            'Topic :: Software Development :: Compilers',
            'Topic :: Software Development :: Code Generators',
            'Operating System :: POSIX :: Linux',
            'Operating System :: MacOS :: MacOS X'],
        zip_safe=False,
        author="Bryan Catanzaro, Michael Garland",
        author_email="bcatanzaro@nvidia.com, mgarland@nvidia.com",
        license = "Apache 2.0",
        package_dir = {'':'stage'},   # packages are under stage
        packages=['copperhead', 'copperhead.runtime', 'copperhead.compiler'],
        install_requires=["codepy>=2012.1.2"],
        package_data={
            'copperhead': build_products,
        },
        url="http://github.com/copperhead"
        )



