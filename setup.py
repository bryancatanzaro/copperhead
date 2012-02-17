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

from distutils.command.build_ext import build_ext as BuildExtCommand
from distutils.command.clean import clean as CleanCommand
from distutils.cmd import Command

import subprocess

# Call custom build routines to create Python extensions
class CopperheadBuildExt(Command):
    user_options=[]
    description = BuildExtCommand.description

    def initialize_options(self):pass
    def finalize_options(self):pass
    def get_source_files(self): return []

    def run(self):
        try:
            subprocess.check_call(['scons'], shell=True)
        except subprocess.CalledProcessError:
            raise CompileError("Error while building Python Extensions")
        self.extensions=[]

# Call custom clean command to forward call to SCons
class CopperheadClean(CleanCommand):

    def run(self):
        CleanCommand.run(self)
        try:
             subprocess.check_call(['scons', '--remove'])
        except subprocess.CalledProcessError:
            raise CompileError("Error while cleaning Python Extensions")

##

setup(name="copperhead",
        version="0.2a1",
        description="Data Parallel Python",
        long_description="Copperhead is a Data Parallel Python dialect, with runtime code generation and execution for CUDA Graphics Processors.",
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
        ext_modules=[('copperhead','')], # the name as no meaning as we override the build_ext command to call SCons
        install_requires=["codepy>=2012.1.1"],
        package_data={
            'copperhead': ['library/*/*.h'],
            'copperhead.compiler' : ['backendcompiler.so',
                'backendsyntax.so',
                'backendtypes.so'],
            'copperhead.runtime' : ['cudata.so', 'load.so', 'cuda_info.so',
                'libcunp.*','libcopperhead.*']
            },
        url="http://code.google.com/p/copperhead",
        cmdclass = { 'build_ext' : CopperheadBuildExt, 'clean': CopperheadClean },
        test_suite = 'copperhead.tests.test_all',
        )
