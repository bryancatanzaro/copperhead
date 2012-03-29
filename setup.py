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

#ensure stage directories exist
stage_directories = ('stage',
                     os.path.join('stage', 'copperhead'),
                     os.path.join('stage', 'copperhead', 'runtime'),
                     os.path.join('stage', 'copperhead', 'compiler'))
for d in stage_directories:
    if not os.path.isdir(d):
        os.mkdir(d)

# Call custom build routines to copy Python files appropriately
class CopperheadBuildPy(BuildPyCommand):
    def build_packages(self, *args, **kwargs):
        try:
            subprocess.check_call(['scons build_py'], shell=True)
        except subprocess.CalledProcessError:
            raise CompileError("Error while building Python Extensions")
        BuildPyCommand.build_packages(self)

# Call custom build routines to create Python extensions
class CopperheadBuildExt(BuildExtCommand):
    user_options=[]
    description = BuildExtCommand.description

    def get_source_files(self): return []

    def run(self):
        try:
            subprocess.check_call(['scons build_ext'], shell=True)
        except subprocess.CalledProcessError:
            raise CompileError("Error while building Python Extensions")
        acceptable_library_extensions = ['dll', 'so', 'dylib']
        for e in self.extensions:
            name = e.name
            sources = e.sources
            input_dir = os.path.join('stage', *name.split('.'))
            output_dir = os.path.join(self.build_lib, *name.split('.'))
            for s in sources:
                for ext in acceptable_library_extensions:
                    materialized = s + '.' + ext
                    if os.path.exists(os.path.join(input_dir, materialized)):
                        break
                self.copy_file(os.path.join(input_dir, materialized),
                               os.path.join(output_dir, materialized))

# Call custom clean command to forward call to SCons
class CopperheadClean(CleanCommand):

    def run(self):
        #Doesn't appear to clean up...  =(
        CleanCommand.run(self)
        try:
             subprocess.check_call(['scons', '--remove'])
        except subprocess.CalledProcessError:
            raise CompileError("Error while cleaning Python Extensions")



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
        install_requires=["codepy>=2012.1.1"],
        #This is a nonstandard ext_modules, we do our own copying
        ext_modules=[distutils.extension.Extension(name='copperhead.compiler',
                                                   sources=['backendcompiler',
                                                            'backendsyntax',
                                                            'backendtypes']),
                     distutils.extension.Extension(name='copperhead.runtime',
                                                   sources=['cudata',
                                                            'load',
                                                            'tags',
                                                            'cuda_info',
                                                            'libcunp',
                                                            'libcopperhead'])
                                                            ], 
        package_data={
            'copperhead': ['inc/prelude/*.h',
                           'inc/prelude/basic/*.h',
                           'inc/prelude/primitives/*.h',
                           'inc/prelude/runtime/*.hpp',
                           'inc/prelude/runtime/*.h',
                           'inc/prelude/sequences/*.h']
        },
        url="http://github.com/copperhead",
        cmdclass = { 'build_py' : CopperheadBuildPy,
                     'build_ext' : CopperheadBuildExt,
                     'clean': CopperheadClean },
        test_suite = 'copperhead.tests.test_all',
        )



