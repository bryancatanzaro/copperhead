#!/usr/bin/env python
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

import distribute_setup
distribute_setup.use_setuptools()

from setuptools import setup


setup(name="copperhead",
      version="0.1a1",
      description="Data Parallel Python",
      long_description="""
      Copperhead is a Data Parallel Python dialect, with runtime code
      generation and execution for CUDA Graphics Processors.
      """,
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
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows :: Windows NT/2000'],
      zip_safe=False,
      author="Bryan Catanzaro, Michael Garland",
      author_email="catanzar@cs.berkeley.edu, mgarland@nvidia.com",
      license = "Apache 2.0",
      packages=['copperhead', 'copperhead.runtime', 'copperhead.compiler', 'copperhead.thrust'],
      package_data={
        'copperhead': ['include/*.h'],
        'copperhead.thrust': ['wrappers/*.h'],
        },
      url="http://code.google.com/p/copperhead",
      )
