#
#   Copyright 2012      NVIDIA Corporation
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
import binarygenerator
import parsetypes
import passes
import pltools
import pyast
import rewrites
import typeinference
import unifier
import utility
import visitor
import imp as _imp
import os as _os
import glob as _glob
_cur_dir, _cur_file = _os.path.split(__file__)

def _find_module(name):
    _ext_poss = [ path for path in _glob.glob(_os.path.join(_cur_dir, name+'*')) if _os.path.splitext(path)[1] in ['.so', '.dll'] ]
    if len(_ext_poss) != 1:
        raise ImportError(name)
    return _imp.load_dynamic(name, _ext_poss[0])

backendcompiler = _find_module('backendcompiler')
backendsyntax = _find_module('backendsyntax')
backendtypes = _find_module('backendtypes')
import conversions
