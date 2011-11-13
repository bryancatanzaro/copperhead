#import binarygenerator
import inference
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
    _ext_poss = _glob.glob(_os.path.join(_cur_dir, name+'*'))
    if len(_ext_poss) != 1:
        raise ImportError(name)
    return _imp.load_dynamic(name, _ext_poss[0])

print("Importing compiler")
backendcompiler = _find_module('backendcompiler')
backendsyntax = _find_module('backendsyntax')
backendtypes = _find_module('backendtypes')
