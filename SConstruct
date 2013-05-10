from __future__ import print_function
import os
import re
import subprocess
from distutils.errors import CompileError
import operator
import fnmatch

#For Python 2.6 compatibility
def check_output(*popenargs, **kwargs):
    r"""Run command with arguments and return its output as a byte string.

    Backported from Python 2.7 as it's implemented as pure python on stdlib.

    >>> check_output(['/usr/bin/python', '--version'])
    Python 2.6.2
    """
    process = subprocess.Popen(stdout=subprocess.PIPE, *popenargs, **kwargs)
    output, unused_err = process.communicate()
    retcode = process.poll()
    if retcode:
        cmd = kwargs.get("args")
        if cmd is None:
            cmd = popenargs[0]
        error = subprocess.CalledProcessError(retcode, cmd)
        error.output = output
        raise error
    return output

# What are we doing?
# Figure out whether we're copying Python files
# Or building C++ extensions, or both
python_build = False
ext_build = False
if 'build_py' in COMMAND_LINE_TARGETS:
    python_build = True
if 'build_ext' in COMMAND_LINE_TARGETS:
    ext_build = True
if not python_build and not ext_build:
    python_build = True
    ext_build = True


# try to import an environment first
try:
    Import('env')
except:
    exec open(os.path.join('config', "build-env.py"))
    env = Environment()

# Check availability of a program, with given version.
def CheckVersion(context, cmd, exp, required, extra_error=''):
    context.Message("Checking {cmd} version... ".format(cmd=cmd.split()[0]))

    try:
        log = context.sconf.logstream if context.sconf.logstream else file('/dev/null','w')
        vsout = check_output([cmd], shell=True, stderr=log)
    except subprocess.CalledProcessError:
        context.Result('%s was not found %s' % (cmd.split()[0], extra_error) )
        return False
    match = exp.search(vsout)
    if not match:
        context.Result('%s returned unexpected output %s' % (cmd, extra_error) )
        return False
    version = match.group(1)
    exploded_version = map(int, version.split('.')[:len(required)])
    def to_number(exploded):
        return sum(map(lambda (idx, val): val*(10**(len(required) - idx)),
                       enumerate(exploded)))
    numeric_version = to_number(exploded_version)
    numeric_require = to_number(required)
  
    if not numeric_version >= numeric_require:
        context.Result("%s returned version %s, but we need version %s or better. %s" % (cmd, version, '.'.join(map(str, required)), extra_error) )
        return False
    context.Result(version)
    return True

thrust_version_check_file = """
#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_OMP
#include <thrust/version.h>
#include <iostream>
int main() {
  std::cout << THRUST_MAJOR_VERSION << std::endl;
  std::cout << THRUST_MINOR_VERSION << std::endl;
  std::cout << THRUST_SUBMINOR_VERSION << std::endl;
  return 0;
}
"""

def CheckThrustVersion(context, required_version):
    context.Message("Checking Thrust version...")
    int_required_version = [int(x) for x in required_version]
    result = context.TryRun(thrust_version_check_file, ".cpp")[1]
    returned_version = result.splitlines(False)
    version = '.'.join(returned_version)
    context.Result(version)

    int_returned_version = [int(x) for x in returned_version]
    return all(map(operator.le, int_required_version, int_returned_version))

def autoconf():
    conf=Configure(env, custom_tests = {'CheckVersion':CheckVersion,
                                        'CheckThrustVersion':CheckThrustVersion})
    siteconf = {}
    
    
    # Check we have numpy
    try:
        import numpy
        np_inc_path = numpy.get_include()
        siteconf['NP_INC_DIR'] = np_inc_path

    except ImportError:
        raise CompileError("numpy must be installed before building copperhead")

    siteconf['CUDA_LIB_DIR'], siteconf['CUDA_INC_DIR'] = env['CUDA_PATHS']
    siteconf['BOOST_INC_DIR'] = None
    siteconf['BOOST_LIB_DIR'] = None
    siteconf['BOOST_PYTHON_LIBNAME'] = None
    siteconf['THRUST_DIR'] = None
    siteconf['TBB_INC_DIR'] = None
    siteconf['TBB_LIB_DIR'] = None
    siteconf['CXX'] = env['CXX']
    siteconf['CC'] = env['CC']
    # Check to see if the user has written down siteconf stuff
    
    if os.path.exists("siteconf.py"):
        glb = {}
        execfile("siteconf.py", glb, siteconf)
    else:
        print("""
*************** siteconf.py not found ***************
We will try building anyway, but may not succeed.
Read the README for more details.
""")

        
    f = open("siteconf.py", 'w')
    print("""#! /usr/bin/env python
#
# Configuration file.
# Use Python syntax, e.g.:
# VARIABLE = "value"
# 
# The following information can be recorded:
#
# CXX : path and name of the host c++ compiler, eg: /usr/bin/g++-4.5
#
# CC : path and name of the host c compiler, eg: /usr/bin/gcc
#
# BOOST_INC_DIR : Directory where the Boost include files are found.
#
# BOOST_LIB_DIR : Directory where Boost shared libraries are found.
#
# BOOST_PYTHON_LIBNAME : Name of Boost::Python shared library.
#   NOTE: Boost::Python must be compiled using the same compiler
#   that was used to build your Python.  Strange errors will
#   ensue if this is not true.
# CUDA_INC_DIR : Directory where CUDA include files are found
#
# CUDA_LIB_DIR : Directory where CUDA libraries are found
#
# NP_INC_DIR : Directory where Numpy include files are found.
#
# TBB_INC_DIR : Directory where TBB include files are found
#
# TBB_LIB_DIR : Directory where TBB libraries are found
#
# THRUST_DIR : Directory where Thrust include files are found.
#

""", file=f)


    for k, v in sorted(siteconf.items()):
        if v:
            v = '"' + str(v) + '"'
        print('%s = %s' % (k, v), file=f)

    f.close()

    Export('siteconf')
    if siteconf['CXX']:
        env.Replace(CXX=siteconf['CXX'])
    if siteconf['CC']:
        env.Replace(CC=siteconf['CC'])
    host_compiler = env['CXX']
    #Ensure we have g++ >= 4.5
    
    gpp_re = re.compile(r'.*g\+\+.* \(.*\) ([\d\.]+)')
    compiler_support = conf.CheckVersion('%s --version' % host_compiler, gpp_re, (4,5))
    if not compiler_support:
        print('g++ 4.5 or better required, please add path to siteconf.py')
        Exit(1)
        
    #g++ comes with openmp support builtin
    omp_support = True
    Export('omp_support')
    
    # Check to see if we have nvcc >= 4.1
    nv_re = re.compile(r'release ([\d\.]+)')
    cuda_support=conf.CheckVersion('nvcc --version',
                                   nv_re,
                                   (4,1),
                                   extra_error="nvcc was not found. No CUDA support will be included.")
    if cuda_support:
        conf.CheckLib('cudart', language='C++', autoadd=0)
    Export('cuda_support')

    if siteconf['BOOST_INC_DIR']:
        env.Append(CPPPATH=siteconf['BOOST_INC_DIR'])
    if siteconf['BOOST_LIB_DIR']:
        env.Append(LIBPATH=siteconf['BOOST_LIB_DIR'])
    if siteconf['THRUST_DIR']:
        #Must prepend because build-env.py might have found an old system Thrust
        env.Prepend(CPPPATH=siteconf['THRUST_DIR'])
        
    # Check we have boost::python
    from distutils.sysconfig import get_python_lib, get_python_version
    env.Append(LIBPATH=os.path.join(get_python_lib(0,1),"config"))


    if not conf.CheckLib('python'+get_python_version(), language="C++"):
        print("You need the python development library to compile this program")
        Exit(1)

    if not siteconf['BOOST_PYTHON_LIBNAME']:
        siteconf['BOOST_PYTHON_LIBNAME'] = 'boost_python'

    if not conf.CheckLib(siteconf['BOOST_PYTHON_LIBNAME'], language="C++", autoadd=0):
        print("You need the Boost::Python library to compile this program")
        print("Consider installing it, or changing BOOST_PYTHON_LIBNAME in siteconf.py")
        Exit(1)

    #Check we have a Thrust installation
    if not conf.CheckCXXHeader('thrust/host_vector.h'):
        print("You need Thrust to compile this program")
        print("Consider installing it, or changing THRUST_DIR in siteconf.py")
        Exit(1)

    #Ensure Thrust Version > 1.6
    if not conf.CheckThrustVersion((1,6)):
        print("You need Thrust version 1.6 or greater")
        print("Change THRUST_DIR in siteconf.py to point to your Thrust installation.")
        Exit(1)
    if siteconf['TBB_INC_DIR']:
        env.Append(CPPPATH=siteconf['TBB_INC_DIR'])
    if siteconf['TBB_LIB_DIR']:
        env.Append(LIBPATH=siteconf['TBB_LIB_DIR'])
    
    #Check for TBB Support
    tbb_support = conf.CheckCXXHeader('tbb/tbb.h') and \
        conf.CheckLib('tbb', language="C++", autoadd=0)
    Export('tbb_support')
    # MacOS Support
    if env['PLATFORM'] == 'darwin':
        env.Append(SHLINKFLAGS = '-undefined dynamic_lookup')

    print('********************* BACKEND SUPPORT *********************')
    print('CUDA backend: %s' % cuda_support)
    print('OpenMP backend: %s' % omp_support)
    print('TBB backend: %s' % tbb_support)
    print('***********************************************************')


# Configuration is expensive, only do it if we need to
if ext_build:
    autoconf()

env.Append(CCFLAGS = ['-O3'])
    
Export('env')

#Build backend
#Ensure backend exists.
if not os.path.exists(os.path.join('backend', 'SConstruct')):
    try:
        subprocess.check_call(['git submodule init'], shell=True)
    except subprocess.CalledProcessError:
        raise CompileError("Error while downloading backend")
    try:
        subprocess.check_call(['git submodule update'], shell=True)
    except subprocess.CalledProcessError:
        raise CompileError("Error while downloading backend")

build_ext_targets = []

libcopperhead = SConscript(os.path.join('backend', 'src', 'SConscript'),
                           variant_dir=os.path.join('backend','build'),
                           duplicate=0)

Export('libcopperhead')


build_ext_targets.append(env.Install(os.path.join('stage', 'copperhead', 'runtime'), libcopperhead))

extensions = SConscript(os.path.join('src', 'SConscript'),
                            variant_dir='stage',
                            duplicate=0)
build_ext_targets.extend(extensions)

def recursive_glob(pattern, dir=os.curdir):
    files = Glob(dir+'/'+pattern)
    if files:
        files += recursive_glob("*/"+pattern,dir)
    return files

def explode_path(path):
    head, tail = os.path.split(path)
    return explode_path(head) + [tail] \
        if head and head != path \
        else [head or tail]


python_files = recursive_glob('*.py','copperhead')
build_py_targets = []

for x in python_files:
    head, tail = os.path.split(str(x))
    build_py_targets.append(env.Install(os.path.join('stage', head), x))
    
library_files = recursive_glob('*.h*', os.path.join(
        'backend', 'inc', 'prelude'))

for x in library_files:
    exploded_path = explode_path(str(x))[:-1]
    exploded_path[0] = os.path.join('stage', 'copperhead')
    install_path = os.path.join(*exploded_path)
    build_py_targets.append(env.Install(install_path, x))

frontend_library_files = recursive_glob('*.hpp', os.path.join(
    'src', 'copperhead', 'runtime'))

backend_library_files = [os.path.join('backend','inc', x+'.hpp') for x in \
                         ['type', 'monotype', 'polytype', 'ctype']]

for x in frontend_library_files + backend_library_files:
    install_path = os.path.join('stage','copperhead', 'inc', 'prelude', 'runtime')
    build_py_targets.append(env.Install(install_path, x))

thrust_patch_files = []
for root, dirs, files in os.walk(
    os.path.join('backend', 'inc', 'thrust_patch')):
    [thrust_patch_files.append(os.path.join(root, _file))\
         for _file in fnmatch.filter(files, '*.h')]  
for x in thrust_patch_files:
    #chop off backend/inc/thrust_patch
    exploded_path = explode_path(x)[2:-1]
    #and replace it with stage/copperhead/inc
    exploded_path[0] = os.path.join('stage', 'copperhead', 'inc')
    install_path = os.path.join(*exploded_path)
    build_py_targets.append(env.Install(install_path, x))
siteconf_file = 'siteconf.py'
build_py_targets.append(env.Install(os.path.join('stage', 'copperhead', 'runtime'),
                                    siteconf_file))

if 'build_py' in COMMAND_LINE_TARGETS:
    env.Alias('build_py', build_py_targets)
    env.Default(build_py_targets)
if 'build_ext' in COMMAND_LINE_TARGETS:
    env.Alias('build_ext', build_ext_targets)
    env.Default(build_ext_targets)
