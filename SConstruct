import os
import inspect
import fnmatch

# try to import an environment first
try:
    Import('env')
except:
    exec open(os.path.join('build', "build-env.py"))
    env = Environment()

import multiprocessing
n_jobs = multiprocessing.cpu_count()
SetOption('num_jobs', n_jobs)
    
#Build backend    
libcopperhead = SConscript('backend/SConscript')

#Install backend library
env.Install('build/lib', libcopperhead)


#XXX This shouldn't be hardcoded. Fix!
if env['PLATFORM'] == 'darwin':
    env.Append(CPPPATH = "/Users/catanzar/boost_1_47_0")
    env.Append(LIBPATH="/Users/catanzar/boost_1_47_0/stage/lib")
    env.Append(LINKFLAGS="-F/System/Library/Frameworks/ -framework Python")
    
env.Append(CPPPATH = "backend/inc")
env.Append(CPPPATH = "/usr/include/python2.7")
env.Append(LIBS = ['boost_python'])

cppenv = env.Clone()
cudaenv = env.Clone()


cppenv.Append(CCFLAGS = "-std=c++0x -Wall")
cppenv.Append(LIBPATH = "backend/build/lib")
cppenv.Append(LIBS = "copperhead")

if env['PLATFORM'] == 'darwin':
    cudaenv.Append(CPPPATH = "/System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/numpy/core/include")
else:
    cudaenv.Append(CPPPATH = "/usr/lib/pymodules/python2.7/numpy/core/include")    


cudaenv.Append(LIBS = ['cudart'])
cudaenv.Append(CPPPATH = "backend/library/prelude")

    
#build cudata.os from a CUDA source file
cudata_source = 'src/cudata.cu'
head, tail = os.path.split(cudata_source)
basename, extension = os.path.splitext(tail)
target = os.path.join(os.path.join("build", "obj"), os.path.join(head, basename))
cudaenv.SharedObject(target=target,source=cudata_source)
cudata_object = [target + '.os']


def make_ext(working_env, x, objs):
    basename, extension = os.path.splitext(str(x))
    head, tail = os.path.split(basename)
    dest = os.path.join(os.path.join("build", "lib"), tail)
    working_env.SharedLibrary(target=dest, source=[x]+objs, SHLIBPREFIX='', SHLIBSUFFIX='.so')


for x in ['src/compiler.cpp',
          'src/coresyntax.cpp',
          'src/coretypes.cpp']:
    make_ext(cppenv, x, [])

for x in ['src/cudata.cpp']:
    make_ext(cudaenv, x, [cudata_object])

