from __future__ import print_function
import os
import re
import subprocess
from distutils.errors import CompileError

# try to import an environment first
try:
    Import('env')
except:
    exec open(os.path.join('config', "build-env.py"))
    env = Environment()

# Check availability of a program, with given version.
def CheckVersion(context, cmd, exp, required, extra_error=''):
    context.Message("Checking {cmd} version... ".format(cmd=cmd.split()[0]))

    def version_geq(required, received):
        for x, y in zip(required, received):
            if y < x:
                return False
        return True
    try:
        log = context.sconf.logstream if context.sconf.logstream else file('/dev/null','w')
        vsout = subprocess.check_output([cmd], shell=True, stderr=log)
    except subprocess.CalledProcessError:
        context.Result('%s was not found %s' % (cmd.split()[0], extra_error) )
        return False
    match = exp.search(vsout)
    if not match:
        context.Result('%s returned unexpected output' % (cmd, extra_error) )
        return False
    version = match.group(1)
    exploded_version = version.split('.')
    if not version_geq(required, exploded_version):
        context.Result("%s returned version %s, but we need version %s or better." % (cmd, version, '.'.join(required), extra_error) )
        return False
    context.Result(version)
    return True

# Check dependencies
conf=Configure(env, custom_tests = {'CheckVersion':CheckVersion})
siteconf = {}

#Ensure we have g++ >= 4.5
gpp_re = re.compile(r'g\+\+ \(.*\) ([\d\.]+)')
conf.CheckVersion('g++ --version', gpp_re, (4,5))

# Check to see if we have nvcc >= 4.1
nv_re = re.compile(r'release ([\d\.]+)')
cuda_support=conf.CheckVersion('nvcc --version', nv_re, (4,1), extra_error="nvcc was not found. No CUDA support will be included.")
          
# Check we have numpy
try:
    import numpy
    np_path, junk = os.path.split(numpy.__file__)
    np_inc_path = os.path.join(np_path, 'core', 'include')
    siteconf['NP_INC_PATH'] = np_inc_path

except ImportError:
    raise CompileError("numpy must be installed before building copperhead")

# Check to see if the user has written down siteconf stuff
if os.path.exists("siteconf.py"):
    glb = {}
    try:
        execfile("siteconf.py", glb, siteconf)
    except Exception:
        pass
else:

    print("""
*************** siteconf.py not found ***************
We will try building anyway, but may not succeed.
Read the README for more details.
""")
        
    siteconf['BOOST_INC_DIR'] = None
    siteconf['BOOST_LIB_DIR'] = None
    siteconf['BOOST_PYTHON_LIBNAME'] = None
    
    f = open("siteconf.py", 'w')
    for k, v in siteconf.items():
        if v:
            v = '"' + str(v) + '"'
        print('%s = %s' % (k, v), file=f)
        
    f.close()

Export('siteconf')


# Check we have boost::python
from distutils.sysconfig import get_python_lib, get_python_version
env.Append(LINKFLAGS='-L'+get_python_lib(0,1)+"/config")
if not conf.CheckLib('python'+get_python_version(), language="C++"):
    print("You need the python development library to compile this program")
    Exit(1)

if not siteconf['BOOST_PYTHON_LIBNAME']:
    siteconf['BOOST_PYTHON_LIBNAME'] = 'boost_python'

if not conf.CheckLib( siteconf['BOOST_PYTHON_LIBNAME'] , language="C++"):
    print("You need the boost_python library to compile this program")
    print("Consider installing it, or changing BOOST_PYTHON_LIBNAME in siteconfig.py")
    Exit(1)


# MacOS Support
if env['PLATFORM'] == 'darwin':
    env.Append(SHLINKFLAGS = '-undefined dynamic_lookup')


#Parallelize the build maximally
import multiprocessing
n_jobs = multiprocessing.cpu_count()
SetOption('num_jobs', n_jobs)

Export('env')
Export('cuda_support')
    

#Build backend    
libcopperhead = SConscript(os.path.join('backend',
                                         os.path.join('src', 'SConscript')),
                           variant_dir=os.path.join('backend','build'),
                           duplicate=0)

Export('libcopperhead')

env.Install(os.path.join('stage', 'copperhead', 'runtime'), libcopperhead)

extensions = SConscript(os.path.join('src', 'SConscript'),
                        variant_dir='stage',
                        duplicate=0)
    
def recursive_glob(pattern, dir=os.curdir):
    files = Glob(dir+'/'+pattern)
    if files:
        files += recursive_glob("*/"+pattern,dir)
    return files

python_files = recursive_glob('*.py','copperhead')
for x in python_files:
    head, tail = os.path.split(str(x))
    env.Install(os.path.join('stage', head), x)
    
library_files = recursive_glob('*.h', os.path.join(
    'backend', 'library', 'prelude')) + \
    recursive_glob('*.h', os.path.join(
        'backend', 'library', 'thrust_wrappers'))

def explode_path(path):
    head, tail = os.path.split(path)
    return explode_path(head) + [tail] \
        if head and head != path \
        else [head or tail]

for x in library_files:
    exploded_path = explode_path(str(x))[:-1]
    exploded_path[0] = os.path.join('stage', 'copperhead')
    install_path = os.path.join(*exploded_path)
    env.Install(install_path, x)

frontend_library_files = recursive_glob('*.h', os.path.join(
    'src', 'copperhead', 'runtime'))

backend_library_files = [os.path.join('backend', 'inc', 'cudata.h')]


for x in frontend_library_files + backend_library_files:
    install_path = os.path.join('stage','copperhead','library','prelude')
    env.Install(install_path, x)


    

siteconf_file = 'siteconf.py'
env.Install(os.path.join('stage', 'copperhead', 'runtime'),
            siteconf_file)
