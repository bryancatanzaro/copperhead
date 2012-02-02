import os
import inspect
import fnmatch
import string
import re

# try to import an environment first
try:
    Import('env')
except:
    exec open(os.path.join('config', "build-env.py"))
    env = Environment()

siteconf = {}


import subprocess
def check_version(cmd, exp, required):
    def version_geq(required, received):
        for x, y in zip(required, received):
            if y < x:
                return False
        return True
    try:
        vsout = subprocess.check_output([cmd], shell=True)
    except subprocess.CalledProcessError:
        raise CompileError('%s was not found' % cmd)
    match = exp.search(vsout)
    if not match:
        raise CompileError('%s returned unexpected output' % cmd)
    version = match.group(1)
    exploded_version = version.split('.')
    if not version_geq(required, exploded_version):
        raise CompileError("%s returned version %s, but we need version %s or better" % (cmd, version, '.'.join(required)))
    print('%s returned version %s' % (cmd, version))

#Ensure we have g++ >= 4.5
gpp_re = re.compile(r'g\+\+ \(.*\) ([\d\.]+)')
check_version('g++ --version', gpp_re, (4,5))

#Ensure we have nvcc >= 4.1
nv_re = re.compile(r'release ([\d\.]+)')
check_version('nvcc --version', nv_re, (4,1))
                           

try:
    import numpy
    np_path, junk = os.path.split(numpy.__file__)
    np_inc_path = os.path.join(np_path, 'core', 'include')
    siteconf['NP_INC_PATH'] = np_inc_path

except ImportError:
    raise CompileError("numpy must be installed before building copperhead")

#Check to see if the user has written down siteconf stuff
if os.path.exists("siteconf.py"):
    glb = {}
    try:
        execfile("siteconf.py", glb, siteconf)
    except Exception:
        pass
else:
    #Prompt the user for locations

    def norm_path(p):
        if not p:
            return None
        q = os.path.normpath(os.path.expanduser(p))
        if os.path.isabs(q):
            return q
        else:
            return os.path.abspath(q)
    print("""
************** CONFIGURATION REQUIRED **************
siteconf.py configuration file not found.
Please enter paths, siteconf.py will be generated.
If you enter a blank path, system defaults will be used.

NOTE: It is critical that the Boost Python Library you specify
was compiled with the same compiler as your Python installation.
Using a different compiler version may cause unexplained and mysterious
crashes.
""")
    
    print("(Directory of Boost include files) BOOST_INC_DIR [None]: ")
    bid = norm_path(raw_input())
    print("(Directory containing Boost Python Library) [None]: BOOST_LIB_DIR")
    bld = norm_path(raw_input())
    print("(Name of boost_python shared library (e.g. boost_python-mt)) [boost_python]: ")
    bpl = raw_input()

    #Throw away extension, if given
    if bpl:
        dot = string.rfind(bpl, '.')
        if dot > 0:
            bpl = bpl[:dot]
    else:
        bpl = 'boost_python'
        

    siteconf['BOOST_INC_DIR'] = bid
    siteconf['BOOST_LIB_DIR'] = bld
    siteconf['BOOST_PYTHON_LIBNAME'] = bpl
    
    if bid:
        #Check for sanity
        if not os.path.exists(
                os.path.join(
                    os.path.join(bid, 'boost'),
                    'python.hpp')):
            raise IOError('BOOST_INC_DIR (%s) does not appear to point to a valid Boost include directory' % bid)

    if bld:
        
        #Check for sanity
        import glob
        bpls = glob.glob(os.path.join(siteconf['BOOST_LIB_DIR'], '*%s*' % bpl))
        if not bpls:
            raise IOError('BOOST_LIB_DIR (%s) does not appear to point to a directory containing a Boost Python Library' % bld)
    
    f = open("siteconf.py", 'w')
    for k, v in siteconf.items():
        if v:
            v = '"' + str(v) + '"'
        print >> f, '%s = %s' % (k, v)
        
    f.close()

Export('siteconf')

#Parallelize the build maximally
import multiprocessing
n_jobs = multiprocessing.cpu_count()
SetOption('num_jobs', n_jobs)

Export('env')

    

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

for x in frontend_library_files:
    install_path = os.path.join('stage','copperhead','library','prelude')
    env.Install(install_path, x)
    

siteconf_file = 'siteconf.py'
env.Install(os.path.join('stage', 'copperhead', 'runtime'),
            siteconf_file)
