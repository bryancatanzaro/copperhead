import os
import inspect
import fnmatch

# try to import an environment first
try:
    Import('env')
except:
    exec open(os.path.join('config', "build-env.py"))
    env = Environment()

#Parallelize the build maximally
#Perhaps we want this not to override the user's wishes?...
import multiprocessing
n_jobs = multiprocessing.cpu_count()
SetOption('num_jobs', n_jobs)

Export('env')

#Build backend    
libcopperhead = SConscript(os.path.join('backend',
                                         os.path.join('src', 'SConscript')),
                           variant_dir=os.path.join('backend','build'),
                           duplicate=0)

#Install backend library
#cuinstall = env.Install(os.path.join(os.path.join('stage', 'copperhead'),
#                        'compiler'), libcopperhead)
Export('libcopperhead')

extensions = SConscript(os.path.join('src', 'SConscript'),
                        variant_dir='build',
                        duplicate=0)

for x, y in extensions:
    head, tail = os.path.split(x)
    env.Install(os.path.join('stage', head), y)
    
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
    'backend', os.path.join('library', 'prelude'))) + \
    recursive_glob('*.h', os.path.join(
        'backend', os.path.join('library', 'thrust_wrappers')))

def explode_path(path):
    head, tail = os.path.split(path)
    return explode_path(head) + [tail] \
        if head and head != path \
        else [head or tail]

for x in library_files:
    exploded_path = explode_path(str(x))[:-1]
    exploded_path[0] = 'stage'
    install_path = os.path.join(*exploded_path)
    env.Install(install_path, x)

#Make this opt-in to avoid forcing everyone to install Doxygen    
if 'doc' in COMMAND_LINE_TARGETS:
    env.Alias('doc', [env.AlwaysBuild(env.Doxygen('backend/doc/copperhead.dox'))])
