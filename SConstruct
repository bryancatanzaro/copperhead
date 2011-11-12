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
libcopperhead = SConscript('backend/src/SConscript',
                           variant_dir='backend/build',
                           duplicate=0)

#Install backend library
cuinstall = env.Install(os.path.join(os.path.join('stage', 'copperhead'),
                        'compiler'), libcopperhead)
Export('cuinstall')

extensions = SConscript('src/SConscript',
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
