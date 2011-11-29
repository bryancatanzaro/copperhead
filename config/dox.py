#Barebones Doxygen Builder

import os
import SCons.Builder

def DoxygenEmitter(source, target, env):
    input_dox_file = str(source[0])
    head, tail = os.path.split(input_dox_file)
    html_dir = env.Dir(os.path.join(head, 'html'))
    env.Precious(html_dir)
    env.Clean(html_dir, html_dir)
    return ([html_dir], source)

def generate(env):
    doxygen_builder = SCons.Builder.Builder(
        action = "cd ${SOURCE.dir} && ${DOXYGEN} ${SOURCE.file}",
        emitter = DoxygenEmitter,
        target_factory = env.fs.Entry,
        single_source = True)
    env.Append(BUILDERS = {'Doxygen' : doxygen_builder})
    env.AppendUnique(DOXYGEN='doxygen')

def exists(env):
    return env.Detect('doxygen')
