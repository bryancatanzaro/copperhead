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
