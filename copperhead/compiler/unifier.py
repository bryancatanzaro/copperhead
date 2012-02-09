#
#   Copyright 2008-2012 NVIDIA Corporation
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

"""
Procedures for unification of syntactic forms.
"""

def unify(t1, t2, tcon):
    # (1) Instantiate with fresh variables where appropriate
    t1 = tcon.instantiate(t1)
    t2 = tcon.instantiate(t2)

    # (2) If t1/t2 are variables, resolve their values in current environment
    t1 = tcon.resolve_variable(t1)
    t2 = tcon.resolve_variable(t2)

    # (3a) For variables, update the environment
    if tcon.is_variable(t1):
        # Do nothing if t1 and t2 are identical typevars
        if t1!=t2:
            tcon.occurs_check(t1, t2)
            tcon.typings[t1] = t2
    elif tcon.is_variable(t2):
        tcon.occurs_check(t2, t1)
        tcon.typings[t2] = t1

    # (3b) For other forms, check that constructors are compatible and
    #      then recursively unify parameters
    else:
        if t1.name != t2.name or len(t1.parameters) != len(t2.parameters):
            tcon.error('type mismatch %s and %s' % (t1,t2))
       
        for (u,v) in zip(t1.parameters, t2.parameters):
            unify(u, v, tcon)
