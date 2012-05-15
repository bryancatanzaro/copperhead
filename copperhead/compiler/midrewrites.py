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


class Iterizer(S.SyntaxRewrite):
    def __init__(self):
        pass
    def _Procedure(self, proc):
        self.proc_name = proc.variables[0].id
        self.body = []
        self.current_bin = self.body
        self.tail_recursive = False
        self.rewrite_children(proc)
        if not self.tail_recursive:
            return proc
        def convert_recursion(stmt):
            if not isinstance(stmt, S.Bind):
                return stmt
            if not isinstance(stmt.id, S.Name):
                return stmt
            if stmt.id.id == C.anonymousReturnValue.id:
                arguments = proc.variables[1:]
                apply = stmt.parameters[0]
                assert apply.parameters[0].id == self.proc_name
                newArguments = apply.parameters[1:]
                return [S.Bind(x, y) for x, y in zip(arguments, newArguments)] 
            else:
                return stmt
        recursive_branch = list(flatten([convert_recursion(x) for x in self.recursive_branch]))
        recursive_branch = filter(lambda x: isinstance(x, S.Bind), recursive_branch)
        whileLoop = M.While(self.condition, [recursive_branch + self.body])
        cleanup = self.nonrecursive_branch
        proc.parameters = self.body + [whileLoop] + cleanup
        return proc
        
    def _Bind(self, stmt):
        self.current_bin.append(stmt)
        self.binder = stmt.id
        self.rewrite_children(stmt)
        return stmt

    def _Return(self, stmt):
        self.current_bin.append(stmt)
        return stmt
    
    def _Apply(self, apply):
        applyFnName = apply.parameters[0].id
        if applyFnName == self.proc_name:
            self.recursion_detected = True
            if not self.binder.id == C.anonymousReturnValue.id:
                raise SyntaxError('Recursive function detected, but recursion not in tail position')
        return apply
    def _Cond(self, cond):
        self.recursion_detected = False
        self.current_branch = []
        self.current_bin = self.current_branch
        thenBranch = self.rewrite(cond.parameters[1])
        if self.recursion_detected:
            self.recursive_branch = self.current_branch
            self.tail_recursive = True
            self.condition = cond.parameters[0]
        else:
            self.nonrecursive_branch = self.current_branch
        self.recursion_detected = False
        self.current_branch = []
        self.current_bin = self.current_branch
        elseBranch = self.rewrite(cond.parameters[2])
        if self.recursion_detected:
            self.recursive_branch = self.current_branch
            self.tail_recursive = True
            self.condition = S.Apply(S.Name('op_not'), [cond.parameters[0]])
        else:
            self.nonrecursive_branch = self.current_branch
        return cond

    
    
def remove_recursion(stmt):
    import pdb
    pdb.set_trace()
    iterizer = Iterizer()
    return iterizer.rewrite(stmt)
