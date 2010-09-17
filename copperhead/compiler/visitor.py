#
#  Copyright 2008-2009 NVIDIA Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#


class Visitor(object):

    def __init__(self):
        super(Visitor,self).__init__()
        self.fallback = self.unknown_node

    def visit(self, tree):
        if isinstance(tree, list):
            return [self.visit(i) for i in tree]
        elif isinstance(tree, tuple): 
            return [self.visit(i) for i in list(tree)]
        else:
            name = "_"+tree.__class__.__name__
            if hasattr(self, name):
                fn = getattr(self, name)
                return fn(tree)
            else:
                return self.fallback(tree)

    def __call__(self, tree): return self.visit(tree)

    def unknown_node(self, tree):
        raise ValueError, "visiting unknown node: %s " % tree
