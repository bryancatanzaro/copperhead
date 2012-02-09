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
Parsing Copperhead types from strings.
"""

import coretypes as T
import pyast
import re

class _TypeConversion(pyast._astVisitor):
    "Convert Pythonified type expressions into internal form"

    def _Expression(self, tree): return self.visit(tree.body)

    def _Name(self, tree):
        id = tree.id

        if (id in T.__dict__) and isinstance(T.__dict__[id], T.Monotype):
            return T.__dict__[id]
        else:
            return tree.id

    def _BinOp(self, tree):
        if type(tree.op) != pyast.ast.RShift:
            raise SyntaxError, "illegal type operator (%s)" % tree.op
        L = self.visit(tree.left)
        R = self.visit(tree.right)
        return T.Fn(L, R)

    def _Lambda(self, tree):
        args = self.visit(tree.args.args)
        body = self.visit(tree.body)
        print "BODY=", repr(body)
        return T.Polytype(args, body)

    def _Call(self, tree):
        conx = self.visit(tree.func)
        args = self.visit(tree.args)

        if (id in T.__dict__) and isinstance(T.__dict__[id], T.Monotype):
            return T.__dict__[id](*args)
        else:
            return T.Monotype(conx, *args)

    def _Tuple(self, tree):  return T.Tuple(*self.visit(tree.elts))
    def _List(self, tree):  return T.Seq(*self.visit(tree.elts))

def _pythonify_type(text):
    """
    Convert valid Copperhead type expression to valid Python expression.

    The Copperhead type language is very nearly syntactically valid
    Python.  To make parsing types relatively painless, we shamelessly
    convert type expressions into similar Python expressions, which we
    then feed into the Python parser.  Producing a syntactically valid
    Python expression involves the following conversions:

        - substitute 'lambda' for 'ForAll'
        - substitute '>>' for the '->' function operator
    """

    text = re.sub(r'ForAll(?=\s)', 'lambda', text)
    text = re.sub(r'->', '>>', text)

    return text.strip()

_convert_type = _TypeConversion()

def type_from_text(text):
    past = pyast.ast.parse(_pythonify_type(text), "<type string>", mode='eval')
    return T.quantify_type(_convert_type(past))



if __name__ == "__main__":

    def trial(text):
        t = type_from_text(text)
        print
        print text, "==", str(t)
        print "        ", repr(t)

    trial("ForAll a, b : a -> b")
    trial("Int -> Bool")
    trial("Point(Float, Float)")
    trial("(Int, Bool, Float)")
    trial("[(Int,Bool) -> Float]")
