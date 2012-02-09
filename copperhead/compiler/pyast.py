#
#   Copyright 2008-2012 NVIDIA Corporation
#  Copyright 2009-2010 University of California
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

# Converting Python's native AST to internal Copperhead form


# There are several internal Python AST representations.  This is the
# most recent and the simplest.  It is only supported in Python 2.5 and
# later.

try:
    # Try the Python 2.6 interface
    import ast
except ImportError:
    # Fall back on the Python 2.5 interface, faking the 2.6 interface
    # where possible
    import _ast as ast
    ast.parse = lambda expr, filename='<unknown>', mode='exec': \
                    compile(expr, filename, mode, ast.PyCF_ONLY_AST)


from visitor import Visitor
from coresyntax import *


# Map from Python operator AST classes to function names
op_table = {
        # Binary operators
        ast.Add : 'op_add',
        ast.Sub : 'op_sub',
        ast.Mult : 'op_mul',
        ast.Div  : 'op_div',
        ast.Mod  : 'op_mod',
        ast.Pow  : 'op_pow',
        ast.LShift : 'op_lshift',
        ast.RShift : 'op_rshift',
        ast.BitOr  : 'op_or',
        ast.BitXor : 'op_xor',
        ast.BitAnd : 'op_and',

        # Boolean operators
        ast.Eq    : 'cmp_eq',
        ast.NotEq : 'cmp_ne',
        ast.Lt    : 'cmp_lt',
        ast.LtE   : 'cmp_le',
        ast.Gt    : 'cmp_gt',
        ast.GtE   : 'cmp_ge',

        # Unary operators
        ast.Invert : 'op_invert',
        ast.Not    : 'op_not',
        ast.UAdd   : 'op_pos',
        ast.USub   : 'op_neg',
}




class _astVisitor(Visitor):
    def __init__(self):  super(_astVisitor,self).__init__()

    def name_list(self, names):  return [name.id for name in names]

    def extract_arguments(self, args):  return self.name_list(args.args)

class Printer(_astVisitor):

    def __init__(self):
        super(Printer,self).__init__()

    def _Module(self, tree):
        print "BEGIN module"
        self.visit(tree.body)

    def _FunctionDef(self, tree):
        argnames = self.extract_arguments(tree.args)
        print "def %s(%s):" % (tree.name, argnames)
        self.visit(tree.body)

    def _Name(self, tree):
        print tree.id

    def _Call(self, tree):
        name = tree.func
        args = tree.args
        print "%s(%s)" % (name.id, self.visit(args))

    def _Lambda(self, tree):
        argnames = self.extract_arguments(tree.args)
        print "lambda %s: ", argnames
        self.visit(tree.body)
        
    def _Return(self, tree):
        print "return "
        if tree.value:
            self.visit(tree.value)

class ExprConversion(_astVisitor):
    "Convert Python's ast expression trees to Copperhead's AST"

    def __init__(self):  super(ExprConversion,self).__init__()

    def _Expression(self, tree): return self.visit(tree.body)

    def _Num(self, tree):  return Number(tree.n)
    def _Name(self, tree): return Name(tree.id)

    def _BoolOp(self, tree):
        op = type(tree.op)
        if op==ast.And:
            return And(*self.visit(tree.values))
        elif op==ast.Or:
            return Or(*self.visit(tree.values))
        else:
            self.unknown_node(tree)

    def _BinOp(self, tree):
        L = self.visit(tree.left)
        R = self.visit(tree.right)
        op = Name(op_table[type(tree.op)])
        return Apply(op, [L, R])

    def _UnaryOp(self, tree):
        L = self.visit(tree.operand)
        op = Name(op_table[type(tree.op)])
        return Apply(op, [L])

    def _Compare(self, tree):
        if tree.ops[1:]:
            raise SyntaxError, "can't accept multiple comparisons"
        else:
            L = self.visit(tree.left)
            R = self.visit(tree.comparators[0])
            op = Name(op_table[type(tree.ops[0])])
            return Apply(op, [L, R])

    def _Call(self, tree):
        fn = self.visit(tree.func)
        args = self.visit(tree.args)
        if type(fn)==Name and fn.id=='map':
            return Map(args)
        else:
            return Apply(fn, args)

    def _Lambda(self, tree):
        args = self.visit(tree.args.args)
        body = self.visit(tree.body)
        return Lambda(args, body)

    def _IfExp(self, tree):
        test = self.visit(tree.test)
        body = self.visit(tree.body)
        alt  = self.visit(tree.orelse)
        return If(test, body, alt)

    def _Tuple(self, tree):
        return Tuple(*self.visit(tree.elts))

    def _Subscript(self, tree):
        value = self.visit(tree.value)
        slice = self.visit(tree.slice)
        return Subscript(value, slice)

    def _Index(self, tree):
        return Index(self.visit(tree.value))

    def _Slice(self, tree):
        raise SyntaxError, "array slicing is not yet supported"

    def _ListComp(self, tree):
        E      = self.visit(tree.elt)
        target = tree.generators[0].target
        iter   = tree.generators[0].iter
        ifs    = tree.generators[0].ifs

        if len(tree.generators) > 1:
            raise SyntaxError, \
                  "can't have multiple generators in comprehensions"
        if len(ifs)>0:
            raise SyntaxError, \
                    "predicated comprehensions are not supported"

        target = self.visit(target)
        iter = self.visit(iter)

        if isinstance(target, Name):
            return Map([Lambda([target], E), iter])
        elif isinstance(target, Tuple):
            def is_zip(t):
                return isinstance(t, Apply) and t.function().id == "zip"

            if not is_zip(iter):
                raise SyntaxError, \
                        "multivariable comprehensions work only with zip()"

            return Map([Lambda(list(target.parameters), E)] + list(iter.arguments()))

        else:
            raise SyntaxError, \
                    "unsupported list comprehension form"



convert_expression = ExprConversion()

def expression_from_text(text, source="<string>"):
    past = ast.parse(text, source, mode='eval')
    return convert_expression(past)



class StmtConversion(_astVisitor):
    "Convert Python's _ast statement trees to Copperhead's AST"

    def __init__(self):  super(StmtConversion,self).__init__()

    def _Module(self, tree):  return self.visit(tree.body)

    def _Return(self, tree):
        return Return(convert_expression(tree.value))

    def _Assign(self, tree):
        if len(tree.targets)>1:
            raise SyntaxError, 'multiple assignments not supported'

        id = convert_expression(tree.targets[0])
        value = convert_expression(tree.value)
        return Bind(id, value)

    def _FunctionDef(self, tree):
        id = tree.name
        args = convert_expression(tree.args.args)

        if tree.args.vararg:
            raise SyntaxError, 'varargs not allowed in function definitions'
        if tree.args.kwarg:
            raise SyntaxError, 'kwargs not allowed in function definitions'
        if tree.args.defaults:
            raise SyntaxError, 'argument defaults not allowed in function definitions'

        #Remove Docstring before converting to Copperhead
        if tree.body[0].__class__.__name__ == 'Expr':
            if tree.body[0].value.__class__.__name__ == 'Str':
                body = tree.body[1:]
            else:
                body = tree.body
        else:
            body = tree.body
            
        body = self.visit(body)

        return Procedure(Name(id), args, body)

    def _If(self, tree):
        test   = convert_expression(tree.test)
        body   = self.visit(tree.body)
        orelse = self.visit(tree.orelse)
        return Cond(test, body, orelse)

convert_statement = StmtConversion()

def statement_from_text(text, source="<string>"):
    past = ast.parse(text, source, mode='exec')
    return convert_statement(past)

