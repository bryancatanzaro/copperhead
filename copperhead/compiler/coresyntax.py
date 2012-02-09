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

# The Copperhead AST

from pltools import strlist
import copy
import utility as U

def selectCopy(dest, template, items):
    for item in items:
        if hasattr(template, item):
            setattr(dest, item, copy.copy(getattr(template, item))) 

def _indent(text, spaces=4):
    'Indent every line in the given text by a fixed number of spaces'
    from re import sub
    return sub(r'(?m)^', ' '*spaces, text)

def _bracket(text, bracket):
    'Wrap a string in brackets.'
    return bracket[0]+text+bracket[1]

def _paren(text):
    'Wrap a string in parentheses.'
    return _bracket(text, '()')

class AST(object):
    """
    Base class for objects representing syntax trees.

    This class should not be instanced, only subclassed.  Valid
    subclasses should define:

    - self.parameters: a list of nodes that (may) need to be
                       evaluated in order to evaluate this node.

    - self.variables:  a list of variables bound within this node,
                       e.g., the formal arguments of a lambda expression.

    Optionally, subclasses can define:

    - self.type: Type information about this node

    - self.phase: Phase information about this node
    """

    def children(self):  return getattr(self, 'parameters', [])
    def bindings(self):  return getattr(self, 'variables', [])

    def __repr__(self):
        return self.__class__.__name__ + \
               strlist(self.children(), '()', form=repr)

########################################################################
#
# Expression syntax
#

class Expression(AST):
    pass

# XXX Should phase out slots that alias elements in self.parameters
#     This can lead to confusion when/if we do in-place modifications of
#     expression objects.

# Literal expressions
class Literal(Expression):
    pass

class Number(Literal):
    def __init__(self, val):
        self.val = val
        self.parameters = []

    def __repr__(self): return "Number(%r)" % self.val
    def __str__(self):  return str(self.val)

class Name(Literal):
    def __init__(self, id):
        self.id = id
        self.parameters = []
    def __repr__(self): return "Name(%r)" % self.id
    def __str__(self):  return str(self.id)

# Function definition and application

class Apply(Expression):
    def __init__(self, fn, args):
        self.parameters = [fn] + args

    def __repr__(self):
        fn, args = self.parameters[0], self.parameters[1:]
        return "Apply(%r, %s)" % (fn, strlist(args, '[]', form=repr))

    def __str__(self):
        fn, args = self.parameters[0], self.parameters[1:]
        op = str(fn)
        if not isinstance(fn, (str,Name)): op = _paren(op)
        return op + strlist(args, '()', form=str)

    def function(self): return self.parameters[0]
    def arguments(self): return self.parameters[1:]

class Lambda(Expression):
    def __init__(self, args, body):
        self.variables = args
        self.parameters = [body]

    def __repr__(self):
        v, e = self.variables, self.parameters[0]
        return "Lambda(%s, %r)" % (strlist(v, '[]', form=repr), e)

    def __str__(self):
        v, e = self.variables, self.parameters[0]
        return "lambda " + strlist(v, form=str) + ": " + str(e)

    def formals(self):  return self.variables
    def body(self):     return self.parameters[0]


class Closure(Expression):
    def __init__(self, args, body):
        self.variables = args
        if not isinstance(body, list):
            self.parameters = [body]
        else:
            self.parameters = body

    def __repr__(self):
        v, e = self.variables, self.parameters[0]
        return "Closure(%s, %r)" % (strlist(v,'[]',form=repr), e)

    def __str__(self):
        v, e = self.variables, self.parameters[0]
        return "closure(%s, %s)" % (strlist(v,'[]',form=str), e)

    # XXX somewhat ugly special case for Closure nodes
    def children(self):
        return self.variables + self.parameters

    def closed_over(self):  return self.variables
    def body(self):         return self.parameters[0]

# Compound expressions

class If(Expression):
    def __init__(self, test, body, orelse):
        self.parameters = [test, body, orelse]

    def __str__(self):
        t, b, e = str(self.test()), str(self.body()), str(self.orelse())

        if isinstance(self.body(), Lambda):   b = _paren(b)
        if isinstance(self.orelse(), Lambda): e = _paren(e)

        return "%s if %s else %s" % (b, t, e)

    def test(self):   return self.parameters[0]
    def body(self):   return self.parameters[1]
    def orelse(self): return self.parameters[2]

# Special forms whose semantics differ from usual function call

class Tuple(Expression):
    def __init__(self, *args):
        self.parameters = args

    def __str__(self):
        return strlist(self.parameters, '()', form=str)

    def __iter__(self):
        for i in self.parameters:
            yield i

class And(Expression):
    def __init__(self, *args):
        self.parameters = args

    def __str__(self):
        return strlist(self.parameters, sep=' and ', form=str)

    def arguments(self):
        return self.parameters

class Or(Expression):
    def __init__(self, *args):
        self.parameters = args

    def __str__(self):
        return strlist(self.parameters, sep=' or ', form=str)

    def arguments(self):
        return self.parameters
    
class Map(Expression):
    def __init__(self, args):
        self.parameters = args

    def __str__(self):
        return 'map' + strlist(self.parameters, '()', form=str)

    def function(self): return self.parameters[0]
    def inputs(self):   return self.parameters[1:]

class Subscript(Expression):
    def __init__(self, value, slice):
        self.parameters = [value, slice]

    def __str__(self):
        return str(self.parameters[0]) + '[' + str(self.parameters[1]) + ']'

    def value(self): return self.parameters[0]
    def slice(self): return self.parameters[1]

class Index(Expression):

    def __init__(self, value):
        self.parameters = [value]

    def __str__(self):
        return strlist(self.parameters, '', sep=', ', form=str)

    def value(self): return self.parameters[0]
    
########################################################################
#
# Statement syntax
#

class Statement(AST):
   pass

class Return(Statement):
    'Return a value from the enclosing procedure definition'
    def __init__(self, value):
        self.parameters = [value]

    def value(self): return self.parameters[0]

    def __str__(self):  return 'return %s' % self.value()

class Bind(Statement):
    'Bind a value to an identifier in the current scope'
    def __init__(self, id, value):
        self.id = id
        self.parameters = [value]

    def binder(self): return self.id
    def value(self): return self.parameters[0]

    def __repr__(self):
        return 'Bind(%r, %r)' % (self.binder(), self.value())

    def __str__(self):
        return '%s = %s' % (self.binder(), self.value())

    
class Cond(Statement):
    'Conditional statement'
    def __init__(self, test, body, orelse):
        self.parameters = [test, body, orelse]

    def __str__(self):
        test   = str(self.test())
        body   = _indent(strlist(self.body(),   sep='\n', form=str))
        orelse = _indent(strlist(self.orelse(), sep='\n', form=str))
                            
        return 'if %s:\n%s\nelse:\n%s' % (test, body, orelse)

    def test(self):   return self.parameters[0]
    def body(self):   return self.parameters[1]
    def orelse(self): return self.parameters[2]

class Procedure(Statement):
    'Define a new procedure'
    def __init__(self, id, args, body, template=None):
        self.variables = [id] + args
        self.parameters = body
        if template:
            selectCopy(self, template, ['entry_point', 'master', 'type', 'phase', 'context', 'typings', 'phases'])

    def __repr__(self):
        id, args = self.variables[0], self.variables[1:]
        body = self.parameters
        return 'Procedure(%r, %r, %r)' % (id, args, body)

    def __str__(self):
        id = self.variables[0]
        args = strlist(self.variables[1:], '()', form=str)
        body   = _indent(strlist(self.parameters, sep='\n', form=str))
        return 'def %s%s:\n%s' % (id, args, body)

    def name(self):    return self.variables[0]
    def formals(self): return self.variables[1:]
    def body(self):    return self.parameters

class Null(Statement):
    def __init__(self):
        pass
    def __repr__(self):
        return 'Null()'
    def __str__(self):
        return ''


########################################################################
#
# Standard tools for processing syntax trees
#

def walk(*roots):
    from collections import deque

    pending = deque(roots)
    while pending:
        next = pending.popleft()
        pending.extend(next.children())
        yield next


class SyntaxVisitor(object):

    def visit_children(self, x):  return self.visit(x.children())

    def visit(self, x):
        from itertools import chain
        if isinstance(x, (list, tuple)):
            return [self.visit(y) for y in x]
        else:
            name = "_"+x.__class__.__name__
            fn = getattr(self, name, self._default)
            return fn(x)

    def _default(self, x):
        if not hasattr(x, 'children'):
            raise ValueError, "can't visit node: %r" % x
        else:
            return self.visit_children(x)

class SyntaxFlattener(object):

    def visit_children(self, x):  return self.visit(x.children())

    def visit(self, x):
        from itertools import chain
        if isinstance(x, (list, tuple)):
            # NOTE: the 'or []' is not necessary in Python 2.6, but is
            #       needed in Python 2.5.
            return chain(*[self.visit(y) or [] for y in x])
        else:
            name = "_"+x.__class__.__name__
            fn = getattr(self, name, self._default)
            return fn(x)

    def _default(self, x):
        if not hasattr(x, 'children'):
            raise ValueError, "can't visit node: %r" % x
        else:
            return self.visit_children(x)


class SyntaxRewrite(object):

    def rewrite_children(self, x):
        x.parameters = self.rewrite(x.parameters)

        # XXX ugly special case for Closure nodes!
        if isinstance(x, Closure):
            x.variables = self.rewrite(x.variables)

        return x
        
    def rewrite(self, x):
        if isinstance(x, (list, tuple)):
            return  [self.rewrite(y) for y in x]
        else:
            name = "_"+x.__class__.__name__
            fn = getattr(self, name, self._default)
            x_copy = copy.copy(x)
            rewritten = fn(x_copy)
            return rewritten

    def _default(self, x):
        if not hasattr(x, 'parameters'):
            raise ValueError, "can't rewrite node: %r" % x
        else:
            return self.rewrite_children(x)


class FreeVariables(SyntaxFlattener):

    def __init__(self, env):
        self.env = env
      
    def _Name(self, x):
        if x.id not in self.env:
            yield x.id
            
    def _Bind(self, x):
        names = U.flatten(x.binder())
        result = list(self.visit(x.value()))
        for name in names:
            self.env[name.id] = name.id
        return result
    
    def filter_bindings(self, x):
        from itertools import ifilter
        bound = [v.id if hasattr(v, 'id') else v for v in x.variables]
        return ifilter(lambda id: id not in bound, self.visit_children(x))

    def _Lambda(self, x):     return self.filter_bindings(x)
    def _Procedure(self, x):
        self.env.begin_scope()
        result = list(self.filter_bindings(x))
        self.env.end_scope()
        return result


class VariableSubstitute(SyntaxRewrite):

    def __init__(self, subst):  self.subst = subst

    def _Name(self, x):
        if x.id in self.subst:
            return copy.copy(self.subst[x.id])
        return x

    def _Lambda(self, x):
        self.subst.begin_scope()
        for v in x.variables:
            # Prevent variables bound by Lambda from being substituted
            self.subst[v.id] = v

        self.rewrite_children(x)
        self.subst.end_scope()
        return x

    def _Procedure(self, x): return self._Lambda(x)

    def _Bind(self, bind):
        newId = self.rewrite(bind.id)
        self.rewrite_children(bind)
        return Bind(newId, bind.parameters[0])


def print_source(ast, step=4, indentation=0, prefix=''):
    lead = prefix + ' '*indentation

    if isinstance(ast, Procedure):
        name = ast.variables[0]
        args = strlist(ast.variables[1:], '()', form=str)
        body = ast.parameters
        print "%sdef %s%s:" % (lead, name, args)
        print_source(body, step, indentation+step, prefix)
    elif isinstance(ast, list):
        for s in ast:
            print_source(s, step, indentation, prefix)
    else:
        print "%s%s" % (lead, ast)

def free_variables(e, env={}):
    'Generates all freely occurring identifiers in E not bound in ENV.'
    from pltools import Environment
    return FreeVariables(Environment(env)).visit(e)

def toplevel_procedures(ast):
    'Generate names of all top-level procedures in the given code block.'
    from itertools import ifilter
    if isinstance(ast, list):
        for p in ifilter(lambda x: isinstance(x,Procedure), ast):
            yield p.name().id
    elif isinstance(ast, Procedure):
        yield ast.name().id

def substituted_expression(e, env):
    """
    Return an expression with all freely occurring identifiers in E
    replaced by their corresponding value in ENV.
    """
    from pltools import Environment
    subst = Environment(env)
    rewriter = VariableSubstitute(subst)
    return rewriter.rewrite(e)

def mark_user(name):
    return Name('_' + name.id)
    
