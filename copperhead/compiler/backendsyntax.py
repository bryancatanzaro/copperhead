#
#  Copyright 2008-2010 NVIDIA Corporation
#  Copyright 2009-2010 University of California
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

#The Copperhead Backend AST

from coresyntax import *
import coretypes as T
import backtypes as BT
import coresyntax as S
from pltools import strlist
from ..runtime import cudata 

class IndentManager(object):
    def __init__(self, level = 0, sep = '  '):
        self.level = level
        self.sep = sep
    def indent(self):
        self.level = self.level + 1
    def dedent(self):
        self.level = self.level - 1
        if (self.level < 0):
            self.level = 0
    def prefix(self):
        return self.level * self.sep

indentation = IndentManager()

class CBlock(AST):
    begin_block = '{'
    end_block = '}'


class StringBuilder(object):
    def __init__(self, contents = ''):
        self.contents = contents
    def add(self, x, post='', pre = ''):
        self.contents = self.contents + pre + str(x)
        if not isinstance(x, CBlock):
            self.contents = self.contents + post
        return self.contents
    def add_(self, x):
        return self.add(x, ' ')
    def addP_(self, x):
        return self.add(x, ' ', indentation.prefix())
    def addP(self, x):
        return self.add(x, '', indentation.prefix())
    def addN(self, x):
        return self.add(x, '\n')
    def addSN(self, x):
        return self.add(x, ';\n')
    def addS(self, x):
        return self.add(x, ';')
    def addPN(self, x):
        return self.add(x, '\n', indentation.prefix())
    def addPSN(self, x):
        return self.add(x, ';\n', indentation.prefix())

class Reference(Expression):
    def __init__(self, arg):
        self.parameters = [arg]
        self.id = str(arg)

    def __repr__(self): return "Reference(%r)" % self.id
    def __str__(self): return '&' + str(self.id)

class Template(Expression):
    def __init__(self, templateTypes, scope):
        self.parameters = scope
        self.variables = templateTypes
    def __repr__(self): return "Template<%r>(%r)" %(strlist(self.variables, form=repr), strlist(self.parameters, '', form=repr))
    def __str__(self):
        def typenamify(item):
            return 'typename ' + str(item)
        builder = StringBuilder()
        builder.addN("template<%s >" % strlist(self.variables, form=typenamify))
        builder.addPN(strlist(self.parameters, form=str))
        return builder.contents
    
class TemplateInst(Expression):
    def __init__(self, type, templateTypes):
        self.parameters = templateTypes
        self.id = type
    def __str__(self):
        return "%s<%s >" %(str(self.id), strlist(self.parameters, sep = ', ', form=str))
    def __repr__(self):
        return "%r<%r >" %(repr(self.id), strlist(self.parameters, sep = ', ', form=repr))

def flatten_sequence_type(type, depth=-1):
    if hasattr(type, 'unbox'):
        return flatten_sequence_type(type.unbox(), depth + 1)
    return (type, depth)
    
    
class CType(Expression):
    def __init__(self, type):
        self.cu_type = type
        self.lowered = False
        self.uniform = False
        
        if isinstance(type, T.Polytype):
            self.monotype = type.monotype()
        else:
            self.monotype = type
        if isinstance(type, BT.DependentType):
            dependent_params = self.monotype.parameters[1:]
            new_params = [x if isinstance(x, CType) else CType(x) for x in dependent_params]
            complete_params = [self.monotype.parameters[0]] + new_params
            self.monotype.parameters = complete_params
            


        (atomic_cu_type, self.depth) = flatten_sequence_type(self.monotype)
        if isinstance(atomic_cu_type, T.Tuple):
            atomic_types = [self.translate(x) for x in atomic_cu_type]
            self.cu_type = BT.DependentType(S.Name('typename thrust::tuple'),
                                         atomic_types)
            atomic_cu_type = self.cu_type
        self.atomic_type = self.translate(atomic_cu_type)

    def translate(self, cu_type):
        cu_str_type = str(cu_type)
        if cu_str_type in cudata.cu_to_c_types:
            return cudata.cu_to_c_types[cu_str_type]
        else:
            return cu_str_type
        
    def lower(self):
        self.lowered = True
    def mark_uniform(self):
        self.uniform = True
    def __str__(self):
        if self.depth < 0:
            return self.atomic_type
        if self.uniform:
            return 'uniform_nested_sequence<%s, %s >' % (self.atomic_type,
                                                       self.depth)
        if self.depth == 0:
            if self.lowered:
                return "lowered_sequence<%s >" % self.atomic_type
            else:
                return "stored_sequence<%s >" % self.atomic_type
        return "nested_sequence<%s, %s>" % (self.atomic_type, self.depth)
    def __repr__(self):
        return "CType<%r>" % self.cu_type
        

class CTypeDecl(Expression):
    
    def __init__(self, type, name, declspec=None):

        self.declspec = declspec
        self.type = type
        self.name = name
        self.ctype = CType(type)
        self.parameters = [self.ctype, name]
    def update_ctype(self, ctype):
        self.ctype = ctype
        self.parameters[0] = ctype
    def lower(self):
        self.ctype.lower()
    def uniform(self):
        self.ctype.mark_uniform()
    def __repr__(self):
        return "CTypeDecl(%r)" % strlist(self.parameters, '', form=repr)
    def __str__(self):
        declaration = strlist(self.parameters, sep=' ', form=str)
        if self.declspec is not None:
            declaration = self.declspec + ' ' + declaration 
        return declaration

class CStatement(AST):
    pass

class CBind(CStatement, Bind):
    def __repr__(self):
        return 'CBind(%r, %r)' % (self.binder(), self.value())

class CTypedef(CStatement):
    def __init__(self, type, name):
        self.decl = CTypeDecl(type, name, "typedef")
    def __repr__(self):
        return "CTypedef(%r)" % self.decl
    def __str__(self):
        return str(self.decl)

class CTypename(Expression):
    def __init__(self, type):
        self.type = type
    def __repr__(self):
        return "CTypename(%r)" % self.type
    def __str__(self):
        return 'typename ' + str(self.type)
    
class CConstructor(AST):
    def __init__(self, id, parameters):
        self.id = id
        self.parameters = parameters
    def __str__(self):
        declaration = str(self.id) + strlist(self.parameters, bracket = '()', sep = ', ', form=str)
        return declaration
    def arguments(self):
        return self.parameters
    
class CNamespace(Expression):
    def __init__(self, *qualifiers):
        self.qualifiers = qualifiers
    def __repr__(self):
        return "CNamespace(%r)" % strlist(self.qualifiers, sep = ', ',
                                          form=repr)
    def __str__(self):
        return strlist(self.qualifiers, sep = '::', form = str)

class CMember(Expression):
    def __init__(self, source, member):
        self.parameters = [source, member]
    def __repr__(self):
        return "CMember(%r)" % strlist(self.parameters, sep = ', ', form = repr)
    def __str__(self):
        return strlist(self.parameters, sep = '.', form = str)
    
class CIntrinsic(AST):
    def __init__(self, id, parameters=None, cstr='', infix=False, unary=False):
        self.id = id
        self.parameters = parameters
        if cstr=='':
            self.cstr = id
        else:
            self.cstr = cstr
        self.infix = infix
        self.unary = unary
    def __repr__(self):
        return "CIntrinsic(%s, %r)" % (self.id, strlist(self.parameters, '', form=repr))
    def __str__(self):
        if self.infix:
            return '(' + self.cstr.join([str(x) for x in self.parameters]) + ')'
        elif self.unary:
            return self.cstr + str(self.parameters[0])
        else:
            if self.parameters != None:
                return self.cstr + '(' + ', '.join([str(x) for x in self.parameters]) + ')'
            else:
                return self.cstr


    
class CFunction(CBlock):
    def __init__(self, proc, return_type='void', cuda_kind=None):
        self.id = proc.variables[0].id
        self.arguments = proc.variables[1:]
        self.parameters = proc.parameters
        self.return_type = return_type
        self.type = proc.type
        if proc.entry_point:
            self.cuda_kind = '__global__'
        else:
            self.cuda_kind = '__device__'
        self.entry_point = proc.entry_point
    def __repr__(self):
        result = 'CFunction(' + strlist(self.parameters, sep = ',', form=repr) + ')'
        return result
    def __str__(self):
        builder = StringBuilder()
        if self.cuda_kind != None:
            builder.add_(self.cuda_kind)
        if self.return_type != None:
            builder.add_(self.return_type)
        builder.add(self.id)
        argumentDeclaration = '(' + strlist(self.arguments, sep = ', ', form=str) + ')'
        builder.add_(argumentDeclaration)
        builder.addN(self.begin_block)
        indentation.indent()
        strlist(self.parameters, sep = '', form=builder.addPSN)
        indentation.dedent()
        builder.addPN(self.end_block)
        return builder.contents
    def name(self):
        return self.id
class CLines(CBlock):
    def __init__(self, lines):
        self.parameters = lines
    def __str__(self):
        builder = StringBuilder()
        builder.addN('')
        [builder.addPSN(str(line)) for line in self.parameters]
        return builder.contents
    def __repr__(self):
        return 'CLines' + strlist(self.parameters, sep=',')
    
class CFor(CBlock):
    def __init__(self, init, test, increment, body):
        self.parameters = [init, test, increment, body]
    def __str__(self):
        builder = StringBuilder()
        builder.add_('for')
        builder.add_(strlist([self.parameters[0], self.parameters[1], self.parameters[2]], bracket = '()', sep = '; ', form=str))
        builder.addN(self.begin_block)
        indentation.indent()
        strlist(self.parameters[3], sep = '', form = builder.addPSN)
        indentation.dedent()
        builder.addPN(self.end_block)
        return builder.contents

class CWhile(CBlock):
    def __init__(self, test, body):
        self.parameters = [test, body]
    def __str__(self):
        builder = StringBuilder()
        builder.add_('while')
        builder.add_('(%s)' % self.parameters[0])
        builder.addN(self.begin_block)
        indentation.indent()
        strlist(self.parameters[1], sep = '', form = builder.addPSN)
        indentation.dedent()
        builder.addPN(self.end_block)
        return builder.contents
    
class CIf(CBlock):
    def __init__(self, test, bodyTrue, bodyFalse=[Null()]):

        if not isinstance(bodyTrue, (list, tuple)):
            bodyTrue = [bodyTrue]
        if not isinstance(bodyFalse, (list, tuple)):
            bodyFalse = [bodyFalse]
        self.parameters = [test, bodyTrue, bodyFalse]
            
    def __str__(self):
        builder = StringBuilder()
        builder.add_('if')
        # XXX Are the brackets here necessary?  They perhaps could be removed
        builder.add_(strlist([self.parameters[0]], bracket = '()', sep = ' ', form=str))
        builder.addN(self.begin_block)
        indentation.indent()
        strlist(self.parameters[1], sep = '', form = builder.addPSN)
        indentation.dedent()
        if (len(self.parameters[2]) > 1) or not isinstance(self.parameters[2][0], Null):
            builder.addP_(self.end_block)
            builder.add_('else')
            builder.addN(self.begin_block)
            indentation.indent()
            strlist(self.parameters[2], sep = '', form = builder.addPSN)
            indentation.dedent()
        builder.addPN(self.end_block)
        return builder.contents
        

class CExtern(CBlock):
    def __init__(self, body):
        self.parameters = [body]
    def body(self):
        return self.parameters[0]
    def __str__(self):
        builder = StringBuilder()
        builder.addN('extern "C" {')
        builder.addN('')
        builder.add(str(self.parameters[0]))
        builder.addN('')
        builder.addN('}')
        return builder.contents


class CStruct(AST):
    def __init__(self, name, proc):
        self.variables = [name]
        self.parameters = proc
    def __str__(self):
        builder = StringBuilder()
        builder.add_('struct')
        builder.add_(str(self.variables[0]))
        builder.addN('{')
        indentation.indent()
        strlist(self.parameters, sep = '', form = builder.addP)
        indentation.dedent()
        builder.addPSN('}')
        return builder.contents


class CCommentBlock(CBlock):
    width = 70
    def __init__(self, name, comment, body):
        comments = comment.split('\n')
        def blockify(comment):
            length = CCommentBlock.width - len(comment) - 6
            return '/* ' + comment + ' ' * length + ' */'
        comments = [blockify(x) for x in comments]

        self.variables = [name] + comments
        self.parameters = body

    
        
    def __str__(self):
        def makeSeparator(label):
            length = (CCommentBlock.width - len(label))/2
            starsA = '*' * length
            separator = '/' + starsA + ' ' + label + ' '
            starsBLength = CCommentBlock.width - len(separator) - 1
            separator = separator + '*' * starsBLength + '/'
            return separator
        
        builder = StringBuilder()
        builder.addN(makeSeparator('Begin ' + self.variables[0]))
        for comment in self.variables[1:]:
            builder.addPN(comment)
        #builder.addN('')
        strlist(self.parameters, sep = '', form=builder.addPSN)
        #builder.addN('')
        builder.addPN(makeSeparator('End ' + self.variables[0]))
        return builder.contents

class CInclude(CBlock):
    def __init__(self, name):
        self.name = name
        self.parameters = []
    def __str__(self):
        return '#include <' + self.name + '>'
