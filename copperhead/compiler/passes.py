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

"""
Top-level coordination of compiler passes.

The Copperhead compiler is organized as a collection of passes.  This
module handles the coordination of passes necessary to compile a
Copperhead source program, and also provides the facilities to define
custom compilation pipelines.

A compiler pass is any callable object that accepts two parameters:

    - a Copperhead AST

    - a Compilation object

and returns an AST suitable for passing to subsequent passes.
Compilation objects encapsulate all persistent state that needs to be
passed between passes.
"""

import __builtin__

from pltools import strlist, Environment

import typeinference

import rewrites as Front
import conversions
import binarygenerator as Binary
import coresyntax as S
import backend as Back

def ast_to_string(ast):
    return strlist(ast, sep='\n', form=str)

class Compilation(object):
    """
    Compilation objects hold persistent state accumulated in the compiler.
    """

    def __init__(self,
                 source=str(),
                 globals=None,
                 input_types=dict(),
                 ):
        

        self.input_types = input_types
        self.entry_points = self.input_types.keys()
        self.source_text = source
        self.globals = globals
        self.type_context = typeinference.TypingContext(globals=globals)
        
        
class Pipeline(object):

    def __init__(self, name, passes):
        self.name = name
        self.__name__ = name
        self.passes = passes

        self.capture = None

    def emit(self, name, ast, M):
        if self.capture:
            self.capture.send( (name, ast, M) )

    def __call__(self, ast, M):
        self.emit('BEGIN '+self.__name__, ast, M)

        for P in self.passes:
            try:
                ast = P(ast, M)
            except:
                print
                print "ERROR during compilation in", P.__name__
                print S._indent(ast_to_string(ast))
                raise

            self.emit(P.__name__, ast, M)

        self.emit('END '+self.__name__, ast, M)
        return ast

def parse(source, mode='exec', **opts):
    'Convert string containing Copperhead code to an AST'
    from pyast import expression_from_text, statement_from_text
    if mode is 'exec':
        return statement_from_text(source)
    elif mode is 'eval':
        return expression_from_text(source)
    else:
        raise ValueError, "illegal parsing mode (%s)" % mode



def xform(fn):
    'Decorator indicating a procedure which is a compiler pass.'
    return fn

########################################################################
#
# FRONT-END PASSES
#

@xform
def gather_source(ast, M):
    'Gather source code for this function'
    return Front.gather_source(ast, M)

@xform
def mark_identifiers(ast, M):
    'Mark all user-provided identifiers'
    return Front.mark_identifiers(ast, M)

@xform
def lower_variadics(ast, M):
    'Convert variadic function calls into a lowered form'
    return Front.lower_variadics(ast)

@xform
def scrub_literals(ast, M):
    'Removes type information from implicitly typed literals'
    return Front.scrub_literals(ast)

@xform
def single_assignment_conversion(ast, M):
    'Convert text to Copperhead AST'
    return Front.single_assignment_conversion(ast)

@xform
def closure_conversion(ast, M):
    'Perform single assignment conversion'
    env = Environment(M.globals, __builtin__.__dict__)
    return Front.closure_conversion(ast, env)

@xform
def lambda_lift(ast, M):
    'Promote lambda functions to real procedures'
    return Front.lambda_lift(ast)

@xform
def procedure_flatten(ast, M):
    'Turn nested procedures into sets of procedures'
    return Front.procedure_flatten(ast)

@xform
def expression_flatten(ast, M):
    'Make every statement an atomic expression (no nested expressions)'
    return Front.expression_flatten(ast)

@xform
def protect_conditionals(ast, M):
    'Wrap branches of conditional expressions with lambdas'
    return Front.ConditionalProtector().rewrite(ast)

@xform
def inline(ast, M):
    return Front.procedure_prune(Front.inline(ast), M.entry_points)

@xform
def type_assignment(ast, M):
    typeinference.infer(ast, context=M.type_context, input_types=M.input_types)
    return ast

@xform
def type_globalize(ast, M):
    return typeinference.globalize(ast, context=M.type_context)

@xform
def backend_compile(ast, M):
    return Back.execute(ast, M)

@xform
def prepare_compilation(ast, M):
    return Binary.prepare_compilation(M)

@xform
def make_binary(ast, M):
    return Binary.make_binary(M)
    



frontend = Pipeline('frontend', [gather_source,
                                 mark_identifiers,
                                 scrub_literals,
                                 closure_conversion,
                                 single_assignment_conversion,
                                 protect_conditionals,  # XXX temporary fix
                                 lambda_lift,
                                 procedure_flatten,
                                 expression_flatten,
                                 inline,
                                 lower_variadics,
                                 type_assignment,
                                 type_globalize])

backend = Pipeline('backend', [backend_compile])

binarize = Pipeline('binarize', [prepare_compilation,
                                 make_binary])

to_binary = Pipeline('to_binary', [frontend,
                                   backend,
                                   binarize])

def run_compilation(target, suite, M):
    """
    Internal compilation interface.

    This will run the target compilation pipeline over the given suite
    of declarations with metadata state M.
    """
    return target(suite, M)


def compile(source,
            input_types={},
            globals=None,
            target=to_binary, **opts):

    M = Compilation(source=source,
                    globals=globals,
                    input_types=input_types)
    if isinstance(source, str):
        source = parse(source, mode='exec')
    M.arity = len(source[0].formals())
    M.time = opts.pop('time', False)
    return run_compilation(target, source, M)


