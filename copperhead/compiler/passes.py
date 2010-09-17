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
import shapeinference

import rewrites as Front, midrewrites as Mid, backrewrites as Back, binarygenerator as Binary
import coresyntax as S, backendsyntax as B
import intrinsics as I

def ast_to_string(ast):
    return strlist(ast, sep='\n', form=str)

class Compilation(object):
    """
    Compilation objects hold persistent state accumulated in the compiler.
    """

    def __init__(self,
                 source=str(),
                 globals=None,
                 inputTypes=dict(),
                 inputShapes=dict(),
                 inputPlaces=dict(),
                 functors=set(),
                 ):
        

        self.inputTypes = inputTypes
        self.inputShapes = inputShapes
        self.inputPlaces = inputPlaces
        self.entry_points = self.inputTypes.keys()

        self.source_text = source
        self.globals = globals
        self.functors = functors
        self.type_context = typeinference.TypingContext(globals=globals)
        self.shape_context = shapeinference.ShapingContext(globals=globals)
        
        
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
def collect_toplevel(ast, M):
    'Collect all top-level declarations for later reference.'
    M.toplevel = list(S.toplevel_procedures(ast))
    return ast

@xform
def gather_source(ast, M):
    'Gather source code for this function'
    return Front.gather_source(ast, M)

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
def collect_local_typings(suite, M):
    'For each top-level procedure, collect typings of all local variables'
    return typeinference.collect_local_typings(suite, M)

def collect_local_shapes(suite, M):
    'For each top-level procedure, collect shapes of all local variables'
    return shapeinference.collect_local_shapes(suite, M)


########################################################################
#
# MID-END PASSES
#

@xform
def remove_recursion(ast, M):
    'Convert tail recursion to loops'
    return Mid.remove_recursion(ast)

@xform
def type_assignment(ast, M):
    typeinference.infer(ast, context=M.type_context)
    collect_local_typings(ast, M)
    return ast

@xform
def shape_assignment(ast, M):
    shapeinference.infer(ast, context=M.shape_context)
    collect_local_shapes(ast, M)
    return ast


@xform 
def sequentialize(ast, M):
    'Choose sequential or parallel execution'
    return Mid.sequentialize(ast, M.entry_points, M.p_hier)

@xform
def inline(ast, M):
    'Inline function calls'
    return Front.inline(ast)

@xform
def procedure_prune(ast, M):
    'Remove procedures which are orphaned after inlining'
    return Front.procedure_prune(ast, M.toplevel)

@xform
def unzip_transform(ast, M):
    'Select variant being used'
    return Mid.unzip_transform(ast, M.entry_points)

@xform
def select_variant(ast, M):
    'Select variant being used'
    return Mid.select_variant(ast, M.globals)

@xform
def phase_analysis(ast, M):
    'Find phase boundaries'
    return Mid.phase_analysis(ast, M.globals)

@xform
def phase_rewrite(ast, M):
    'Find phase boundaries'
    return Mid.phase_rewrite(ast, M.entry_points)


########################################################################
#
# BACK-END PASSES
#

@xform
def thrust_filter(ast, M):
    return Back.thrust_filter(ast, M.entry_points)

@xform
def execution_shapes(ast, M):
    'Compute execution shapes for resulting kernels'
    return Back.execution_shape(ast, M.inputShapes)
    
@xform
def reference_conversion(ast, M):
    'Turn returns into assignments'
    #
    # XXX reference_conversion might need to be rewritten to operate on globals
    # rather than the old "typings" entry of the type inference engine
    return Back.reference_conversion(ast, Environment(M.globals))


@xform
def closure_lift(ast, M):
    return Back.closure_lift(ast, M)

@xform
def allocate_conversion(ast, M):
    'Make memory allocation explicit'
    return Back.allocate_conversion(ast)

@xform
def structify(ast, M):
    'Turn function arguments into templated structs.'
    return Back.structify(ast, M)

@xform
def sequential_fusion(ast, M):
    return Back.sequential_fusion(ast, M)

@xform
def tuplify(ast, M):
    return Back.tuplify(ast, M);

@xform
def ctype_conversion(ast, M):
    'Add C Type declarations'
    return Back.ctype_conversion(ast)

@xform
def cnode_conversion(ast, M):
    'Convert to C Nodes'
    return Back.cnode_conversion(ast, M.p_hier)

@xform
def intrinsic_conversion(ast, M):
    'Convert Copperhead intrinsics to C Nodes'
    return Back.intrinsic_conversion(ast)

@xform
def uniform_conversion(ast, M):
    'Find all declarations of uniform_nested_sequences'
    return Back.uniform_conversion(ast, M.uniforms)

@xform
def pycuda_wrap(ast, M):
    'Wrap functions to be accessible from PyCUDA'
    return Back.pycuda_wrap(ast, M.inputTypes, M.fn_types, M.time)
@xform
def rename_templates(ast, M):
    'Rename template types to avoid clashes with variable names'
    return Back.rename_templates(ast)

########################################################################
#
# BINARIZING PASSES
#
@xform
def final_python_code(ast, M):
    'Generate python code'
    return Binary.final_python_code(ast, M)

@xform
def final_cuda_code(ast, M):
    'Generate cuda code'
    return Binary.final_cuda_code(ast, M)

@xform
def find_entry_points(ast, M):
    'Find final entry points'
    return Binary.find_entry_points(ast, M)

@xform
def find_master(ast, M):
    'Find master Python coordination function'
    return Binary.find_master(ast, M)

@xform
def make_binary(ast, M):
    'Generate executable function'
    return Binary.make_binary(ast, M)

@xform
def print_cuda_code(ast, M):
    'Terminate pass by returning CUDA code'
    return M.cuda_code

@xform
def print_python_code(ast, M):
    'Terminate pass by returning Python code'
    return M.python_code

frontend = Pipeline('frontend', [collect_toplevel,
                                 gather_source,
                                 closure_conversion,
                                 single_assignment_conversion,
                                 protect_conditionals,  # XXX temporary fix
                                 lambda_lift,
                                 procedure_flatten,
                                 expression_flatten,
                                 type_assignment,
                                 collect_local_typings ] )

midend = Pipeline('midend', [remove_recursion,
                             inline,
                             procedure_prune,
                             sequentialize,
                             type_assignment,
                             shape_assignment,
                             unzip_transform,
                             select_variant,
                             phase_analysis,
                             phase_rewrite
                             ] )

backend = Pipeline('backend', [procedure_flatten,
                               thrust_filter,
                               execution_shapes,
                               reference_conversion,
                               closure_lift,
                               sequential_fusion,
                               tuplify,
                               structify,
                               intrinsic_conversion,
                               ctype_conversion,
                               cnode_conversion,
                               uniform_conversion,
                               pycuda_wrap,
                               rename_templates,
                               allocate_conversion] )

binarizer = Pipeline('compiler', [final_cuda_code,
                                  final_python_code,
                                  find_entry_points,
                                  find_master,
                                  make_binary])


through_frontend = Pipeline('through_frontend', [frontend,
                                                 final_python_code,
                                                 print_python_code])

through_midend   = Pipeline('through_midend',   [frontend,
                                                 midend,
                                                 final_python_code,
                                                 print_python_code])

to_cuda = Pipeline('cuda_pipeline', [frontend,
                                     midend,
                                     backend,
                                     binarizer])

functorize = Pipeline('functorize',
                      [collect_toplevel,
                       single_assignment_conversion,
                       protect_conditionals,
                       closure_conversion,
                       lambda_lift,
                       procedure_flatten,
                       expression_flatten,
                       type_assignment,
                       inline,
                       procedure_prune,
                       type_assignment,
                       phase_analysis,
                       phase_rewrite,
                       procedure_flatten,
                       thrust_filter,
                       reference_conversion,
                       tuplify,
                       structify,
                       intrinsic_conversion,
                       ctype_conversion,
                       cnode_conversion,
                       rename_templates,
                       final_cuda_code,
                       print_cuda_code])

def run_compilation(target, suite, M):
    """
    Internal compilation interface.

    This will run the target compilation pipeline over the given suite
    of declarations with metadata state M.
    """
    return target(suite, M)


def compile(source,
            inputTypes={}, inputShapes={}, inputPlaces={}, uniforms=[],
            globals=None,
            target=to_cuda,
            functors=set(), **opts):

    M = Compilation(source=source,
                    globals=globals,
                    inputTypes=inputTypes,
                    inputShapes=inputShapes,
                    inputPlaces=inputPlaces,
                    functors=functors)
    if isinstance(source, str):
        source = parse(source, mode='exec')
    M.save_code = opts.pop('save_code', False)
    M.block_size = opts.pop('block_size', (256, 1, 1))
    M.time = opts.pop('time', False)
    M.p_hier = opts.pop('p_hier', (I.distributed, I.sequential))
    M.uniforms = uniforms
    return run_compilation(target, source, M)


def get_functor(fn):
    return compile(fn.get_ast(), target=functorize, globals=fn.get_globals(), functors=set([fn.__name__]))
