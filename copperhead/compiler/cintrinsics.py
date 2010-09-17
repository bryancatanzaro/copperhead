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

import backendsyntax as B
import codesnippets as C
import coresyntax as S
import coretypes as T
import backtypes as BT
import pltools as P

def scalar(fn):
    fn.scalar = True
    return fn

class CIntrinsicFactory(object):
    scalar = True
    def __init__(self, id, cstr = '', infix=False, unary=False):
        self.id = id
        self.cstr = cstr
        self.infix = infix
        self.unary = unary
    def __call__(self, bind, typings, write_out):
        app = bind.value()
        parameters = app.arguments()
        return S.Bind(bind.binder(), B.CIntrinsic(self.id, parameters, self.cstr, self.infix, self.unary))



_True     = CIntrinsicFactory('True', 'true')
_False    = CIntrinsicFactory('False', 'false')
_None     = CIntrinsicFactory('None', 'void')
_op_add   = CIntrinsicFactory('op_add', ' + ', True)
_op_sub   = CIntrinsicFactory('op_sub', ' - ', True)
_op_mul   = CIntrinsicFactory('op_mul', ' * ', True)
_op_div   = CIntrinsicFactory('op_div', ' / ', True)
_op_mod   = CIntrinsicFactory('op_mod', ' % ', True)
_op_pow   = CIntrinsicFactory('op_pow', 'pow')
_op_lshift= CIntrinsicFactory('op_lshift', ' << ', True)
_op_rshift= CIntrinsicFactory('op_rshift', ' >> ', True)
_op_or    = CIntrinsicFactory('op_or', ' | ', True)
_op_xor   = CIntrinsicFactory('op_xor', ' ^ ', True)
_op_and   = CIntrinsicFactory('op_and', ' & ', True)
_op_band  = CIntrinsicFactory('op_band', ' && ', True)
_op_bor   = CIntrinsicFactory('op_bor', ' || ', True)
_op_pos   = CIntrinsicFactory('op_pos', '+', unary=True)
_op_neg   = CIntrinsicFactory('op_neg', '-', unary=True)
_op_not   = CIntrinsicFactory('op_not', '~', unary=True)
_cmp_eq   = CIntrinsicFactory('cmp_eq', ' == ', True)
_cmp_ne   = CIntrinsicFactory('cmp_ne', ' != ', True)
_cmp_lt   = CIntrinsicFactory('cmp_lt', ' < ', True)
_cmp_le   = CIntrinsicFactory('cmp_lt', ' <= ', True)
_cmp_gt   = CIntrinsicFactory('cmp_gt', ' > ', True)
_cmp_ge   = CIntrinsicFactory('cmp_gt', ' >= ', True)

def intrinsic_call(name, bind):
    intrinsic = globals().get('_' + name, False)
    if intrinsic:
        return intrinsic(bind, None, False)
    return False


    

def declaration(type, name):
    return B.CTypeDecl(type, name)

def assign(dest, exp):
    return S.Bind(dest, exp)

def convert(assign):
    name = assign.value().function().id
    return intrinsic_call(name, assign)

boilerplateStatements = [assign(declaration(C.tileIndexType, C.tileIndex), C.threadIdx),
               assign(declaration(C.tileIdType, C.tileId), C.blockIdx),
               convert(assign(declaration(C.tileBeginType, C.tileBegin), S.Apply(S.Name('op_mul'), [C.tileId, C.blockDim]))),
               convert(assign(declaration(C.globalIndexType, C.globalIndex), S.Apply(S.Name('op_add'), [C.tileBegin, C.tileIndex])))]


boilerplate = [B.CCommentBlock('Copperhead boilerplate', 'Generally useful indices.', boilerplateStatements)]

syncthreads = S.Apply(S.Name('__syncthreads'), [])

headers = [B.CInclude('copperhead.h')]

intrinsic_name_supply = P.name_supply()


def _len(bind, typings, write_out):
    app = bind.value()
    sequence = app.parameters[1]
    size = B.CMember(sequence, S.Apply(S.Name("size"), []))
    bind.parameters[0] = size
    return bind

def _range(bind, typings, write_out):
    typings[bind.binder().id] = T.Monotype("index_sequence")
    bound = apply.parameters[1]
    bind.parameters[0] = S.Apply(S.Name("index_sequence"), [bound])
    return bind
    
def _indices(bind, typings, write_out):
    app = bind.value()
    if write_out:
        new_dest_id = C.markGenerated(bind.binder().id + '_'  + \
                                      intrinsic_name_supply.next())
        typings[new_dest_id] = T.Monotype("index_sequence")
        source = app.parameters[1]
        declaration = S.Bind(S.Name(new_dest_id),
                             S.Apply(S.Name("index_sequence"),
                                     [B.CMember(source, S.Apply(S.Name("size"),
                                                                []))]))
        declaration.no_return_convert = True
        copy = S.Apply(S.Name('copy'),
                       [bind.binder(),
                       S.Name(new_dest_id),
                       C.globalIndex])
        return declaration, copy
    
    typings[bind.binder().id] = T.Monotype("index_sequence")
    source = app.parameters[1]
    bind.parameters[0] = S.Apply(S.Name("index_sequence"),
                                 [B.CMember(source, S.Apply(S.Name("size"), []))])
    return bind
            
                                                        
def _shift(bind, typings, write_out):
    app = bind.value()
    source = app.parameters[1]
    offset = app.parameters[2]
    fill = app.parameters[3]
    binder_type = typings[bind.binder().id]
    shift_type = BT.DependentType("shifted_sequence", binder_type.parameters)
    typings[bind.binder().id] = shift_type
    ctype = B.CType(shift_type)
    constructor = B.CConstructor(ctype, [source, offset, fill])
    bind.parameters[0] = constructor
    return bind

@scalar
def _exp(bind, typings, write_out):
    app = bind.value()
    operand = app.parameters[1]
    bind.parameters[0] = S.Apply(S.Name('exp'), [operand])
    return bind
