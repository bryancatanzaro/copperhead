import backendsyntax as B
import coresyntax as S
import codesnippets as C
import coretypes as T
import backtypes as BT
import pdb

class CIntrinsicFactory(object):
    scalar = True
    def __init__(self, id, cstr = '', infix=False, unary=False):
        self.id = id
        self.cstr = cstr
        self.infix = infix
        self.unary = unary
    def __call__(self, bind):
        app = bind.value()
        parameters = app.arguments()
        if bind.allocate:
            declaration = B.CTypeDecl(bind.binder().type, bind.binder())
            result = declaration
        else:
            result = bind.binder()
        return S.Bind(result, B.CIntrinsic(self.id, parameters, self.cstr, self.infix, self.unary))



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

def _reduce(bind):
    if not bind.allocate:
        return bind
    declaration = B.CTypeDecl(bind.binder().type, bind.binder())
    bind.id = declaration
    return bind

def _sum(bind):
    return _reduce(bind)

def _indices(bind):
    name = bind.binder()
    source = bind.value().arguments()[0]
    ctype = B.CType(T.Monotype('index_sequence'))
    declaration = B.CTypeDecl(ctype,
                              B.CConstructor(name,
                                             [B.CApply(
                                                 B.CMember(source, \
                                                           S.Name('size')),
                                                 [])]))
    constructed_typename = C.type_from_name(name)
    typedef = B.CTypedef(ctype, constructed_typename)
    result = [declaration, typedef]
    return result

def _zip(bind):
    name = bind.binder()
    sources = bind.value().arguments()
    source_iterators = [B.CNamespace(C.type_from_name(x), S.Name('iterator')) \
                        for x in sources]
    iterator_tuple_type = B.CType(T.Tuple(*source_iterators))
    zip_iterator_type = BT.DependentType(S.Name('typename thrust::zip_iterator'),
                                         [iterator_tuple_type])
    zip_iterator_type_name = C.type_from_name(name.id + '_zip_iterator')
    zip_iterator_typedef = B.CTypedef(zip_iterator_type,
                                      zip_iterator_type_name)
    iterator_sequence_type = BT.DependentType(T.Monotype('iterator_sequence'),
                                              [S.Name(zip_iterator_type_name)])
    iterator_sequence_typedef = B.CTypedef(iterator_sequence_type,
                                           C.type_from_name(name))
    source_beginnings = [B.CApply(B.CMember(x, S.Name('begin')),
                                  []) for x in sources]
    source_tuple = B.CApply(S.Name('thrust::make_tuple'), source_beginnings)
    size = B.CApply(B.CMember(sources[0], S.Name('size')), [])
    
    
    constructor = B.CTypeDecl(C.type_from_name(name),
                              B.CConstructor(name,
                                             [source_tuple, size]))
    result = [zip_iterator_typedef, iterator_sequence_typedef,
              constructor]
    return result

def _zip4(bind):
    return _zip(bind)

def _scan(bind):
    arg = bind.binder()
    appl = bind.value()
    source = appl.arguments()[1]
    argname = str(arg)
    typ = arg.type
    assert(isinstance(typ, T.Seq))
    atomic_type = B.CType(arg.type.unbox())
    tv_name = S.Name(C.markGenerated(argname + '_tv'))
    tv_type = S.Name('thrust::device_vector<' + str(atomic_type) + ' >')
    extent = B.CApply(B.CMember(source, S.Name('size')), [])
    allocation = B.CTypeDecl(tv_type, B.CConstructor(tv_name,
                                                     [extent]))
    
    raw_pointer = B.CApply(S.Name('thrust::raw_pointer_cast'),
                                       [S.Subscript(
                                           B.Reference(tv_name),
                                           S.Number(0))
                                           ])
    sequence = B.CTypeDecl(typ, B.CConstructor(arg, [raw_pointer,
                                                      extent]))
    call = B.CApply(appl.function(), [B.CApply(appl.arguments()[0], []),
                    source, arg])
    if bind.allocate:
        return [allocation, sequence, call]
    else:
        return [call]
def _rscan(bind):
    return _scan(bind)

def _permute(bind):
    arg = bind.binder()
    appl = bind.value()
    source = appl.arguments()[0]
    idxes = appl.arguments()[1]
    argname = str(arg)
    typ = arg.type
    assert(isinstance(typ, T.Seq))
    atomic_type = B.CType(arg.type.unbox())
    tv_name = S.Name(C.markGenerated(argname + '_tv'))
    tv_type = S.Name('thrust::device_vector<' + str(atomic_type) + ' >')
    extent = B.CApply(B.CMember(source, S.Name('size')), [])
    allocation = B.CTypeDecl(tv_type, B.CConstructor(tv_name,
                                                     [extent]))
    
    raw_pointer = B.CApply(S.Name('thrust::raw_pointer_cast'),
                                       [S.Subscript(
                                           B.Reference(tv_name),
                                           S.Number(0))
                                           ])
    sequence = B.CTypeDecl(typ, B.CConstructor(arg, [raw_pointer,
                                                      extent]))
    call = B.CApply(appl.function(), [source, idxes, arg])
    if bind.allocate:
        return [allocation, sequence, call]
    else:
        return [call]
    return bind
