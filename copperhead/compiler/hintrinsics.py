import backendsyntax as B
import coresyntax as S
import codesnippets as C
import coretypes as T
import backtypes as BT
import pdb

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
