import backendtypes as ET
import coretypes as T

import backendsyntax as ES
import coresyntax as S

# This dictionary enumerates the variant in backend/inc/type.hpp/type_base
# This mechanism allows us to convert between boost::variant dynamic types
# And Python's natural dynamic types

# XXX
# HAZARD HAZARD HAZARD
# This can lead to segfaults and other mysteries if the variant is altered
# and this table is not amended!

# XXX It would be nice to create this table automatically from the
# definition of type_base

# XXX Alternatively, it'd be great to make a metaclass that did this
# for us

which_to_back = {
   0 : ET.retrieve_monotype_t,
   1 : ET.retrieve_polytype_t,
   2 : ET.retrieve_sequence_t,
   3 : ET.retrieve_tuple_t,
   4 : ET.retrieve_fn_t }

def unvariate(x):
    """Converts between boost::variant dynamic typing used by backend
    and Python's natural dynamic typing"""
    return which_to_back[ET.which(x)](x)


def back_to_front_type(x):
    concrete = unvariate(x)
    if isinstance(concrete, ET.Sequence):
        sub = back_to_front_type(concrete.sub())
        return T.Seq(sub)
    # XXX Python iteration over ET.Tuple causes segfault. Investigate.
    #elif isinstance(concrete, ET.Tuple):
    #    return ET.Tuple(*[back_to_front_type(y) for y in concrete])
    elif isinstance(concrete, ET.Monotype):
        name = str(x)
        if name == 'Int32':
            return T.Int
        elif name == 'Int64':
            return T.Long
        elif name == 'Bool':
            return T.Bool
        elif name == 'Float32':
            return T.Float
        elif name == 'Float64':
            return T.Double
        else:
            raise ValueError("Unknown monotype %s" % name)
    else:
        raise ValueError("Unknown type")


def front_to_back_type(x):
    if isinstance(x, T.Polytype):
        variables = [ET.Monotype(str(y)) for y in x.variables]
        sub = front_to_back_type(x.monotype())
        return ET.Polytype(variables, sub)
    elif isinstance(x, T.Tuple):
        subs = [front_to_back_type(y) for y in x.parameters]
        return ET.Tuple(*subs)
    elif isinstance(x, T.Fn):
        args = front_to_back_type(x.parameters[0])
        result = front_to_back_type(x.parameters[1])
        return ET.Fn(args, result)
    elif isinstance(x, T.Seq):
        sub = front_to_back_type(x.unbox())
        return ET.Sequence(sub)
    elif isinstance(x, T.Monotype):
        if str(x) == str(T.Int):
            return ET.Int32
        elif str(x) == str(T.Long):
            return ET.Int64
        elif str(x) == str(T.Float):
            return ET.Float32
        elif str(x) == str(T.Double):
            return ET.Float64
        elif str(x) == str(T.Bool):
            return ET.Bool
        elif str(x) == str(T.Void):
            return ET.Void
    elif isinstance(x, str):
        return ET.Monotype(str(x))
    raise ValueError("Can't convert %s to backendtypes" % str(x))

def front_to_back_node(x):
    if isinstance(x, list):
        subs = [front_to_back_node(y) for y in x]
        return ES.Suite(subs)
    elif isinstance(x, S.Name):
        name = ES.Name(x.id)
        name.type = front_to_back_type(x.type)
        return name
    elif isinstance(x, S.Number):
        literal = ES.Literal(str(x))
        literal.type = front_to_back_type(x.type)
        return literal
    elif isinstance(x, S.Tuple):
        subs = [front_to_back_node(y) for y in x]
        tup = ES.Tuple(subs)
        tup.type = front_to_back_type(x.type)
        return tup
    elif isinstance(x, S.Apply):
        fn = front_to_back_node(x.function())
        args = [front_to_back_node(y) for y in x.arguments()]
        appl = ES.Apply(fn, ES.Tuple(args))
        return appl
    elif isinstance(x, S.Bind):
        lhs = front_to_back_node(x.binder())
        rhs = front_to_back_node(x.value())
        return ES.Bind(lhs, rhs)
    elif isinstance(x, S.Return):
        val = front_to_back_node(x.value())
        return ES.Return(val)
    elif isinstance(x, S.Cond):
        test = front_to_back_node(x.test())
        body = front_to_back_node(x.body())
        orelse = front_to_back_node(x.orelse())
        return ES.Cond(test, body, orelse)
    elif isinstance(x, S.Lambda):
        args = [front_to_back_node(y) for y in x.formals()]
        body = front_to_back_node(x.body())
        lamb = ES.Lambda(ES.Tuple(args), body)
        lamb.type = front_to_back_type(x.type)
        return lamb
    elif isinstance(x, S.Closure):
        closed_over = [front_to_back_node(y) for y in x.closed_over()]
        body = front_to_back_node(x.body())
        closure = ES.Closure(ES.Tuple(closed_over), body)
        closure.type = front_to_back_type(x.type)
        return closure
    elif isinstance(x, S.Procedure):
        name = front_to_back_node(x.name())
        formals = [front_to_back_node(y) for y in x.formals()]
        body = front_to_back_node(x.body())
        proc = ES.Procedure(name, ES.Tuple(formals), body)
        proc.type = name.type
        return proc
