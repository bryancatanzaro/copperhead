import backendtypes as BT
import coretypes as T

# This dictionary enumerates the variant in backend/inc/type.hpp/type_base
# This mechanism allows us to convert between boost::variant dynamic types
# And Python's natural dynamic types

# XXX
# HAZARD HAZARD HAZARD
# This can lead to segfaults and other mysteries if the variant is altered
# and this table is not amended!

# XXX It would be nice to create this table automatically from the
# definition of type_base

which_to_back = {
   0 : BT.retrieve_monotype_t,
   1 : BT.retrieve_polytype_t,
   2 : BT.retrieve_sequence_t,
   3 : BT.retrieve_tuple_t,
   4 : BT.retrieve_fn_t }

def unvariate(x):
    """Converts between boost::variant dynamic typing used by backend
    and Python's natural dynamic typing"""
    return which_to_back[BT.which(x)](x)


def back_to_front(x):
    concrete = unvariate(x)
    if isinstance(concrete, BT.Sequence):
        sub = back_to_front(concrete.sub())
        return T.Seq(sub)
    # XXX Python iteration over BT.Tuple causes segfault. Investigate.
    #elif isinstance(concrete, BT.Tuple):
    #    return BT.Tuple(*[back_to_front(y) for y in concrete])
    elif isinstance(concrete, BT.Monotype):
        name = str(x)
        if name == 'Int32':
            return T.Int
        elif name == 'Int64':
            return T.Long
        elif name == 'Bool':
            return T.Bool
        elif name == 'Float32':
            return T.Float
        elif name == 'Double':
            return T.Double
        else:
            raise ValueError("Unknown monotype %s" % name)
    else:
        raise ValueError("Unknown type")

