import numpy as np
import copperhead.compiler.backendtypes as ET
import copperhead.compiler.coretypes as T
from copperhead.compiler.conversions import back_to_front_type


def to_numpy(ary):
    front_type = back_to_front_type(ary.type)
    if not isinstance(front_type, T.Seq):
        raise ValueError("Not convertible to numpy")
    sub = front_type.unbox()
    if str(sub) == str(T.Int):
        return np.fromiter(ary, dtype=np.int32, count=-1)
    elif str(sub) == str(T.Long):
        return np.fromiter(ary, dtype=np.int64, count=-1)
    elif str(sub) == str(T.Float):
        return np.fromiter(ary, dtype=np.float32, count=-1)
    elif str(sub) == str(T.Double):
        return np.fromiter(ary, dtype=np.float64, count=-1)
    elif str(sub) == str(T.Bool):
        return np.fromiter(ary, dtype=np.bool, count=-1)
    else:
        raise ValueError("Not convertible to numpy")
    
