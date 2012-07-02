from copperhead import *

@cu
def infinite_loop(val):
    return infinite_loop(val+1)

@cu
def nested_non_tail(val, limit_a, limit_b):
    if (val < limit_a):
        if (val < limit_b):
            return nested_non_tail(val+1, limit_a, limit_b)
        else:
            return val
    else:
        if (val > limit_b):
            return val
        else:
            return nested_non_tail(val+1, limit_a, limit_b)

@cu
def non_tail_recursive(val, limit):
    if (val < limit):
        return non_tail_recursive(val, limit) + 1
    else:
        return val

@cu
def anticount(val, limit):
    inc_val = val+1
    if (val < limit):
        return anticount(inc_val, limit)
    else:
        return val


@cu
def count(val, limit):
    if (val == limit):
        return val
    else:
        return count(val+1, limit)

    #print(nested_non_tail(0, 1, 2))
    #print(infinite_loop(0))
    #print(non_tail_recursive(0))


@cu
def vinc(x, val, limit):
    if (val == limit):
        return x
    else:
        return vinc(map(lambda xi: xi+1, x), val+1, limit)

print(anticount(0, 10))
print(count(0, 10))
#print(vinc([0,0,0], 0, 5))

@cu
def divergent(limit):
    def divergent_sub(val):
        if (val == limit):
            return val
        else:
            return divergent_sub(val+1)
    return map(divergent_sub, range(limit))

print(divergent(10))
    

#XXX Things to do:
# Fix pruning for loop-carried dependences
# Fix phase inference for loop-carried dependences
# Fix calling closures recursively
