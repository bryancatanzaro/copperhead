from copperhead import *


@cu
def enumerate(x):
    return zip(indices(x), x)

@cu
def argmax(x):
    def argmax_el((ia, xa), (ib, xb)):
        if xa > xb:
            return ia, xa
        else:
            return ib, xb
    return reduce(argmax_el, enumerate(x), (-1, min_bound_el(x)))

@cu
def choose(x):
    if x % 2 == 0:
        return x / 2
    else:
        return 3 * x + 1


@cu
def evaluate(x, i):
    if x == 1:
        return i
    else:
        return evaluate(choose(x), i + 1)

@cu
def start(x):
    return evaluate(x, 1)

@cu
def p14(n):
    lengths = map(start, bounded_range(1, n))
    index, value = argmax(lengths)
    return index+1, value


print p14(1000000, verbose=True)
