from copperhead import *

# @cu
# def test_bounded_range(a, b):
#     return bounded_range(a, b)

# print test_bounded_range(10, 15)


@cu
def test(x):
    y = bounded_range(10, x)
    return [yi + 1 for yi in y]

import copperhead.runtime.intermediate as I
with I.tracing(action=I.print_and_pause):
    print test(15)

# @cu
# def inline_closure_literal_test(x):
#     def scale(b, y):
#         return [yi + b for yi in y]
#     return scale(10, x)

# print inline_closure_literal_test([0, 1])
