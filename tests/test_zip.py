from copperhead import *

@cu
def test_zip(x, y):
    return zip(x, y)

@cu
def test_unzip(x):
    y, z = unzip(x)
    return y

x = [1, 2, 3, 4, 5]
y = [3, 4, 5, 6, 7]


# @cu
# def demux(x):
#     return [(xi, xi) for xi in x]

# @cu
# def shift_zip(x, y, z, d):
#     a = zip(x, y)
#     b = shift(a, d, z)
#     return b


z = test_zip(x, y)

# q = shift_zip(x, y, (-1, -2), 1)
# r = shift_zip(x, y, (-3, -4), -1)

# print(repr(z))
# print(repr(q))

# print(repr(r))

print(repr(test_unzip(z)))

#q = demux([1,2,3])
#a, b = test_unzip(q)
