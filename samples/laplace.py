from copperhead import *
from numpy import zeros

dx = 0.1
dy = 0.1
dx2 = dx*dx
dy2 = dy*dy

@cu
def initialize(nx, ny):
    def el(i):
        y = i / nx
        if y==0:
            return 1.0
        else:
            return 0.0
    return map(el, range(nx * nx))

@cu
def update(u, nx, ny, dx2, dy2):
    def el(i):
        x = i % nx
        y = i / nx
        if x == 0 or x == nx-1 or y == 0 or y == ny-1:
            return u[i]
        else:
            return ((u[i-1]+u[i+1])*dy2 + \
                        (u[i-nx]+u[i+nx])*dx2)/(2*(dx2+dy2))
        
    
    return map(el, indices(u))

N = 100
u = initialize(N, N)

import time
start = time.time()
for x in xrange(8000):
    u = update(u, N, N, dx2, dy2)
end = time.time()
print(end - start)

result = np.reshape(to_numpy(u), [N, N])
import matplotlib.pyplot as plt
plt.imshow(result)
plt.show()
