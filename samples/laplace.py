from copperhead import *
from numpy import zeros

@cu
def initialize(N):
    nx, ny = N
    def el(i):
        y = i / nx
        if y==0:
            return 1.0
        else:
            return 0.0
    return map(el, range(nx * nx))

@cu
def solve(u, N, D2, it):
    nx, ny = N
    dx2, dy2 = D2
    
    def el(i):
        x = i % nx
        y = i / nx
        if x == 0 or x == nx-1 or y == 0 or y == ny-1:
            return u[i]
        else:
            return ((u[i-1]+u[i+1])*dy2 + \
                        (u[i-nx]+u[i+nx])*dx2)/(2*(dx2+dy2))
        
            
    if it > 0:
        u = map(el, indices(u))
        return solve(u, N, D2, it-1)
    else:
        return u
    
dx = 0.1
dy = 0.1
dx2 = dx*dx
dy2 = dy*dy
N = (100,100)
D2 = (dx2, dy2)

p = runtime.places.default_place

with p:
    u = initialize(N)
    print("starting timer")
    import time
    start = time.time()
    #Solve
    u = solve(u, N, D2, 8000)
    #Force result to be finalized at execution place
    #Otherwise, timing loop may not account for all computation
    u = force(u, p)
    end = time.time()
    print("Computation time: %s seconds" %(end - start))

result = np.reshape(to_numpy(u), N)

try:
    import matplotlib.pyplot as plt
    plt.imshow(result)
    plt.show()
except:
    pass
