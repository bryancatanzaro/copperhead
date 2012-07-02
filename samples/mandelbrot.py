from copperhead import *

@cu
def z_square(z):
    real, imag = z
    return real * real - imag * imag, 2 * real * imag

@cu
def z_magnitude(z):
    real, imag = z
    return sqrt(real * real + imag * imag)

@cu
def z_add((z0r, z0i), (z1r, z1i)):
    return z0r + z1r, z0i + z1i

@cu
def mandelbrot_iteration(z0, z, i, m, t):
    z = z_add(z_square(z), z0)
    escaped = z_magnitude(z) > m
    converged = i > t
    done = escaped or converged
    if not done:
        return mandelbrot_iteration(z0, z, i+1, m, t)
    else:
        return i


@cu
def mandelbrot(lb, scale, (x, y), m, t):

    def mandelbrot_el(zi):
        return mandelbrot_iteration(zi, zi, 0, m, t)

    def index(i):
        scale_x, scale_y = scale
        lb_x, lb_y = lb
        return float32(i % x) * scale_x + lb_x, float32(i / x) * scale_y + lb_y

    
    two_d_points = map(index, range(x*y))

    return map(mandelbrot_el, two_d_points)
    


lb = (np.float32(-2.5), np.float32(-2.0))
ub = (np.float32(1.5), np.float32(2.0))
x, y = 1000, 1000

scale = ((ub[0]-lb[0])/np.float32(x), (ub[0]-lb[0])/np.float32(y))

max_iterations = 100
diverge_threshold = np.float32(4.0)

print("Calculating...")
result = mandelbrot(lb, scale, (x,y), diverge_threshold, max_iterations)
print("Plotting...")

import matplotlib.pyplot as plt
im_result = to_numpy(result).reshape([x, y])
plt.imshow(im_result)
plt.show()
