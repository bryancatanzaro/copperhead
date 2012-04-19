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
def mandelbrot(x, y, m, t):
    def mandelbrot_iteration(z0, z, i):
    if i > t:
        return t
    else:
        z = z_add(z_square(z), z0)
        if z_magnitude(z) < m:
            return mandelbrot_iteration(z0, z, i+1, m, t)
        else:
            return i

    def mandelbrot_el(xi, yi):
        return mandelbrot_iteration((xi, yi), (xi, yi), 0)

    return map(mandelbrot_el, x, y)

        
