#
#  Copyright 2008-2010 NVIDIA Corporation
#  Copyright 2009-2010 University of California
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

from copperhead import *

import numpy as np
import matplotlib as mat
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import plac
import urllib

@cu
def axpy(a, x, y):
    return [a * xi + yi for xi, yi in zip(x, y)]

@cu
def dot(x, y):
    return sum(map(op_mul, x, y))

@cu
def vadd(x, y):
    return map(op_add, x, y)

@cu
def vmul(x, y):
    return map(op_mul, x, y)

@cu
def vsub(x, y):
    return map(op_sub, x, y)


@cu
def of_spmv(du, dv, width, m1, m2, m3, m4, m5, m6, m7):
    e = vadd(vmul(m1, du), vmul(m2, dv))
    f = vadd(vmul(m2, du), vmul(m3, dv))
    e = vadd(e, vmul(m4, shift(du, -width, 0.0)))
    f = vadd(f, vmul(m4, shift(dv, -width, 0.0)))
    e = vadd(e, vmul(m5, shift(du, -1, 0.0)))
    f = vadd(f, vmul(m5, shift(dv, -1, 0.0)))
    e = vadd(e, vmul(m6, shift(du, 1, 0.0)))
    f = vadd(f, vmul(m6, shift(dv, 1, 0.0)))
    e = vadd(e, vmul(m7, shift(du, width, 0.0)))
    f = vadd(f, vmul(m7, shift(dv, width, 0.0)))
    return (e, f)

@cu
def zeros(x):
    return [0.0 for xi in x]

eps = 1e-6

@cu
def init_cg(ux, vx, du, dv, width, m1, m2, m3, m4, m5, m6, m7):
    u, v = of_spmv(ux, vx, width, m1, m2, m3, m4, m5, m6, m7)
    ur = vsub(du, u)
    vr = vsub(dv, v)
    return ur, vr

@cu
def precondition(u, v, p1, p2, p3):
    e = vadd(vmul(p1, u), vmul(p2, v))
    f = vadd(vmul(p2, u), vmul(p3, v))
    return e, f

@cu
def pre_cg_iteration(ux, vx, ur, vr, ud, vd, uz, vz, width, m1, m2, m3, m4, m5, m6, m7, p1, p2, p3):
    uAdi, vAdi = of_spmv(ud, vd, width, m1, m2, m3, m4, m5, m6, m7)
    urnorm = dot(ur, uz)
    vrnorm = dot(vr, vz)
    rnorm = urnorm + vrnorm
    udtAdi = dot(ud, uAdi)
    vdtAdi = dot(vd, vAdi)
    dtAdi = udtAdi + vdtAdi
    alpha = rnorm / dtAdi
    ux = axpy(alpha, ud, ux)
    vx = axpy(alpha, vd, vx)
    urp1 = axpy(-alpha, uAdi, ur)
    vrp1 = axpy(-alpha, vAdi, vr)
    uzp1, vzp1 = precondition(urp1, vrp1, p1, p2, p3)
    urp1norm = dot(urp1, uzp1)
    vrp1norm = dot(vrp1, vzp1)
    beta = (urp1norm + vrp1norm)/rnorm
    udp1 = axpy(beta, uzp1, urp1)
    vdp1 = axpy(beta, vzp1, vrp1)
    return ux, vx, urp1, vrp1, uzp1, vzp1, udp1, vdp1, rnorm



@cu
def form_preconditioner(m1, m2, m3):
    def indet(a, b, c):
        return 1.0/(a * c - b * b)
    indets = map(indet, m1, m2, m3)
    p1 = map(op_mul, indets, m3)
    p2 = map(lambda a, b: -a * b, indets, m2)
    p3 = map(op_mul, indets, m1)
    return p1, p2, p3


def cg(it, A, width, ux, vx, du, dv):
    m1 = CuArray(A[0])
    m2 = CuArray(A[1])
    m3 = CuArray(A[2])
    m4 = CuArray(A[3])
    m5 = CuArray(A[4])
    m6 = CuArray(A[5])
    m7 = CuArray(A[6])
  

    ux = CuArray(ux)
    vx = CuArray(vx)
    du = CuArray(du)
    dv = CuArray(dv)


    
    p1, p2, p3 = form_preconditioner(m1, m2, m3)
    ur, vr= init_cg(ux, vx, du, dv, width, m1, m2, m3, m4, m5, m6, m7)
    uz, vz = precondition(ur, vr, p1, p2, p3)
    ud = ur
    vd = vr
    ur0, vr0 = ur, vr
    pre_res = []
    ur, vr = ur0, vr0
    ud, vd = ur0, vr0
    for i in range(it):
        ux, vx, ur, vr, uz, vz, ud, vd, rnorm = \
            pre_cg_iteration(ux, vx, ur, vr, ud, vd, uz, vz, width, \
                             m1, m2, m3, m4, m5, m6, m7, p1, p2, p3)
        pre_res.append(rnorm.value)
        print("PCG Iteration: %i" %i)
    
    
        
    return ux, vx, pre_res

def initialize_data(file_name):
    print("Reading data from file")
    if not file_name:
        file_name, headers = urllib.urlretrieve('http://www.cs.berkeley.edu/~catanzar/Urban331.npz')
    npz = np.load(file_name)
    width = npz['width'].item()
    height = npz['height'].item()
    npixels = width * height
    m1 = npz['m1']
    m2 = npz['m2']
    m3 = npz['m3']
    m4 = npz['m4']
    m5 = npz['m5']
    m6 = npz['m6']
    m7 = npz['m7']
    du = npz['du']
    dv = npz['dv']
    A = [m1, m2, m3, m4, m5, m6, m7]
    ux = np.zeros(npixels, dtype=np.float32)
    vx = np.zeros(npixels, dtype=np.float32)
    img = npz['img']
    return(A, ux, vx, du, dv, width, height, img)

def plot_data(image, width, height, res, ux, vx):
    plt.subplot(121)
    plt.imshow(image[10:110, 10:110])
    plt.subplot(122)

    u = ux.numpy()
    v = vx.numpy()
    u = np.reshape(u, [height,width])
    v = np.reshape(v, [height,width])
    x, y = np.meshgrid(np.arange(0, 100), np.arange(99, -1, -1))

    plt.quiver(x, y, u[10:110,10:110], v[10:110, 10:110], angles='xy')
    plt.show()

@plac.annotations(data_file="""Filename of Numpy data file for this problem.
If none is found, a default dataset will be loaded from
http://www.cs.berkeley.edu/~catanzar/Urban331.npz""")
def main(data_file=None):
    """Performs a Preconditioned Conjugate Gradient solver for a particular
    problem found in Variational Optical Flow methods for video analysis."""
    A, ux, vx, du, dv, width, height, image = initialize_data(data_file)

    with places.gpu0:
        ux, vx, res = cg(100, A, width, ux, vx, du, dv)

    plot_data(image, width, height, res, ux, vx)



if __name__ == '__main__':
    plac.call(main)

