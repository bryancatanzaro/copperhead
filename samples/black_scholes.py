from copperhead import *
import numpy as np

@cu
def cnd(d):
    A1 = 0.31938153
    A2 = -0.356563782
    A3 = 1.781477937
    A4 = -1.821255978
    A5 = 1.330274429
    RSQRT2PI = 0.39894228040143267793994605993438

    K = 1.0 / (1.0 + 0.2316419 * abs(d))

    cnd = RSQRT2PI * exp(- 0.5 * d * d) * \
        (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))))

    if d > 0:
        return 1.0 - cnd
    else:
        return cnd


@cu
def black_scholes(S, X, T, R, V):
    def black_scholes_el(si, xi, ti):
        sqrt_ti = sqrt(ti)
        d1 = (log(si/xi) + (R + .5 * V * V) * ti) / (V * sqrt_ti)
        d2 = d1 - V * sqrt_ti
        cnd_d1 = cnd(d1)
        cnd_d2 = cnd(d2)
        exp_Rti = exp(-R * ti)
        call_result = si * cnd_d1 - xi * exp_Rti * cnd_d2;
        put_result = xi * exp_Rti * (1.0 - cnd_d2) - si * (1.0 - cnd_d1)
        return call_result, put_result
    return map(black_scholes_el, S, X, T)

def rand_floats(n, min, max):
    diff = np.float32(max) - np.float32(min)
    rands = np.array(np.random.random(n), dtype=np.float32)
    rands = rands * diff
    rands = rands + np.float32(min)
    return cuarray(rands)


n = 100


S = rand_floats(n, 5, 30)
X = rand_floats(n, 1, 100)
T = rand_floats(n, .25, 10)
R = np.float32(.02)
V = np.float32(.3)

r = black_scholes(S, X, T, R, V)

