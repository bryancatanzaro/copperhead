#
#  Copyright 2008-2010 NVIDIA Corporation
#  Copyright 2008-2010 University of California
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
from copperhead import *

import numpy as np
import re
import os
import itertools
import urllib2

# plac is the command-line parser 
import plac

def find_padding(size, alignment):
    """Return the number of padding elements needed to make a sequence of
    length size have a padded length which is a multiple of alignment"""
    return alignment - size % alignment

def read_svm(f):
    """Read an SVM training set from a file using the LIBSVM format"""
    y = []
    data = []
    label = re.compile('^([1\.\0\-\+]+)')
    unpack = re.compile('[\d]+\:([e\+\-\d\.]+)')
    alignment = 32 #How many floats do we want to align things at?
    n_dim = None
    for line in f:
        y.append(np.float32(label.search(line).group(1)))
        point = [np.float32(x) for x in unpack.findall(line)]
        if n_dim is None:
            n_dim = len(point)
        data.append(point)
    n_points = len(data)
    # Form row-major data
    padding = find_padding(n_dim, alignment)
    r_data = []
    for point in data:
        r_data.append(point + [np.float32(0)] * padding)
    r_data = np.array(r_data)
    r_data = r_data.reshape(((n_dim + padding) * n_points,))
    r_data = CuUniform(r_data, (n_points, n_dim), strides=(n_dim + padding, 1))

    
    # Add empty data points to make the data padded to alignment
    padding = find_padding(n_points, alignment)
    data.extend([[np.float32(0)] * n_dim] * padding)
    # Transpose from row-major to column-major format
    data = np.array(data)
    data = data.transpose()
    data = data.reshape(((n_points + padding) * n_dim,))
    # Form Uniform Nested Sequence
    c_data = CuUniform(data, (n_points, n_dim), strides=(1, n_points + padding))

    return r_data, c_data, CuArray(y)

def write_mdl(filename, data, labels, alpha, gamma, eps, b):
    """Write a SVM Model to a file, using LIBSVM format"""
    f = open(filename, 'w')
    psv = []
    nsv = []
    for point, label, weight in zip(data, labels, alpha):
        if weight >= eps:
            if label > 0.0:
                psv.append((weight, point))
            else:
                nsv.append((-weight, point))
    print >> f, 'svm_type c_svc'
    print >> f, 'kernel_type rbf'
    print >> f, 'gamma %s' % gamma
    print >> f, 'nr_class 2'
    print >> f, 'total_sv %s' % (len(psv) + len(nsv))
    print >> f, 'rho %s' % -b
    print >> f, 'label 1 -1'
    print >> f, 'nr_sv %s %s' % (len(psv), len(nsv))
    print >> f, 'SV'
    
    for weight, point in psv + nsv:
        print >> f, '%s ' % weight,
        for idx, dim in zip(itertools.count(1), point):
            print >> f, '%s:%s ' %(idx, dim),
        print >> f
    f.close()

@cu
def norm2_diff(x, y):
    def el(xi, yi):
        diff = xi - yi
        return diff * diff
    return sum(map(el, x, y))

@cu
def rbf(gamma, x, y):
    return exp(-gamma * norm2_diff(x,y))


    
@cu
def argextrema((a_l_idx, a_l_val, a_h_idx, a_h_val),
               (b_l_idx, b_l_val, b_h_idx, b_h_val)):
    if a_l_val < b_l_val:
        if a_h_val > b_h_val:
            return (a_l_idx, a_l_val, a_h_idx, a_h_val)
        else:
            return (a_l_idx, a_l_val, b_h_idx, b_h_val)
    else:
        if a_h_val > b_h_val:
            return (b_l_idx, b_l_val, a_h_idx, a_h_val)
        else:
            return (b_l_idx, b_l_val, b_h_idx, b_h_val)


@cu
def train_iteration(data, labels, gamma, high, low, \
              alpha, f, d_a_high, d_a_low, idxes, eps, ceps, inf, extid):
    def kernel_evaluation(x):
        high_kernel = rbf(gamma, x, high)
        low_kernel = rbf(gamma, x, low)
        return (high_kernel, low_kernel)
    kernels = map(kernel_evaluation, data)
    def f_evaluation(fi, (hi, li)):
        return fi + d_a_high * hi + d_a_low * li
    f_p = map(f_evaluation, f, kernels)
    
    
    def high_membership(ai, yi, fi):
        if (ai >= eps and ai <= ceps) or \
            (yi > 0.0 and ai < eps) or \
            (yi < 0.0 and ai > ceps):
            return fi
        else:
            return inf
        
    def low_membership(ai, yi, fi):
        if (ai >= eps and ai <= ceps) or \
            (yi > 0.0 and ai > ceps) or \
            (yi < 0.0 and ai < eps):
            return fi
        else:
            return -inf
    
    high_values = map(high_membership, alpha, labels, f_p)
    low_values = map(low_membership, alpha, labels, f_p)

    extremes = reduce(argextrema, zip4(idxes, high_values, idxes, low_values), extid)
    
    return f_p, extremes

@cu
def vneg(x):
    return [-xi for xi in x]

@plac.annotations(
    data_file="""Input Data Filename.  Data should be in LIBSVM format.
    Defaults to a sample file at http://www.cs.berkeley.edu/~catanzar/abaloneData.svm""",
    model_file="""Output Model Filename.
    Defaults to the input filename with the extension replaced by .mdl""",
    gamma='Parameter for RBF Kernel Function.  Defaults to 0.125',
    C='SVM Cost Parameter.  Defaults to 10.0',
    eps='Support Vector Tolerance.  Defaults to 1e-3',
    tau='Convergence Tolerance.  Defaults to 1e-3')
def main(data_file=None, model_file=None, gamma=0.125, C=10.0, eps=1e-3, tau=1e-3):
    """Support Vector Machine training using the SMO algorithm, RBF kernel,
    and first order working set selection heuristic."""
    eps = np.float32(eps)
    C = np.float32(C)
    ceps = C - eps
    tau = np.float32(tau)
    gamma = np.float32(gamma)
    inf = np.float32(float('inf'))
    extid = (0, inf, 0, -inf)

    if data_file:
        f = open(data_file)
        if not model_file:
            (base, svm_ext) = os.path.splitext(data_file)
            model_file = base + '.mdl'
            print("Loading data from %s" % data_file)
    else:
        print("Loading sample dataset")
        dataset_name = 'abaloneData'
        data_file = dataset_name + '.svm'
        model_file = dataset_name + '.mdl'
        default_data_url = 'http://www.cs.berkeley.edu/~catanzar/' + data_file
        f = urllib2.urlopen(default_data_url)

    
    r_data, c_data, labels = read_svm(f)
    n_points = r_data.shape.extents[0]
    n_dim = r_data.shape.element.extents[0]
    print("%s points, %s dimensions" % (n_points, n_dim))
    
    print("RBF Kernel, gamma = %s, tau = %s, epsilon = %s" % (gamma, tau, eps))

    i_low = 0
    while labels[i_low] > 0:
        i_low = i_low + 1
    i_high = 0
    while labels[i_high] < 0:
        i_high = i_high + 1


    alpha = CuArray(np.zeros(n_points, dtype=np.float32))
   
    high = r_data[i_high]
    low = r_data[i_low]
    
    kernel = rbf(gamma, high, low)
    eta = np.float32(2.0) - np.float32(2.0) * kernel
    alpha_n = np.float32(2.0)/eta
    if alpha_n > C:
        alpha_n = C
    d_a_high = -alpha_n
    d_a_low = alpha_n
    alpha.update([i_low, i_high], [alpha_n, alpha_n])
    f = vneg(labels)
    idxes = CuArray(np.arange(n_points, dtype=np.int32))
    b_low_p = np.float32(1.0)
    b_high_p = np.float32(-1.0)
    i_low_p = i_low
    i_high_p = i_high

    gap = np.float32(2.0)
    tau = np.float32(1e-3)
    iteration = 0
    while b_low_p > (b_high_p + 2 * tau):
        f, (i_high_p, b_high_p, i_low_p, b_low_p) = \
           train_iteration(c_data, labels, gamma, high, low, \
                           alpha, f, d_a_high, d_a_low, idxes, \
                           eps, ceps, inf, extid)
        iteration = iteration + 1
        if iteration % 1024 == 0:
            print("Iteration: %s, gap: %s" % (iteration, gap))
        gap = b_low_p - b_high_p
        high = r_data[i_high_p]
        low = r_data[i_low_p]
        kernel = rbf(gamma, high, low)
        eta = np.float32(2.0) - np.float32(2.0) * kernel
        local_alphas = alpha.extract([i_low_p, i_high_p])
        alpha_low = local_alphas[0]
        alpha_high = local_alphas[1]
        
        alpha_diff = alpha_low - alpha_high
        low_label = labels[i_low_p]
        high_label = labels[i_high_p]
        sign = high_label * low_label
        if sign < 0:
            if alpha_diff < 0:
                alpha_low_lb = np.float32(0.0)
                alpha_low_ub = C + alpha_diff
            else:
                alpha_low_lb = alpha_diff
                alpha_low_ub = C
        else:
            alpha_sum = alpha_low + alpha_high
            if alpha_sum < C:
                alpha_low_lb = np.float32(0.0)
                alpha_low_ub = alpha_sum
            else:
                alpha_low_lb = alpha_sum - C
                alpha_low_ub = C
        alpha_low_p = alpha_low + low_label * (b_high_p - b_low_p) / eta
        if alpha_low_p < alpha_low_lb:
            alpha_low_p = alpha_low_lb
        elif alpha_low_p > alpha_low_ub:
            alpha_low_p = alpha_low_ub
        d_a_low = alpha_low_p - alpha_low
        alpha_high_p = alpha_high - sign * d_a_low
        if alpha_high_p > C:
            # This may occur due to FP precision issues
            alpha_high_p = C
        elif alpha_high_p < 0:
            # This may also occur due to FP precision issues
            alpha_high_p = np.float32(0.0)
        d_a_low = d_a_low * low_label
        d_a_high = (alpha_high_p - alpha_high) * high_label
        
        alpha.update([i_high_p, i_low_p], [alpha_high_p, alpha_low_p])
      
    print("Converged after %s iterations" % iteration)
    b = (b_high_p + b_low_p) / 2.0
    print("Saving model in %s" % model_file)
    write_mdl(model_file, r_data, labels, alpha, gamma, eps, b)



if __name__ == '__main__':
    plac.call(main)






