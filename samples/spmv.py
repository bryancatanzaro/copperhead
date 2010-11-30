#
#  Copyright 2008-2010 NVIDIA Corporation
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

import numpy
import plac
from collections import namedtuple
from blas import norm_diff
from scipy.io import mmread
from numpy import array, array_split, ones
import urllib
from copperhead import *
import copperhead.runtime.intermediate as I

dia_matrix = namedtuple('DIA', 'data offsets')
ell_matrix = namedtuple('ELL', 'data indices')
csr_matrix = namedtuple('CSR', 'data indices')

@cu
def spvv_csr(x, cols, y):
    """
    Multiply a sparse row vector x -- whose non-zero values are in the
    specified columns -- with a dense column vector y.
    """
    z = gather(y, cols)
    return sum(map(lambda a, b: a * b, x, z))
 
@cu
def spmv_csr(Ax, Aj, x):
    """
    Compute y = Ax for CSR matrix A and dense vector x.
 
    Ax and Aj are nested sequences where Ax[i] are the non-zero entries
    for row i and Aj[i] are the corresponding column indices.
    """
    return map(lambda y, cols: spvv_csr(y, cols, x), Ax, Aj)

@cu
def spmv_ell(data, idx, x):
    def kernel(i):
        return sum(map(lambda Aj, J: Aj[i] * x[J[i]], data, idx))
    return map(kernel, indices(x))

def csr_to_ell(S):
    # max nnz/row
    K = numpy.diff(S.indptr).max()

    # allocate space for ELL
    ELL_indices = numpy.zeros((S.shape[0],K), dtype=np.int32)
    ELL_values  = numpy.zeros((S.shape[0],K), dtype=S.dtype)

    # write non-zero columns into ELL structure
    for n,row in enumerate(numpy.array_split(S.indices, S.indptr[1:-1])):
        k = len(row)
        ELL_indices[n,:k] = row

    # write non-zero values into ELL structure
    for n,row in enumerate(numpy.array_split(S.data, S.indptr[1:-1])):
        k = len(row)
        ELL_values[n,:k] = row

    # Copperhead dosn't handle 2d arrays yet, so we convert to lists of
    # 1d arrays instead
    return ell_matrix(data=list(ELL_values.transpose()),
                      indices=list(ELL_indices.transpose()))

def print_error(x, y):
    print("-------- Error: %s" % norm_diff(x, y))

@plac.annotations(data_file="""Filename of Matrix Market file holding sparse matrix.
Defaults to the matrix at http://www.cs.berkeley.edu/~catanzar/ex.mtx .""",
                  ell=("Try converting this matrix to ELLPACK Format", 'flag',
                       'e'),
                  single=("Force single precision", 'flag', 's'))
def main(data_file=None, ell=False, single=False):
    if not data_file:
        (data_file, headers) = urllib.urlretrieve("http://www.cs.berkeley.edu/~catanzar/ex.mtx")
        ell = True
        single = True
    dtype = np.float32 if single else np.float64
    print "---- Reading MTX data from file", data_file
    A = mmread(data_file).astype(dtype).tocsr()
    
    nrows, ncols = A.shape
    print("-------- Matrix found of dimension %s" % str(A.shape))
    
    print "---- Converting matrix data"
    csrA = csr_matrix(data=array_split(A.data, A.indptr[1:-1]),
                      indices=[np.array(x, dtype=np.int32) for x in \
                               array_split(A.indices, A.indptr[1:-1])])
        
    if ell:
        ellA = csr_to_ell(A)
    x  = ones(ncols, dtype=dtype)

    print "---- CSR SpMV in NumPy"
    y_ref = A*x

    print "---- CSR SpMV in Python interpreter"
    with places.here:
        y = spmv_csr(csrA.data, csrA.indices, x)
        print_error(y_ref, y)

        
    print "---- CSR SpMV on GPU"
    with places.gpu0:
        y = spmv_csr(csrA.data, csrA.indices, x)
        print_error(y_ref, y)
        
    if ell:
        print "---- ELL SpMV in Python interpreter"
        with places.here:
            y = spmv_ell(ellA.data, ellA.indices, x)
            print_error(y_ref, y)

        print "---- ELL SpMV on GPU"
        with places.gpu0:
            y = spmv_ell(ellA.data, ellA.indices, x)
            print_error(y_ref, y)
            
if __name__ == '__main__':
    plac.call(main)
