#include "cunp.h"
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>
#include <stdexcept>

void initialize_cunp() {
    //This initializes Numpy
    import_array();
}

BRIDGE_TYPE np_to_br(NPY_TYPES n) {
    switch(n) {
    case NPY_BOOL:
        return CUNP_BOOL;       
    case NPY_INT:
        return CUNP_INT;
    case NPY_LONG:
        return CUNP_LONG;
    case NPY_FLOAT:
        return CUNP_FLOAT;
    case NPY_DOUBLE:
        return CUNP_DOUBLE;
    default:
        return CUNP_NOTYPE;
    }
}

ssize_t sizeof_br(BRIDGE_TYPE n) {
    switch(n) {
    case CUNP_BOOL:
        return sizeof(bool);
    case CUNP_INT:
        return sizeof(int);
    case CUNP_LONG:
        return sizeof(long);
    case CUNP_FLOAT:
        return sizeof(float);
    case CUNP_DOUBLE:
        return sizeof(double);
    default:
        return 0;
    }
}

NPY_TYPES br_to_np(BRIDGE_TYPE n) {
    switch(n) {
    case CUNP_BOOL:
        return NPY_BOOL;       
    case CUNP_INT:
        return NPY_INT;
    case CUNP_LONG:
        return NPY_LONG;
    case CUNP_FLOAT:
        return NPY_FLOAT;
    case CUNP_DOUBLE:
        return NPY_DOUBLE;
    default:
        return NPY_NOTYPE;
    }
}

array_info::array_info(void* _d, ssize_t _n, BRIDGE_TYPE _t) :
    d(_d), n(_n), t(_t) {}

array_info inspect_array(PyObject* in) {
    //PyObject* input_array = PyArray_FROM_OTF(in,
    //                                         NPY_NOTYPE,
    //                                         NPY_IN_ARRAY);
    if (!(PyArray_Check(in))) {
        throw std::invalid_argument("Input was not a numpy array");
    }
   
    NPY_TYPES dtype = NPY_TYPES(PyArray_TYPE(in));
    
    PyArrayObject *vecin = (PyArrayObject*)PyArray_ContiguousFromObject(in, dtype, 1, 1);
    if (vecin == NULL) {
        //int nd = PyArray_NDIM(input_array);
        //if (nd != 1) {
        throw std::invalid_argument("Can't create CuArray from this object");
    }
    ssize_t n = vecin->dimensions[0];
    void* d = vecin->data;
    //npy_intp* dims = PyArray_DIMS(input_array);
    //ssize_t n = dims[0];
    //void* d = PyArray_DATA(input_array);
    return array_info(d, n, np_to_br(dtype));
}

PyObject* make_array(array_info i) {
    npy_intp dims[1];
    dims[0] = i.n;
    PyObject* return_array = PyArray_SimpleNew(1, dims, br_to_np(i.t));
    memcpy(PyArray_DATA(return_array), i.d, i.n * sizeof_br(i.t));
    return return_array;
}

//Instantiate scalar packings
PyObject* make_scalar(const float& s) {
    PyObject* result = PyArrayScalar_New(Float);
    PyArrayScalar_ASSIGN(result, Float, s);
    return result;
}

PyObject* make_scalar(const double& s) {
    PyObject* result = PyArrayScalar_New(Double);
    PyArrayScalar_ASSIGN(result, Double, s);
    return result;
}

PyObject* make_scalar(const int& s) {
    PyObject* result = PyArrayScalar_New(Int);
    PyArrayScalar_ASSIGN(result, Int, s);
    return result;
}

PyObject* make_scalar(const long& s) {
    PyObject* result = PyArrayScalar_New(Long);
    PyArrayScalar_ASSIGN(result, Long, s);
    return result;
}

PyObject* make_scalar(const bool& s) {
    if (s) {
        PyArrayScalar_RETURN_TRUE;
    } else {
        PyArrayScalar_RETURN_FALSE;
    }
}


float unpack_scalar_float(PyObject* s) {
    return PyArrayScalar_VAL(s, Float);
}

double unpack_scalar_double(PyObject* s) {
    return PyArrayScalar_VAL(s, Double);
}

int unpack_scalar_int(PyObject* s) {
    return PyArrayScalar_VAL(s, Int);
}

long unpack_scalar_long(PyObject* s) {
    return PyArrayScalar_VAL(s, Long);
}

bool unpack_scalar_bool(PyObject* s) {
    return PyArrayScalar_VAL(s, Bool);
}
