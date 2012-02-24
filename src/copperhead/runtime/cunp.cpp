#include "cunp.hpp"
#include "np_inspect.hpp"
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>
#include <stdexcept>
#include "monotype.hpp"
#include "type_printer.hpp"


using std::shared_ptr;
using namespace backend;
using std::make_tuple;
using std::make_shared;

extern "C" {
void initialize_cunp() {
    //This initializes Numpy
    import_array();
}
}

shared_ptr<type_t> np_to_cu(NPY_TYPES n) {
    switch(n) {
    case NPY_BOOL:
        return bool_mt;       
    case NPY_INT:
        return int32_mt;
    case NPY_LONG:
        return int64_mt;
    case NPY_FLOAT:
        return float32_mt;
    case NPY_DOUBLE:
        return float64_mt;
    default:
        return void_mt;
    }
}

// ssize_t sizeof_cu(CUTYPE n) {
//     switch(n) {
//     case CUBOOL:
//         return sizeof(bool);
//     case CUINT32:
//         return sizeof(int);
//     case CUINT64:
//         return sizeof(long);
//     case CUFLOAT32:
//         return sizeof(float);
//     case CUFLOAT64:
//         return sizeof(double);
//     default:
//         return 0;
//     }
// }

// NPY_TYPES cu_to_np(CUTYPE n) {
//     switch(n) {
//     case CUBOOL:
//         return NPY_BOOL;       
//     case CUINT32:
//         return NPY_INT;
//     case CUINT64:
//         return NPY_LONG;
//     case CUFLOAT32:
//         return NPY_FLOAT;
//     case CUFLOAT64:
//         return NPY_DOUBLE;
//     default:
//         return NPY_NOTYPE;
//     }
// }

np_array_info inspect_array(PyObject* in) {
    //PyObject* input_array = PyArray_FROM_OTF(in,
    //                                         NPY_NOTYPE,
    //                                         NPY_IN_ARRAY);
    if (!(PyArray_Check(in))) {
        throw std::invalid_argument("Input was not a numpy array");
    }
   
    NPY_TYPES dtype = NPY_TYPES(PyArray_TYPE(in));
    
    PyArrayObject *vecin = (PyArrayObject*)PyArray_ContiguousFromObject(in, dtype, 1, 1);
    if (vecin == NULL) {
        throw std::invalid_argument("Can't create CuArray from this object");
    }
    ssize_t n = vecin->dimensions[0];
    void* d = vecin->data;
    shared_ptr<type_t> t = np_to_cu(dtype);
    return make_tuple(d, n, make_shared<sequence_t>(t));
}

// PyObject* make_array(array_info i) {
//     npy_intp dims[1];
//     dims[0] = i.n;
//     PyObject* return_array = PyArray_SimpleNew(1, dims, cu_to_np(i.t));
//     memcpy(PyArray_DATA(return_array), i.d, i.n * sizeof_cu(i.t));
//     return return_array;
// }

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
