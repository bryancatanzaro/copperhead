#include <boost/python.hpp>
#include <Python.h>
#include <numpy/arrayobject.h>
#include "cudata/cudata.h"

#include <boost/shared_ptr.hpp>

sp_cuarray_var make_cuarray_PyObject(PyObject* in) {
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
    switch (dtype) {
    case NPY_BOOL:
        return make_cuarray(n, (bool*)d);
    case NPY_INT:
        return make_cuarray(n, (int*)d);
    case NPY_LONG:
        return make_cuarray(n, (long*)d);
    case NPY_FLOAT:
        return make_cuarray(n, (float*)d);
    case NPY_DOUBLE:
        return make_cuarray(n, (double*)d);
    default:
        throw std::invalid_argument("Can't create CuArray from this object");
    }
}

BOOST_PYTHON_MODULE(cudata) {
    //This initializes Numpy so we can examine types of Numpy arrays
    import_array();
    
    using namespace boost::python;
    
    class_<cuarray_var, boost::shared_ptr<cuarray_var> >("CuArray", no_init)
        .def("__init__", make_constructor(make_cuarray_PyObject))
        .def("__repr__", repr_cuarray);
}
