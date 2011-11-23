#include <boost/python.hpp>
#include <Python.h>
#include <numpy/arrayobject.h>
#include "cudata.h"

#include <boost/shared_ptr.hpp>

#include "type.hpp"
#include "monotype.hpp"

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

struct type_deriver
    : public boost::static_visitor<std::shared_ptr<backend::type_t> > {
    result_type operator()(const cuarray<bool> &b) const {
        return std::make_shared<backend::sequence_t>(
            backend::bool_mt);
    }
    result_type operator()(const cuarray<int> &b) const {
        return std::make_shared<backend::sequence_t>(
            backend::int32_mt);
    }
    result_type operator()(const cuarray<long> &b) const {
        return std::make_shared<backend::sequence_t>(
            backend::int64_mt);
    }
    result_type operator()(const cuarray<float> &b) const {
        return std::make_shared<backend::sequence_t>(
            backend::float32_mt);
    }
    result_type operator()(const cuarray<double> &b) const {
        return std::make_shared<backend::sequence_t>(
            backend::float64_mt);
    }
};

std::shared_ptr<backend::type_t> type_derive(const cuarray_var& in) {
    return boost::apply_visitor(type_deriver(), in);
}

BOOST_PYTHON_MODULE(cudata) {
    //This initializes Numpy so we can examine types of Numpy arrays
    import_array();
    
    using namespace boost::python;
    
    class_<cuarray_var, boost::shared_ptr<cuarray_var> >("CuArray", no_init)
        .def("__init__", make_constructor(make_cuarray_PyObject))
        .def("__repr__", repr_cuarray)
        .add_property("type", type_derive);
}
