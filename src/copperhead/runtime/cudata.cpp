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

struct element_typer
    : public boost::static_visitor<std::shared_ptr<backend::type_t> > {
    result_type operator()(const cuarray<bool> &b) const {
        return backend::bool_mt;
    }
    result_type operator()(const cuarray<int> &b) const {
        return backend::int32_mt;
    }
    result_type operator()(const cuarray<long> &b) const {
        return backend::int64_mt;
    }
    result_type operator()(const cuarray<float> &b) const {
        return backend::float32_mt;
    }
    result_type operator()(const cuarray<double> &b) const {
        return backend::float64_mt;
    }
};

std::shared_ptr<backend::type_t> type_derive(const cuarray_var& in) {
    return std::make_shared<backend::sequence_t>(boost::apply_visitor(element_typer(), in));
}

struct get_viewer
    : public boost::static_visitor<void*> {
    template<typename E>
    result_type operator()(cuarray<E> &b) const {
        return b.get_view();
    }
};

struct get_sizer
    : public boost::static_visitor<int> {
    template<typename E>
    result_type operator()(const cuarray<E> &b) const {
        return (int)b.get_size();
    }
};

PyObject* np(cuarray_var& in) {
    std::shared_ptr<backend::type_t> el_type = boost::apply_visitor(element_typer(), in);
    NPY_TYPES dtype;
    int size_of_el;
    if (el_type == backend::bool_mt) {
        dtype = NPY_BOOL;
        size_of_el = sizeof(bool);
    }
    else if (el_type == backend::int32_mt) {
        dtype = NPY_INT;
        size_of_el = sizeof(int);
    } else if (el_type == backend::int64_mt) {
        dtype = NPY_LONG;
        size_of_el = sizeof(long);
    } else if (el_type == backend::float32_mt) {
        dtype = NPY_FLOAT;
        size_of_el = sizeof(float);
    } else if (el_type == backend::float64_mt) {
        dtype = NPY_DOUBLE;
        size_of_el = sizeof(double);
    } else {
        throw std::invalid_argument("Can't create numpy array from this object");
    }
    void *view = boost::apply_visitor(get_viewer(), in);
    npy_intp dims[1];
    dims[0] = boost::apply_visitor(get_sizer(), in);
    PyObject* return_array = PyArray_SimpleNew(1, dims, dtype);
    memcpy(PyArray_DATA(return_array), view, size_of_el * dims[0]);
    
    return return_array;
}

BOOST_PYTHON_MODULE(cudata) {
    //This initializes Numpy so we can examine types of Numpy arrays
    import_array();
    
    using namespace boost::python;
    
    class_<cuarray_var, boost::shared_ptr<cuarray_var> >("CuArray", no_init)
        .def("__init__", make_constructor(make_cuarray_PyObject))
        .def("__repr__", repr_cuarray)
        .add_property("type", type_derive)
        .def("np", np);
    
}
