#include <boost/python.hpp>
#include <numpy/arrayobject.h>
#include <memory>
#include "cudata.h"
#include "cunp.h"
#include "type.hpp"
#include "monotype.hpp"


using std::shared_ptr;
using std::make_shared;

typedef std::shared_ptr<cuarray> sp_cuarray;

template<typename T>
sp_cuarray make_cuarray(ssize_t n, T* d) {
    return make_shared<cuarray>(n, d);
}

sp_cuarray make_cuarray_PyObject(PyObject* in) {
    array_info in_props = inspect_array(in);
    switch (in_props.t) {
    case CUBOOL:
        return make_cuarray(in_props.n, (bool*)in_props.d);
    case CUINT32:
        return make_cuarray(in_props.n, (int*)in_props.d);
    case CUINT64:
        return make_cuarray(in_props.n, (long*)in_props.d);
    case CUFLOAT32:
        return make_cuarray(in_props.n, (float*)in_props.d);
    case CUFLOAT64:
        return make_cuarray(in_props.n, (double*)in_props.d);
    default:
        throw std::invalid_argument("Can't create CuArray from this object");
    }
}


std::shared_ptr<backend::type_t> type_derive(const cuarray& in) {
    switch(in.t) {
    case CUBOOL:
        return std::make_shared<backend::sequence_t>(backend::bool_mt);
    case CUINT32:
        return std::make_shared<backend::sequence_t>(backend::int32_mt);
    case CUINT64:
        return std::make_shared<backend::sequence_t>(backend::int64_mt);
    case CUFLOAT32:
        return std::make_shared<backend::sequence_t>(backend::float32_mt);
    case CUFLOAT64:
        return std::make_shared<backend::sequence_t>(backend::float64_mt);
    default:
        throw std::invalid_argument("Can't derive type from this object");
    }
}


PyObject* np(cuarray& in) {
    auto i = in.get_local_r();
    return make_array(array_info(i.first, i.second, in.t));
}


class cuarray_iterator {
private:
    cuarray& source;
    ssize_t index;
public:
    cuarray_iterator(cuarray& _source) : source(_source), index(0) {}
    PyObject* next() {
        auto i = source.get_local_r();
        if (index >= i.second) {
            PyErr_SetString(PyExc_StopIteration, "No more data.");
            boost::python::throw_error_already_set();
        }
        switch(source.t) {
        case CUBOOL:
            {
                bool* l_d = (bool*)i.first;
                bool result = l_d[index++];
                return make_scalar(result);
            }
        case CUINT32:
            {
                int* l_d = (int*)i.first;
                int result = l_d[index++];
                return make_scalar(result);
            }
        case CUINT64:
            {
                long* l_d = (long*)i.first;
                long result = l_d[index++];
                return make_scalar(result);
            }
        case CUFLOAT32:
            {
                float* l_d = (float*)i.first;
                float result = l_d[index++];
                return make_scalar(result);
            }
        case CUFLOAT64:
            {
                double* l_d = (double*)i.first;
                double result = l_d[index++];
                return make_scalar(result);
            }
        default:
            throw std::invalid_argument("Untyped array");
        }
    }
};

shared_ptr<cuarray_iterator>
make_iterator(shared_ptr<cuarray>& in) {
    return make_shared<cuarray_iterator>(*in);
}


template<class T>
T* get_pointer(shared_ptr<T> const &p) {
    return p.get();
}


BOOST_PYTHON_MODULE(cudata) {
    //This initializes Numpy
    initialize_cunp();
    
    using namespace boost::python;
    
    class_<cuarray, shared_ptr<cuarray> >("CuArray", no_init)
        .def("__init__", make_constructor(make_cuarray_PyObject))
        //.def("__repr__", repr_cuarray)
        .add_property("type", type_derive)
        .def("np", np)
        .def("__iter__", make_iterator);
    class_<cuarray_iterator, shared_ptr<cuarray_iterator> >
        ("CuArrayIterator", no_init)
        .def("next", &cuarray_iterator::next);
    
}
