#include <boost/python.hpp>
#include <Python.h>
#include <numpy/arrayobject.h>
#include "cudata.h"
#include "cunp.h"
#include <boost/shared_ptr.hpp>


#include "type.hpp"
#include "monotype.hpp"

sp_cuarray_var make_cuarray_PyObject(PyObject* in) {
    array_info in_props = inspect_array(in);
    switch (in_props.t) {
    case CUNP_BOOL:
        return make_cuarray(in_props.n, (bool*)in_props.d);
    case CUNP_INT:
        return make_cuarray(in_props.n, (int*)in_props.d);
    case CUNP_LONG:
        return make_cuarray(in_props.n, (long*)in_props.d);
    case CUNP_FLOAT:
        return make_cuarray(in_props.n, (float*)in_props.d);
    case CUNP_DOUBLE:
        return make_cuarray(in_props.n, (double*)in_props.d);
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
    BRIDGE_TYPE dtype;
    if (el_type == backend::bool_mt) {
        dtype = CUNP_BOOL;
    }
    else if (el_type == backend::int32_mt) {
        dtype = CUNP_INT;
    } else if (el_type == backend::int64_mt) {
        dtype = CUNP_LONG;
    } else if (el_type == backend::float32_mt) {
        dtype = CUNP_FLOAT;
    } else if (el_type == backend::float64_mt) {
        dtype = CUNP_DOUBLE;
    } else {
        throw std::invalid_argument("Can't create numpy array from this object");
    }
    void *view = boost::apply_visitor(get_viewer(), in);

    return make_array(array_info(view, boost::apply_visitor(get_sizer(), in), dtype));
    
}


class cuarray_indexer
    : public boost::static_visitor<PyObject*> {
private:
    ssize_t index;
public:
    cuarray_indexer(const ssize_t& _index=0) : index(_index) {}
    template<typename T>
    PyObject* operator()(cuarray<T>& in) {
        if (index >= in.get_size()) {
            PyErr_SetString(PyExc_StopIteration, "No more data.");
            boost::python::throw_error_already_set();
        }
        T el = in[index++];
        return make_scalar(el);
    }
};

class cuarray_iterator {
private:
    cuarray_var& source;
    cuarray_indexer indexer;
public:
    cuarray_iterator(cuarray_var& _source) : source(_source), indexer(0) {}
    PyObject* next() {
        return boost::apply_visitor(indexer, source);
    }
};

boost::shared_ptr<cuarray_iterator>
make_iterator(boost::shared_ptr<cuarray_var>& in) {
    return boost::shared_ptr<cuarray_iterator>(new cuarray_iterator(*in));
}




BOOST_PYTHON_MODULE(cudata) {
    //This initializes Numpy
    initialize_cunp();
    
    using namespace boost::python;
    
    class_<cuarray_var, boost::shared_ptr<cuarray_var> >("CuArray", no_init)
        .def("__init__", make_constructor(make_cuarray_PyObject))
        .def("__repr__", repr_cuarray)
        .add_property("type", type_derive)
        .def("np", np)
        .def("__iter__", make_iterator);
    class_<cuarray_iterator, boost::shared_ptr<cuarray_iterator> >
        ("CuArrayIterator", no_init)
        .def("next", &cuarray_iterator::next);
    
}
