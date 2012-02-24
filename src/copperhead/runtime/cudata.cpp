#include <boost/python.hpp>
#include "cuarray.hpp"
#include "make_cuarray.hpp"
#include "make_cuarray_impl.hpp"
#include "make_sequence.hpp"
#include "make_sequence_impl.hpp"
#include "cunp.hpp"
#include "np_inspect.hpp"
#include "type.hpp"
#include "monotype.hpp"
#include "type_printer.hpp"
#include <sstream>

using std::shared_ptr;
using std::make_shared;
using std::ostringstream;
using std::string;

typedef boost::shared_ptr<cuarray> sp_cuarray;

template<typename T>
sp_cuarray make_cuarray_copy(T* d, ssize_t n) {
    sp_cuarray new_array = make_cuarray<T>(n);
    sequence<T> new_seq = make_sequence<sequence<T> >(new_array, true, true);
    memcpy(new_seq.m_d, d, sizeof(T) * n);
    return new_array;
}

sp_cuarray make_cuarray_PyObject(PyObject* in) {
    np_array_info in_props = inspect_array(in);
    shared_ptr<backend::sequence_t> t =
        std::static_pointer_cast<backend::sequence_t>(std::get<2>(in_props));
    if (t->p_sub() == backend::bool_mt) {
        return make_cuarray_copy((bool*)std::get<0>(in_props),
                                 std::get<1>(in_props));
    } else if (t->p_sub() == backend::int32_mt) {
        return make_cuarray_copy((int*)std::get<0>(in_props),
                                 std::get<1>(in_props));
    } else if (t->p_sub() == backend::int64_mt) {
        return make_cuarray_copy((long*)std::get<0>(in_props),
                                 std::get<1>(in_props));
    } else if (t->p_sub() == backend::float32_mt) {
        return make_cuarray_copy((float*)std::get<0>(in_props),
                                 std::get<1>(in_props));
    } else if (t->p_sub() == backend::float64_mt) {
        return make_cuarray_copy((double*)std::get<0>(in_props),
                                 std::get<1>(in_props));
    } else {
        throw std::invalid_argument("Can't create CuArray from this object");
    }
}


std::shared_ptr<backend::type_t> type_derive(const cuarray& in) {
    return in.m_t;
}



class cuarray_iterator {
private:
    sp_cuarray source;
    ssize_t index;
public:
    cuarray_iterator(sp_cuarray& _source) : source(_source), index(0) {}
    PyObject* next() {
        if (index >= ssize_t(source->m_l[0])) {
            PyErr_SetString(PyExc_StopIteration, "No more data.");
            boost::python::throw_error_already_set();
        }
        std::shared_ptr<backend::sequence_t> seq_t =
            std::static_pointer_cast<backend::sequence_t>(source->m_t);
        std::shared_ptr<backend::type_t> sub_t = seq_t->p_sub();
        if (sub_t == backend::int32_mt) {
            sequence<int> s = make_sequence<sequence<int> >(source, true, false);
            int result = s[index++];
            return make_scalar(result);
        } else if (sub_t == backend::int64_mt) {
            sequence<long> s = make_sequence<sequence<long> >(source, true, false);
            long result = s[index++];
            return make_scalar(result);
        } else if (sub_t == backend::float32_mt) {
            sequence<float> s = make_sequence<sequence<float> >(source, true, false);
            float result = s[index++];
            return make_scalar(result);
        } else if (sub_t == backend::float64_mt) {
            sequence<double> s = make_sequence<sequence<double> >(source, true, false);
            double result = s[index++];
            return make_scalar(result);
        } else if (sub_t == backend::bool_mt) {
            sequence<bool> s = make_sequence<sequence<bool> >(source, true, false);
            bool result = s[index++];
            return make_scalar(result);
        }
        throw std::invalid_argument("Can't print arrays of this type");
    }
};

shared_ptr<cuarray_iterator>
make_iterator(sp_cuarray& in) {
    return make_shared<cuarray_iterator>(in);
}


template<class T>
T* get_pointer(shared_ptr<T> const &p) {
    return p.get();
}


void print_array(sp_cuarray& in, ostringstream& os) {
    std::shared_ptr<backend::sequence_t> seq_t = std::static_pointer_cast<backend::sequence_t>(in->m_t);
    std::shared_ptr<backend::type_t> sub_t = seq_t->p_sub();
    if (sub_t == backend::int32_mt) {
        sequence<int> s = make_sequence<sequence<int> >(in, true, false);
        os << s;
    } else if (sub_t == backend::int64_mt) {
        sequence<long> s = make_sequence<sequence<long> >(in, true, false);
        os << s;
    } else if (sub_t == backend::float32_mt) {
        sequence<float> s = make_sequence<sequence<float> >(in, true, false);
        os << s;
    } else if (sub_t == backend::float64_mt) {
        sequence<double> s = make_sequence<sequence<double> >(in, true, false);
        os << s;
    } else if (sub_t == backend::bool_mt) {
        sequence<bool> s = make_sequence<sequence<bool> >(in, true, false);
        os << s;
    }
}

void print_type(sp_cuarray&in, ostringstream& os) {
    backend::repr_type_printer tp(os);
    boost::apply_visitor(tp, *(in->m_t));
}

string repr_cuarray(sp_cuarray& in) {
    ostringstream os;
    os << "cuarray(";
    print_array(in, os);
    os << ", type=";
    print_type(in, os);
    os << ")";
    return os.str();
}

string str_cuarray(sp_cuarray& in) {
    ostringstream os;
    print_array(in, os);
    return os.str();
}

BOOST_PYTHON_MODULE(cudata) {
    using namespace boost::python;
    
    class_<cuarray, boost::shared_ptr<cuarray>, boost::noncopyable >("cuarray", no_init)
        .def("__init__", make_constructor(make_cuarray_PyObject))
        .def("__repr__", repr_cuarray)
        .def("__str__", str_cuarray)
        .add_property("type", type_derive)
    //     .def("np", np)
        .def("__iter__", make_iterator);
    class_<cuarray_iterator, shared_ptr<cuarray_iterator> >
        ("cuarrayiterator", no_init)
        .def("next", &cuarray_iterator::next)
        ;
    
}
