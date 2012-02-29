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
#include <boost/python/slice.hpp>
#include "utility/isinstance.hpp"


using std::shared_ptr;
using std::make_shared;
using std::ostringstream;
using std::string;

typedef boost::shared_ptr<cuarray> sp_cuarray;

namespace detail {
template<typename T>
bool isinstance(PyObject* in) {
    boost::python::extract<T> extractor(in);
    if (extractor.check()) {
        return true;
    } else {
        return false;
    }
}
}

template<typename T>
sp_cuarray make_cuarray_copy(T* d, ssize_t n) {
    sp_cuarray new_array = make_cuarray<T>(n);
    sequence<T> new_seq = make_sequence<sequence<T> >(new_array, true, true);
    memcpy(new_seq.m_d, d, sizeof(T) * n);
    return new_array;
}

void desc_lens(PyObject* in, std::vector<size_t>& lens, size_t level=0) {
    if (PyList_Check(in)) {
        size_t cur_len = PyList_Size(in);
        lens[level] += cur_len;
        for(size_t i = 0; i < cur_len; i++) {
            desc_lens(PyList_GetItem(in, i), lens, level+1);
        }
    } else {
        np_array_info in_props = inspect_array(in);
        lens[level] += std::get<1>(in_props);
    }
}

size_t get_el_size(shared_ptr<backend::type_t> p) {
    if (p == backend::int32_mt) {
        return sizeof(int);
    } else if (p == backend::int64_mt) {
        return sizeof(long);
    } else if (p == backend::float32_mt) {
        return sizeof(float);
    } else if (p == backend::float64_mt) {
        return sizeof(double);
    } else if (p == backend::bool_mt) {
        return sizeof(bool);
    }
    return 0;
}

void populate_array(PyObject* in, std::vector<std::shared_ptr<chunk<host_alloc> > >& locals,
                    size_t el_size,
                    std::vector<size_t>& offsets, size_t level = 0) {
    if (PyList_Check(in)) {
        size_t cur_len = PyList_Size(in);
        size_t* desc = (size_t*)locals[level]->ptr();
        for(size_t i = 0; i < cur_len; i++) {
            desc[offsets[level]] = offsets[level+1];
            ++offsets[level];
            populate_array(PyList_GetItem(in, i), locals, el_size, offsets, level+1);
        }
    } else {
        np_array_info in_props = inspect_array(in);
        char* local = (char*)locals[level]->ptr();
        size_t size = std::get<1>(in_props);
        memcpy(local + (offsets[level] * el_size), std::get<0>(in_props), size * el_size);
        offsets[level] += size;
    }
}

shared_ptr<backend::type_t> examine_leaf_array(PyObject* leaf) {
    np_array_info leaf_props = inspect_array(leaf);
    if (std::get<0>(leaf_props) == NULL) {
        throw std::invalid_argument("Can't create CuArray from this object");
    }
    
    //Derive type
    shared_ptr<backend::sequence_t> seq_type =
        std::static_pointer_cast<backend::sequence_t>(std::get<2>(leaf_props));
    
    shared_ptr<backend::type_t> el_type = seq_type->p_sub();
    if (el_type == backend::void_mt) {
        throw std::invalid_argument("Can't create CuArray from this object");
    }
    return el_type;
}

sp_cuarray make_cuarray_PyObject(PyObject* in) {

    if (detail::isinstance<sp_cuarray>(in)) {
        //XXX Do a deep copy (following numpy)
        
        return boost::python::extract<sp_cuarray>(in);
    }

    sp_cuarray result = sp_cuarray(new cuarray());
    
    //Establish nesting depth
    std::cout << "Establish nesting depth" << std::endl;
    int depth = -1;
    PyObject* previous = in;
    PyObject* patient = in;

    shared_ptr<backend::type_t> el_type;
    shared_ptr<backend::type_t> in_type;
    size_t el_size;
    
    
    while (PyList_Check(patient)) {
        ++depth;
        previous = patient;
        patient = PyList_GetItem(patient, 0);
    }

    bool leaf_conversion = false;
    std::vector<PyObject*> leaves;
    if (isnumpyarray(patient)) {
        std::cout << "Leaf was numpy" << std::endl;
        //Leaf element is a numpy array
        ++depth;
        el_type = examine_leaf_array(patient);
    } else {
        std::cout << "Leaf was not numpy" << std::endl;
        //Leaf element is not a numpy array,
        //back up one level in nesting and try to convert
        //it to a numpy array
        el_type = examine_leaf_array(previous);
        leaf_conversion = true;
        
    }

    
    el_size = get_el_size(el_type);
    in_type = el_type;
    for(int i = -1; i < depth; i++) {
        in_type = std::make_shared<backend::sequence_t>(in_type);
    }
    std::cout << "Type was derived: ";
    backend::repr_type_printer rp(std::cout);
    boost::apply_visitor(rp, *in_type);
    std::cout << std::endl;

    //Derive descriptor lengths
    std::vector<size_t> lens(depth+1, 1);
    lens[depth] = 0;
    desc_lens(in, lens);
    std::cout << "Descriptor lengths derived" << std::endl;


    //Allocate descriptors
    std::vector<std::shared_ptr<chunk<host_alloc> > > local_chunks;
#ifdef CUDA_SUPPORT
    std::vector<std::shared_ptr<chunk<cuda_alloc> > > remote_chunks;
#endif
    for(int i = 0; i < depth; i++) {
        local_chunks.push_back(std::make_shared<chunk<host_alloc> >(host_alloc(), sizeof(size_t) * lens[i]));
#ifdef CUDA_SUPPORT
        remote_chunks.push_back(std::make_shared<chunk<cuda_alloc> >(cuda_alloc(), sizeof(size_t) * lens[i]));
#endif
    }
    std::cout << "Descriptors allocated" << std::endl;

    //Allocate data
    local_chunks.push_back(std::make_shared<chunk<host_alloc> >(host_alloc(), el_size * lens[depth]));
#ifdef CUDA_SUPPORT
    remote_chunks.push_back(std::make_shared<chunk<cuda_alloc> >(cuda_alloc(), el_size * lens[depth]));
#endif
    std::cout << "Data allocated" << std::endl;

    //Create descriptors and data
    std::vector<size_t> offsets(depth+1,0);
    populate_array(in, local_chunks, el_size, offsets);
    //Tack on trailing lengths to descriptors
    for(int i = 0; i < depth; i++) {
        size_t* local = (size_t*)local_chunks[i]->ptr();
        local[lens[i]-1] = lens[i+1];
    }
    std::cout << "Data structures built" << std::endl;
    
    //Populate result
    result->m_t = in_type;
    result->m_l = std::move(lens);
    result->m_local = std::move(local_chunks);
#ifdef CUDA_SUPPORT
    result->m_remote = std::move(remote_chunks);
    result->m_clean_local = true;
    result->m_clean_remote = false;
#endif

    std::cout << "Cuarray populated" << std::endl;
    
    return result;
    
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

PyObject* getitem_idx(sp_cuarray& in, size_t index) {
    std::shared_ptr<backend::sequence_t> seq_t = std::static_pointer_cast<backend::sequence_t>(in->m_t);
    std::shared_ptr<backend::type_t> sub_t = seq_t->p_sub();
    if (sub_t == backend::int32_mt) {
        sequence<int> s = make_sequence<sequence<int> >(in, true, false);
        return make_scalar(s[index]);
    } else if (sub_t == backend::int64_mt) {
        sequence<long> s = make_sequence<sequence<long> >(in, true, false);
        return make_scalar(s[index]);
    } else if (sub_t == backend::float32_mt) {
        sequence<float> s = make_sequence<sequence<float> >(in, true, false);
        return make_scalar(s[index]);
    } else if (sub_t == backend::float64_mt) {
        sequence<double> s = make_sequence<sequence<double> >(in, true, false);
        return make_scalar(s[index]);
    } else if (sub_t == backend::bool_mt) {
        sequence<bool> s = make_sequence<sequence<bool> >(in, true, false);
        return make_scalar(s[index]);
    }
    return boost::python::object().ptr(); //None

    //To return a boost::shared_ptr<T> as a PyObject*
    //return boost::python::converter::shared_ptr_to_python(in);

}

void setitem_idx(sp_cuarray& in, size_t index, PyObject* value) {
    std::shared_ptr<backend::sequence_t> seq_t = std::static_pointer_cast<backend::sequence_t>(in->m_t);
    std::shared_ptr<backend::type_t> sub_t = seq_t->p_sub();
    if (sub_t == backend::int32_mt) {
        sequence<int> s = make_sequence<sequence<int> >(in, true, false);
        s[index] = unpack_scalar_int(value);
    } else if (sub_t == backend::int64_mt) {
        sequence<long> s = make_sequence<sequence<long> >(in, true, false);
        s[index] = unpack_scalar_long(value);
    } else if (sub_t == backend::float32_mt) {
        sequence<float> s = make_sequence<sequence<float> >(in, true, false);
        s[index] = unpack_scalar_float(value);
    } else if (sub_t == backend::float64_mt) {
        sequence<double> s = make_sequence<sequence<double> >(in, true, false);
        s[index] = unpack_scalar_double(value);
    } else if (sub_t == backend::bool_mt) {
        sequence<bool> s = make_sequence<sequence<bool> >(in, true, false);
        s[index] = unpack_scalar_bool(value);
    }
    
}



BOOST_PYTHON_MODULE(cudata) {
    using namespace boost::python;
    
    class_<cuarray, boost::shared_ptr<cuarray>, boost::noncopyable >("cuarray", no_init)
        .def("__init__", make_constructor(make_cuarray_PyObject))
        .def("__repr__", repr_cuarray)
        .def("__str__", str_cuarray)
        .def("__getitem__", &getitem_idx)
        .def("__setitem__", &setitem_idx)
        .add_property("type", type_derive)
        .def("__iter__", make_iterator);
    
    class_<cuarray_iterator, shared_ptr<cuarray_iterator> >
        ("cuarrayiterator", no_init)
        .def("next", &cuarray_iterator::next)
        ;
    
}
