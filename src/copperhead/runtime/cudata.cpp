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
using std::ostream;
using std::vector;
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

void desc_lens(PyObject* in, vector<size_t>& lens,
               shared_ptr<backend::type_t>& el_type,
               vector<np_array_info>& leaves,
               size_t level=0) {

    //Are we at a leaf level?
    if (level == lens.size() - 1) {
        np_array_info in_props = inspect_array(in);
        if (std::get<0>(in_props) == NULL) {
            throw std::invalid_argument("Can't create cuarray from this object");
        }
        shared_ptr<backend::type_t> obs_el_type = std::get<2>(in_props);
        if (backend::detail::isinstance<backend::sequence_t>(*obs_el_type)) {
            obs_el_type = (std::static_pointer_cast<backend::sequence_t>(obs_el_type))->p_sub();
        }
        //If element type isn't correct
        if ((obs_el_type != el_type) &&
            //And we didn't find an empty leaf (which is allowed)
            (std::get<1>(in_props) != 0)) {
            throw std::invalid_argument("Can't create cuarray from this object, it was not homogenously typed");
        }
        lens[level] += std::get<1>(in_props);
        leaves.push_back(in_props);
    } else if (PyList_Check(in)) {
        size_t cur_len = PyList_Size(in);
        lens[level] += cur_len;
        for(size_t i = 0; i < cur_len; i++) {
            desc_lens(PyList_GetItem(in, i), lens, el_type, leaves, level+1);
        }
    } else {
        throw std::invalid_argument("Can't create cuarray from this object, it was not homogeneously nested");
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

void populate_array(PyObject* in,
                    vector<std::shared_ptr<chunk<host_alloc> > >& locals,
                    size_t el_size,
                    vector<np_array_info>::const_iterator& leaves,
                    vector<size_t>& offsets, size_t level = 0) {
    //Are we at a leaf level?
    if (level == offsets.size() - 1) {
        np_array_info in_props = *leaves;
        ++leaves;
        char* local = (char*)locals[level]->ptr();
        size_t size = std::get<1>(in_props);
        memcpy(local + (offsets[level] * el_size), std::get<0>(in_props), size * el_size);
        offsets[level] += size;
    } else {
        size_t cur_len = PyList_Size(in);
        size_t* desc = (size_t*)locals[level]->ptr();
        for(size_t i = 0; i < cur_len; i++) {
            desc[offsets[level]] = offsets[level+1];
            ++offsets[level];
            populate_array(PyList_GetItem(in, i), locals, el_size, leaves, offsets, level+1);
        }
    } 
}

shared_ptr<backend::type_t> examine_leaf_array(PyObject* leaf) {
    np_array_info leaf_props = inspect_array(leaf);
    if (std::get<0>(leaf_props) == NULL) {
        throw std::invalid_argument("Can't create cuarray from this object");
    }

    shared_ptr<backend::type_t> indicated_type = std::get<2>(leaf_props);
    if (indicated_type == backend::void_mt) {
        size_t indicated_length = std::get<1>(leaf_props);
        if (indicated_length != 0) {
            throw std::invalid_argument("Can't create cuarray from this object");
        }
        //An empty sequence. Following numpy, declare it do be a Float64 sequence.
        return backend::float64_mt;
    }
        
    //Derive type
    shared_ptr<backend::sequence_t> seq_type =
        std::static_pointer_cast<backend::sequence_t>(std::get<2>(leaf_props));
    
    shared_ptr<backend::type_t> el_type = seq_type->p_sub();
    if (el_type == backend::void_mt) {
        throw std::invalid_argument("Can't create cuarray from this object");
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
    int depth = -1;
    PyObject* previous = in;
    PyObject* patient = in;

    shared_ptr<backend::type_t> el_type;
    shared_ptr<backend::type_t> in_type;
    size_t el_size;

    while ((patient != NULL) && (PyList_Check(patient))) {
        ++depth;
        previous = patient;

        size_t length = PyList_Size(patient);
        if (length > 0) {
            patient = PyList_GetItem(patient, 0);
        } else {
            patient = NULL;
        }
    }
    
    if (((patient != NULL) && isnumpyarray(patient))) {
        //Leaf element is a numpy array
        ++depth;
        el_type = examine_leaf_array(patient);
    } else {
        //Leaf element is not a numpy array,
        //back up one level in nesting and try to convert
        //it to a numpy array
        el_type = examine_leaf_array(previous);
    }

    
    el_size = get_el_size(el_type);
    in_type = el_type;
    for(int i = -1; i < depth; i++) {
        in_type = std::make_shared<backend::sequence_t>(in_type);
    }

    //Derive descriptor lengths, type check, derive leaf pointers
    vector<size_t> lens(depth+1, 1);
    lens[depth] = 0;
    vector<np_array_info> leaves;

    desc_lens(in, lens, el_type, leaves);

    //Allocate descriptors
    vector<std::shared_ptr<chunk<host_alloc> > > local_chunks;
#ifdef CUDA_SUPPORT
    vector<std::shared_ptr<chunk<cuda_alloc> > > remote_chunks;
#endif
    for(int i = 0; i < depth; i++) {
        local_chunks.push_back(std::make_shared<chunk<host_alloc> >(host_alloc(), sizeof(size_t) * lens[i]));
#ifdef CUDA_SUPPORT
        remote_chunks.push_back(std::make_shared<chunk<cuda_alloc> >(cuda_alloc(), sizeof(size_t) * lens[i]));
#endif
    }

    //Allocate data
    local_chunks.push_back(std::make_shared<chunk<host_alloc> >(host_alloc(), el_size * lens[depth]));
#ifdef CUDA_SUPPORT
    remote_chunks.push_back(std::make_shared<chunk<cuda_alloc> >(cuda_alloc(), el_size * lens[depth]));
#endif

    //Create descriptors and data
    vector<size_t> offsets(depth+1,0);
    auto leaves_iterator = leaves.cbegin();
    populate_array(in, local_chunks, el_size, leaves_iterator, offsets);
   
    //Tack on trailing lengths to descriptors
    for(int i = 0; i < depth; i++) {
        size_t* local = (size_t*)local_chunks[i]->ptr();
        local[lens[i]-1] = offsets[i+1];
    }

    //Populate result
    result->m_t = in_type;
    result->m_l = std::move(lens);
    result->m_local = std::move(local_chunks);
#ifdef CUDA_SUPPORT
    result->m_remote = std::move(remote_chunks);
    result->m_clean_local = true;
    result->m_clean_remote = false;
#endif
    result->m_o = 0;
    return result;
}


std::shared_ptr<backend::type_t> type_derive(const cuarray& in) {
    return in.m_t;
}






template<class T>
T* get_pointer(shared_ptr<T> const &p) {
    return p.get();
}


sp_cuarray make_index_view(sp_cuarray& in, long index) {
    //Handle negative indices as Python does
    size_t len = in->m_l[0];
    if (index < 0) {
        index += len;
    }
    //Bounds check
    if (index >= long(len)) {
         PyErr_SetString(PyExc_IndexError, "Index out of range");
         boost::python::throw_error_already_set();
    }

    //This function only operates on nested sequences. Ensure type complies
    std::shared_ptr<backend::sequence_t> seq_t =
        std::static_pointer_cast<backend::sequence_t>(in->m_t);
    std::shared_ptr<backend::type_t> sub_t = seq_t->p_sub();
    if (!backend::detail::isinstance<backend::sequence_t>(*sub_t)) {
        throw std::invalid_argument("Internal error, can't index sequence");
    }

    //Begin assembling view
    std::vector<std::shared_ptr<chunk<host_alloc> > > local;
#ifdef CUDA_SUPPORT
    std::vector<std::shared_ptr<chunk<cuda_alloc> > > remote;
#endif
    std::vector<size_t> lengths;

    //Index into outermost descriptor
    size_t* root_desc = (size_t*)in->m_local[0]->ptr() + in->m_o;
    size_t begin = root_desc[index];
    size_t end = root_desc[index+1];
    size_t length = end - begin;
    //Will return a nested sequence?
    if (in->m_l.size() > 2) {
        length++; //Account for tail descriptor entry
    }
    lengths.push_back(length);
    //Copy remaining lengths
    for(size_t i = 2;
        i < in->m_local.size();
        i++) {
        lengths.push_back(in->m_l[i]);
    }
    //Copy buffers for view
    for(size_t i = 1;
        i < in->m_local.size();
        i++) {
        local.push_back(in->m_local[i]);
#ifdef CUDA_SUPPORT
        remote.push_back(in->m_remote[i]);
#endif
    }
    //Assemble resulting view
    sp_cuarray result(new cuarray());
    result->m_local = std::move(local);
#ifdef CUDA_SUPPORT
    result->m_remote = std::move(remote);
    result->m_clean_local = in->m_clean_local;
    result->m_clean_remote = in->m_clean_remote;
#endif
    result->m_l = std::move(lengths);
    result->m_t = sub_t;
    result->m_o = begin;
    return result;
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
    } else {
        sp_cuarray sub_array = make_index_view(in, index);
        return boost::python::converter::shared_ptr_to_python(sub_array);
    }
}

// void setitem_idx(sp_cuarray& in, size_t index, PyObject* value) {
//     std::shared_ptr<backend::sequence_t> seq_t = std::static_pointer_cast<backend::sequence_t>(in->m_t);
//     std::shared_ptr<backend::type_t> sub_t = seq_t->p_sub();
//     if (sub_t == backend::int32_mt) {
//         sequence<int> s = make_sequence<sequence<int> >(in, true, false);
//         s[index] = unpack_scalar_int(value);
//     } else if (sub_t == backend::int64_mt) {
//         sequence<long> s = make_sequence<sequence<long> >(in, true, false);
//         s[index] = unpack_scalar_long(value);
//     } else if (sub_t == backend::float32_mt) {
//         sequence<float> s = make_sequence<sequence<float> >(in, true, false);
//         s[index] = unpack_scalar_float(value);
//     } else if (sub_t == backend::float64_mt) {
//         sequence<double> s = make_sequence<sequence<double> >(in, true, false);
//         s[index] = unpack_scalar_double(value);
//     } else if (sub_t == backend::bool_mt) {
//         sequence<bool> s = make_sequence<sequence<bool> >(in, true, false);
//         s[index] = unpack_scalar_bool(value);
//     }
    
// }


class cuarray_iterator {
private:
    sp_cuarray source;
    size_t index;
    size_t length;
public:
    cuarray_iterator(sp_cuarray& _source) : source(_source), index(0) {
        length = source->m_l[0];
    }
    PyObject* next() {
        if (index >= length) {
            PyErr_SetString(PyExc_StopIteration, "No more data.");
            boost::python::throw_error_already_set();
        }
        return getitem_idx(source, index++);
    }
};

shared_ptr<cuarray_iterator>
make_iterator(sp_cuarray& in) {
    return make_shared<cuarray_iterator>(in);
}


void print_array(sp_cuarray& in, ostream& os) {
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
    } else {
        os << "[";
        size_t length = in->m_l[0] - 1;
        for(size_t i = 0; i < length; i++) {
            sp_cuarray el = make_index_view(in, i);
            print_array(el, os);
            if (i + 1 < length) {
                os << ", ";
            }
        }
        os << "]";
        
    }
}

void print_type(sp_cuarray&in, ostream& os) {
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
        .def("__getitem__", &getitem_idx)
        //.def("__setitem__", &setitem_idx)
        .add_property("type", type_derive)
        .def("__iter__", make_iterator);
    
    class_<cuarray_iterator, shared_ptr<cuarray_iterator> >
        ("cuarrayiterator", no_init)
        .def("next", &cuarray_iterator::next)
        ;
    
}
