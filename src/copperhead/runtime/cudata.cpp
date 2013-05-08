#include <boost/python.hpp>
#include <prelude/runtime/chunk.hpp>
#include <prelude/runtime/cuarray.hpp>
#include <prelude/runtime/type_holder.hpp>
#include <prelude/runtime/make_sequence.hpp>
#include <prelude/runtime/mempool.hpp>
#include "cunp.hpp"
#include "np_inspect.hpp"
#include "type.hpp"
#include "monotype.hpp"
#include "type_printer.hpp"
#include <sstream>
#include <boost/python/slice.hpp>
#include "utility/isinstance.hpp"
#include <boost/scoped_ptr.hpp>

#include <thrust/system/omp/memory.h>
#include <boost/version.hpp>

using std::shared_ptr;
using std::make_shared;
using std::static_pointer_cast;
using std::ostringstream;
using std::ostream;
using std::vector;
using std::string;
using std::map;

typedef boost::shared_ptr<copperhead::cuarray> sp_cuarray;

namespace copperhead {

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
               shared_ptr<const backend::type_t>& el_type,
               vector<np_array_info>& leaves,
               size_t level=0) {

    //Are we at a leaf level?
    if (level == lens.size() - 1) {
        np_array_info in_props = inspect_array(in);
        if (std::get<0>(in_props) == NULL) {
            throw std::invalid_argument("Can't create cuarray from this object");
        }
        shared_ptr<const backend::type_t> obs_el_type = std::get<2>(in_props);
        if (backend::detail::isinstance<backend::sequence_t>(*obs_el_type)) {
            obs_el_type = (static_pointer_cast<const backend::sequence_t>(obs_el_type))->sub().ptr();
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

size_t get_el_size(shared_ptr<const backend::type_t> p) {
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
                    vector<boost::shared_ptr<chunk> >& locals,
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

shared_ptr<const backend::type_t> examine_leaf_array(PyObject* leaf) {
    np_array_info leaf_props = inspect_array(leaf);
    if (std::get<0>(leaf_props) == NULL) {
        throw std::invalid_argument("Can't create cuarray from this object");
    }

    shared_ptr<const backend::type_t> indicated_type = std::get<2>(leaf_props);
    if (indicated_type == backend::void_mt) {
        size_t indicated_length = std::get<1>(leaf_props);
        if (indicated_length != 0) {
            throw std::invalid_argument("Can't create cuarray from this object");
        }
        //An empty sequence. Following numpy, declare it do be a Float64 sequence.
        return backend::float64_mt;
    }
        
    //Derive type
    shared_ptr<const backend::sequence_t> seq_type =
        static_pointer_cast<const backend::sequence_t>(std::get<2>(leaf_props));
    
    shared_ptr<const backend::type_t> el_type = seq_type->sub().ptr();
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
    
    //Establish nesting depth
    int depth = -1;
    PyObject* previous = in;
    PyObject* patient = in;

    shared_ptr<const backend::type_t> el_type;
    shared_ptr<const backend::type_t> in_type;
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

    //Construct array
    type_holder* th = new type_holder();
    th->m_t = in_type;

    sp_cuarray result(
        new cuarray(th));
    //Allocate descriptors
    for(int i = 0; i < depth; i++) {
        result->add_chunk(boost::shared_ptr<chunk>(new chunk(cpp_tag(), sizeof(size_t) * lens[i])), true);
#ifdef CUDA_SUPPORT
        result->add_chunk(boost::shared_ptr<chunk>(new chunk(cuda_tag(), sizeof(size_t) * lens[i])), false);
#endif
    }

    //Allocate data
    result->add_chunk(boost::shared_ptr<chunk>(new chunk(cpp_tag(), el_size * lens[depth])), true);
#ifdef CUDA_SUPPORT
    result->add_chunk(boost::shared_ptr<chunk>(new chunk(cuda_tag(), el_size * lens[depth])), false);
#endif

    std::vector<boost::shared_ptr<chunk> >& local_chunks = result->get_chunks(cpp_tag(), false);
    //Create descriptors and data
    vector<size_t> offsets(depth+1, 0);
    auto leaves_iterator = leaves.cbegin();
    populate_array(in, local_chunks, el_size, leaves_iterator, offsets);
   
    //Tack on tail descriptors
    for(int i = 0; i < depth; i++) {
        size_t* local = (size_t*)local_chunks[i]->ptr();
        local[lens[i]-1] = offsets[i+1];
    }

    result->m_l = lens;
        
    return result;
}


shared_ptr<backend::type_t> type_derive(const cuarray& in) {
    //const_pointer_cast necessary here because boost::python
    //doesn't deal well with shared_ptr<const T>.
    //However, since we don't expose any methods for mutating
    //backend::type_t objects in Python, this shouldn't cause
    //issues
    return std::const_pointer_cast<backend::type_t>(in.m_t->m_t);
}

sp_cuarray make_index_view(sp_cuarray& in, long index) {

    //This function only operates on nested sequences. Ensure type complies
    shared_ptr<const backend::sequence_t> seq_t =
        static_pointer_cast<const backend::sequence_t>(in->m_t->m_t);
    shared_ptr<const backend::type_t> sub_t = seq_t->sub().ptr();
    if (!backend::detail::isinstance<backend::sequence_t>(*sub_t)) {
        throw std::invalid_argument("Internal error, can't index sequence (index_view)");
    }

    //Assemble resulting view
    type_holder* th = new type_holder();
    th->m_t = sub_t;

    sp_cuarray result(new cuarray(th));

    bool local_valid = in->clean(cpp_tag());
    std::vector<boost::shared_ptr<chunk> >& local_chunks = in->get_chunks(cpp_tag(), false);
#ifdef CUDA_SUPPORT
    bool remote_valid = in->clean(cuda_tag());
    std::vector<boost::shared_ptr<chunk> >& remote_chunks = in->get_chunks(cuda_tag(), false);
#endif
    //Index into outermost descriptor
    size_t* root_desc = (size_t*)local_chunks[0]->ptr() + in->m_o;
    size_t begin = root_desc[index];
    result->m_o = begin;
    size_t end = root_desc[index+1];
    size_t length = end - begin;
    
    //Will return a nested sequence?
    if (in->m_l.size() > 2) {
        length++; //Account for tail descriptor entry
    }
    result->push_back_length(length);
    //Copy remaining lengths
    for(size_t i = 2;
        i < in->m_l.size();
        i++) {
        result->push_back_length(in->m_l[i]);
    }
    //Copy buffers for view
    for(size_t i = 1;
        i < in->m_l.size();
        i++) {
        result->add_chunk(local_chunks[i], local_valid);
#ifdef CUDA_SUPPORT
        result->add_chunk(remote_chunks[i], remote_valid);
#endif
    }
    return result;
}

template<typename T>
PyObject* deref_scalar(const boost::shared_ptr<chunk>& s, long index) {
    const T* src = reinterpret_cast<T*>(s->ptr());
    return make_scalar(src[index]);
}

PyObject* deref_zip_sequence(shared_ptr<const backend::type_t> t, 
                             typename vector<boost::shared_ptr<chunk> >::const_iterator& leaf,
                             long index) {
    if (t == backend::int32_mt) {
        PyObject* result = deref_scalar<int>(*leaf, index);
        leaf++;
        return result;
    } else if (t == backend::int64_mt) {
        PyObject* result = deref_scalar<long>(*leaf, index);
        leaf++;
        return result;
    } else if (t == backend::float32_mt) {
        PyObject* result = deref_scalar<float>(*leaf, index);
        leaf++;
        return result;
    } else if (t == backend::float64_mt) {
        PyObject* result = deref_scalar<double>(*leaf, index);
        leaf++;
        return result;
    } else if (t == backend::bool_mt) {
        PyObject* result = deref_scalar<bool>(*leaf, index);
        leaf++;
        return result;
    } else if (backend::detail::isinstance<backend::tuple_t>(*t)) {
        const backend::tuple_t& tup = boost::get<const backend::tuple_t&>(*t);
        int arity = tup.arity();
        PyObject* result = PyTuple_New(arity);
        int i = 0;
        for(auto j = tup.begin(); j!= tup.end(); i++, j++) {
            PyTuple_SetItem(result, i, deref_zip_sequence(j->ptr(), leaf, index));
        }
        return result;
    } else {
        throw std::invalid_argument("Internal error, can't index sequence (deref_zip_sequence)");
    }
        
}


PyObject* getitem_idx(sp_cuarray& in, long index) {
    //Handle negative indices as Python does
    size_t length = in->size();
    
    if (index < 0) {
        index += length;
    }
    //Bounds check
    if ((index < 0) || (index >= long(length))) {
         PyErr_SetString(PyExc_IndexError, "Index out of range");
         boost::python::throw_error_already_set();
    }

    
    shared_ptr<const backend::sequence_t> seq_t = static_pointer_cast<const backend::sequence_t>(in->m_t->m_t);
    shared_ptr<const backend::type_t> sub_t = seq_t->sub().ptr();
    
    if (sub_t == backend::int32_mt) {
        sequence<cpp_tag, int> s = make_sequence<sequence<cpp_tag, int> >(in, cpp_tag(), false);
        return make_scalar(s[index]);
    } else if (sub_t == backend::int64_mt) {
        sequence<cpp_tag, long> s = make_sequence<sequence<cpp_tag, long> >(in, cpp_tag(), false);
        return make_scalar(s[index]);
    } else if (sub_t == backend::float32_mt) {
        sequence<cpp_tag, float> s = make_sequence<sequence<cpp_tag, float> >(in, cpp_tag(), false);
        return make_scalar(s[index]);
    } else if (sub_t == backend::float64_mt) {
        sequence<cpp_tag, double> s = make_sequence<sequence<cpp_tag, double> >(in, cpp_tag(), false);
        return make_scalar(s[index]);
    } else if (sub_t == backend::bool_mt) {
        sequence<cpp_tag, bool> s = make_sequence<sequence<cpp_tag, bool> >(in, cpp_tag(), false);
        return make_scalar(s[index]);
    } else if (backend::detail::isinstance<backend::tuple_t>(*sub_t)) {
        auto leaf = in->get_chunks(cpp_tag(), false).cbegin();
        return deref_zip_sequence(sub_t, leaf, index);
    } else {
        sp_cuarray sub_array = make_index_view(in, index);
        return boost::python::converter::shared_ptr_to_python(sub_array);
    }
}

// void setitem_idx(sp_cuarray& in, size_t index, PyObject* value) {
//     shared_ptr<backend::sequence_t> seq_t = static_pointer_cast<backend::sequence_t>(in->m_t);
//     shared_ptr<backend::type_t> sub_t = seq_t->p_sub();
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

#if BOOST_VERSION < 105300

template<class T>
T* get_pointer(shared_ptr<T> const &p) {
    return p.get();
}

#endif

class cuarray_iterator {
private:
    sp_cuarray source;
    size_t index;
    size_t length;
public:
    cuarray_iterator(sp_cuarray& _source) : source(_source), index(0) {
        length = source->size();
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

template<typename T>
std::ostream& operator<<(std::ostream& os, sequence<cpp_tag, T, 0>& in) {
    os << "[";
    for(size_t i = 0; i < in.size(); i++) {
        os << in[i];
        if (i + 1 != in.size()) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

template<typename T, int D>
std::ostream& operator<<(std::ostream& os, sequence<cpp_tag, T, D>& in) {
    os << "[";
    for(size_t i = 0; i < in.size(); i++) {
        sequence<cpp_tag, T, D-1> cur = in[i];
        os << cur;
        if (i + 1 != in.size()) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}



void print_zip_element(sp_cuarray& in, size_t index, ostream& os) {
    shared_ptr<const backend::sequence_t> seq_t = static_pointer_cast<const backend::sequence_t>(in->m_t->m_t);
    shared_ptr<const backend::type_t> sub_t = seq_t->sub().ptr();
    auto leaf = in->get_chunks(cpp_tag(), false).cbegin();
    PyObject* el = deref_zip_sequence(sub_t, leaf, index);
    PyObject* repr = PyObject_Repr(el);
    Py_DECREF(el);
    os << PyString_AsString(repr);
    Py_DECREF(repr);
}

void print_array(sp_cuarray& in, ostream& os) {
    shared_ptr<const backend::sequence_t> seq_t = static_pointer_cast<const backend::sequence_t>(in->m_t->m_t);
    shared_ptr<const backend::type_t> sub_t = seq_t->sub().ptr();
    if (sub_t == backend::int32_mt) {
        sequence<cpp_tag, int> s = make_sequence<sequence<cpp_tag, int> >(in, cpp_tag(), false);
        os << s;
    } else if (sub_t == backend::int64_mt) {
        sequence<cpp_tag, long> s = make_sequence<sequence<cpp_tag, long> >(in, cpp_tag(), false);
        os << s;
    } else if (sub_t == backend::float32_mt) {
        sequence<cpp_tag, float> s = make_sequence<sequence<cpp_tag, float> >(in, cpp_tag(), false);
        os << s;
    } else if (sub_t == backend::float64_mt) {
        sequence<cpp_tag, double> s = make_sequence<sequence<cpp_tag, double> >(in, cpp_tag(), false);
        os << s;
    } else if (sub_t == backend::bool_mt) {
        sequence<cpp_tag, bool> s = make_sequence<sequence<cpp_tag, bool> >(in, cpp_tag(), false);
        os << s;
    } else if (backend::detail::isinstance<backend::tuple_t>(*sub_t)) {
        os << "[";
        size_t length = in->m_l[0];
        for(size_t i = 0; i < length; i++) {
            print_zip_element(in, i, os);
            if (i + 1 < length) {
                os << ", ";
            }
        }
        os << "]";
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
    boost::apply_visitor(tp, *(in->m_t->m_t));
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

bool clean(cuarray& in, boost::python::object place) {
    boost::python::object bpo_tag = place.attr("tag")();
    //Extra parentheses here are for C++11/boost::variant WAR needed
    //for Boost 1.53.0
    copperhead::system_variant tag =
        boost::python::extract<copperhead::system_variant>(bpo_tag)();
    return in.clean(tag);
}

void cuarray_copy(cuarray& in, boost::python::object place, bool invalidate) {
    boost::python::object bpo_tag = place.attr("tag")();
    //Extra parentheses here are for C++11/boost::variant WAR needed
    //for Boost 1.53.0
    copperhead::system_variant tag = boost::python::extract<copperhead::system_variant>(bpo_tag)();
    in.get_chunks(tag, invalidate);
}

sp_cuarray force(sp_cuarray &in, boost::python::object place) {
    cuarray_copy(*in, place, false);
    //XXX What we really need here is an event system
    //This is much too big a hammer
#ifdef CUDA_SUPPORT
    cudaThreadSynchronize();
#endif
    return in;
}

}

BOOST_PYTHON_MODULE(cudata) {
    using namespace boost::python;
    using namespace copperhead;
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
    def("take_down", &take_down);
    def("force", &force);
}
