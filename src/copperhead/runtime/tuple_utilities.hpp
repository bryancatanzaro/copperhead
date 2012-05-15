#include <prelude/runtime/cunp.hpp>
#include <prelude/runtime/cuarray.hpp>
#include <thrust/tuple.h>

#include <stdexcept>
#include <vector>

//Note: This would be prettier if we could use boost::python
//object/tuple here instead of the Python C API.
//But since boost::python is incompatible with nvcc,
//we can't do so, and must use the lowest common denominator here.

namespace copperhead {

namespace detail {

template<typename T>
struct unpack_tuple_impl;

template<>
struct unpack_tuple_impl<bool> {
    static bool fun(PyObject* o, int i) {
        return unpack_scalar_bool(o);
    }
};

template<>
struct unpack_tuple_impl<int> {
    static int fun(PyObject* o, int i) {
        return unpack_scalar_int(o);
    }
};

template<>
struct unpack_tuple_impl<long> {
    static long fun(PyObject* o, int i) {
        return unpack_scalar_long(o);
    }
};

template<>
struct unpack_tuple_impl<float> {
    static float fun(PyObject* o, int i) {
        return unpack_scalar_float(o);
    }
};

template<>
struct unpack_tuple_impl<double> {
    static double fun(PyObject* o, int i) {
        return unpack_scalar_double(o);
    }
};

template<>
struct unpack_tuple_impl<sp_cuarray> {
    static sp_cuarray fun(PyObject* o, int i) {
        return unpack_array(o);
    }
};

template<typename HT, typename TT>
struct unpack_tuple_impl<thrust::detail::cons<HT, TT> > {
    static thrust::detail::cons<HT, TT> fun(PyObject* t, int i) {
        if (!PyTuple_Check(t)) {
            throw std::invalid_argument("Tuple expected");
        }
                
        return thrust::detail::cons<HT, TT>(
            unpack_tuple_impl<HT>::fun(PyTuple_GetItem(t, i), 0),
            unpack_tuple_impl<TT>::fun(t, i+1));
    }
};

template<typename T0,
         typename T1,
         typename T2,
         typename T3,
         typename T4,
         typename T5,
         typename T6,
         typename T7,
         typename T8,
         typename T9>
struct unpack_tuple_impl<
    thrust::tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> > {
    typedef thrust::tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> result_type;
    static result_type fun(
        PyObject* t, int i) {
        return unpack_tuple_impl<
            thrust::detail::cons<
                typename result_type::head_type,
                typename result_type::tail_type> >::fun(t, i);
        
    }
};
         
         

template<>
struct unpack_tuple_impl<thrust::null_type> {
    static thrust::null_type fun(PyObject* t, int i) {
        return thrust::null_type();
    }
};

}

template<typename Tuple>
Tuple unpack_tuple(PyObject* t) {
    return detail::unpack_tuple_impl<Tuple>::fun(t, 0);
}

namespace detail {

template<typename T>
struct pack_tuple_impl;

template<>
struct pack_tuple_impl<bool> {
    static PyObject* fun(const bool& x) {
        return make_scalar(x);
    }
};

template<>
struct pack_tuple_impl<int> {
    static PyObject* fun(const int& x) {
        return make_scalar(x);
    }
};

template<>
struct pack_tuple_impl<long> {
    static PyObject* fun(const long& x) {
        return make_scalar(x);
    }
};

template<>
struct pack_tuple_impl<float> {
    static PyObject* fun(const float& x) {
        return make_scalar(x);
    }
};

template<>
struct pack_tuple_impl<double> {
    static PyObject* fun(const double& x) {
        return make_scalar(x);
    }
};

template<>
struct pack_tuple_impl<sp_cuarray> {
    static PyObject* fun(const sp_cuarray& x) {
        return objectify(x);
    }
};

template<typename HT, typename TT>
struct pack_tuple_impl<thrust::detail::cons<HT, TT> > {
    static PyObject* fun(const thrust::detail::cons<HT, TT>& x) {
        PyObject* head = pack_tuple_impl<HT>::fun(x.get_head());
        PyObject* tail = pack_tuple_impl<TT>::fun(x.get_tail());
        if (tail == NULL) {
            return PyTuple_Pack(1, head);
        } else {
            assert(PyTuple_Check(tail));
            int tuple_len = 1 + PyTuple_Size(tail);
            PyObject* result = PyTuple_New(tuple_len);
            PyTuple_SetItem(result, 0, head);
            for(int i = 1; i < tuple_len; i++) {
                PyObject* tail_item = PyTuple_GetItem(tail, i-1);
                Py_INCREF(tail_item);
                PyTuple_SetItem(result, i, tail_item);
            }
            Py_DECREF(tail);
            return result;
        }
    }
};

template<typename T0,
         typename T1,
         typename T2,
         typename T3,
         typename T4,
         typename T5,
         typename T6,
         typename T7,
         typename T8,
         typename T9>
struct pack_tuple_impl<
    thrust::tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> > {
    typedef thrust::tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> input_type;
    static PyObject* fun(
        const input_type& t) {
        return pack_tuple_impl<
            thrust::detail::cons<
                typename input_type::head_type,
                typename input_type::tail_type> >::fun(t);
        
    }
};
         
         

template<>
struct pack_tuple_impl<thrust::null_type> {
    static PyObject* fun(thrust::null_type) {
        return NULL;
    }
};

}

template<typename Tuple>
PyObject* make_python_tuple(const Tuple& t) {
    return detail::pack_tuple_impl<Tuple>::fun(t);
}

}
