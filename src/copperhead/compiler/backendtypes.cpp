#include <boost/python.hpp>
#include <boost/python/implicit.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/make_constructor.hpp>
#include <boost/variant.hpp>
#include <boost/python/copy_const_reference.hpp>
#include <boost/python/return_value_policy.hpp>
#include "type_printer.hpp"
#include "utility/isinstance.hpp"
#include "raw_constructor.hpp"



namespace backend {

template<class T>
T* get_pointer(std::shared_ptr<T> const &p) {
    return p.get();
}


template<class T, class P=repr_type_printer>
const std::string to_string(std::shared_ptr<T> &p) {
    std::ostringstream os;
    P pp(os);
    pp(*p);
    return os.str();
}

template<class T, class P=repr_type_printer>
const std::string to_string_apply(std::shared_ptr<T> &p) {
    std::ostringstream os;
    P pp(os);
    boost::apply_visitor(pp, *p);
    return os.str();
}

template<class T>
const std::string repr(std::shared_ptr<T> &p) {
    return to_string<T, repr_type_printer>(p);
}
template<class T>
const std::string repr_apply(std::shared_ptr<T> &p) {
    return to_string_apply<T, repr_type_printer>(p);
}

}

using namespace boost::python;
using namespace backend;

template<typename S, typename T, typename I>
static std::shared_ptr<T> make_from_iterable(const I& vals) {
    std::vector<std::shared_ptr<S> > values;
    boost::python::ssize_t n = len(vals);
    for(boost::python::ssize_t i=0; i<n; i++) {
        object elem = vals[i];
        std::shared_ptr<S> p_elem = extract<std::shared_ptr<S> >(elem);
        values.push_back(p_elem);
    }
    auto result = std::shared_ptr<T>(new T(std::move(values)));
    return result;
}

template<typename S, typename T>
static std::shared_ptr<T> make_from_list(boost::python::list vals) {
    return make_from_iterable<S, T, boost::python::list>(vals);
}

template<typename S, typename T>
static std::shared_ptr<T> make_from_tuple(boost::python::tuple vals) {
    return make_from_iterable<S, T, boost::python::tuple>(vals);
}

template <typename S, typename T>
static std::shared_ptr<T> make_from_args(boost::python::tuple args,
                                         boost::python::dict kwargs) {
    if(len(args) == 1) {
        boost::python::object first_arg = args[0];
        if (PyList_Check(first_arg.ptr())) {
            return make_from_list<S, T>(list(args[0]));
        } else if (PyTuple_Check(first_arg.ptr())) {
            return make_from_tuple<S, T>(boost::python::tuple(args[0]));
        }
    }
    return make_from_iterable<S, T, boost::python::tuple>(args);
}

static std::shared_ptr<polytype_t> make_polytype(list vals,
                                                 std::shared_ptr<monotype_t> sub) {
    std::vector<std::shared_ptr<monotype_t> > values;
    boost::python::ssize_t n = len(vals);
    for(boost::python::ssize_t i=0; i<n; i++) {
        object elem = vals[i];
        std::shared_ptr<monotype_t> p_elem = extract<std::shared_ptr<monotype_t> >(elem);
        values.push_back(p_elem);
    }
    auto result = std::make_shared<polytype_t>(std::move(values), sub);
    return result;
}

std::shared_ptr<monotype_t> retrieve_monotype_t(const std::shared_ptr<type_t>& in) {
    return std::static_pointer_cast<monotype_t>(in);
}
std::shared_ptr<polytype_t> retrieve_polytype_t(const std::shared_ptr<type_t>& in) {
    return std::static_pointer_cast<polytype_t>(in);
}
std::shared_ptr<sequence_t> retrieve_sequence_t(const std::shared_ptr<type_t>& in) {
    return std::static_pointer_cast<sequence_t>(in);
}
std::shared_ptr<tuple_t> retrieve_tuple_t(const std::shared_ptr<type_t>& in) {
    return std::static_pointer_cast<tuple_t>(in);
}
std::shared_ptr<fn_t> retrieve_fn_t(const std::shared_ptr<type_t>& in) {
    return std::static_pointer_cast<fn_t>(in);
}

int which(const std::shared_ptr<type_t>& in) {
    return in->which();
}

BOOST_PYTHON_MODULE(backendtypes) {
    def("which", &which);
    def("retrieve_monotype_t", &retrieve_monotype_t);
    def("retrieve_polytype_t", &retrieve_polytype_t);
    def("retrieve_sequence_t", &retrieve_sequence_t);
    def("retrieve_tuple_t", &retrieve_tuple_t);
    def("retrieve_fn_t", &retrieve_fn_t);
    class_<type_t, std::shared_ptr<type_t>, boost::noncopyable >("Type", no_init)
        .def("__repr__", &backend::repr_apply<type_t>);
    class_<monotype_t, std::shared_ptr<monotype_t>, bases<type_t> >("Monotype", init<std::string>())
        .def("__repr__", &backend::repr<monotype_t>);
    class_<polytype_t, std::shared_ptr<polytype_t>, bases<type_t> >("Polytype", no_init)
        .def("__init__", make_constructor(make_polytype))
        .def("__repr__", &backend::repr<polytype_t>);
    scope().attr("Int32") = backend::int32_mt;
    scope().attr("Int64") = backend::int64_mt;
    scope().attr("Uint32") = backend::uint32_mt;
    scope().attr("Uint64") = backend::uint64_mt;
    scope().attr("Float32") = backend::float32_mt;
    scope().attr("Float64") = backend::float64_mt;
    scope().attr("Bool") = backend::bool_mt;
    scope().attr("Void") = backend::void_mt;
    class_<sequence_t, std::shared_ptr<sequence_t>, bases<monotype_t, type_t> >("Sequence", init<std::shared_ptr<type_t> >())
        .def("__repr__", &backend::repr<sequence_t>)
        .def("sub", &backend::sequence_t::p_sub);
    class_<tuple_t, std::shared_ptr<tuple_t>, bases<monotype_t, type_t> >("Tuple", no_init)
        .def("__init__", raw_constructor(make_from_args<type_t, tuple_t>))
        .def("__iter__", range(&tuple_t::p_begin, &tuple_t::p_end))
        .def("__repr__", &backend::repr<tuple_t>);
    class_<fn_t, std::shared_ptr<fn_t>, bases<monotype_t, type_t> >("Fn", init<std::shared_ptr<tuple_t>, std::shared_ptr<type_t> >())
        .def("__repr__", &backend::repr<fn_t>);
    implicitly_convertible<std::shared_ptr<backend::monotype_t>, std::shared_ptr<backend::type_t> >();
    implicitly_convertible<std::shared_ptr<backend::polytype_t>, std::shared_ptr<backend::type_t> >();
    implicitly_convertible<std::shared_ptr<backend::sequence_t>, std::shared_ptr<backend::type_t> >();
    implicitly_convertible<std::shared_ptr<backend::sequence_t>, std::shared_ptr<backend::monotype_t> >();
    implicitly_convertible<std::shared_ptr<backend::tuple_t>, std::shared_ptr<backend::type_t> >();
    implicitly_convertible<std::shared_ptr<backend::tuple_t>, std::shared_ptr<backend::monotype_t> >();
    implicitly_convertible<std::shared_ptr<backend::fn_t>, std::shared_ptr<backend::type_t> >();
    implicitly_convertible<std::shared_ptr<backend::fn_t>, std::shared_ptr<backend::monotype_t> >();
}
