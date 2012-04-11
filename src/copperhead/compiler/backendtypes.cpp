#include <boost/python.hpp>
#include <boost/python/implicit.hpp>
#include <boost/python/make_constructor.hpp>
#include <boost/variant.hpp>
#include <boost/python/copy_const_reference.hpp>
#include <boost/python/return_value_policy.hpp>
#include "type_printer.hpp"
#include "utility/isinstance.hpp"
#include "raw_constructor.hpp"
#include "shared_ptr_util.hpp"
#include "auxiliary_constructors.hpp"
#include "to_string.hpp"
#include "register_conversions.hpp"

namespace backend {

template<class T>
const std::string repr(std::shared_ptr<T> &p) {
    return to_string<T, repr_type_printer>(p);
}

}

using namespace boost::python;
using namespace backend;

std::shared_ptr<monotype_t> retrieve_monotype_t(const std::shared_ptr<type_t>& in) {
    return std::static_pointer_cast<monotype_t>(in);
}
std::shared_ptr<polytype_t> retrieve_polytype_t(std::shared_ptr<type_t>& in) {
    return std::static_pointer_cast<polytype_t>(in);
}
std::shared_ptr<sequence_t> retrieve_sequence_t(std::shared_ptr<type_t>& in) {
    return std::static_pointer_cast<sequence_t>(in);
}
std::shared_ptr<tuple_t> retrieve_tuple_t(std::shared_ptr<type_t>& in) {
    return std::static_pointer_cast<tuple_t>(in);
}
std::shared_ptr<fn_t> retrieve_fn_t(std::shared_ptr<type_t>& in) {
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
        .def("__repr__", &backend::repr<type_t>);
    class_<monotype_t, std::shared_ptr<monotype_t>, bases<type_t> >("Monotype", init<std::string>())
        .def("__repr__", &backend::repr<monotype_t>);
    class_<polytype_t, std::shared_ptr<polytype_t>, bases<type_t> >("Polytype", no_init)
        .def("__init__", make_constructor(&make_from_iterable<
                                          monotype_t,
                                          polytype_t,
                                          boost::python::list,
                                          monotype_t>))
        .def("__repr__", &backend::repr<polytype_t>);

    //The following const_pointer_casts are a workaround for boost::python's poor support
    // of C++ objects held as std::shared_ptr<const T>
    //Treating them as mutable in this context doesn't cause problems because we haven't provided
    //any methods by which to mutate them in Python.
    scope().attr("Int32") = std::const_pointer_cast<monotype_t>(backend::int32_mt);
    scope().attr("Int64") = std::const_pointer_cast<monotype_t>(backend::int64_mt);
    scope().attr("Uint32") = std::const_pointer_cast<monotype_t>(backend::uint32_mt);
    scope().attr("Uint64") = std::const_pointer_cast<monotype_t>(backend::uint64_mt);
    scope().attr("Float32") = std::const_pointer_cast<monotype_t>(backend::float32_mt);
    scope().attr("Float64") = std::const_pointer_cast<monotype_t>(backend::float64_mt);
    scope().attr("Bool") = std::const_pointer_cast<monotype_t>(backend::bool_mt);
    scope().attr("Void") = std::const_pointer_cast<monotype_t>(backend::void_mt);


    class_<sequence_t, std::shared_ptr<sequence_t>, bases<monotype_t, type_t> >("Sequence", init<std::shared_ptr<type_t> >())
        .def("__repr__", &backend::repr<sequence_t>);
    class_<tuple_t, std::shared_ptr<tuple_t>, bases<monotype_t, type_t> >("Tuple", no_init)
        .def("__init__", raw_constructor(make_from_args<type_t, tuple_t>))
        .def("__repr__", &backend::repr<tuple_t>);
    class_<fn_t, std::shared_ptr<fn_t>, bases<monotype_t, type_t> >("Fn", init<std::shared_ptr<tuple_t>, std::shared_ptr<type_t> >())
        .def("__repr__", &backend::repr<fn_t>);

    register_conversions<backend::monotype_t, backend::type_t>();
    register_conversions<backend::polytype_t, backend::type_t>();
    register_conversions<backend::sequence_t, backend::type_t>();
    register_conversions<backend::sequence_t, backend::monotype_t>();
    register_conversions<backend::tuple_t, backend::type_t>();
    register_conversions<backend::tuple_t, backend::monotype_t>();
    register_conversions<backend::fn_t, backend::type_t>();
    register_conversions<backend::fn_t, backend::monotype_t>();
    
}
