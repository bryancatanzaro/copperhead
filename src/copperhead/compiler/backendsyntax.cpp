#include <boost/python.hpp>
#include <boost/python/make_constructor.hpp>
#include <boost/variant.hpp>
#include <boost/python/copy_const_reference.hpp>
#include <boost/python/return_value_policy.hpp>
#include "node.hpp"
#include "expression.hpp"
#include "py_printer.hpp"
#include "repr_printer.hpp"
#include "cpp_printer.hpp"
#include "shared_ptr_util.hpp"
#include "auxiliary_constructors.hpp"
#include "to_string.hpp"
#include "register_conversions.hpp"

#include <sstream>
#include <iostream>

namespace backend {

template<class T>
const std::string str(std::shared_ptr<T> &p) {
    return to_string<T, py_printer>(p);
}
    
template<class T>
const std::string repr(std::shared_ptr<T> &p) {
    return to_string<T, repr_printer>(p);
}

}
using namespace boost::python;
using namespace backend;
    
BOOST_PYTHON_MODULE(backendsyntax) {
    class_<node, std::shared_ptr<node>, boost::noncopyable>("Node", no_init)
        .def("__str__", &backend::str<node>)
        .def("__repr__", &backend::repr<node>);
    
    class_<expression, std::shared_ptr<expression>, bases<node>, boost::noncopyable>("Expression", no_init)
        .def("__str__", &backend::str<expression>)
        .def("__repr__", &backend::repr<expression>);
    
    class_<literal, std::shared_ptr<literal>, bases<expression, node> >("Literal", init<std::string, std::shared_ptr<const type_t> >())
        .def("__str__", &backend::str<literal>)
        .def("__repr__", &backend::repr<literal>);
    
    class_<name, std::shared_ptr<name>, bases<expression, node> >("Name", init<std::string, std::shared_ptr<const type_t> >())
        .def("__str__", &backend::str<name>)
        .def("__repr__", &backend::repr<name>);
    
    class_<backend::tuple, std::shared_ptr<backend::tuple>, bases<expression, node> >("Tuple", no_init)
        .def("__init__", make_constructor(&backend::make_from_iterable<
                                              backend::expression,
                                              backend::tuple,
                                              boost::python::list,
                                              backend::type_t>))
        .def("__str__", &backend::str<backend::tuple>)
        .def("__repr__", &backend::repr<backend::tuple>);
    
    class_<apply, std::shared_ptr<apply>, bases<expression, node> >("Apply", init<std::shared_ptr<const name>, std::shared_ptr<const backend::tuple> >())
        .def("__str__", &backend::str<apply>)
        .def("__repr__", &backend::repr<apply>);
                                                                  
    class_<lambda, std::shared_ptr<lambda>, bases<expression, node> >("Lambda", init<std::shared_ptr<const backend::tuple>, std::shared_ptr<const expression>, std::shared_ptr<const type_t> >())
        .def("__str__", &backend::str<lambda>)
        .def("__repr__", &backend::repr<lambda>);
                                                                  
    class_<closure, std::shared_ptr<closure>, bases<expression, node> >("Closure", init<std::shared_ptr<const backend::tuple>, std::shared_ptr<const expression>, std::shared_ptr<const type_t> >())
        .def("__str__", &backend::str<closure>)
        .def("__repr__", &backend::repr<closure>);

    class_<subscript, std::shared_ptr<subscript>, bases<expression, node> >("Subscript", init<std::shared_ptr<const backend::name>, std::shared_ptr<const expression>, std::shared_ptr<const type_t> >())
        .def("__str__", &backend::str<subscript>)
        .def("__repr__", &backend::repr<subscript>);

    class_<statement, std::shared_ptr<statement>, bases<node>, boost::noncopyable>("Statement", no_init)
        .def("__str__", &backend::str<statement>)
        .def("__repr__", &backend::repr<statement>);

    class_<ret, std::shared_ptr<ret>, bases<statement, node> >("Return", init<std::shared_ptr<const expression> >())
        .def("__str__", &backend::str<ret>)
        .def("__repr__", &backend::repr<ret>);
    
    class_<bind, std::shared_ptr<bind>, bases<statement, node> >("Bind", init<std::shared_ptr<const expression>, std::shared_ptr<expression> >())
        .def("__str__", &backend::str<bind>)
        .def("__repr__", &backend::repr<bind>);
    
    class_<procedure, std::shared_ptr<procedure>, bases<statement, node> >("Procedure", init<std::shared_ptr<const name>, std::shared_ptr<const backend::tuple>, std::shared_ptr<const suite> , std::shared_ptr<const type_t> >())
        .def("__str__", &backend::str<procedure>)
        .def("__repr__", &backend::repr<procedure>);
    
    class_<conditional, std::shared_ptr<conditional>, bases<statement, node> >("Cond", init<std::shared_ptr<const expression>, std::shared_ptr<const suite>, std::shared_ptr<const suite> >())
        .def("__str__", &backend::str<conditional>)
        .def("__repr__", &backend::repr<conditional>);
    
    class_<suite, std::shared_ptr<suite>, bases<node> >("Suite", no_init)
        .def("__init__", make_constructor(&backend::make_from_iterable<
                                              backend::statement,
                                              backend::suite,
                                              boost::python::list>))
        .def("__str__", &backend::str<backend::suite>)
        .def("__repr__", &backend::repr<backend::suite>);
    
    class_<structure, std::shared_ptr<structure>, bases<statement, node> >("Structure", init<std::shared_ptr<const name>, std::shared_ptr<const suite> >())
        .def("__str__", &backend::str<structure>)
        .def("__repr__", &backend::repr<structure>);

    register_conversions<backend::expression, backend::node>();

    register_conversions<backend::literal, backend::node>();
    register_conversions<backend::literal, backend::expression>();

    register_conversions<backend::name, backend::node>();
    register_conversions<backend::name, backend::expression>();
    register_conversions<backend::name, backend::literal>();

    register_conversions<backend::tuple, backend::node>();
    register_conversions<backend::tuple, backend::expression>();

    register_conversions<backend::apply, backend::node>();
    register_conversions<backend::apply, backend::expression>();
    
    register_conversions<backend::lambda, backend::node>();
    register_conversions<backend::lambda, backend::expression>();

    register_conversions<backend::closure, backend::node>();
    register_conversions<backend::closure, backend::expression>();
    
    register_conversions<backend::subscript, backend::node>();
    register_conversions<backend::subscript, backend::expression>();

    register_conversions<backend::statement, backend::node>();

    register_conversions<backend::ret, backend::node>();
    register_conversions<backend::ret, backend::statement>();

    register_conversions<backend::bind, backend::node>();
    register_conversions<backend::bind, backend::statement>();

    register_conversions<backend::procedure, backend::node>();
    register_conversions<backend::procedure, backend::statement>();

    register_conversions<backend::suite, backend::node>();

    register_conversions<backend::structure, backend::node>();
    register_conversions<backend::structure, backend::statement>();

    register_conversions<backend::conditional, backend::node>();
    register_conversions<backend::conditional, backend::statement>();
    
}
