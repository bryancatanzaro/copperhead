#include <boost/python.hpp>
#include <boost/python/implicit.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/make_constructor.hpp>
#include <boost/variant.hpp>
#include <boost/python/copy_const_reference.hpp>
#include <boost/python/return_value_policy.hpp>
#include "node.hpp"
#include "expression.hpp"
#include "py_printer.hpp"
#include "repr_printer.hpp"
#include "cuda_printer.hpp"

#include "wrappers.hpp"

#include <sstream>
#include <iostream>

namespace backend {

template<class T>
T* get_pointer(std::shared_ptr<T> const &p) {
    return p.get();
}


template<class T, class P=py_printer>
const std::string to_string(std::shared_ptr<T> &p) {
    std::ostringstream os;
    P pp(os);
    pp(*p);
    return os.str();
}

template<class T, class P=py_printer>
const std::string to_string_apply(std::shared_ptr<T> &p) {
    std::ostringstream os;
    P pp(os);
    boost::apply_visitor(pp, *p);
    return os.str();
}


template<class T>
const std::string str(std::shared_ptr<T> &p) {
    return to_string<T, py_printer>(p);
}
    
template<class T>
const std::string repr(std::shared_ptr<T> &p) {
    return to_string<T, repr_printer>(p);
}

template<class T>
const std::string str_apply(std::shared_ptr<T> &p) {
    return to_string_apply<T, py_printer>(p);
}

template<class T>
const std::string repr_apply(std::shared_ptr<T> &p) {
    return to_string_apply<T, repr_printer>(p);
}



}
using namespace boost::python;
using namespace backend;

template<typename S, typename T>
static std::shared_ptr<T> make_from_list(list vals) {
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



BOOST_PYTHON_MODULE(coresyntax) {
    class_<node, std::shared_ptr<node>, boost::noncopyable>("Node", no_init)
        .def("__str__", &backend::str_apply<node>)
        .def("__repr__", &backend::repr_apply<node>);
    class_<expression, std::shared_ptr<expression>, bases<node>, boost::noncopyable>("Expression", no_init)
        .def("__str__", &backend::str_apply<expression>)
        .def("__repr__", &backend::repr_apply<expression>);
    class_<literal_wrap, std::shared_ptr<literal_wrap>, bases<expression, node> >("Literal", init<std::string>())
        .def("id", &literal_wrap::id)
        .def("__str__", &backend::str<literal_wrap>)
        .def("__repr__", &backend::repr<literal_wrap>)
        .add_property("type", &literal_wrap::p_type, &literal_wrap::set_type);
    //.add_property("ctype", &number_wrap::p_ctype, &number_wrap::set_ctype);
    class_<name_wrap, std::shared_ptr<name_wrap>, bases<expression, node> >("Name", init<std::string>())
        .def("id", &name_wrap::id)
        .def("__str__", &backend::str<name_wrap>)
        .def("__repr__", &backend::repr<name_wrap>)
        .add_property("type", &name_wrap::p_type, &name_wrap::set_type);
        //.add_property("ctype", &name_wrap::p_ctype, &name_wrap::set_ctype);
    class_<backend::tuple_wrap, std::shared_ptr<backend::tuple_wrap>, bases<expression, node> >("Tuple")
        .def("__init__", make_constructor(make_from_list<expression, backend::tuple_wrap>))
        .def("__iter__", range(&tuple_wrap::p_begin, &tuple_wrap::p_end))
        .def("__str__", &backend::str<backend::tuple_wrap>)
        .def("__repr__", &backend::repr<backend::tuple_wrap>)
        .add_property("type", &tuple_wrap::p_type, &tuple_wrap::set_type);
        //.add_property("ctype", &tuple_wrap::p_ctype, &tuple_wrap::set_ctype);
    
    class_<apply_wrap, std::shared_ptr<apply_wrap>, bases<expression, node> >("Apply", init<std::shared_ptr<name>, std::shared_ptr<backend::tuple> >())
        .def("fn", &apply_wrap::p_fn) 
        .def("args", &apply_wrap::p_args)
        .def("__str__", &backend::str<apply_wrap>)
        .def("__repr__", &backend::repr<apply_wrap>);
    class_<lambda_wrap, std::shared_ptr<lambda_wrap>, bases<expression, node> >("Lambda", init<std::shared_ptr<backend::tuple>, std::shared_ptr<expression> >())
        .def("args", &lambda_wrap::p_args)
        .def("body", &lambda_wrap::p_body)
        .def("__str__", &backend::str<lambda_wrap>)
        .def("__repr__", &backend::repr<lambda_wrap>)
        .add_property("type", &lambda_wrap::p_type, &lambda_wrap::set_type);
    class_<closure_wrap, std::shared_ptr<closure_wrap>, bases<expression, node> >("Closure", init<std::shared_ptr<backend::tuple>, std::shared_ptr<expression> >())
        .def("args", &closure_wrap::p_args)
        .def("body", &closure_wrap::p_body)
        .def("__str__", &backend::str<closure_wrap>)
        .def("__repr__", &backend::repr<closure_wrap>)
        .add_property("type", &closure_wrap::p_type, &closure_wrap::set_type);
    class_<statement, std::shared_ptr<statement>, bases<node>, boost::noncopyable>("Statement", no_init)
        .def("__str__", &backend::str_apply<statement>)
        .def("__repr__", &backend::repr_apply<statement>);
    class_<ret_wrap, std::shared_ptr<ret_wrap>, bases<statement, node> >("Return", init<std::shared_ptr<expression> >())
        .def("val", &ret_wrap::p_val)
        .def("__str__", &backend::str<ret_wrap>)
        .def("__repr__", &backend::repr<ret_wrap>);
    class_<bind_wrap, std::shared_ptr<bind_wrap>, bases<statement, node> >("Bind", init<std::shared_ptr<expression>, std::shared_ptr<expression> >())
        .def("lhs", &bind_wrap::p_lhs)
        .def("rhs", &bind_wrap::p_rhs)
        .def("__str__", &backend::str<bind_wrap>)
        .def("__repr__", &backend::repr<bind_wrap>);
    class_<procedure_wrap, std::shared_ptr<procedure_wrap>, bases<statement, node> >("Procedure", init<std::shared_ptr<name>, std::shared_ptr<backend::tuple>, std::shared_ptr<suite> >())
        .def("id", &procedure_wrap::p_id)
        .def("args", &procedure_wrap::p_args)
        .def("stmts", &procedure_wrap::p_stmts)
        .def("__str__", &backend::str<procedure_wrap>)
        .def("__repr__", &backend::repr<procedure_wrap>)
        .add_property("type", &procedure_wrap::p_type, &procedure_wrap::set_type);
        //.add_property("ctype", &procedure_wrap::p_ctype, &procedure_wrap::set_ctype);
    class_<conditional_wrap, std::shared_ptr<conditional_wrap>, bases<statement, node> >("Conditional", init<std::shared_ptr<expression>, std::shared_ptr<suite>, std::shared_ptr<suite> >())
        .def("cond", &conditional_wrap::p_cond)
        .def("then", &conditional_wrap::p_then)
        .def("orelse", &conditional_wrap::p_orelse)
        .def("__str__", &backend::str<conditional_wrap>)
        .def("__repr__", &backend::repr<conditional_wrap>);
    class_<suite_wrap, std::shared_ptr<suite_wrap>, bases<node> >("Suite")
        .def("__init__", make_constructor(make_from_list<statement, suite_wrap>))
        .def("__iter__", range(&suite_wrap::p_begin, &suite_wrap::p_end))
        .def("__str__", &backend::str<backend::suite_wrap>)
        .def("__repr__", &backend::repr<backend::suite_wrap>);
    class_<structure_wrap, std::shared_ptr<structure_wrap>, bases<statement, node> >("Structure", init<std::shared_ptr<name>, std::shared_ptr<suite> >())
        .def("id", &structure_wrap::p_id)
        .def("stmts", &structure_wrap::p_stmts)
        .def("__str__", &backend::str<structure_wrap>)
        .def("__repr__", &backend::repr<structure_wrap>);
    

   
    
    implicitly_convertible<std::shared_ptr<backend::expression>, std::shared_ptr<backend::node> >();
    implicitly_convertible<std::shared_ptr<backend::literal>, std::shared_ptr<backend::node> >();
    implicitly_convertible<std::shared_ptr<backend::literal>, std::shared_ptr<backend::expression> >();
    implicitly_convertible<std::shared_ptr<backend::literal_wrap>, std::shared_ptr<backend::node> >();
    implicitly_convertible<std::shared_ptr<backend::literal_wrap>, std::shared_ptr<backend::expression> >();
    implicitly_convertible<std::shared_ptr<backend::literal_wrap>, std::shared_ptr<backend::literal> >();
    implicitly_convertible<std::shared_ptr<backend::name>, std::shared_ptr<backend::node> >();
    implicitly_convertible<std::shared_ptr<backend::name>, std::shared_ptr<backend::expression> >();
    implicitly_convertible<std::shared_ptr<backend::name>, std::shared_ptr<backend::literal> >();
    implicitly_convertible<std::shared_ptr<backend::name_wrap>, std::shared_ptr<backend::node> >();
    implicitly_convertible<std::shared_ptr<backend::name_wrap>, std::shared_ptr<backend::expression> >();
    implicitly_convertible<std::shared_ptr<backend::name_wrap>, std::shared_ptr<backend::literal> >();
    implicitly_convertible<std::shared_ptr<backend::name_wrap>, std::shared_ptr<backend::name> >();
    implicitly_convertible<std::shared_ptr<backend::tuple>, std::shared_ptr<backend::node> >();
    implicitly_convertible<std::shared_ptr<backend::tuple>, std::shared_ptr<backend::expression> >();
    implicitly_convertible<std::shared_ptr<backend::tuple_wrap>, std::shared_ptr<backend::node> >();
    implicitly_convertible<std::shared_ptr<backend::tuple_wrap>, std::shared_ptr<backend::expression> >();
    implicitly_convertible<std::shared_ptr<backend::tuple_wrap>, std::shared_ptr<backend::tuple> >();
    implicitly_convertible<std::shared_ptr<backend::apply>, std::shared_ptr<backend::node> >();
    implicitly_convertible<std::shared_ptr<backend::apply>, std::shared_ptr<backend::expression> >();
    implicitly_convertible<std::shared_ptr<backend::apply_wrap>, std::shared_ptr<backend::node> >();
    implicitly_convertible<std::shared_ptr<backend::apply_wrap>, std::shared_ptr<backend::expression> >();
    implicitly_convertible<std::shared_ptr<backend::apply_wrap>, std::shared_ptr<backend::apply> >();
    implicitly_convertible<std::shared_ptr<backend::lambda>, std::shared_ptr<backend::node> >();
    implicitly_convertible<std::shared_ptr<backend::lambda>, std::shared_ptr<backend::expression> >();
    implicitly_convertible<std::shared_ptr<backend::lambda_wrap>, std::shared_ptr<backend::node> >();
    implicitly_convertible<std::shared_ptr<backend::lambda_wrap>, std::shared_ptr<backend::expression> >();
    implicitly_convertible<std::shared_ptr<backend::lambda_wrap>, std::shared_ptr<backend::lambda> >();
    implicitly_convertible<std::shared_ptr<backend::closure>, std::shared_ptr<backend::node> >();
    implicitly_convertible<std::shared_ptr<backend::closure>, std::shared_ptr<backend::expression> >();
    implicitly_convertible<std::shared_ptr<backend::closure_wrap>, std::shared_ptr<backend::node> >();
    implicitly_convertible<std::shared_ptr<backend::closure_wrap>, std::shared_ptr<backend::expression> >();
    implicitly_convertible<std::shared_ptr<backend::closure_wrap>, std::shared_ptr<backend::closure> >();
    implicitly_convertible<std::shared_ptr<backend::closure>, std::shared_ptr<backend::expression> >();
    implicitly_convertible<std::shared_ptr<backend::statement>, std::shared_ptr<backend::node> >();
    implicitly_convertible<std::shared_ptr<backend::ret>, std::shared_ptr<backend::statement> >();
    implicitly_convertible<std::shared_ptr<backend::ret>, std::shared_ptr<backend::node> >();
    implicitly_convertible<std::shared_ptr<backend::ret_wrap>, std::shared_ptr<backend::statement> >();
    implicitly_convertible<std::shared_ptr<backend::ret_wrap>, std::shared_ptr<backend::node> >();
    implicitly_convertible<std::shared_ptr<backend::ret_wrap>, std::shared_ptr<backend::ret> >();
    implicitly_convertible<std::shared_ptr<backend::bind>, std::shared_ptr<backend::statement> >();
    implicitly_convertible<std::shared_ptr<backend::bind>, std::shared_ptr<backend::node> >();
    implicitly_convertible<std::shared_ptr<backend::bind_wrap>, std::shared_ptr<backend::statement> >();
    implicitly_convertible<std::shared_ptr<backend::bind_wrap>, std::shared_ptr<backend::node> >();
    implicitly_convertible<std::shared_ptr<backend::bind_wrap>, std::shared_ptr<backend::bind> >();
    implicitly_convertible<std::shared_ptr<backend::procedure>, std::shared_ptr<backend::statement> >();
    implicitly_convertible<std::shared_ptr<backend::procedure>, std::shared_ptr<backend::node> >();
    implicitly_convertible<std::shared_ptr<backend::procedure_wrap>, std::shared_ptr<backend::statement> >();
    implicitly_convertible<std::shared_ptr<backend::procedure_wrap>, std::shared_ptr<backend::node> >();
    implicitly_convertible<std::shared_ptr<backend::procedure_wrap>, std::shared_ptr<backend::procedure> >();
    implicitly_convertible<std::shared_ptr<backend::suite>, std::shared_ptr<backend::node> >();
    implicitly_convertible<std::shared_ptr<backend::suite_wrap>, std::shared_ptr<backend::node> >();
    implicitly_convertible<std::shared_ptr<backend::suite_wrap>, std::shared_ptr<backend::suite> >();
    implicitly_convertible<std::shared_ptr<backend::structure>, std::shared_ptr<backend::node> >();
    implicitly_convertible<std::shared_ptr<backend::structure>, std::shared_ptr<backend::statement> >();
    implicitly_convertible<std::shared_ptr<backend::structure_wrap>, std::shared_ptr<backend::node> >();
    implicitly_convertible<std::shared_ptr<backend::structure_wrap>, std::shared_ptr<backend::statement> >();
    implicitly_convertible<std::shared_ptr<backend::structure_wrap>, std::shared_ptr<backend::structure> >();
    implicitly_convertible<std::shared_ptr<backend::conditional>, std::shared_ptr<backend::node> >();
    implicitly_convertible<std::shared_ptr<backend::conditional>, std::shared_ptr<backend::statement> >();
    implicitly_convertible<std::shared_ptr<backend::conditional_wrap>, std::shared_ptr<backend::node> >();
    implicitly_convertible<std::shared_ptr<backend::conditional_wrap>, std::shared_ptr<backend::statement> >();
    implicitly_convertible<std::shared_ptr<backend::conditional_wrap>, std::shared_ptr<backend::conditional> >();
}
