#include <boost/python.hpp>

#include "compiler.hpp"
#include "wrappers.hpp"
#include "cuda_printer.hpp"
#include "type_printer.hpp"
#include "utility/up_get.hpp"
#include "utility/isinstance.hpp"

namespace backend {
template<class T>
T* get_pointer(std::shared_ptr<T> const &p) {
    return p.get();
}


std::string compile(std::shared_ptr<compiler> &c,
                    std::shared_ptr<suite_wrap> &s) {
    //Compile
    std::shared_ptr<suite> result = c->operator()(*s);
    std::string entry_point = c->entry_point();
    std::ostringstream os;
    //Convert to string
    backend::cuda_printer p(entry_point, c->reg(), os);
    p(*result);
    std::string device_code = os.str();
    return device_code;
}

std::string wrap_name(std::shared_ptr<compiler> &c) {
    std::shared_ptr<procedure> p = c->p_wrap_decl();
    return p->id().id();
}

std::string wrap_result_type(std::shared_ptr<compiler> &c) {
    std::shared_ptr<procedure> p = c->p_wrap_decl();
    std::ostringstream os;
    backend::ctype::ctype_printer cp(os);
    const ctype::type_t& n_t = p->ctype(); 
    assert(detail::isinstance<ctype::monotype_t>(n_t));
    const ctype::monotype_t& n_mt =
        detail::up_get<ctype::monotype_t>(n_t);
    assert(std::string("void") !=
           n_mt.name());
    assert(detail::isinstance<ctype::fn_t>(n_t));
    const ctype::fn_t &n_f = boost::get<const ctype::fn_t&>(n_t);
    const ctype::type_t& ret_t = n_f.result();
    boost::apply_visitor(cp, ret_t);
    return os.str();
}

boost::python::list wrap_arg_types(std::shared_ptr<compiler> &c) {
    std::shared_ptr<procedure> p = c->p_wrap_decl();
    std::ostringstream os;
    backend::ctype::ctype_printer cp(os);
    boost::python::list result;
    const backend::tuple &args = p->args();
    for(auto i = args.begin();
        i != args.end();
        i++) {
        boost::apply_visitor(cp, i->ctype());
        result.append(os.str());
        os.str("");
    }
    return result;   
}

boost::python::list wrap_arg_names(std::shared_ptr<compiler> &c) {
    std::shared_ptr<procedure> p = c->p_wrap_decl();
    boost::python::list result;
    const backend::tuple& args = p->args();
    for(auto i = args.begin();
        i != args.end();
        i++) {
        assert(detail::isinstance<backend::name>(*i));
        const backend::name& arg = boost::get<const backend::name&>(*i);
        result.append(arg.id());
    }
    return result;
}
}

using namespace boost::python;
using namespace backend;

BOOST_PYTHON_MODULE(backendcompiler) {
    class_<compiler, std::shared_ptr<compiler> >("Compiler", init<std::string>())
        .def("__call__", &compile)
        .def("wrap_name", &wrap_name)
        .def("wrap_result_type", &wrap_result_type)
        .def("wrap_arg_types", &wrap_arg_types)
        .def("wrap_arg_names", &wrap_arg_names);
    
    
}
