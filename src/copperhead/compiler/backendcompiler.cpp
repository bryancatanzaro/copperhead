#include <boost/python.hpp>

#include "compiler.hpp"
#include "wrappers.hpp"
#include "cuda_printer.hpp"
#include "type_printer.hpp"
#include "utility/up_get.hpp"
#include "utility/isinstance.hpp"

using std::string;
using std::shared_ptr;
using std::ostringstream;
using boost::python::list;

namespace backend {
template<class T>
T* get_pointer(shared_ptr<T> const &p) {
    return p.get();
}


string compile(compiler &c,
               suite_wrap &s) {
    //Compile
    shared_ptr<suite> result = c(s);
    string entry_point = c.entry_point();
    ostringstream os;
    //Convert to string
    backend::cuda_printer p(entry_point, c.reg(), os);
    p(*result);
    string device_code = os.str();
    return device_code;
}

string wrap_name(compiler &c) {
    shared_ptr<procedure> p = c.p_wrap_decl();
    return p->id().id();
}

string wrap_result_type(compiler &c) {
    shared_ptr<procedure> p = c.p_wrap_decl();
    ostringstream os;
    backend::ctype::ctype_printer cp(os);
    const ctype::type_t& n_t = p->ctype(); 
    assert(detail::isinstance<ctype::monotype_t>(n_t));
    const ctype::monotype_t& n_mt =
        detail::up_get<ctype::monotype_t>(n_t);
    assert(string("void") !=
           n_mt.name());
    assert(detail::isinstance<ctype::fn_t>(n_t));
    const ctype::fn_t &n_f = boost::get<const ctype::fn_t&>(n_t);
    const ctype::type_t& ret_t = n_f.result();
    boost::apply_visitor(cp, ret_t);
    return os.str();
}

list wrap_arg_types(compiler &c) {
    shared_ptr<procedure> p = c.p_wrap_decl();
    ostringstream os;
    backend::ctype::ctype_printer cp(os);
    list result;
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

list wrap_arg_names(compiler &c) {
    shared_ptr<procedure> p = c.p_wrap_decl();
    list result;
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
    
    class_<compiler, shared_ptr<compiler> >("Compiler", init<string>())
        .def("__call__", &compile)
        .def("wrap_name", &wrap_name)
        .def("wrap_result_type", &wrap_result_type)
        .def("wrap_arg_types", &wrap_arg_types)
        .def("wrap_arg_names", &wrap_arg_names);
    
    
}
