#include <boost/python.hpp>

#include "compiler.hpp"
#include "cpp_printer.hpp"
#include "type_printer.hpp"
#include "utility/up_get.hpp"
#include "utility/isinstance.hpp"

#include "python_wrap.hpp"
#include "namespace_wrap.hpp"

#include <prelude/runtime/tags.h>
#include "shared_ptr_util.hpp"

using std::string;
using std::hash;
using std::shared_ptr;
using std::set;
using std::static_pointer_cast;
using std::ostringstream;
using boost::python::list;

namespace backend {

shared_ptr<const procedure> wrapper;
string hash_value;
copperhead::system_variant target;

string compile(compiler &c,
               const suite &s) {
    //Compile
    target = c.target();
    shared_ptr<const suite> result = c(s);

    python_wrap python_wrapper(c.target(), c.entry_point());
    shared_ptr<const suite> wrapped =
        static_pointer_cast<const suite>(boost::apply_visitor(python_wrapper, *result));
    wrapper = python_wrapper.wrapper();
    const string entry_point = c.entry_point();

    ostringstream os;
    //Convert to string
    backend::cpp_printer p(c.target(), entry_point, c.reg(), os);
    boost::apply_visitor(p, *wrapped);
    string device_code = os.str();
    //Hash code
    hash<string> std_hasher;
    ostringstream hash_os;
    hash_os << entry_point << '_' << std_hasher(device_code);
    hash_value = hash_os.str();
    
    namespace_wrap namespace_wrapper(hash_value);
    shared_ptr<const suite> namespaced =
        static_pointer_cast<const suite>(boost::apply_visitor(namespace_wrapper, *wrapped));
    ostringstream final_os;
    //Convert to string
    backend::cpp_printer final_p(c.target(), entry_point, c.reg(), final_os);
    boost::apply_visitor(final_p, *namespaced);
    string final_code = final_os.str();
    return final_code;
}

string wrap_name() {
    return wrapper->id().id();
}

string wrap_result_type() {
    ostringstream os;
    backend::ctype::ctype_printer cp(target, os);
    const ctype::type_t& n_t = wrapper->ctype(); 
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

list wrap_arg_types() {
    ostringstream os;
    backend::ctype::ctype_printer cp(target, os);
    list result;
    const backend::tuple &args = wrapper->args();
    for(auto i = args.begin();
        i != args.end();
        i++) {
        boost::apply_visitor(cp, i->ctype());
        result.append(os.str());
        os.str("");
    }
    return result;   
}

list wrap_arg_names() {
    list result;
    const backend::tuple& args = wrapper->args();
    for(auto i = args.begin();
        i != args.end();
        i++) {
        assert(detail::isinstance<backend::name>(*i));
        const backend::name& arg = boost::get<const backend::name&>(*i);
        result.append(arg.id());
    }
    return result;
}

string module_hash() {
    return hash_value;
}

}



using namespace boost::python;
using namespace backend;

BOOST_PYTHON_MODULE(backendcompiler) {
    
    class_<compiler, shared_ptr<compiler> >("Compiler", init<string, copperhead::system_variant>())
        .def("__call__", &compile);
    def("wrap_name", &wrap_name);
    def("wrap_result_type", &wrap_result_type);
    def("wrap_arg_types", &wrap_arg_types);
    def("wrap_arg_names", &wrap_arg_names);
    def("hash", &module_hash);
}
