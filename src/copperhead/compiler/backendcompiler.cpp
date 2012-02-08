#include <boost/python.hpp>

#include "compiler.hpp"
#include "wrappers.hpp"
#include "cuda_printer.hpp"
#include "type_printer.hpp"
#include "utility/up_get.hpp"
#include "utility/isinstance.hpp"

#include "python_wrap.hpp"
#include "namespace_wrap.hpp"



using std::string;
using std::shared_ptr;
using std::static_pointer_cast;
using std::ostringstream;
using boost::python::list;

namespace backend {
template<class T>
T* get_pointer(shared_ptr<T> const &p) {
    return p.get();
}

shared_ptr<procedure> wrapper;
string hash_value;

string compile(compiler &c,
               suite_wrap &s) {
    //Compile
    shared_ptr<suite> result = c(s);
    python_wrap python_wrapper(c.entry_point());
    shared_ptr<suite> wrapped =
        static_pointer_cast<suite>(boost::apply_visitor(python_wrapper, *result));
    wrapper = python_wrapper.wrapper();
    string entry_point = c.entry_point();

    entry_hash entry_hasher(c.entry_point());
    boost::apply_visitor(entry_hasher, *wrapped);
    hash_value = entry_hasher.hash();
    namespace_wrap namespace_wrapper(hash_value);
    shared_ptr<suite> namespaced =
        static_pointer_cast<suite>(boost::apply_visitor(namespace_wrapper, *wrapped));

    
    ostringstream os;
    //Convert to string
    backend::cuda_printer p(entry_point, c.reg(), os);
    boost::apply_visitor(p, *namespaced);
    string device_code = os.str();
        
    return device_code;
}

string wrap_name() {
    return wrapper->id().id();
}

string wrap_result_type() {
    ostringstream os;
    backend::ctype::ctype_printer cp(os);
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
    backend::ctype::ctype_printer cp(os);
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

string hash() {
    return hash_value;
}


}



using namespace boost::python;
using namespace backend;

BOOST_PYTHON_MODULE(backendcompiler) {
    
    class_<compiler, shared_ptr<compiler> >("Compiler", init<string>())
        .def("__call__", &compile);
    def("wrap_name", &wrap_name);
    def("wrap_result_type", &wrap_result_type);
    def("wrap_arg_types", &wrap_arg_types);
    def("wrap_arg_names", &wrap_arg_names);
    def("hash", &hash);
    
}
