#include <boost/python.hpp>

#include "compiler.hpp"
#include "wrappers.hpp"
#include "cuda_printer.hpp"

namespace backend {
template<class T>
T* get_pointer(std::shared_ptr<T> const &p) {
    return p.get();
}


boost::python::tuple compile(std::shared_ptr<compiler> &c,
                             std::shared_ptr<suite_wrap> &s) {
    //Compile
    c->operator()(*s);
    std::string entry_point = c->entry_point();
    std::ostringstream os;
    //Convert to string
    backend::cuda_printer p(entry_point, c->reg(), os);
    p(*(c->p_device_code()));
    std::string device_code = os.str();
    os.str("");
    p(*(c->p_host_code()));
    std::string host_code = os.str();
    boost::python::list l;
    l.append(device_code);
    l.append(host_code);
    return boost::python::tuple(l);
}

}

using namespace boost::python;
using namespace backend;

BOOST_PYTHON_MODULE(backendcompiler) {
    class_<compiler, std::shared_ptr<compiler> >("Compiler", init<std::string>())
        .def("__call__", &compile)
        .def("host_code", &backend::compiler::p_host_code)
        .def("device_code", &backend::compiler::p_device_code);
}
