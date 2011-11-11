#include <boost/python.hpp>

#include "compiler.hpp"
#include "wrappers.hpp"
#include "cuda_printer.hpp"

namespace backend {
template<class T>
T* get_pointer(std::shared_ptr<T> const &p) {
    return p.get();
}


std::string compile(std::shared_ptr<compiler> &c,
                    std::shared_ptr<suite_wrap> &s) {
    std::shared_ptr<suite> rewritten = c->operator()(*s);
    std::cout << "Done compiling" << std::endl;
    std::string entry_point = c->entry_point();
    std::ostringstream os;
    backend::cuda_printer p(entry_point, c->reg(), os);
    p(*rewritten);
    return os.str();
}

}

using namespace boost::python;
using namespace backend;

BOOST_PYTHON_MODULE(compiler) {
    class_<compiler, std::shared_ptr<compiler> >("Compiler", init<std::string>())
        .def("__call__", &compile);
}
