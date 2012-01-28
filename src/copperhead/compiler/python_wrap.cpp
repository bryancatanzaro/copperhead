#include "python_wrap.hpp"


using std::string;
using std::vector;
using std::shared_ptr;
using std::make_shared;
using std::static_pointer_cast;
using std::move;
using backend::utility::make_vector;

namespace backend {

python_wrap::python_wrap(const string& entry_point) : m_entry_point(entry_point),
                                             m_wrapping(false),
                                             m_wrapper() {}


python_wrap::result_type python_wrap::operator()(const procedure &n) {
    return this->rewriter::operator()(n);

}
python_wrap::result_type python_wrap::operator()(const ret& n) {
        return this->rewriter::operator()(n);
}
python_wrap::result_type python_wrap::operator()(const suite&n) {
    return this->rewriter::operator()(n);
}


}
