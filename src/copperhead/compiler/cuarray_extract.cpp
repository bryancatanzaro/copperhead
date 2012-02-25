#include "cuarray_extract.hpp"
#include "utility/isinstance.hpp"
using std::string;
using std::ostringstream;
using std::vector;
using std::shared_ptr;
using std::make_shared;
using std::static_pointer_cast;
using std::move;
using std::set;

namespace backend {
cuarray_extract::cuarray_extract(const std::string& entry_point, const registry& reg)
    : m_registry(reg), m_entry_point(entry_point) {}

void cuarray_extract::operator()(const suite& n) {
    for(auto i = n.begin(); i!= n.end(); i++) {
        boost::apply_visitor(*this, *i);
    }
}
void cuarray_extract::operator()(const procedure& n) {
    boost::apply_visitor(*this, n.stmts());
}
void cuarray_extract::operator()(const typedefn& n) {
    if (detail::isinstance<ctype::sequence_t>(n.origin())) {
        std::ostringstream os;
        os << "template sp_cuarray make_cuarray<";
        ctype::ctype_printer tp(os);
        boost::apply_visitor(tp, boost::get<const ctype::sequence_t&>(n.origin()).sub());
        os << ">(size_t);";
        m_extractions.insert(os.str());
    }
}
const set<string>& cuarray_extract::extractions() const {
    return m_extractions;
}

}
