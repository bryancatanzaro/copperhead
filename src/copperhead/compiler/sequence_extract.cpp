#include "sequence_extract.hpp"

using std::string;
using std::ostringstream;
using std::vector;
using std::shared_ptr;
using std::make_shared;
using std::static_pointer_cast;
using std::move;
using std::set;

namespace backend {
sequence_extract::sequence_extract(const std::string& entry_point, const registry& reg)
    : m_registry(reg), m_entry_point(entry_point) {}

void sequence_extract::operator()(const suite& n) {
    for(auto i = n.begin(); i!= n.end(); i++) {
        boost::apply_visitor(*this, *i);
    }
}
void sequence_extract::operator()(const procedure& n) {
    boost::apply_visitor(*this, n.stmts());
}
void sequence_extract::operator()(const bind& n) {
    boost::apply_visitor(*this, n.rhs());
}
void sequence_extract::operator()(const tuple& n) {
    for(auto i=n.begin(); i != n.end(); i++) {
        boost::apply_visitor(*this, *i);
    }
}
void sequence_extract::operator()(const apply& n) {
    boost::apply_visitor(*this, n.fn());
    boost::apply_visitor(*this, n.args());
}
void sequence_extract::operator()(const templated_name& n) {
    if (n.id() == detail::make_sequence()) {        
        std::ostringstream os;
        os << "template ";
        ctype::ctype_printer tp(os);
        boost::apply_visitor(tp, *n.template_types().begin());
        os << ' ';
        cuda_printer cp(m_entry_point, m_registry, os);
        boost::apply_visitor(cp, n);
        os << "(sp_cuarray&, bool, bool);";
        m_extractions.insert(os.str());
    }
}
const set<string>& sequence_extract::extractions() const {
    return m_extractions;
}

}
