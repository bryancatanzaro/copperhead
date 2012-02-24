#include "sequence_extract.hpp"

using std::string;
using std::ostringstream;
using std::vector;
using std::shared_ptr;
using std::make_shared;
using std::static_pointer_cast;
using std::move;
using std::set;
using std::make_tuple;

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
        std::ostringstream extract_apply;
        cuda_printer cp(m_entry_point, m_registry, extract_apply);
        boost::apply_visitor(cp, n);

        std::ostringstream extract_result;
        ctype::ctype_printer tp(extract_result);
        boost::apply_visitor(tp, *n.template_types().begin());
        
        m_extractions.insert(make_tuple(extract_apply.str(), extract_result.str()));
    }
}
const set<std::tuple<string, string> >& sequence_extract::extractions() const {
    return m_extractions;
}

}
