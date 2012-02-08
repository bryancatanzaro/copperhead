#include "namespace_wrap.hpp"

using namespace backend;
using std::string;
using std::ostringstream;
using std::make_shared;
using std::shared_ptr;
using std::vector;

string type_summarizer::operator()(const monotype_t& t) const {
    ostringstream os;
    os << t.name();
    for(auto i = t.begin();
        i != t.end();
        i++) {
        os << boost::apply_visitor(*this, *i);
    }
    return os.str();
}

string type_summarizer::operator()(const polytype_t& t) const {
    return "p";
}

entry_hash::entry_hash(const string& n) : m_entry_point(n) {}

entry_hash::result_type entry_hash::operator()(const procedure& n) {
    if (n.id().id() == m_entry_point) {
        type_summarizer t;
        m_entry_hash = n.id().id() + boost::apply_visitor(t, n.type());
    }
    return this->rewriter::operator()(n);
}

const string& entry_hash::hash() const {
    return m_entry_hash;
}

namespace_wrap::namespace_wrap(const std::string& entry_hash)
    : m_entry_hash(entry_hash) {}

namespace_wrap::result_type namespace_wrap::operator()(const suite& n) {
    vector<shared_ptr<statement> > stmts;
    vector<shared_ptr<statement> > wrapping;
    for(auto i = n.begin();
        i != n.end();
        i++) {
        //Exclude #include statements from private namespace
        //Put everything else inside namespace blocks
        if (detail::isinstance<include>(*i)) {
            if (wrapping.size() > 0) {
                stmts.push_back(
                    make_shared<namespace_block>(
                        m_entry_hash,
                        make_shared<suite>(
                            std::move(wrapping))));
                wrapping = vector<shared_ptr<statement> >();                                                 
            }
            stmts.push_back(get_node_ptr(*i));
        } else {
            wrapping.push_back(get_node_ptr(*i));
        }       
    }
    if (wrapping.size() > 0) {
        stmts.push_back(
            make_shared<namespace_block>(
                m_entry_hash,
                make_shared<suite>(
                    std::move(wrapping))));
    }
    return make_shared<suite>(std::move(stmts));
}
