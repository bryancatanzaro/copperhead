#include "namespace_wrap.hpp"

using namespace backend;
using std::string;
using std::make_shared;
using std::static_pointer_cast;
using std::shared_ptr;
using std::vector;

namespace_wrap::namespace_wrap(const std::string& hash)
    : m_hash(hash) {}

namespace_wrap::result_type namespace_wrap::operator()(const suite& n) {
    vector<shared_ptr<const statement> > stmts;
    for(auto i = n.begin();
        i != n.end();
        i++) {
        //Exclude #include statements from wrapping
        if (detail::isinstance<include>(*i)) {
            stmts.push_back(static_pointer_cast<const statement>(i->ptr()));
        } else {
            vector<shared_ptr<const statement> > wrapping;
            for(; i != n.end() && !detail::isinstance<include>(*i); i++) {
                wrapping.push_back(i->ptr());
            }
            i--;
            stmts.push_back(
                make_shared<const namespace_block>(
                    m_hash,
                    make_shared<const suite>(
                        std::move(wrapping))));
        
        }
    }
    return make_shared<const suite>(std::move(stmts));
}
