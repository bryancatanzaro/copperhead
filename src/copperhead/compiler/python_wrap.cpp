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
                                                      m_wrap_result(false),
                                                      m_wrapper(),
                                                      m_scalars() {}


python_wrap::result_type python_wrap::operator()(const procedure &n) {
    if (n.id().id() == detail::wrap_proc_id(m_entry_point)) {
        m_wrapping = true;
        vector<shared_ptr<expression> > wrapper_args;
        for(auto i = n.args().begin();
            i != n.args().end();
            i++) {
            if (detail::isinstance<sequence_t>(i->type())) {
                //Sequences get wrapped in the default wrapper, so pass through
                shared_ptr<expression> passed =
                    static_pointer_cast<expression>(
                        boost::apply_visitor(*this, *i));
                wrapper_args.push_back(passed);
            } else {
                //We can only handle names in procedure arguments
                //at this time
                bool is_node = detail::isinstance<name, node>(*i);
                assert(is_node);
                const name& arg_name = boost::get<const name&>(*i);
                shared_ptr<name> pyobject_name =
                    make_shared<name>(arg_name.id(),
                                      arg_name.p_type(),
                                      make_shared<ctype::monotype_t>("PyObject*"));
                wrapper_args.push_back(pyobject_name);
                m_scalars.insert(arg_name.id());
            }

        }

        //Type must be a fn type
        assert(detail::isinstance<fn_t>(n.type()));
        const fn_t &proc_t = boost::get<const fn_t &>(n.type());

        const type_t &proc_res_t = proc_t.result();
        //Ctype must be a fn type
        const ctype::fn_t& proc_ctype =
            boost::get<const ctype::fn_t&>(n.ctype());
        shared_ptr<ctype::type_t> proc_p_ctype = n.p_ctype();
        
        if (!(detail::isinstance<sequence_t>(proc_res_t))) {
            m_wrap_result = true;
            proc_p_ctype = make_shared<ctype::fn_t>(
                proc_ctype.p_args(),
                make_shared<ctype::monotype_t>("PyObject*"));
        }
        
        auto result = make_shared<procedure>(
            n.p_id(),
            make_shared<tuple>(move(wrapper_args)),
            static_pointer_cast<suite>(
                boost::apply_visitor(*this, n.stmts())),
            n.p_type(),
            proc_p_ctype,
            "");
        m_wrapping = false;
        m_wrapper = result;
        return result;
    } else {
        return this->rewriter::operator()(n);
    }
}

python_wrap::result_type python_wrap::operator()(const name& n) {
    if (m_wrapping && (m_scalars.find(n.id()) != m_scalars.end())) {
        
        return make_shared<apply>(
            make_shared<templated_name>(
                "unpack_scalar",
                make_shared<ctype::tuple_t>(
                    make_vector<shared_ptr<ctype::type_t> >(n.p_ctype()))),
            
            make_shared<tuple>(
                make_vector<shared_ptr<expression> >(
                    get_node_ptr(n)))); 
    } else {
        return this->rewriter::operator()(n);
    }
}

python_wrap::result_type python_wrap::operator()(const ret& n) {
    if (m_wrapping && m_wrap_result) {
        return make_shared<ret>(
            make_shared<apply>(
                make_shared<name>("make_scalar"),
                make_shared<tuple>(
                    make_vector<shared_ptr<expression> >(
                        n.p_val()))));
    } else {
        return this->rewriter::operator()(n);
    }
}

shared_ptr<procedure> python_wrap::wrapper() const {
    return m_wrapper;
}

}
