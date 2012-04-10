#include "python_wrap.hpp"


using std::string;
using std::vector;
using std::shared_ptr;
using std::make_shared;
using std::static_pointer_cast;
using std::move;
using backend::utility::make_vector;

namespace backend {

python_wrap::python_wrap(const copperhead::system_variant& t,
                         const string& entry_point)
    : m_t(t),
      m_entry_point(entry_point),
      m_wrapping(false),
      m_wrap_result(false),
      m_wrapper(),
      m_scalars() {}


python_wrap::result_type python_wrap::operator()(const procedure &n) {
    if (n.id().id() == detail::wrap_proc_id(m_entry_point)) {
        m_wrapping = true;
        vector<shared_ptr<const expression> > wrapper_args;
        for(auto i = n.args().begin();
            i != n.args().end();
            i++) {
            if (detail::isinstance<sequence_t>(i->type())) {
                //Sequences get wrapped in the default wrapper, so pass through
                shared_ptr<const expression> passed =
                    static_pointer_cast<const expression>(
                        boost::apply_visitor(*this, *i));
                wrapper_args.push_back(passed);
            } else {
                //We can only handle names in procedure arguments
                //at this time
                bool is_node = detail::isinstance<name, node>(*i);
                assert(is_node);
                const name& arg_name = boost::get<const name&>(*i);
                shared_ptr<const name> pyobject_name =
                    make_shared<const name>(arg_name.id(),
                                            arg_name.type().ptr(),
                                            make_shared<const ctype::monotype_t>("PyObject*"));
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
        shared_ptr<const ctype::type_t> proc_p_ctype = n.ctype().ptr();
        
        if (!(detail::isinstance<sequence_t>(proc_res_t))) {
            m_wrap_result = true;
            proc_p_ctype = make_shared<const ctype::fn_t>(
                static_pointer_cast<const ctype::tuple_t>(proc_ctype.args().ptr()),
                make_shared<const ctype::monotype_t>("PyObject*"));
        }
        
        auto result = make_shared<const procedure>(
            static_pointer_cast<const name>(n.ptr()),
            make_shared<const tuple>(move(wrapper_args)),
            static_pointer_cast<const suite>(
                boost::apply_visitor(*this, n.stmts())),
            n.type().ptr(),
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
        std::ostringstream os;
        backend::ctype::ctype_printer ctp(m_t, os);
        boost::apply_visitor(ctp, n.ctype());
        
        return make_shared<const apply>(
            make_shared<const name>(
                string("unpack_scalar") + "_" + os.str()),            
            make_shared<const tuple>(
                make_vector<shared_ptr<const expression> >(
                    static_pointer_cast<const name>(n.ptr())))); 
    } else {
        return this->rewriter::operator()(n);
    }
}

python_wrap::result_type python_wrap::operator()(const ret& n) {
    if (m_wrapping && m_wrap_result) {
        return make_shared<const ret>(
            make_shared<const apply>(
                make_shared<const name>("make_scalar"),
                make_shared<const tuple>(
                    make_vector<shared_ptr<const expression> >(
                        static_pointer_cast<const expression>(n.ptr())))));
    } else {
        return this->rewriter::operator()(n);
    }
}

shared_ptr<const procedure> python_wrap::wrapper() const {
    return m_wrapper;
}

}
