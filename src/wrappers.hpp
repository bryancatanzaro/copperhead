#pragma once
#include "expression.hpp"
#include "statement.hpp"
#include "cppnode.hpp"

namespace backend {

class literal_wrap
    : public literal
{
public:
    literal_wrap(const std::string& id)
        : literal(id)
        {}
    inline const std::string id(void) const {
        return m_val;
    }
    inline const std::shared_ptr<type_t> p_type(void) const {
        return m_type;
    }
    // inline const std::shared_ptr<type_t> p_ctype(void) const {
    //     return m_ctype;
    // }
    inline void set_type(std::shared_ptr<type_t> type) {
        m_type = type;
    }
    // inline void set_ctype(std::shared_ptr<type_t> type) {
    //     m_ctype = type;
    // }
};


class name_wrap
    : public name
{
public:
    name_wrap(const std::string& id)
        : name(id)
        {}
        inline const std::string id(void) const {
        return m_val;
    }
    inline const std::shared_ptr<type_t> p_type(void) const {
        return m_type;
    }
    // inline const std::shared_ptr<type_t> p_ctype(void) const {
    //     return m_ctype;
    // }
    inline void set_type(std::shared_ptr<type_t> type) {
        m_type = type;
    }
    // inline void set_ctype(std::shared_ptr<type_t> type) {
    //     m_ctype = type;
    // }
};


    
class apply_wrap
    : public apply
{
public:
    apply_wrap(const std::shared_ptr<name> &fn,
               const std::shared_ptr<tuple> &args)
        : apply(fn, args)
        {}
    inline const std::shared_ptr<name> p_fn(void) const {
        return m_fn;
    }
    inline const std::shared_ptr<tuple> p_args(void) const {
        return m_args;
    }
    inline const std::shared_ptr<type_t> p_type(void) const {
        return m_type;
    }
    // inline const std::shared_ptr<type_t> p_ctype(void) const {
    //     return m_ctype;
    // }
    inline void set_type(std::shared_ptr<type_t> type) {
        m_type = type;
    }
    // inline void set_ctype(std::shared_ptr<type_t> type) {
    //     m_ctype = type;
    // }
};

class lambda_wrap
    : public lambda
{
public:
    lambda_wrap(const std::shared_ptr<tuple> &args,
                const std::shared_ptr<expression> &body)
        : lambda(args, body)
        {}
    inline const std::shared_ptr<tuple> p_args(void) const {
        return m_args;
    }
    inline const std::shared_ptr<expression> p_body(void) const {
        return m_body;
    }
    inline const std::shared_ptr<type_t> p_type(void) const {
        return m_type;
    }
    // inline const std::shared_ptr<type_t> p_ctype(void) const {
    //     return m_ctype;
    // }
    inline void set_type(std::shared_ptr<type_t> type) {
        m_type = type;
    }
    // inline void set_ctype(std::shared_ptr<type_t> type) {
    //     m_ctype = type;
    // }
};

class closure_wrap
    : public closure
{
public:
    closure_wrap(const std::shared_ptr<tuple> &args,
                const std::shared_ptr<expression> &body)
        : closure(args, body)
        {}
    inline const std::shared_ptr<tuple> p_args(void) const {
        return m_args;
    }
    inline const std::shared_ptr<expression> p_body(void) const {
        return m_body;
    }

    inline const std::shared_ptr<type_t> p_type(void) const {
        return m_type;
    }

    inline void set_type(std::shared_ptr<type_t> type) {
        m_type = type;
    }

};

class tuple_wrap
    : public backend::tuple
{
public:
    tuple_wrap(std::vector<std::shared_ptr<expression> > &&values)
        : backend::tuple(std::move(values))
        {}
    tuple_wrap()
        : tuple({})
        {}
    typedef decltype(m_values.cbegin()) const_ptr_iterator;
    const_ptr_iterator p_begin() const {
        return m_values.cbegin();
    }
    const_ptr_iterator p_end() const {
        return m_values.cend();
    }
    inline const std::shared_ptr<type_t> p_type(void) const {
        return m_type;
    }
    // inline const std::shared_ptr<type_t> p_ctype(void) const {
    //     return m_ctype;
    // }
    inline void set_type(std::shared_ptr<type_t> type) {
        m_type = type;
    }
    // inline void set_ctype(std::shared_ptr<type_t> type) {
    //     m_ctype = type;
    // }
};
    
class ret_wrap
    : public ret
{
public:
    ret_wrap(const std::shared_ptr<expression> &val)
        : ret(val)
        {}
    inline std::shared_ptr<expression> p_val(void) const {
        return m_val;
    }
    // inline const std::shared_ptr<type_t> p_type(void) const {
    //     return m_type;
    // }
    // inline const std::shared_ptr<type_t> p_ctype(void) const {
    //     return m_ctype;
    // }
};
    
class bind_wrap
    : public bind
{
public:
    bind_wrap(const std::shared_ptr<expression> &lhs,
              const std::shared_ptr<expression> &rhs)
        : bind(lhs, rhs)
        {}
    inline const std::shared_ptr<expression> p_lhs(void) const {
        return m_lhs;
    }
    inline const std::shared_ptr<expression> p_rhs(void) const {
        return m_rhs;
    }
};
    
class procedure_wrap
    : public procedure
{
public:
    procedure_wrap(const std::shared_ptr<name> &id,
                   const std::shared_ptr<tuple> &args,
                   const std::shared_ptr<suite> &stmts)
        : procedure(id, args, stmts)
        {}
    inline const std::shared_ptr<name> p_id(void) const {
        return m_id;
    }
    inline const std::shared_ptr<tuple> p_args(void) const {
        return m_args;
    }
    inline const std::shared_ptr<suite> p_stmts(void) const {
        return m_stmts;
    }
    inline const std::shared_ptr<type_t> p_type(void) const {
        return m_type;
    }
    // inline const std::shared_ptr<type_t> p_ctype(void) const {
    //     return m_ctype;
    // }
    inline void set_type(std::shared_ptr<type_t> type) {
        m_type = type;
    }
    // inline void set_ctype(std::shared_ptr<type_t> type) {
    //     m_ctype = type;
    // }
};    

class conditional_wrap
    : public conditional
{
public:
    conditional_wrap(const std::shared_ptr<expression> &cond,
                     const std::shared_ptr<suite> &then,
                     const std::shared_ptr<suite> &orelse)
        : conditional(cond, then, orelse) {}
    inline const std::shared_ptr<expression> p_cond(void) const {
        return m_cond;
    }
    inline const std::shared_ptr<suite> p_then(void) const {
        return m_then;
    }
    inline const std::shared_ptr<suite> p_orelse(void) const {
        return m_orelse;
    }
    
};


class suite_wrap
    : public suite
{
public:
    suite_wrap(std::vector<std::shared_ptr<statement> > &&stmts)
        : suite(std::move(stmts))
        {}
    suite_wrap()
        : suite({})
        {}
    typedef decltype(m_stmts.cbegin()) const_ptr_iterator;
    const_ptr_iterator p_begin() const {
        return m_stmts.cbegin();
    }
        
    const_ptr_iterator p_end() const {
        return m_stmts.cend();
    }
};
    
class structure_wrap
    : public structure
{
public:
    structure_wrap(const std::shared_ptr<name> &name,
                   const std::shared_ptr<suite> &stmts)
        : structure(name, stmts)
        {}
        
    inline const std::shared_ptr<name> p_id(void) const {
        return m_id;
    }
        
    inline const std::shared_ptr<suite> p_stmts(void) const {
        return m_stmts;
    }
        
};
    
}


