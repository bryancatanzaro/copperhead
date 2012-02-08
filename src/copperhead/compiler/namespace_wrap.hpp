#pragma once
#include "node.hpp"
#include "type.hpp"
#include "ctype.hpp"
#include "utility/isinstance.hpp"
#include "utility/markers.hpp"
#include "utility/snippets.hpp"
#include "utility/initializers.hpp"
#include "py_printer.hpp"
#include "type_printer.hpp"
#include "rewriter.hpp"
#include <set>
#include <sstream>
#include <string>

namespace backend {

class type_summarizer
    : public boost::static_visitor<std::string> {
public:
    result_type operator()(const monotype_t& t) const;
    result_type operator()(const polytype_t& t) const;
};

class entry_hash
    : public rewriter {

private:
    const std::string m_entry_point;
    std::string m_entry_hash;
public:
    entry_hash(const std::string& n);

    using rewriter::operator();

    result_type operator()(const procedure&);

    const std::string& hash() const;
};

class namespace_wrap
    : public rewriter {
private:
    const std::string m_entry_hash;
public:
    namespace_wrap(const std::string& entry_hash);

    using rewriter::operator();

    result_type operator()(const suite&);
};


}
