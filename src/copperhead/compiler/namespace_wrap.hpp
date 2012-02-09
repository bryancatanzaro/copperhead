/*
 *   Copyright 2012      NVIDIA Corporation
 * 
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 * 
 *       http://www.apache.org/licenses/LICENSE-2.0
 * 
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 * 
 */
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
