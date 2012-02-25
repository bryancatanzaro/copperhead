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
#include "expression.hpp"
#include "statement.hpp"
#include "cppnode.hpp"
#include "utility/snippets.hpp"
#include "cuda_printer.hpp"
#include "type_printer.hpp"
#include "import/library.hpp"
#include <sstream>
#include <set>

namespace backend {

/*!
  \addtogroup visitors
  @{
*/

//! A visitor pass to discover sequence extractions that need to be instantiated specially
/*! This pass can be removed once nvcc can pass boost::python and C++11 features through
*/
class cuarray_extract
    : public no_op_visitor<>
{
private:
    std::set<std::string> m_extractions;
    
    const registry& m_registry;
    const std::string& m_entry_point;
public:
    cuarray_extract(const std::string& entry_point, const registry& reg);
    using no_op_visitor::operator();
    //! Visit rule for \p suite nodes
    result_type operator()(const suite &n);
    //! Visit rule for \p procedure nodes
    result_type operator()(const procedure &n);
    //! Rewrite rule for \p typedefn nodes
    result_type operator()(const typedefn& n);
    //! Grab extractions
    const std::set<std::string>& extractions() const;
};

/*!
  @}
*/

}
