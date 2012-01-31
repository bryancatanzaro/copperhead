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

namespace backend {

/*!
  \addtogroup rewriters
  @{
*/

//! A rewrite pass which constructs the entry point wrapper
/*! The entry point is a little special. It needs to operate on
  containers that are held by the broader context of the program,
  whereas the rest of the program operates solely on views.  This pass
  adds a wrapper which operates on containers, derives views, and then
  calls the body of the entry point.
  
*/
class python_wrap
    : public rewriter
{
private:
    const std::string& m_entry_point;
    bool m_wrapping;
    bool m_wrap_result;
    std::shared_ptr<procedure> m_wrapper;
    std::set<std::string> m_scalars;
public:
    //! Constructor
/*! 
  
  \param entry_point Name of the entry point procedure
*/
    python_wrap(const std::string& entry_point);
    
    using rewriter::operator();
    //! Rewrite rule for \p procedure nodes
    result_type operator()(const procedure &n);
    //! Rewrite rule for \p ret nodes
    result_type operator()(const ret& n);
    //! Rewrite rule for \p name nodes
    result_type operator()(const name& n);
    //! Grab wrapper
    std::shared_ptr<procedure> wrapper() const;
};

/*!
  @}
*/

}
