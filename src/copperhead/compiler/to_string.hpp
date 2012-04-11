#pragma once
#include <memory>

namespace backend {

template<class T, class P>
const std::string to_string(std::shared_ptr<T> &p) {
    std::ostringstream os;
    P pp(os);
    boost::apply_visitor(pp, *p);
    return os.str();
}

}
