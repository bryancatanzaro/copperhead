#pragma once
#include <boost/version.hpp>

#if BOOST_VERSION < 105300
namespace backend {
template<class T>
T* get_pointer(std::shared_ptr<T> const &p) {
    return p.get();
}
}

#endif
