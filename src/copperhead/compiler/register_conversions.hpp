#pragma once
#include <boost/python.hpp>
#include <boost/python/implicit.hpp>
#include <memory>


namespace backend {

template<typename Derived, typename Super>
void register_conversions() {
    //It's ok to convert from a shared_ptr<T> to a shared_ptr<const T>
    boost::python::implicitly_convertible<
        std::shared_ptr<Derived>,
        std::shared_ptr<const Derived> >();
    
    //It's ok to convert from a shared_ptr<Derived> to a shared_ptr<Super>
    boost::python::implicitly_convertible<
        std::shared_ptr<Derived>,
        std::shared_ptr<Super> >();

    //It's ok to convert from a shared_ptr<Derived> to a shared_ptr<const Super>
    boost::python::implicitly_convertible<
        std::shared_ptr<Derived>,
        std::shared_ptr<const Super> >();

    //It's ok to convert from a shared_ptr<const Derived> to a shared_ptr<const Super>
    boost::python::implicitly_convertible<
        std::shared_ptr<const Derived>,
        std::shared_ptr<const Super> >();

    //Note the missing conversions:
    //It's not ok to convert from a shared_ptr<const T> to a shared_ptr<T>
    //It's not ok to convert from a shared_ptr<Super> to a shared_ptr<Derived>
    //And their various combinations.
}

}
