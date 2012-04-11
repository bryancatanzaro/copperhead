#pragma once
namespace backend {

template<class T>
T* get_pointer(std::shared_ptr<T> const &p) {
    return p.get();
}

//WAR boost::python not dealing with const pointers well
//Note: this is evil
template<class T>
T* get_pointer(std::shared_ptr<const T> const &p) {
    return const_cast<T*>(p.get());
}
}

namespace boost {
namespace python {
template<class T>
struct pointee<std::shared_ptr<const T> > {
    typedef T type;
};
}
}
