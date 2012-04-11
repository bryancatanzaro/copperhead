#pragma once
namespace backend {
template<class T>
T* get_pointer(std::shared_ptr<T> const &p) {
    return p.get();
}
}
