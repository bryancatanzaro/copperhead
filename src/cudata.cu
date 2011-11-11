#include "cudata/cudata.h"
#include "cudata/cuarray.h"


template<typename T>
sp_cuarray_var make_cuarray_impl(ssize_t n, T* d) {
    return sp_cuarray_var(new cuarray_var(*new cuarray<T>(n, d)));
}


sp_cuarray_var make_cuarray(ssize_t n, bool* d) {
    return make_cuarray_impl<bool>(n, d);
}
sp_cuarray_var make_cuarray(ssize_t n, int* d) {
    return make_cuarray_impl<int>(n, d);
}
sp_cuarray_var make_cuarray(ssize_t n, long* d) {
    return make_cuarray_impl<long>(n, d);
}
sp_cuarray_var make_cuarray(ssize_t n, float* d) {
    return make_cuarray_impl<float>(n, d);
}
sp_cuarray_var make_cuarray(ssize_t n, double* d) {
    return make_cuarray_impl<double>(n, d);
}

std::string repr_cuarray(const sp_cuarray_var &in) {
    repr_cuarray_printer rp;
    return boost::apply_visitor(rp, *in);
}
