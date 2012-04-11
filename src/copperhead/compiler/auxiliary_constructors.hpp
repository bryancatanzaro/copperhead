#pragma once

#include <boost/python.hpp>
#include <boost/python/extract.hpp>

namespace backend {

template<typename S, typename I>
std::vector<std::shared_ptr<const S> > extract_iterable(const I& l) {
    std::vector<std::shared_ptr<const S> > values;
    boost::python::ssize_t n = len(l);
    for(boost::python::ssize_t i=0; i<n; i++) {
        boost::python::object elem = l[i];
        std::shared_ptr<S> p_elem = boost::python::extract<std::shared_ptr<S> >(elem);
        values.push_back(p_elem);
    }
    return values;
}

template<typename S, typename T, typename I>
std::shared_ptr<T> make_from_iterable(const I& iterable) {
    return std::make_shared<T>(std::move(extract_iterable<S>(iterable)));
}

template<typename S, typename T, typename I, typename TT>
std::shared_ptr<T> make_from_iterable(const I& iterable,
                                      const std::shared_ptr<TT>& arg) {
    return std::make_shared<T>(std::move(extract_iterable<S>(iterable)),
                               arg);
}

template <typename S, typename T>
std::shared_ptr<T> make_from_args(boost::python::tuple args,
                                  boost::python::dict kwargs) {
    if(len(args) == 1) {
        boost::python::object first_arg = args[0];
        if (PyList_Check(first_arg.ptr())) {
            return make_from_iterable<S, T, boost::python::list>(boost::python::list(args[0]));
        } else if (PyTuple_Check(first_arg.ptr())) {
            return make_from_iterable<S, T, boost::python::tuple>(boost::python::tuple(args[0]));
        }
    }
    return make_from_iterable<S, T, boost::python::tuple>(args);
}

}
