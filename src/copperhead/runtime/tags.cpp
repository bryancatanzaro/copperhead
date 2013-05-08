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

#include <prelude/runtime/tags.h>

//XXX WAR
//NVCC includes features.h, which Python.h then partially overrides
//Including this here keeps us from seeing warnings 
#ifdef _XOPEN_SOURCE
#undef _XOPEN_SOURCE
#endif
#ifdef _POSIX_C_SOURCE
#undef _POSIX_C_SOURCE
#endif
#include <boost/python.hpp>
#include <boost/python/make_constructor.hpp>


using namespace boost::python;

namespace copperhead {
namespace detail {

boost::shared_ptr<copperhead::system_variant> cpp_tag(
    new copperhead::system_variant(copperhead::cpp_tag()));
#ifdef CUDA_SUPPORT
boost::shared_ptr<copperhead::system_variant> cuda_tag(
    new copperhead::system_variant(copperhead::cuda_tag()));
#endif
#ifdef OMP_SUPPORT
boost::shared_ptr<copperhead::system_variant> omp_tag(
    new copperhead::system_variant(copperhead::omp_tag()));
#endif
#ifdef TBB_SUPPORT
boost::shared_ptr<copperhead::system_variant> tbb_tag(
    new copperhead::system_variant(copperhead::tbb_tag()));
#endif

boost::shared_ptr<copperhead::system_variant> system_variant_from_int(int i) {
#ifdef CUDA_SUPPORT
    if (i == 1) {
        return cuda_tag;
    }
#endif
#ifdef OMP_SUPPORT
    if (i == 2) {
        return omp_tag;
    }
#endif
#ifdef TBB_SUPPORT
    if (i == 3) {
        return tbb_tag;
    }
#endif
    return cpp_tag;
}

struct system_variant_to_int_visitor
    : public boost::static_visitor<int> {
    int operator()(const copperhead::cpp_tag&) const {
        return 0;
    }
#ifdef CUDA_SUPPORT
    int operator()(const copperhead::cuda_tag&) const {
        return 1;
    }
#endif
#ifdef OMP_SUPPORT
    int operator()(const copperhead::omp_tag&) const {
        return 2;
    }
#endif
#ifdef TBB_SUPPORT
    int operator()(const copperhead::tbb_tag&) const {
        return 3;
    }
#endif
};

int system_variant_to_int(const copperhead::system_variant& t) {
    return boost::apply_visitor(system_variant_to_int_visitor(), t);
}

struct system_variant_pickle_suite
    : boost::python::pickle_suite {
    static boost::python::tuple
    getinitargs(const copperhead::system_variant& t) {
        return boost::python::make_tuple(system_variant_to_int(t));
    }
};

namespace tag_cmp {
bool lt(const system_variant& l, const system_variant& r) {
    return copperhead::system_variant_less()(l, r);
}
bool eq(const system_variant& l, const system_variant& r) {
    return !lt(l, r) && !lt(r, l);
}
bool ne(const system_variant& l, const system_variant& r) {
    return lt(l, r) || lt(r, l);
}
bool gt(const system_variant& l, const system_variant& r) {
    return lt(r, l);
}
bool ge(const system_variant& l, const system_variant& r) {
    return !lt(l, r);
}
bool le(const system_variant& l, const system_variant& r) {
    return !lt(r, l);
}



}
}
}

BOOST_PYTHON_MODULE(tags) {
    class_<copperhead::system_variant,
           boost::shared_ptr<copperhead::system_variant> >("system_variant",
                                                           no_init)
        .def("__init__", make_constructor(
                 &copperhead::detail::system_variant_from_int))
        .def_pickle(
            copperhead::detail::system_variant_pickle_suite())
        .def("__str__", &copperhead::to_string)
        .def("__eq__", &copperhead::detail::tag_cmp::eq)
        .def("__ne__", &copperhead::detail::tag_cmp::ne)
        .def("__lt__", &copperhead::detail::tag_cmp::lt)
        .def("__gt__", &copperhead::detail::tag_cmp::gt)
        .def("__le__", &copperhead::detail::tag_cmp::le)
        .def("__ge__", &copperhead::detail::tag_cmp::ge)
        ;
    scope current;
    current.attr("cpp") = copperhead::detail::cpp_tag;
#ifdef CUDA_SUPPORT
    current.attr("cuda") = copperhead::detail::cuda_tag;
#endif
#ifdef OMP_SUPPORT
    current.attr("omp") = copperhead::detail::omp_tag;
#endif
#ifdef TBB_SUPPORT
    current.attr("tbb") = copperhead::detail::tbb_tag;
#endif
}


