#include <boost/python.hpp>
#include <dlfcn.h>
#include <iostream>
#include <string>
#include <sstream>

//Dynamically load a shared library into the process
//This avoids having to tinker with the LD_LIBRARY_PATH machinery
//In order to use libcopperhead.so, libcunp.so, etc:
//The Copperhead runtime loads these auxiliary shared libraries
//explicitly with this module first, before importing
//cudata.so, backendcompiler.so and other modules which use
//these shared libraries.  This avoids needing to store and initialize
//a new copy of the numpy runtime in every module Copperhead builds.

//Tested on Linux and OS X

//XXX Make this work on Windows

//! 
/*! 
  
  \param f Filename (full path) to shared library to be loaded 
  \param i Optional name of initialization function for library. If
  set to an empty string, no initialization will be performed.
*/
void load_library(std::string f, std::string i) {
    void* p_lib = dlopen(f.c_str(), RTLD_LAZY | RTLD_GLOBAL);
    if (p_lib == NULL) {
        std::ostringstream os;
        os << "Library not loaded: " << f;
        PyErr_SetString(PyExc_ImportError, os.str().c_str());
        boost::python::throw_error_already_set();
    }
    if (i.size() > 0) {
        typedef void (*init_fn_t)();
        init_fn_t init_fn =
            reinterpret_cast<init_fn_t>(dlsym(p_lib, i.c_str()));
        if (init_fn == NULL) {
            std::ostringstream os;
            os << "Library not initialized: " << f << ", given init function " << i;
            PyErr_SetString(PyExc_ImportError, os.str().c_str());
            boost::python::throw_error_already_set();
        }
        init_fn();
    }   
}

BOOST_PYTHON_MODULE(load) {
    boost::python::def("load_library", &load_library);
}
