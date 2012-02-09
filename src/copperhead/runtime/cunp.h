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
#pragma once
//XXX WAR
//NVCC includes features.h, which Python.h then partially overrides
//Including this here keeps us from seeing warnings 
#ifdef _XOPEN_SOURCE
#undef _XOPEN_SOURCE
#endif
#ifdef _POSIX_C_SOURCE
#undef _POSIX_C_SOURCE
#endif

#include <Python.h>
#include "cudata.h"

struct array_info {
    void* d;
    ssize_t n;
    CUTYPE t;
    array_info(void* _d, ssize_t _n, CUTYPE _t);
};

extern "C" {
void initialize_cunp();
}
array_info inspect_array(PyObject* in);
PyObject* make_array(array_info i);

PyObject* make_scalar(const bool& s);
PyObject* make_scalar(const int& s);
PyObject* make_scalar(const long& s);
PyObject* make_scalar(const float& s);
PyObject* make_scalar(const double& s);

bool unpack_scalar_bool(PyObject* s);
int unpack_scalar_int(PyObject* s);
long unpack_scalar_long(PyObject* s);
float unpack_scalar_float(PyObject* s);
double unpack_scalar_double(PyObject* s);
