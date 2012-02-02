#pragma once
#include <Python.h>
#include "cudata.h"

struct array_info {
    void* d;
    ssize_t n;
    CUTYPE t;
    array_info(void* _d, ssize_t _n, CUTYPE _t);
};

void initialize_cunp();
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
