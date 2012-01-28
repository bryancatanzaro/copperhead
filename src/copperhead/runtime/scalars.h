#pragma once
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>


PyObject* make_scalar(const float& s);

PyObject* make_scalar(const double& s);

PyObject* make_scalar(const int& s);

PyObject* make_scalar(const long& s);

PyObject* make_scalar(const bool& s);

template<typename T>
T unpack_scalar(PyObject* s) {
    T result;
    PyArray_ScalarAsCtype(s, &result);
    return result;
}
