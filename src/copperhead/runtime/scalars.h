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
    return T();
}

template<>
float unpack_scalar<float>(PyObject* s) {
    return PyArrayScalar_VAL(s, Float);
}

template<>
double unpack_scalar<double>(PyObject* s) {
    return PyArrayScalar_VAL(s, Double);
}

template<>
int unpack_scalar<int>(PyObject* s) {
    return PyArrayScalar_VAL(s, Int);
}

template<>
long unpack_scalar<long>(PyObject* s) {
    return PyArrayScalar_VAL(s, Long);
}

template<>
bool unpack_scalar<bool>(PyObject* s) {
    return PyArrayScalar_VAL(s, Bool);
}
