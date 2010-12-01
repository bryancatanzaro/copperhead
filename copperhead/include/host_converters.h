#pragma once
#include <boost/python.hpp>
#include <boost/python/extract.hpp>
#include <cuda.h>
#include "bridge_structures.h"
#include <iostream>
using namespace boost::python;

CuArray extract_gpuarray(PyObject* gpuarray) {
  PyObject* descShape = PyObject_GetAttrString(gpuarray, "shape");
  PyObject* py_length = PySequence_GetItem(descShape, 0);
  int descLength = extract<int>(py_length);
  PyObject* gpudata = PyObject_GetAttrString(gpuarray, ("gpudata"));
  CUdeviceptr cudaPointer = extract<CUdeviceptr>(gpudata);
  return CuArray(cudaPointer, descLength);
}

void extract_cusequence(PyObject* cuarray, _CuSequence* output) {
  PyObject* remote = PyObject_GetAttrString(cuarray, "remote");
  Py_ssize_t remote_len = PySequence_Length(remote);
  for(int i = 0; i < remote_len; i++) {
    PyObject* gpuarray = PySequence_GetItem(remote, i);
    output->push_back(extract_gpuarray(gpuarray));
  }
}

struct CuSequence_from_py_CuArray {
  CuSequence_from_py_CuArray() {
    converter::registry::push_back(&convertible,
                                    &construct,
                                    type_id<_CuSequence>());
  }
  static void* convertible(PyObject* obj_ptr) {
    if (!PyObject_HasAttrString(obj_ptr, "remote"))
      return 0;
    else
      return obj_ptr;
  }
  static void construct(PyObject* obj_ptr,
                        converter::rvalue_from_python_stage1_data* data) {
    void* storage = ((converter::rvalue_from_python_storage<_CuSequence>*)data)->storage.bytes;
    new (storage) _CuSequence;
    extract_cusequence(obj_ptr,
                       (_CuSequence*)storage);
    data->convertible = storage;
  }                                 
};

void extract_cuuniform(PyObject* cuuniform, _CuUniform* output) {
  IntVector &extents = output->extents;
  IntVector &strides = output->strides;
  PyObject* py_extents = PyObject_GetAttrString(cuuniform, "extents");
  PyObject* py_strides = PyObject_GetAttrString(cuuniform, "strides");
  Py_ssize_t extent_len = PySequence_Length(py_extents);
  for(int i = 0; i < extent_len; i++) {
    extents.push_back(extract<int>(PySequence_GetItem(py_extents, i)));
    strides.push_back(extract<int>(PySequence_GetItem(py_strides, i)));
  }
  output->offset = extract<int>(PyObject_GetAttrString(cuuniform, "offset"));
  PyObject* remote = PyObject_GetAttrString(cuuniform, "remote");
  PyObject* gpudata = PyObject_GetAttrString(remote, "gpudata");
  output->data = extract<CUdeviceptr>(gpudata);
}

struct CuUniform_from_py_CuUniform {
  CuUniform_from_py_CuUniform() {
    converter::registry::push_back(&convertible,
                                    &construct,
                                    type_id<_CuUniform>());
  }
  static void* convertible(PyObject* obj_ptr) {
    bool ok = true;
    ok = ok & PyObject_HasAttrString(obj_ptr, "remote");
    ok = ok & PyObject_HasAttrString(obj_ptr, "extents");
    ok = ok & PyObject_HasAttrString(obj_ptr, "strides");
    if (ok)
      return obj_ptr;
    else
      return 0;
  }
  
  static void construct(PyObject* obj_ptr,
                        converter::rvalue_from_python_stage1_data* data) {
    void* storage = ((converter::rvalue_from_python_storage<_CuUniform>*)data)->storage.bytes;
    new (storage) _CuUniform;
    extract_cuuniform(obj_ptr,
                       (_CuUniform*)storage);
    data->convertible = storage;
  }                                 
};




// Numpy scalars are structs with a format like the one that follows.
// This is a hack, a more principled solution would be to include the
// Numpy headers and use their struct definitions directly.
// However, that complicates the build process somewhat, for little gain.
// Instead, I just define a struct which matches the Numpy scalar struct
// and allows me to dereference it and get the data out:


struct fake_numpy_scalar {
  PyObject_HEAD
  void* obval;
};

struct CuScalar_from_py_CuScalar {
  CuScalar_from_py_CuScalar() {
    converter::registry::push_back(&convertible,
                                    &construct,
                                    type_id<_CuScalar>());
  }
  static void* convertible(PyObject* obj_ptr) {
    bool ok = PyObject_HasAttrString(obj_ptr, "value");
    if (ok)
      return obj_ptr;
    else
      return 0;
  }
  
  static void construct(PyObject* obj_ptr,
                        converter::rvalue_from_python_stage1_data* data) {
    fake_numpy_scalar* numpy_scalar = (fake_numpy_scalar*)((void*)PyObject_GetAttrString(obj_ptr, "value"));
    void* value = &numpy_scalar->obval;
    void* storage = ((converter::rvalue_from_python_storage<_CuScalar>*)data)->storage.bytes;
    new (storage) _CuScalar(value);
    data->convertible = storage;
  }                                 
};

struct CuTuple_from_py_CuTuple {
  CuTuple_from_py_CuTuple() {
    converter::registry::push_back(&convertible,
                                   &construct,
                                   type_id<_CuTuple>());
  }
  static void* convertible(PyObject* obj_ptr) {
    // XXX Empty check here
    // Since we're in charge of this call,
    // We're responsible for ensuring this is legal.
    bool ok = PyObject_HasAttrString(obj_ptr, "value");
    if (ok)
      return obj_ptr;
    else
      return 0;
  }
  
  static void construct(PyObject* obj_ptr,
                        converter::rvalue_from_python_stage1_data* data) {
    PyObject* value = PyObject_GetAttrString(obj_ptr, "value");
    Py_ssize_t len = PySequence_Size(value);
    void* storage = ((converter::rvalue_from_python_storage<_CuTuple>*)data)->storage.bytes;
    new (storage) _CuTuple();
    _CuTuple* p_tuple = (_CuTuple*)storage; 
    for(Py_ssize_t i = 0; i < len; i++) {
      PyObject* sub_ptr = PySequence_GetItem(value, i);
      fake_numpy_scalar* numpy_scalar = (fake_numpy_scalar*)((void*)PyObject_GetAttrString(sub_ptr, "value"));
      void* value = &numpy_scalar->obval;
      _CuScalar scalar(value);
      p_tuple->push_back(scalar);
    }
    data->convertible = storage;
  }
};

void register_converters() {
  CuSequence_from_py_CuArray();
  CuUniform_from_py_CuUniform();
  CuScalar_from_py_CuScalar();
  CuTuple_from_py_CuTuple();
}
