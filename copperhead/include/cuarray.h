#pragma once
#include <boost/python.hpp>
#include <cuda.h>
#include <exception>

class CuException: public std::exception
{
  virtual const char* what() const throw()
  {
    return "Internal Error in Copperhead Runtime";
  }
} cuException;


struct CuArray {
  CUdeviceptr devicePointer;
  int length;
  PyObject* cu_type;
};

static PyObject* pyCuArrayClass = NULL;


void setCuArrayClass(PyObject* cuArrayClass) {
  pyCuArrayClass = cuArrayClass;
}

CuArray extractArray(PyObject* pythonCuArray) {
  if (pyCuArrayClass == NULL) {
    pyCuArrayClass = PyObject_GetAttrString(pythonCuArray, "__class__");
  }
  PyObject* shape = PyObject_GetAttrString(pythonCuArray, "shape");
  PyObject* extents = PyObject_GetAttrString(shape, "extents");
  PyObject* pyLength = PySequence_GetItem(extents, 0);
  PyObject* type = PyObject_GetAttrString(pythonCuArray, "type");
  int length = boost::python::extract<int>(pyLength);
  PyObject* remote = PyObject_GetAttrString(pythonCuArray, "remote");
  PyObject* gpuArray = PySequence_GetItem(remote, 0);
  PyObject* gpuPointer = PyObject_GetAttrString(gpuArray, "gpudata");
  CUdeviceptr cudaPointer = boost::python::extract<CUdeviceptr>(gpuPointer);
  CuArray result;
  result.devicePointer = cudaPointer;
  result.length = length;
  result.cu_type = type;
  return result;
}

PyObject* buildArray(CuArray in) {
  if (pyCuArrayClass == NULL) {
    throw cuException;
  }
  PyObject* args = Py_BuildValue("()");
  PyObject* newShape = Py_BuildValue("[i]", in.length);
  PyObject* remote = Py_BuildValue("i", in.devicePointer);
  PyObject* kwargs = Py_BuildValue("{sOsOsO}", "shape", newShape, "type", in.cu_type, "remote", remote);
  PyObject* cuArray = PyObject_Call(pyCuArrayClass, args, kwargs);
  return cuArray;
}
  
