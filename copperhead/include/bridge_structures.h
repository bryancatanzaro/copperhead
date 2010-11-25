#pragma once
#include <vector>
#include <cuda.h>

// This file contains data structures which connect the host C++ compiler and nvcc
// NVCC can't digest Boost headers, so we have to do the extraction from Python types
// in the host compiler.
// Conversely, the host compiler can't digest __host__/__device__ tags, so we can't
// build the CUDA data structures in code that can extract them from Python.
// Instead, we bridge the gap with these data structures.

struct CuArray {
  CUdeviceptr data;
  int length;

  CuArray(CUdeviceptr _data, int _length) : data(_data), length(_length) { }
};

typedef std::vector<int> IntVector;
typedef std::vector<CuArray> _CuSequence;

struct _CuUniform {
  IntVector strides;
  IntVector extents;
  int offset;
  CUdeviceptr data;
};


struct _CuScalar {
  _CuScalar(void* _value) : value(_value) { }
  _CuScalar() { }
  void* value;
};

typedef std::vector<_CuScalar> _CuTuple;
