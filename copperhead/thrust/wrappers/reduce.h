#pragma once
#include <thrust/reduce.h>
#include <cuda.h>

template<typename T>
T sum(T* deviceData, int length, T prefix) {
  thrust::device_ptr<T> inputPointer((T*)deviceData);
  T result = thrust::reduce(inputPointer, inputPointer + length, prefix);
  return result;
}

template<typename T, class F>
  T reduce(T* deviceData, int length, T prefix, F functor) {
  thrust::device_ptr<T> inputPointer((T*)deviceData);
  T result = thrust::reduce(inputPointer, inputPointer + length, prefix, functor);
  return result;
}
