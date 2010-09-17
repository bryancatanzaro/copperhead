#pragma once
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <cuda.h>

template<typename T>
void sum_scan(T* deviceData, int length, T* deviceOutput) {
  thrust::device_ptr<T> inputPointer(deviceData);
  thrust::device_ptr<T> outputPointer(deviceOutput);
  thrust::inclusive_scan(inputPointer, inputPointer + length,
                         outputPointer);
  return;
}

template<typename T, class F>
  void scan(F functor, T* deviceData, int length, T* deviceOutput) {
  thrust::device_ptr<T> inputPointer(deviceData);
  thrust::device_ptr<T> outputPointer(deviceOutput);
  thrust::inclusive_scan(inputPointer, inputPointer + length, outputPointer, functor);
  return;
}
