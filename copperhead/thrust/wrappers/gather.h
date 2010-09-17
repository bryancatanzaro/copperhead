#pragma once
#include <thrust/gather.h>
#include <thrust/device_vector.h>
#include <cuda.h>

template<typename T>
void gather(int* deviceMap, T* deviceSource, int length, T* deviceOutput) {
  thrust::device_ptr<int> mapPointer(deviceMap);
  thrust::device_ptr<T> sourcePointer(deviceSource);
  thrust::device_ptr<T> output(deviceOutput);
  thrust::gather(mapPointer, mapPointer + length, sourcePointer, output);
  return;
}

