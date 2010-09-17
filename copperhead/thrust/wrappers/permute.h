#pragma once
#include <thrust/scatter.h>
#include <thrust/device_vector.h>
#include <cuda.h>

template<typename T>
void permute(T* deviceSource, int* deviceMap, int length, T* deviceOutput) {
  thrust::device_ptr<int> mapPointer(deviceMap);
  thrust::device_ptr<T> sourcePointer(deviceSource);
  thrust::device_ptr<T> output(deviceOutput);
  thrust::scatter(sourcePointer, sourcePointer + length,
                  mapPointer, output);
}

