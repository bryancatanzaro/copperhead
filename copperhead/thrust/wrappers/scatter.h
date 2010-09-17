#pragma once
#include <thrust/scatter.h>
#include <thrust/device_vector.h>
#include <cuda.h>

template<typename T>
void scatter(T* deviceSource, int* deviceIdx, int idxLength, T* deviceDest, int destLength, T* deviceOutput) {
  thrust::device_ptr<int> idxPointer(deviceIdx);
  thrust::device_ptr<T> sourcePointer(deviceSource);
  thrust::device_ptr<T> destPointer(deviceDest);
  thrust::device_ptr<T> output(deviceOutput);
  cudaMemcpy(deviceOutput, deviceDest, sizeof(T) * destLength, cudaMemcpyDeviceToDevice);
  thrust::scatter(sourcePointer, sourcePointer + idxLength,
                 idxPointer, output);
}

