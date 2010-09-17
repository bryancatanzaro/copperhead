#pragma once
#include <thrust/sort.h>
#include <cuda.h>

template<typename T>
void sort(T* deviceData, int length, T* deviceOutput) {
  thrust::device_ptr<T> inputPointer(deviceData);
  thrust::device_ptr<T> output(deviceOutput);
  cudaMemcpy(deviceOutput, deviceData, length * sizeof(T), cudaMemcpyDeviceToDevice);
  thrust::stable_sort(output, output + length);
}
