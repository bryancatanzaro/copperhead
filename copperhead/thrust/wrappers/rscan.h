#pragma once
#include <thrust/device_vector.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/scan.h>

template<typename T, class F>
  void rscan(F functor, T* deviceData, int length, T* deviceOutput) {
  thrust::device_ptr<T> d(deviceData);
  thrust::device_ptr<T> o(deviceOutput);
  typename thrust::device_vector<T>::reverse_iterator drbegin(&d[length]);
  typename thrust::device_vector<T>::reverse_iterator drend(&d[0]);

  typename thrust::device_vector<T>::reverse_iterator orbegin(&o[length]);
  thrust::inclusive_scan(drbegin, drend, orbegin, functor);
}
