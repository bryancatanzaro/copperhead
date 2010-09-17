#pragma once
struct index_sequence
{
  int length;
  int value;
  typedef int value_type;
  
  __host__ __device__ index_sequence() : length(0), value(0) {}
  __host__ __device__ index_sequence(int _length) : length(_length), value(0) {}
  __host__ __device__ index_sequence(int _length, int start): length(_length), value(start) {}
  __host__ __device__ int operator[](int index) {
    return index + value;
  }
  __host__ __device__ int size() const { return length; }
  __host__ __device__ bool empty() const { return length <= 0; }
  __host__ __device__ int next()
  {
    int x = value;
    length--;
    value++;
    return x;
  }
};
    
  
  
