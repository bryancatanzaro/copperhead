#pragma once

template<typename T>
struct constant_sequence {
  T value;
  int length;
  typedef T value_type;

  __host__ __device__
  constant_sequence(T _value, int _length) : value(_value), length(_length) { }

  __host__ __device__
  T& next() {
    length--;
    return value;
  }

  __host__ __device__
  bool empty() {
    return (length <= 0);
  }

  __host__ __device__
  int size() {
    return length;
  }

  __host__ __device__
  T& operator[](int index) {
    return value;
  }
};
