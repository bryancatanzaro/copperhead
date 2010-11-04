#pragma once
#include <thrust/device_ptr.h>

template<typename T>
struct stored_sequence 
{
    typedef T value_type;
    typedef typename thrust::device_ptr<T> iterator;
    T *data;
    int length;

    __host__ __device__
    stored_sequence() : data(NULL), length(0) {}

    __host__ __device__
    stored_sequence(T *_data, int _length) : data(_data), length(_length) {}

    __host__ __device__
    stored_sequence(T *begin, T *end) : data(begin), length(end-begin) {}

    //
    // Methods supporting stream interface
    //
    __host__ __device__
    bool empty() const { return length<=0; }

    __host__ __device__
    T next()
    {
        T x = *(data++);
        --length;
        return x;
    }

    //
    // Methods supporting sequence interface
    //
    __host__ __device__
    T&       operator[](int index)       { return data[index]; }
    __host__ __device__
    const T& operator[](int index) const { return data[index]; }

    __host__ __device__
    int size() const { return length; }

    __host__ __device__
    thrust::device_ptr<T> begin() {
      return thrust::device_ptr<T>(data);
    }

    __host__ __device__
    thrust::device_ptr<T> end() {
      return thrust::device_ptr<T>(data) + length;
    }
};


template<typename T>
__host__ __device__
stored_sequence<T> slice(stored_sequence<T> seq, int base, int len)
{
    return stored_sequence<T>(&seq[base], len);
}


