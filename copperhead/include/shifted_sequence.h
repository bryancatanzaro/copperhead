#pragma once

template<typename T>
struct stored_sequence;

template<typename T>
struct shifted_sequence 
{
    typedef T value_type;

    T *data;
    T fill;
    int length;
    int offset;

    __host__ __device__
    shifted_sequence() : data(NULL), length(0), fill(0), offset(0) {}

    __host__ __device__
    shifted_sequence(T *_data, int _length, int _offset, T _fill) :
      data(_data), length(_length), fill(_fill), offset(_offset) {}

    __host__ __device__
    shifted_sequence(stored_sequence<T> in, int _offset, T _fill) :
      data(in.data), length(in.length), fill(_fill), offset(_offset) {}
   

    //
    // Methods supporting sequence interface
    //
    // We don't allow mutating shifted sequences
    // (What would mutating an out of bounds element mean? Changing the fill? Yuk.)
  
    __host__ __device__
    T operator[](int index) const {
      int offset_index = index + offset;
      if ((offset_index >= 0) && (offset_index < length)) {
        return data[offset_index];
      } else {
        return fill;
      }
    }
 
    __host__ __device__
    int size() const { return length; }

   
};



