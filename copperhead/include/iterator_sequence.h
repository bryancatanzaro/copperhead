#pragma once

/* This sequence wraps Thrust iterators into a sequence.
   This allows the use of Fancy Thrust iterators in
   Copperhead generated code.  (such as zipped, constant,
   counting, transformed, etc.)
 */
template<typename I>
struct iterator_sequence
{
    typedef typename I::value_type value_type;
    typedef typename I::value_type T;
    typedef I iterator;
  
    I data;
    int length;
  
    __host__ __device__
    iterator_sequence(I _data, int _length) : data(_data), length(_length) {}

    __host__ __device__
    iterator_sequence(I *begin, I *end) : data(begin), length(end-begin) {}

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
    T        operator[](int index)       { return data[index]; }
    __host__ __device__
    const T  operator[](int index) const { return data[index]; }

    __host__ __device__
    int size() const { return length; }

    __host__ __device__
    I begin() {
      return data;
    }
  
    __host__ __device__
    I end() {
      return data + length;
    }
};
