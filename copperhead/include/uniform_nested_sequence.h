#pragma once

template<typename T, int Depth>
  struct uniform_nested_sequence
  {
    typedef T value_type;
    static const int nesting_depth = Depth;
    int stride;
    int length;
    uniform_nested_sequence<T, Depth-1> sub;

    __host__ __device__
    uniform_nested_sequence() { }

    __host__ __device__
    uniform_nested_sequence(int _length,
                            int _stride,
                            uniform_nested_sequence<T, Depth-1> _sub)
      : length(_length), stride(_stride), sub(_sub) { }

    __host__ __device__
    int size() const { return length; }

    __host__ __device__
    uniform_nested_sequence<T, Depth-1> operator[](int i) {
      return slice(sub, i * stride);
    }

    __host__ __device__
    void advance(int i) {
      sub.advance(i);
    }

    __host__ __device__
    bool empty() const {
      return length <= 0;
    }

    __host__ __device__
    uniform_nested_sequence<T, Depth-1> next() {
      uniform_nested_sequence<T, Depth-1> x = operator[](0);
      advance(stride);
      length--;
      return x;
    }
  };

template<typename T>
struct uniform_nested_sequence<T, 0>
  {
    typedef T value_type;
    static const int nesting_depth = 0;
    int stride;
    int offset;
    int length;
    T* data;
  
    __host__ __device__
    uniform_nested_sequence() { }

    __host__ __device__
    uniform_nested_sequence(int _length,
                            int _stride,
                            T* _data,
                            int _offset=0)
      : length(_length), stride(_stride), offset(_offset), data(_data) { }

    __host__ __device__
    int size() const { return length; }

    __host__ __device__
    T& operator[](int i) {
      return data[i * stride + offset];
    }

    __host__ __device__
      void advance(int i) {
      offset += i;
    }

    __host__ __device__
      bool empty() const {
      return length <= 0;
    }

    __host__ __device__
      T next() {
      T x = operator[](0);
      advance(stride);
      length--;
      return x;
    }
  };

template<typename T, int D>
__host__ __device__
  uniform_nested_sequence<T, D> slice(uniform_nested_sequence<T, D> seq, int offset)
{
  return uniform_nested_sequence<T, D>(seq.length, seq.stride, slice(seq.sub, offset));
}

template<typename T>
__host__ __device__
uniform_nested_sequence<T, 0> slice(uniform_nested_sequence<T, 0> seq, int offset)
{
  return uniform_nested_sequence<T, 0>(seq.length, seq.stride, seq.data, seq.offset + offset);
}
