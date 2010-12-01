#pragma once
#include <utility>
#include <vector>
#include "bridge_structures.h"
#include "sequence.h"
#include <cuda.h>
#include <thrust/tuple.h>

namespace {
  typedef _CuSequence CuSequence;
  typedef _CuUniform CuUniform;
  typedef _CuScalar CuScalar;
  typedef _CuTuple CuTuple;
}

template<typename T>
stored_sequence<T> make_stored_sequence(CuArray cuarray) {
  return stored_sequence<T>((T*)cuarray.data, cuarray.length);
}
  
template<typename T>
stored_sequence<T> extract_stored_sequence(CuSequence remote_arrays) {
  CuArray remote_array = remote_arrays[0];
  return make_stored_sequence<T>(remote_array);
}


//Why is this in a struct:
// To get around C++'s lack of partial function template specialization
template<typename T, int Depth>
  struct extract_nested_sequence_impl {
    static nested_sequence<T, Depth> work(CuSequence remote_arrays) {
      int overall_depth = remote_arrays.size() - 1;
      int current_index = overall_depth - Depth;
      CuArray desc_array = remote_arrays[current_index];
      stored_sequence<int> desc = make_stored_sequence<int>(desc_array);
      return nested_sequence<T, Depth>(desc, extract_nested_sequence_impl<T, Depth-1>::work(remote_arrays));
    }
  };

//Why is this in a struct:
// To get around C++'s lack of partial function template specialization
template<typename T>
struct extract_nested_sequence_impl<T, 1> {
  static nested_sequence<T, 1> work(CuSequence remote_arrays) {
    int overall_depth = remote_arrays.size() - 1;
    int current_index = overall_depth - 1;
    CuArray desc_array = remote_arrays[current_index];
    stored_sequence<int> desc = make_stored_sequence<int>(desc_array);
    CuArray data_array = remote_arrays[current_index + 1];
    stored_sequence<T> data = make_stored_sequence<T>(data_array);
    return nested_sequence<T, 1>(desc, data);
  }
};

template<typename T, int Depth>
  nested_sequence<T, Depth> extract_nested_sequence(CuSequence remote_arrays) {
  return extract_nested_sequence_impl<T, Depth>::work(remote_arrays);
}

//Why is this in a struct:
// To get around C++'s lack of partial function template specialization
template<typename T, int Depth>
  struct extract_uniform_nested_sequence_impl {
    static uniform_nested_sequence<T, Depth> work(CuUniform input) {
      IntVector extents = input.extents;
      IntVector strides = input.strides;
      int overall_depth = extents.size();
      int current_index = overall_depth - Depth - 1;
      return uniform_nested_sequence<T, Depth>(extents[current_index],
                                               strides[current_index],
                                               extract_uniform_nested_sequence_impl<T, Depth-1>::work(input));
    }
  };

//Why is this in a struct:
// To get around C++'s lack of partial function template specialization
template<typename T>
struct extract_uniform_nested_sequence_impl<T, 0> {
  static uniform_nested_sequence<T, 0> work(CuUniform input) {
    IntVector extents = input.extents;
    IntVector strides = input.strides;
    int overall_depth = extents.size();
    int current_index = overall_depth - 1;
    T* data = (T*)input.data;
    int offset = input.offset;
    return uniform_nested_sequence<T, 0>(extents[current_index],
                                         strides[current_index],
                                         data,
                                         offset);
  }
};

template<typename T, int Depth>
  uniform_nested_sequence<T, Depth> extract_uniform_nested_sequence(CuUniform input) {
  return extract_uniform_nested_sequence_impl<T, Depth>::work(input);
}


template<typename T>
T& extract_scalar(CuScalar in) {
  return *((T*)in.value);
}

template<typename T0>
thrust::tuple<T0> extract_tuple(CuTuple in) {
  return thrust::make_tuple(extract_scalar<T0>(in[0]));
}

template<typename T0, typename T1>
  thrust::tuple<T0, T1> extract_tuple(CuTuple in) {
  return thrust::make_tuple(extract_scalar<T0>(in[0]),
                            extract_scalar<T1>(in[1]));
}

template<typename T0, typename T1, typename T2>
  thrust::tuple<T0, T1, T2> extract_tuple(CuTuple in) {
  return thrust::make_tuple(extract_scalar<T0>(in[0]),
                            extract_scalar<T1>(in[1]),
                            extract_scalar<T2>(in[2]));
}

template<typename T0, typename T1, typename T2, typename T3>
  thrust::tuple<T0, T1, T2, T3> extract_tuple(CuTuple in) {
  return thrust::make_tuple(extract_scalar<T0>(in[0]),
                            extract_scalar<T1>(in[1]),
                            extract_scalar<T2>(in[2]),
                            extract_scalar<T3>(in[3]));
}


template<typename T0>
void store_tuple(thrust::tuple<T0> in, CuTuple out) {
  T0& out0 = extract_scalar<T0>(out[0]);
  out0 = thrust::get<0>(in);
}

template<typename T0, typename T1>
void store_tuple(thrust::tuple<T0, T1> in, CuTuple out) {
  T0& out0 = extract_scalar<T0>(out[0]);
  T1& out1 = extract_scalar<T1>(out[1]);
  out0 = thrust::get<0>(in);
  out1 = thrust::get<1>(in);
}

template<typename T0, typename T1, typename T2>
void store_tuple(thrust::tuple<T0, T1, T2> in, CuTuple out) {
  T0& out0 = extract_scalar<T0>(out[0]);
  T1& out1 = extract_scalar<T1>(out[1]);
  T2& out2 = extract_scalar<T1>(out[2]);
  out0 = thrust::get<0>(in);
  out1 = thrust::get<1>(in);
  out2 = thrust::get<2>(in);
}

template<typename T0, typename T1, typename T2, typename T3>
void store_tuple(thrust::tuple<T0, T1, T2, T3> in, CuTuple out) {
  T0& out0 = extract_scalar<T0>(out[0]);
  T1& out1 = extract_scalar<T1>(out[1]);
  T2& out2 = extract_scalar<T2>(out[2]);
  T3& out3 = extract_scalar<T3>(out[3]);
  out0 = thrust::get<0>(in);
  out1 = thrust::get<1>(in);
  out2 = thrust::get<2>(in);
  out3 = thrust::get<3>(in);
}
