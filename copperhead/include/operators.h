#pragma once
#include <cmath>

struct op_add {
  template<typename T>
  __device__ __host__ T operator()(T a, T b) {
    return a + b;
  }
  template<typename T>
  __device__ __host__ void operator()(T a, T b, T& o) {
    o = a + b;
  }
};

struct op_sub {
  template<typename T>
  __device__ __host__ T operator()(T a, T b) {
    return a - b;
  }
  template<typename T>
  __device__ __host__ void operator()(T a, T b, T& o) {
    o = a - b;
  }
};

struct op_mul {
  template<typename T>
  __device__ __host__ T operator()(T a, T b) {
    return a * b;
  }
  template<typename T>
  __device__ __host__ void operator()(T a, T b, T& o) {
    o = a * b;
  }
};

struct op_div {
  template<typename T>
  __device__ __host__ T operator()(T a, T b) {
    return a / b;
  }
  template<typename T>
  __device__ __host__ void operator()(T a, T b, T& o) {
    o = a / b;
  }
};


template<typename T>
struct op_mod {
  typedef T value_type;
  __device__ __host__ T operator()(T a, T b) {
    return a % b;
  }
};

template<typename T>
struct op_pow {
  typedef T value_type;
  __device__ __host__ T operator()(T a, T b) {
    return pow(a, b);
  }
};

template<typename T>
struct op_lshift {
  typedef T value_type;
  __device__ __host__ T operator()(T a, T b) {
    return a << b;
  }
};

template<typename T>
struct op_rshift {
  typedef T value_type;
  __device__ __host__ T operator()(T a, T b) {
    return a >> b;
  }
};

template<typename T>
struct op_or {
  typedef T value_type;
  __device__ __host__ T operator()(T a, T b) {
    return a | b;
  }
};

template<typename T>
struct op_xor {
  typedef T value_type;
  __device__ __host__ T operator()(T a, T b) {
    return a ^ b;
  }
};

template<typename T>
struct op_and {
  typedef T value_type;
  __device__ __host__ T operator()(T a, T b) {
    return a & b;
  }
};

template<typename T>
struct op_invert {
  typedef T value_type;
  __device__ __host__ T operator()(T a) {
    return ~a;
  }
};

template<typename T>
struct op_pos {
  typedef T value_type;
  __device__ __host__ T operator()(T a) {
    return +a;
  }
};

template<typename T>
struct op_neg {
  typedef T value_type;
  __device__ __host__ T operator()(T a) {
    return -a;
  }
};

struct op_not {
  typedef bool value_type;
  __device__ __host__ bool operator()(bool a) {
    return !a;
  }
};

template<typename T>
struct cmp_eq {
  typedef bool value_type;
  __device__ __host__ bool operator()(T a, T b) {
    return a == b;
  }
};

template<typename T>
struct cmp_neq {
  typedef bool value_type;
  __device__ __host__ bool operator()(T a, T b) {
    return a != b;
  }
};

template<typename T>
struct cmp_lt {
  typedef bool value_type;
  __device__ __host__ bool operator()(T a, T b) {
    return a < b;
  }
};

template<typename T>
struct cmp_lte {
  typedef bool value_type;
  __device__ __host__ bool operator()(T a, T b) {
    return a <= b;
  }
};

template<typename T>
struct cmp_gt {
  typedef bool value_type;
  __device__ __host__ bool operator()(T a, T b) {
    return a > b;
  }
};

template<typename T>
struct cmp_gte {
  typedef bool value_type;
  __device__ __host__ bool operator()(T a, T b) {
    return a >= b;
  }
};


