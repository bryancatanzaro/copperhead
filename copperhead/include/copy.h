#pragma once

template<typename S0, typename O0>
  __host__ __device__
  void copy(S0 &s0, O0 &o0) {
  while (!s0.empty()) {
    o0.next() = s0.next();
  }
}


template<typename S0, typename S1, typename O0, typename O1>
  __host__ __device__
  inline void copy(S0 &s0, S1 &s1, O0 &o0, O1 &o1) {
  if (s0.size() == s1.size()) {
    //Detecting the sequences are the same size is a critical performance
    //optimization.  If the sequences aren't the same size, the C++
    //compiler will implement the fused loop without fusing the computations,
    //due to the extra conditionals.
    while (!s0.empty()) {
      o0.next() = s0.next();
      o1.next() = s1.next();
    }
  } else {
    bool a;
    bool b;
    while ((a=!s0.empty()) || (b=!s1.empty())) {
      if (a)
        o0.next() = s0.next();
      if (b)
        o1.next() = s1.next();
    }
  }
}
