#pragma once
#include <thrust/reduce.h>
#include <cuda.h>
#include <copperhead.h>
#include <thrust/scan.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/scatter.h>

template<typename Seq>
typename Seq::value_type sum(Seq &input) {
  return thrust::reduce(input.begin(), input.begin() + input.length, (typename Seq::value_type)0);
}

template<typename Seq, typename T, typename F>
T reduce(F fn, Seq &input, T prefix) {
  return thrust::reduce(input.begin(), input.begin() + input.length, prefix, fn);
}

template<typename Seq>
void sum_scan(Seq& input, Seq& output) {
  thrust::inclusive_scan(input.begin(), input.begin() + input.size(),
                         output.begin());
}

template<typename Seq, typename F>
void scan(F functor, Seq& input, Seq& output) {
  thrust::inclusive_scan(input.begin(), input.begin() + input.size(),
                         output.begin(), functor);
}

template<typename Seq, class F>
void rscan(F functor, Seq& input, Seq& output) {
  typedef typename Seq::iterator I;
  thrust::reverse_iterator<I> drbegin(input.end());
  thrust::reverse_iterator<I> drend(input.begin());
  thrust::reverse_iterator<I> orbegin(output.end());
  thrust::inclusive_scan(drbegin, drend, orbegin, functor);
}

template<typename SeqD, typename SeqI>
void permute(SeqD& source, SeqI& map, SeqD& result) {
  thrust::scatter(source.begin(), source.end(),
                  map.begin(), result.begin());

}
