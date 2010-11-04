#pragma once
#include <iterator_sequence.h>
#include <thrust/iterator/counting_iterator.h>


struct index_sequence : public iterator_sequence<thrust::counting_iterator<int> >
{
  __host__ __device__ index_sequence(int _length) :
  iterator_sequence<thrust::counting_iterator<int> >(thrust::counting_iterator<int>(0), _length) { }
};
    
  
  
