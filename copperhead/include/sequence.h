#pragma once
#include <iostream>

#include "stored_sequence.h"
#include "nested_sequence.h"
#include "tiled_sequence.h"
#include "index_sequence.h"
#include "lowered_sequence.h"
#include "shifted_sequence.h"
#include "uniform_nested_sequence.h"
#include "iterator_sequence.h"
#include "constant_sequence.h"
#include "scanned_sequence.h"
#include "copy.h"

template<typename T>
std::ostream& operator<<(std::ostream& out, stored_sequence<T> in)
{
    out << "[";
    while( !in.empty() )  out << " " << in.next();
    return out << " ]";
}

template<typename T, int D>
std::ostream& operator<<(std::ostream& out, nested_sequence<T,D> in)
{
    out << "[";
    for(int i=0; i<in.size(); ++i)
        out << " " << in[i];
    return out << " ]";
}

template<typename T, int D>
std::ostream& operator<<(std::ostream& out, tiled_sequence<T,D> in)
{
    out << "[";
    for(int i=0; i<in.size(); ++i)
        out << " " << in[i];
    return out << " ]";
}


template<typename A, typename B>
  __device__ __host__ void copy(A destination, B source, int index) {
  destination[index] = source[index];
}
