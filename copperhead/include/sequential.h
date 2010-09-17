#pragma once

#include "transform.h"
#include "gather.h"

namespace sequential {

template<typename Seq>
__host__ __device__
typename Seq::value_type sum(Seq x)
{
    typename Seq::value_type total = 0;
    while( !x.empty() )
        total += x.next();
    return total;
}

 template<typename Seq, typename Fn, typename T>
   __host__ __device__
   T reduce(Fn f, Seq x, T prefix) {
   T accumulator = prefix;
   while ( !x.empty() )
     accumulator = f(accumulator, x.next());
   return accumulator;
 }
 
template<typename Values, typename Indices, typename Result>
__host__ __device__
void gather(Values v, Indices i, Result result)
{
    for(int k=0; k<i.size(); ++k)
        result[k] = v[i[k]];
}

 
}
