#pragma once

namespace block {

template<typename Seq>
__device__
typename Seq::value_type sum(Seq x)
{
  typedef typename Seq::value_type T;
  __shared__ T scratch[BLOCKSIZE];
  T accumulator = (T)0;
  int index = threadIdx.x;
  while (index < x.size()) {
    accumulator += x[index];
    index += blockDim.x;
  }
  scratch[threadIdx.x] = accumulator;
  __syncthreads();
  if (BLOCKSIZE > 256) {
    if (threadIdx.x < 256) scratch[threadIdx.x] += scratch[threadIdx.x + 256];
    __syncthreads();
  }
  if (BLOCKSIZE > 128) {
    if (threadIdx.x < 128) scratch[threadIdx.x] += scratch[threadIdx.x + 128];
    __syncthreads();
  }
  if (BLOCKSIZE >  64) {
    if (threadIdx.x <  64) scratch[threadIdx.x] += scratch[threadIdx.x +  64];
    __syncthreads();
  }
  if (BLOCKSIZE >  32) {
    if (threadIdx.x <  32) scratch[threadIdx.x] += scratch[threadIdx.x +  32];
    __syncthreads();
  }
  if (threadIdx.x <  16) scratch[threadIdx.x] += scratch[threadIdx.x +  16];
  if (threadIdx.x <   8) scratch[threadIdx.x] += scratch[threadIdx.x +   8];
  if (threadIdx.x <   4) scratch[threadIdx.x] += scratch[threadIdx.x +   4];
  if (threadIdx.x <   2) scratch[threadIdx.x] += scratch[threadIdx.x +   2];
  if (threadIdx.x <   1) scratch[threadIdx.x] += scratch[threadIdx.x +   1];
  __syncthreads();
  return scratch[0];

}

 template<typename Seq, typename Fn, typename T>
__device__
T reduce(Fn f, Seq x, T prefix)
{
  __shared__ T scratch[BLOCKSIZE];
  int index = threadIdx.x;
  //block_length is the number of valid threads in this block
  //if x.size() is longer than BLOCKSIZE, all threads are valid
  //but if not, we have to keep track of which threads have valid data.
  int block_length = BLOCKSIZE;
  if (x.size() < block_length)
    block_length = x.size();
  //Early exit for degenerate sequences
  if (block_length == 0)
    return prefix;
  // Initialize sequential reduction
  T accumulator;
  if (index < block_length)
    accumulator = x[index];
  index += blockDim.x;
  // Perform sequential reductions
  while (index < x.size()) {
    accumulator = f(accumulator, x[index]);
    index += blockDim.x;
  }
  scratch[threadIdx.x] = accumulator;
  __syncthreads();
      
  // Commutative reduction tree                   
  if (BLOCKSIZE > 256) {
    if (threadIdx.x + 256 < block_length) scratch[threadIdx.x] = f(scratch[threadIdx.x], scratch[threadIdx.x + 256]);
    __syncthreads();
  }
  if (BLOCKSIZE > 128) {
    if (threadIdx.x + 128 < block_length) scratch[threadIdx.x] = f(scratch[threadIdx.x], scratch[threadIdx.x + 128]);
    __syncthreads();
  }
  if (BLOCKSIZE >  64) {
    if (threadIdx.x +  64 < block_length) scratch[threadIdx.x] = f(scratch[threadIdx.x], scratch[threadIdx.x +  64]);
    __syncthreads();
  }
  if (BLOCKSIZE >  32) {
    if (threadIdx.x +  32 < block_length) scratch[threadIdx.x] = f(scratch[threadIdx.x], scratch[threadIdx.x +  32]);
    __syncthreads();
  }
  if (threadIdx.x +  16 < block_length) scratch[threadIdx.x] = f(scratch[threadIdx.x], scratch[threadIdx.x +  16]);
  if (threadIdx.x +   8 < block_length) scratch[threadIdx.x] = f(scratch[threadIdx.x], scratch[threadIdx.x +   8]);
  if (threadIdx.x +   4 < block_length) scratch[threadIdx.x] = f(scratch[threadIdx.x], scratch[threadIdx.x +   4]);
  if (threadIdx.x +   2 < block_length) scratch[threadIdx.x] = f(scratch[threadIdx.x], scratch[threadIdx.x +   2]);
  if (threadIdx.x +   1 < block_length) scratch[threadIdx.x] = f(scratch[threadIdx.x], scratch[threadIdx.x +   1]);
  __syncthreads();
  // We can add the prefix out of order because this is a commutative reduction
  return f(prefix, scratch[0]);

}
 
}
