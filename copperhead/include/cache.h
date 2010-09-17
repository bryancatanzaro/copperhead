/*
 *  Copyright 2008-2009 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
#ifndef CACHE_H
#define CACHE_H
/**
 * This function loads a chunk of an array into shared memory.
 * It doesn't perform a __syncthreads() after it has loaded -
 * so a __syncthreads() must be performed prior to using the cached chunk.
 *
 * Also, it assumes chunkMin and chunkMax point to a valid range of the array.
 *
 * @param array The pointer to the array in global memory
 * @param arrayCached The pointer to the chunk in shared memory
 * @param chunkMin The index of the first element of the array which will
 *   be cached (inclusive)
 * @param chunkMax No element of the array with index >= chunkMax will be
 *   cached.
 */
template<typename T>
__device__ void cacheArrayChunk(T* array, T* arrayCached, unsigned int chunkMin, unsigned int chunkMax, unsigned int globalIndex) {
  int myIndex = globalIndex;
  //Load the data into the cache in parallel, ensuring coalesced access
  while (myIndex < chunkMax) {
    arrayCached[myIndex - chunkMin] = array[myIndex];
    myIndex += blockDim.x;
  }
}


/**
 * This function accesses an array which has been partially cached.
 *
 * This function assumes the index is valid - if it's not, you'll dereference an out of bounds pointer.
 *
 * @param array The pointer to the array in global memory
 * @param arrayCached The pointer to the chunk in shared memory
 * @param chunkMin The index of the first element of the array which is
 *   be cached (inclusive)
 * @param chunkMax No element of the array with index >= chunkMax is
 *   cached.
 * @param index The index which needs to be accessed.
 * @return The element requested.
 */
template<typename T>
__device__ T cacheLookup(T* array, T* arrayCached, unsigned int chunkMin, unsigned int chunkMax, int index) {
  if ((index >= chunkMin) && (index < chunkMax)) {
    //Hit!!
    unsigned int cacheIndex = index - chunkMin;
    return arrayCached[cacheIndex];
  } else {
    return array[index];
  }
}

#endif
