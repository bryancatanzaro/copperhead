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
#ifndef SHIFT_H
#define SHIFT_H


/**
 * Returns an element from the array after shifting by "offset".  Offset
 * shifts to the left, so right shifting is done with negative offset.
 * "boundary" is the value of the elements which will be shifted into the array.
 * @param array The pointer to the array
 * @param arrayLength The length of the array
 * @param index The global index of the thread
 * @param offset The amount to shift
 * @param boundary The value to shift in
 */
template<typename T>
__device__ T shift(T* array, unsigned int arrayLength, int index, int offset, T boundary) {
  int globalIndex = index + offset;
  if ((globalIndex < 0) || (globalIndex >= arrayLength)) {
    return boundary;
  }
  return array[globalIndex];
}

/**
 * Returns an element from the array after shifting by "offset".  Offset
 * shifts to the left, so right shifting is done with negative offset.
 * "boundary" is the value of the elements which will be shifted into the array.
 * This version assumes that a chunk of the array has been cached in shared
 * memory.
 * @param array The pointer to the array in global memory
 * @param arrayLength The length of the array
 * @param arrayCached The pointer to the array chunk in shared memory
 * @param chunkMin The index of the first element of the array which is
 *   cached (inclusive)
 * @param chunkMax No element of the array with index >= chunkMax is
 *   cached.
 * @param index The global index of the thread
 * @param offset The amount to shift
 * @param boundary The value to shift in
 */

template<typename T>
__device__ T shiftCached(T* array, unsigned int arrayLength, T* arrayCached, unsigned int cacheMin, unsigned int cacheMax, unsigned int index, int offset, T boundary) {
  int globalIndex = index + offset;
  if ((globalIndex < 0) || (globalIndex >= arrayLength)) {
    return boundary;
  }
  if ((globalIndex >= cacheMin) && (globalIndex < cacheMax)) {
    return arrayCached[globalIndex - cacheMin];
  }
  return array[globalIndex];
}


/**
 * Returns an element from the array after rotating by "offset".  Offset
 * shifts to the left, so right shifting is done with negative offset.
 * @param array The pointer to the array
 * @param arrayLength The length of the array
 * @param index The global index of the thread
 * @param offset The amount to rotate
 */
template<typename T>
__device__ T rotate(T* array, unsigned int arrayLength, int index, int offset, T boundary) {
  int globalIndex = index + offset;
  while (globalIndex < 0) {
    globalIndex += arrayLength;
  }
  while (globalIndex >= arrayLength) {
    globalIndex -= arrayLength;
  }
  return array[globalIndex];
}

/**
 * Returns an element from the array after rotating by "offset".  Offset
 * rotates to the left, so right shifting is done with negative offset.
 * This version assumes that a chunk of the array has been cached in shared
 * memory.
 * @param array The pointer to the array in global memory
 * @param arrayLength The length of the array
 * @param arrayCached The pointer to the array chunk in shared memory
 * @param chunkMin The index of the first element of the array which is
 *   cached (inclusive)
 * @param chunkMax No element of the array with index >= chunkMax is
 *   cached.
 * @param index The global index of the thread
 * @param offset The amount to rotate
 */

template<typename T>
__device__ T rotateCached(T* array, unsigned int arrayLength, T* arrayCached, unsigned int cacheMin, unsigned int cacheMax, int index, int offset) {
  int globalIndex = index + offset;
  
  while (globalIndex < 0) {
    globalIndex += arrayLength;
  }
  while (globalIndex >= arrayLength) {
    globalIndex -= arrayLength;
  }
  
  if ((globalIndex >= cacheMin) && (globalIndex < cacheMax)) {
    return arrayCached[globalIndex - cacheMin];
  }
  return array[globalIndex];
}


#endif
