#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

template<class T>
T *get_pointer(thrust::device_vector<T>& data)
{
    return thrust::raw_pointer_cast(&data.front());
}

template<class T>
T *get_pointer(thrust::host_vector<T>& data)
{
    return thrust::raw_pointer_cast(&data.front());
}

template<typename T>
stored_sequence<T> get_sequence(thrust::device_vector<T>& data)
{
    return stored_sequence<T>(get_pointer(data), data.size());
}

template<typename T>
stored_sequence<T> get_sequence(thrust::host_vector<T>& data)
{
    return stored_sequence<T>(get_pointer(data), data.size());
}
