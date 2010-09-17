template<typename T>
struct lowered_sequence 
{
    typedef T value_type;

    T data;
    int length;

    __host__ __device__
    lowered_sequence() {}
  
    __host__ __device__
    lowered_sequence(T _data, int _length) : data(_data), length(_length) {}

    //
    // Methods supporting sequence interface
    //
    __host__ __device__
    T&       operator[](int index)       { return data; }
    __host__ __device__
    const T& operator[](int index) const { return data; }

    __host__ __device__
    int size() const { return length; }
};




