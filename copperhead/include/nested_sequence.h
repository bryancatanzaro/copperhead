
template<typename T, int Depth>
struct nested_sequence
{
    typedef T value_type;
    static const int nesting_depth = Depth;

    stored_sequence<int> desc;
    nested_sequence<T, Depth-1> sub;

    __host__ __device__
    nested_sequence() { }
  
    __host__ __device__
    nested_sequence(stored_sequence<int> _desc,
                    nested_sequence<T, Depth-1> _sub)
        : desc(_desc), sub(_sub) { }


    __host__ __device__
    int size() const { return desc.size()-1; }

    __host__ __device__
    nested_sequence<T,Depth-1> operator[](int i)
    {
        int begin=desc[i], end=desc[i+1];
        return slice(sub, begin, end-begin);
    }

    __host__ __device__
    bool empty() const {
      return desc.length <= 1;
    }

    __host__ __device__
    nested_sequence<T, Depth-1> next() {
      nested_sequence<T, Depth-1> x = operator[](0);
      desc.data++;
      desc.length--;
      return x;
    }
};

template<typename T, int D>
__host__ __device__
nested_sequence<T,D> slice(nested_sequence<T,D> seq, int base, int len)
{
    return nested_sequence<T,D>(slice(seq.desc, base, len+1), seq.sub);
}

template<typename T>
struct nested_sequence<T, 1>
{
    typedef T value_type;
    static const int nesting_depth = 1;

    stored_sequence<int> desc;
    stored_sequence<T>   data;

    __host__ __device__
    nested_sequence() { }
    
    __host__ __device__
    nested_sequence(stored_sequence<int> _desc,
                    stored_sequence<T>   _data)
        : desc(_desc), data(_data) { }

    __host__ __device__
    int size() const { return desc.size()-1; }

    __host__ __device__
    stored_sequence<T> operator[](int i)
    {
        int begin=desc[i], end=desc[i+1];
        return slice(data, begin, end-begin);
    }

    __host__ __device__
      bool empty() const {
      return desc.length <= 1;
    }

    __host__ __device__
      stored_sequence<T> next() {
      stored_sequence<T> x = operator[](0);
      desc.data++;
      desc.length--;
      return x;
    }
};


template<typename T>
__host__ __device__
nested_sequence<T,1> slice(nested_sequence<T,1> seq, int base, int len)
{
    return nested_sequence<T,1>(slice(seq.desc, base, len+1), seq.data);
}

template<typename T>
__host__ __device__
nested_sequence<T,1> split(stored_sequence<T> seq, stored_sequence<int> desc)
{
    return nested_sequence<T,1>(desc, seq);
}

template<typename T, int D>
__host__ __device__
nested_sequence<T,D+1> split(nested_sequence<T,D> seq,
                             stored_sequence<int> desc)
{
    return nested_sequence<T,D+1>(desc, seq);
}

template<typename T>
__host__ __device__
stored_sequence<T> join(nested_sequence<T,1> seq) { return seq.data; }

template<typename T, int D>
__host__ __device__
nested_sequence<T,D-1> join(nested_sequence<T,D> seq) { return seq.sub; }
