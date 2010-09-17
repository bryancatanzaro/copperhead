template<typename T, int Depth>
struct tiled_sequence
{
    typedef T value_type;
    static const int nesting_depth = Depth;

    tiled_sequence<T, Depth-1> sub;
    int tilesize;
    int tilecount;
    int offset;

    __host__ __device__
    tiled_sequence(tiled_sequence<T, Depth-1> _sub, int _tilesize)
        : sub(_sub), tilesize(_tilesize),
          tilecount( (_sub.size()+_tilesize-1) / _tilesize ),
          offset(0) { }

    __host__ __device__
    tiled_sequence(tiled_sequence<T, Depth-1> _sub, int _tilesize,
                   int _tilecount, int _offset)
        : sub(_sub), tilesize(_tilesize),
          tilecount(_tilecount),
          offset(_offset) { }


    __host__ __device__
    int size() const { return tilecount; }

    __host__ __device__
    tiled_sequence<T,Depth-1> operator[](int i)
    {
        int base = tilesize*i + offset;
        int limit = min(sub.size()-base, tilesize);

        return slice(sub, base, limit);
    }
};

template<typename T, int D>
__host__ __device__
tiled_sequence<T,D> slice(tiled_sequence<T,D> seq, int base, int len)
{
    return tiled_sequence<T,D>(seq.sub,
                               seq.tilesize,
                               len,
                               seq.tilesize*base);
}

template<typename T>
struct tiled_sequence<T, 1>
{
    typedef T value_type;
    static const int nesting_depth = 1;

    stored_sequence<T> data;
    int tilesize;
    int tilecount;
    int offset;

    __host__ __device__
    tiled_sequence(stored_sequence<T> _data, int _tilesize)
        : data(_data), tilesize(_tilesize),
          tilecount( (_data.size()+_tilesize-1) / _tilesize ),
          offset(0) { }

    __host__ __device__
    tiled_sequence(stored_sequence<T> _data, int _tilesize,
                   int _tilecount, int _offset)
        : data(_data), tilesize(_tilesize),
          tilecount(_tilecount), offset(_offset) { }

    __host__ __device__
    int size() const { return tilecount; }

    __host__ __device__
    stored_sequence<T> operator[](int i)
    {
        int base = tilesize*i + offset;
        int limit = min(data.size()-base, tilesize);

        return slice(data, base, limit);
    }
};


template<typename T>
__host__ __device__
tiled_sequence<T,1> slice(tiled_sequence<T,1> seq, int base, int len)
{
    return tiled_sequence<T,1>(seq.data,
                               seq.tilesize,
                               len,
                               seq.tilesize*base);
}


template<typename T>
__host__ __device__
tiled_sequence<T,1> split(stored_sequence<T> seq, int tilesize)
{
    return tiled_sequence<T,1>(seq, tilesize);
}

template<typename T, int D>
__host__ __device__
tiled_sequence<T,D+1> split(tiled_sequence<T,D> seq, int tilesize)
{
    return tiled_sequence<T,D+1>(seq, tilesize);
}

template<typename T>
__host__ __device__
stored_sequence<T> join(tiled_sequence<T,1> seq)
{
    return slice(seq.data, seq.offset, seq.data.size() - seq.offset);
}

template<typename T, int D>
__host__ __device__
tiled_sequence<T,D-1> join(tiled_sequence<T,D> seq)
{
    return slice(seq.sub, seq.offset, seq.sub.size() - seq.offset);
}
