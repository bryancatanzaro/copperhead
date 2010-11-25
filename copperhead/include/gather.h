#pragma once

template<typename ValueSeq, typename IndexSeq=stored_sequence<int> >
struct _gathered
{
    typedef typename ValueSeq::value_type value_type;

    ValueSeq values;
    IndexSeq indices;

    __host__ __device__
    _gathered(ValueSeq _values, IndexSeq _indices)
        : values(_values), indices(_indices) { }

    __host__ __device__
    int size() const { return indices.size(); }

    __host__ __device__
    bool empty() const { return indices.empty(); }

    __host__ __device__
    value_type operator[](int i) { return values[indices[i]]; }

    __host__ __device__
    value_type next() { return values[indices.next()]; }
};


template<typename Values, typename Indices>
__host__ __device__
_gathered<Values,Indices> gather(Values v, Indices i)
{
    return _gathered<Values,Indices>(v, i);
}
