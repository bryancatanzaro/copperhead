#pragma once
  
template<typename Fn,
         typename SeqA,
         typename T=typename SeqA::value_type>
struct _transformed1
{
    typedef T value_type;

    Fn   f;
    SeqA a;

    __host__ __device__
    _transformed1(Fn _f, SeqA _a) : f(_f), a(_a) { }

    __host__ __device__
    int size() const { return a.size(); }

    __host__ __device__
    bool empty() const { return a.empty(); }

    __host__ __device__
    value_type next() { return f(a.next()); }

    __host__ __device__
    value_type operator[](int index) { return f(a[index]); }
};



template<typename Fn,
         typename SeqA,
         typename SeqB=SeqA,
         typename T=typename SeqA::value_type>
struct _transformed2
{
    typedef T value_type;

    Fn   f;
    SeqA a;
    SeqB b;

    __host__ __device__
    _transformed2(Fn _f, SeqA _a, SeqB _b) : f(_f), a(_a), b(_b) { }

    __host__ __device__
    int size() const { return a.size(); }

    __host__ __device__
    bool empty() const { return a.empty(); }

    __host__ __device__
    value_type next() { return f(a.next(), b.next()); }

    __host__ __device__
    value_type operator[](int index) { return f(a[index], b[index]); }
};

template<typename Fn,
         typename SeqA,
         typename SeqB=SeqA,
         typename SeqC=SeqA,
         typename T=typename SeqA::value_type>
struct _transformed3
{
    typedef T value_type;

    Fn   f;
    SeqA a;
    SeqB b;
    SeqC c;

    __host__ __device__
    _transformed3(Fn _f, SeqA _a, SeqB _b, SeqC _c) : f(_f), a(_a), b(_b), c(_c) { }

    __host__ __device__
    int size() const { return a.size(); }

    __host__ __device__
    bool empty() const { return a.empty(); }

    __host__ __device__
    value_type next() { return f(a.next(), b.next(), c.next()); }

    __host__ __device__
    value_type operator[](int index) { return f(a[index], b[index], c[index]); }
};



template<typename T, typename Fn, typename SeqA>
__host__ __device__
_transformed1<Fn, SeqA, T> transform(Fn f, SeqA a)
{
    return _transformed1<Fn, SeqA, T>(f, a);
}
 
template<typename T, typename Fn, typename SeqA, typename SeqB>
__host__ __device__
_transformed2<Fn, SeqA, SeqB, T> transform(Fn f, SeqA a, SeqB b)
{
    return _transformed2<Fn, SeqA, SeqB, T>(f, a, b);
}

template<typename T, typename Fn, typename SeqA, typename SeqB, typename SeqC>
__host__ __device__
_transformed3<Fn, SeqA, SeqB, SeqC, T> transform(Fn f, SeqA a, SeqB b, SeqC c)
{
    return _transformed3<Fn, SeqA, SeqB, SeqC, T>(f, a, b, c);
}

