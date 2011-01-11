#pragma once

template<typename F, typename S>
struct scanned_sequence {
    S sequence;
    typedef typename S::value_type value_type;
    typedef value_type T;
    T accumulator;
    F fn;
    __host__ __device__
    scanned_sequence(F _fn, S _sequence, T prefix) : fn(_fn), sequence(_sequence), accumulator(prefix) {}
    
    __host__ __device__
    int size() {
        return sequence.size();
    }
    
    __host__ __device__
    bool empty() {
        return (sequence.size() > 0);
    }
    
    __host__ __device__
    T next() {
        T result = fn(accumulator, sequence.next());
        accumulator = result;
        return result;
    }
};

template<typename S>
struct summed_sequence {
    S sequence;
    typedef typename S::value_type value_type;
    typedef value_type T;
    T accumulator;
    __host__ __device__
    summed_sequence(S _sequence) : sequence(_sequence), accumulator(0) {}

    __host__ __device__
    int size() {
        return sequence.size();
    }

    __host__ __device__
    bool empty() {
        return (sequence.size() <= 0);
    }

    __host__ __device__
    T next() {
        T result = accumulator + sequence.next();
        accumulator = result;
        return result;
    }
  
};
