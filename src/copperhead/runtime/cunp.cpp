#include "cunp.hpp"
#include "np_inspect.hpp"
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>
#include <stdexcept>
#include "monotype.hpp"
#include "type_printer.hpp"
#include <boost/python/extract.hpp>


using std::shared_ptr;
using namespace backend;
using std::make_tuple;
using std::make_shared;

extern "C" {
void initialize_cunp() {
    //This initializes Numpy
    import_array();
}
}

shared_ptr<const type_t> np_to_cu(NPY_TYPES n) {
    switch(n) {
    case NPY_BOOL:
        return bool_mt;       
    case NPY_INT:
        return int32_mt;
    case NPY_LONG:
        return int64_mt;
    case NPY_FLOAT:
        return float32_mt;
    case NPY_DOUBLE:
        return float64_mt;
    default:
        return void_mt;
    }
}

bool isnumpyarray(PyObject* in) {
    return PyArray_Check(in);
}

boost::python::object convert_to_array(PyObject* in) {
    PyObject* array = PyArray_FROM_OTF(in, NPY_NOTYPE, NPY_IN_ARRAY);
    
    //Record that we have a reference to the result of PyArray_FROM_OTF
    //Putting it in a boost::python::object ensures that reference counting
    //happens automatically for this object, even in the presence of exceptions
    boost::python::handle<> array_handle(array);
    return boost::python::object(array_handle);
}


np_array_info inspect_array(PyObject* in) {
    boost::python::object bp_object = convert_to_array(in);
    PyObject* input_array = bp_object.ptr();
    
    if (!(PyArray_Check(input_array))) {
        return make_tuple((void*)NULL, 0, void_mt, boost::python::object());
    }
    
    PyArrayObject* input_as_array = (PyArrayObject*)input_array;
    
    ssize_t n = input_as_array->dimensions[0];
    void* d = input_as_array->data;
    NPY_TYPES dtype = NPY_TYPES(PyArray_TYPE(input_as_array));
    shared_ptr<const type_t> t = np_to_cu(dtype);
    
    return make_tuple(d, n, make_shared<const sequence_t>(t), bp_object);
}


//Instantiate scalar packings
PyObject* make_scalar(const float& s) {
    PyObject* result = PyArrayScalar_New(Float);
    PyArrayScalar_ASSIGN(result, Float, s);
    return result;
}

PyObject* make_scalar(const double& s) {
    PyObject* result = PyArrayScalar_New(Double);
    PyArrayScalar_ASSIGN(result, Double, s);
    return result;
}

PyObject* make_scalar(const int& s) {
    PyObject* result = PyArrayScalar_New(Int);
    PyArrayScalar_ASSIGN(result, Int, s);
    return result;
}

PyObject* make_scalar(const long& s) {
    PyObject* result = PyArrayScalar_New(Long);
    PyArrayScalar_ASSIGN(result, Long, s);
    return result;
}

PyObject* make_scalar(const bool& s) {
    if (s) {
        PyArrayScalar_RETURN_TRUE;
    } else {
        PyArrayScalar_RETURN_FALSE;
    }
}


float unpack_scalar_float(PyObject* s) {
    return PyArrayScalar_VAL(s, Float);
}

double unpack_scalar_double(PyObject* s) {
    return PyArrayScalar_VAL(s, Double);
}

int unpack_scalar_int(PyObject* s) {
    return PyArrayScalar_VAL(s, Int);
}

long unpack_scalar_long(PyObject* s) {
    return PyArrayScalar_VAL(s, Long);
}

bool unpack_scalar_bool(PyObject* s) {
    return PyArrayScalar_VAL(s, Bool);
}

copperhead::sp_cuarray unpack_array(PyObject* a) {
    return boost::python::extract<copperhead::sp_cuarray>(a);
}

PyObject* objectify(const copperhead::sp_cuarray& s) {
    return boost::python::converter::shared_ptr_to_python(s);
}
