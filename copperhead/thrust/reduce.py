#
#  Copyright 2009-2010 University of California
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

from copperhead.decorators import cutype, cubox
from copperhead.runtime import nvcc_toolchain, host_toolchain, cudata, CuFunction
from copperhead.runtime.cudata import induct
import copperhead.compiler.passes as p
import codepy.bpl
import codepy.cuda
from codepy.cgen import *
import copperhead.compiler.pltools as PL
import copperhead.compiler.utility as U

name_supply = PL.name_supply()

@cubox
@cutype("[a] -> a")
def sum(array, cache):
    array.using_remote()
    cu_result_type = array.type.unbox()
    result_type = cudata.cu_to_c_types[str(cu_result_type)]
    if result_type in cache:
        compiled_function = cache[result_type]
        return induct(compiled_function(array), cu_result_type)
    
    host_module = codepy.bpl.BoostPythonModule()
    host_module.add_function(
        FunctionBody(
            FunctionDeclaration(Value(result_type, "entryPoint"),
                                [Pointer(Value("PyObject", "cuArray"))]),
            Block([Statement(x) for x in [
                '%s prefix = (%s)0;' % (result_type, result_type),
                'CuArray input = extractArray(cuArray)',
                '%s result = sumInstance(input.devicePointer, input.length, prefix)' % result_type,
                'return result']])))
    host_module.add_to_preamble([Include('cuarray.h'),
                                 Include('boost/python/extract.hpp')])

    device_module = codepy.cuda.CudaModule(host_module)
    device_module.add_to_preamble([Include('wrappers/reduce.h')])
    sumInstance = FunctionBody(
        FunctionDeclaration(Value(result_type, 'sumInstance'),
                            [Value('CUdeviceptr', 'inputPtr'),
                             Value('int', 'length'),
                             Value(result_type, 'prefix')]),
        Block([Statement('return sum<%s>((%s*)inputPtr, length, prefix)' % (result_type, result_type))]))
    device_module.add_function(sumInstance)
    module = device_module.compile(host_toolchain, nvcc_toolchain, debug=True)
    cache[result_type] = module.entryPoint
    result = module.entryPoint(array)
    return induct(result, cu_result_type)
                        
@cubox
@cutype("((a, a) -> a, [a], a) -> a")
def reduce(fn, array, prefix, cache):
    if isinstance(array, tuple):
        arrays = [x for x in array]
        prefixes = [x for x in prefix]
    else:
        arrays = [array]
        prefixes = [prefix]
    input_types = [x.type for x in arrays] + [x.type for x in prefixes]

    for arr in arrays:
        arr.using_remote()
    if isinstance(fn, CuFunction):
        functor_name = fn.fn.__name__
        user_functor = True
    else:
        functor_name = fn.__name__
        user_functor = False
   
    result_types = [x.type for x in prefixes]
    c_result_types = [cudata.cu_to_c_types[str(x)] for x in result_types]
    cu_result_type = prefix.type
    signature = functor_name + ','.join(c_result_types)

    arguments = tuple(arrays + [x.value for x in prefixes])

    if signature in cache:
        compiled_function = cache[signature]
        return induct(compiled_function(arguments), cu_result_type)
    if user_functor:
        functor = p.get_functor(fn)

    

    array_names = [name_supply.next() for x in arrays]
    prefix_names = [name_supply.next() for x in prefixes]
    host_arguments = [Pointer(Value("PyObject", 'args'))]
    array_decls = [Statement('PyObject* %s' % x) for x in array_names]
    prefix_decls = [Statement('%s %s' % (x, y)) for x, y in zip(c_result_types, prefix_names)]
    py_return_types = ''.join((cudata.cu_to_py_types[str(x)] for x in result_types))
    arg_tuple_string = 'O' * len(array_names) + ''.join(py_return_types)
    arg_parse = Statement('PyArg_ParseTuple(args, "%s", ' % arg_tuple_string + \
                          ', '.join(['&'+x for x in array_names] +
                                    ['&'+x for x in prefix_names]) + ')')
    
    cuarray_names = [x + '_array' for x in array_names]
    cuarray_decls = [Statement('CuArray %s = extractArray(%s)' % (x, y)) for x,y \
                     in zip(cuarray_names, array_names)]
   
        
    result_names = [name_supply.next() for x in prefixes]
    result_decls = [Statement('%s %s' % (x, y)) for x, y \
                              in zip(c_result_types, result_names)]
    device_pointers = ['%s.devicePointer' % x for x in cuarray_names]
    device_lengths = ['%s.length' %x for x in cuarray_names]
    
    device_arrays = list(U.interleave(device_pointers, device_lengths))
    device_args = device_arrays + prefix_names + result_names
    instantiation = Statement('reduceInstance(' + ', '.join(device_args) + ')')
   
    return_stmt = Statement('return Py_BuildValue("%s", %s)' % (py_return_types,
                                                  ', '.join(result_names)))
    call = [instantiation, return_stmt]
    
    host_code = array_decls + prefix_decls + [arg_parse] + cuarray_decls + \
                result_decls + call
    host_module = codepy.bpl.BoostPythonModule()
    host_module.add_function(
        FunctionBody(
            FunctionDeclaration(Pointer(Value("PyObject", "entryPoint")),
                                host_arguments),
            Block(host_code)))
    host_module.add_to_preamble([Include('cuarray.h'),
                                 Include('boost/python/extract.hpp')])

    device_module = codepy.cuda.CudaModule(host_module)
    
    if user_functor:
        device_module.add_to_module([Line(functor)])
    else:
        device_module.add_to_preamble([Include('operators.h')])
    device_module.add_to_preamble([Include('thrust/tuple.h'),
                                   Include('thrust/device_ptr.h'),
                                   Include('thrust/reduce.h')])
    device_pointer_names = [name_supply.next() for x in cuarray_names]
    device_length_names = [name_supply.next() for x in cuarray_names]
    device_prefix_names = [name_supply.next() for x in prefixes]
    device_return_names = [name_supply.next() for x in result_names]
    device_args = list(U.interleave([Value('CUdeviceptr', x) \
                                     for x in device_pointer_names],
                                    [Value('int', x) \
                                     for x in device_length_names])) + \
                                     [Value(x, y) for x, y in \
                                      zip(c_result_types, device_prefix_names)] + \
                                      [Reference(Value(x, y)) for x, y in \
                                       zip(c_result_types, device_return_names)]
    thrust_ptr_types = ['typename thrust::device_ptr<%s>' % x for x in c_result_types]
    thrust_ptr_names = [x + '_thrust' for x in device_pointer_names]
    thrust_ptr_decls = [Statement('%s %s((%s*)%s)' % \
                                  (x, y, z, w)) for x, y, z, w in \
                                  zip(thrust_ptr_types, \
                                      thrust_ptr_names,
                                      c_result_types,
                                      device_pointer_names)]
    if len(result_types) == 1:
        thrust_begin_iterator = thrust_ptr_names[0]
        thrust_end_iterator = thrust_ptr_names[0] + ' + ' + device_length_names[0]
        thrust_prefix = device_prefix_names[0]
        thrust_result = device_return_names[0]
        thrust_unpacking = []
    else:
        thrust_begin_iterator = 'thrust::make_zip_iterator(thrust::make_tuple(' + \
                                ', '.join(thrust_ptr_names) + '))'
        thrust_end_iterator = 'thrust::make_zip_iterator(thrust::make_tuple(' + \
                              ', '.join([x + ' + ' + y for x, y in \
                                         zip(thrust_ptr_names, device_length_names)]) + '))'
        thrust_prefix = 'thrust::make_tuple(' + ', '.join(device_prefix_names) + ')'
        thrust_result_name = name_supply.next()
        thrust_result = 'typename thrust::tuple<' + ', '.join(c_result_types) + '> ' + \
                        thrust_result_name
        thrust_unpacking = [Statement(x + ' = thrust::get<%s>(%s) ' \
                                       % (y, thrust_result_name)) \
                                      for (y, x) in enumerate(device_return_names)] 
    thrust_functor = functor_name + '()'
    thrust_call = [Statement(thrust_result + ' = ' + 'thrust::reduce(' + \
                            ', '.join([thrust_begin_iterator,
                                       thrust_end_iterator,
                                       thrust_prefix,
                                       thrust_functor]) + ')')]
    device_body = thrust_ptr_decls + thrust_call + thrust_unpacking
    
    
    reduceInstance = FunctionBody(
        FunctionDeclaration(Value('void', 'reduceInstance'),
                            device_args),
        Block(device_body))
    device_module.add_function(reduceInstance)
    module = device_module.compile(host_toolchain, nvcc_toolchain)
    cache[signature] = module.entryPoint
    result = module.entryPoint(arguments)
    result = cudata.induct(result, prefix.type)
    return result
    
    
    
    
