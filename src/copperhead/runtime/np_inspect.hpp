/*
 *   Copyright 2012      NVIDIA Corporation
 * 
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 * 
 *       http://www.apache.org/licenses/LICENSE-2.0
 * 
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 * 
 */
#pragma once
#include <tuple>
#include "type.hpp"
#include <boost/python.hpp>

typedef std::tuple<void*, size_t, std::shared_ptr<const backend::type_t>, boost::python::object> np_array_info;

np_array_info inspect_array(PyObject* in);
bool isnumpyarray(PyObject* in);
boost::python::object convert_to_array(PyObject* in);
