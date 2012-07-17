#
#   Copyright 2012      NVIDIA Corporation
# 
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
# 
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
# 
#
from copperhead import *
import numpy as np
import unittest
from create_tests import create_tests

@cu
def fn(xa, xb):
    return xa + xb + 1

@cu
def test_scan(x):
    return scan(fn, x)

@cu
def test_rscan(x):
    return rscan(fn, x)

@cu
def test_exclusive_scan(x):
    return exclusive_scan(fn, cast_to_el(0, x), x)

@cu
def test_exclusive_rscan(x):
    return exclusive_rscan(fn, cast_to_el(0, x), x)

class ScanTest(unittest.TestCase):
    def setUp(self):
        self.source = [1,2,3,4,5]

    def run_test(self, target, fn, *args):
        
        python_result = fn(*args, target_place=places.here)
        copperhead_result = fn(*args, target_place=target)
        self.assertEqual(list(python_result), list(copperhead_result))
        
    @create_tests(*runtime.backends)
    def testScan(self, target):
        self.run_test(target, test_scan, self.source)

    @create_tests(*runtime.backends)
    def testRscan(self, target):
        self.run_test(target, test_rscan, self.source)

    @create_tests(*runtime.backends)
    def testExscan(self, target):
        self.run_test(target, test_exclusive_scan, self.source)

    @create_tests(*runtime.backends)
    def testExrscan(self, target):
        self.run_test(target, test_exclusive_rscan, self.source)
        
        
if __name__ == "__main__":
    unittest.main()
