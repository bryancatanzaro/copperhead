from copperhead import *
import numpy as np
import unittest
import random

@cu
def lt_sort(x):
    return sort(cmp_lt, x)

@cu
def gt_sort(x):
    return sort(cmp_gt, x)

class SortTest(unittest.TestCase):
    def setUp(self):
        self.source = np.array([random.random() for x in range(5)], dtype=np.float32)
        
        
    def run_test(self, fn, *args):
        cpuResult = fn(*args, target_place=places.here)
        gpuResult = fn(*args, target_place=places.gpu0)
        self.assertEqual(list(cpuResult), list(gpuResult))
    
    def testLtSort(self):
        self.run_test(lt_sort, self.source)

    def testGtSort(self):
        self.run_test(gt_sort, self.source)


if __name__ == "__main__":
    unittest.main()

