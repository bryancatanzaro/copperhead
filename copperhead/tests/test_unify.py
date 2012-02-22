#! /usr/bin/env python
#
#   Copyright 2008-2012 NVIDIA Corporation
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

import unittest

from copperhead.compiler.typeinference import *
from copperhead.compiler.unifier import unify

a, b, c, d = [T.Typevar(v) for v in "abcd"]

int2 = T.Tuple(T.Int, T.Int)


class UnificationTests(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def succeeds(self, s, t):
        tcon = TypingContext()
        unify(s, t, tcon)

    def fails(self, s, t):
        tcon = TypingContext()
        self.assertRaises(InferenceError, lambda: unify(s, t, tcon))

    def testLiterals(self):
        self.succeeds(T.Int, T.Int)
        self.succeeds(T.Float, T.Float)
        self.fails(T.Int, T.Float)

    def testVars(self):
        self.succeeds(a, T.Int)
        self.succeeds(T.Float, T.Typevar('a'))
        self.succeeds(a, a)
        self.succeeds(a, b)

    def testCombining(self):
        self.succeeds(T.Tuple(T.Int, T.Int), T.Tuple(T.Int, T.Int))
        self.succeeds(T.Tuple(T.Int, T.Int), T.Tuple(a, a))
        self.succeeds(T.Tuple(T.Int, T.Int), T.Tuple(a, b))
        self.fails(T.Tuple(T.Int, T.Int), T.Fn(T.Int, T.Int))


    def testFunctions(self):
        self.succeeds(T.Fn(T.Int, T.Int), T.Fn(T.Int, T.Int)) 
        self.succeeds(T.Fn(T.Int, T.Int), T.Fn([T.Int], T.Int)) 
        self.succeeds(T.Fn(T.Int, T.Int), T.Fn(T.Tuple(T.Int), T.Int)) 

        self.succeeds(T.Fn(T.Int, T.Int), T.Fn(a, a)) 
        self.succeeds(T.Fn(T.Int, T.Int), T.Fn(a, b)) 
        self.succeeds(T.Fn(a, T.Int), T.Fn(T.Int, b)) 


    def testPolytypes(self):
        self.succeeds(T.Polytype([a], T.Tuple(a,a)), int2)
        self.succeeds(T.Polytype(['a'], T.Tuple('a','a')), int2)
        self.succeeds(T.Polytype([a], T.Fn((a,a), a)), T.Fn(int2, T.Int))
        self.fails(T.Polytype([a], T.Fn((a,a), a)), T.Fn(int2, int2))
        self.succeeds(T.Polytype([a], T.Fn((a,a), a)), T.Fn(int2, b))

if __name__ == "__main__":
    unittest.main()
