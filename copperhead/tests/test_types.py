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

"""
Simple tests of type primitives.

These tests exercise the coretypes module of the Copperhead compiler.
They do not depend on any other module, but simply check that the core
functionality provided by the types module are functioning correctly.
"""

import unittest

from copperhead.compiler.coretypes import *

class CoretypeTests(unittest.TestCase):

    def setUp(self):
        self.types = \
            [
                Typevar('x'),
                Tuple(Int, Typevar('a'), Bool, Typevar('b')),
                Fn((Int, Typevar('a')), Bool),
                Fn((Int, Typevar('a')), Typevar('b')),
                Polytype(['a', 'b'], Fn((Int, Typevar('a')), Typevar('b'))),
                Polytype(['a', 'b'], Fn((Int, 'a'), 'b')),
                Polytype(['a', 'b'], Fn((Typevar('c'), Typevar('a')), Typevar('b'))),
            ]

        self.strings = \
            [
                'x'                 ,
                '(Int, a, Bool, b)' ,
                '(Int, a) -> Bool'  ,
                '(Int, a) -> b'     ,
                'ForAll a, b: (Int, a) -> b',
                'ForAll a, b: (Int, a) -> b',
                'ForAll a, b: (c, a) -> b',
            ]

        self.occurring = \
            [
                ['x'],
                ['a', 'b'],
                ['a'],
                ['a', 'b'],
                ['a', 'b'],
                ['a', 'b'],
                ['a', 'b', 'c'],
            ]

        self.free = \
            [
                ['x'],
                ['a', 'b'],
                ['a'],
                ['a', 'b'],
                [],
                [],
                ['c'],
            ]

    def tearDown(self): pass

    def testTypevarAsString(self):
        self.assertEqual(Typevar('x'), 'x')
        self.assertEqual(Typevar('aLongTypeVariable'), 'aLongTypeVariable')

    def testTypeStrings(self):
        for t, s in zip(self.types, self.strings):
            self.assertEqual(str(t), s)

    def testOccurringVariables(self):
        for t, vars in zip(self.types, self.occurring):
            names = sorted(list(names_in_type(t)))
            self.assertEqual(names, vars)

    def testFreeVariables(self):
        for t, vars in zip(self.types, self.free):
            names = sorted(list(free_in_type(t)))
            self.assertEqual(names, vars)

    def testOccursCheck(self):
        self.assert_(occurs('a', 'a'))
        self.assert_(not occurs('a', Fn((Int,Int), Bool)))
        self.assert_(not occurs('a', Fn(('b','c'), 'a0')))
        self.assert_(occurs('a', Fn(('b',Seq('a')), 'b')))
        self.assert_(occurs('a', Polytype(['b','a'], Seq(Tuple('a', 'b')))))
        self.assert_(not occurs('a', Polytype(['b','a'], Seq(Int))))


    def testSubstitution(self):
        S = lambda t: substituted_type(t, {'x':'y','z':'u','zz':'v'})

        self.assertEqual(S('x'), 'y')
        self.assertEqual(S('y'), 'y')
        self.assertEqual(str(S(Polytype(['a','b'], Tuple('a','b','c',
            'z')))),
            'ForAll a, b: (a, b, c, u)')



if __name__ == "__main__":
    unittest.main()
