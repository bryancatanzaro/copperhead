#
#  Copyright 2008-2010 NVIDIA Corporation
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

import coresyntax as S
import coretypes as T
import backendsyntax as B

def markGenerated(variableName):
    if variableName[0] == '_':
        return variableName
    return '_' + variableName

def generateName(id):
    return S.Name(markGenerated(id))

def variableLength(variableName):
    return S.Name(str(variableName) + '.size()')

def variableData(variableName):
    return S.Name(str(variableName) + '.data')

def generateDescData(id, level):
    return S.Name(str(id) + '_desc_%s_data' % level)

def generateDescLength(id, level):
    return S.Name(str(id) + '_desc_%s_length' % level)

def generateDescStored(id, level):
    if level is not None:
        return S.Name(str(id) + '_stored_%s' % level)
    return S.Name(str(id))

def generateDesc(id, level=None):
    if level is not None:
        return S.Name(str(id) + '_desc_%s' % level)
    else:
        return S.Name(str(id))

def generateData(id):
    return S.Name(str(id) + '_data')

def generateLength(id, level=None):
    if level is not None:
        return S.Name(str(id) + '_length_' + str(level))
    return S.Name(str(id) + '_length')

def generateOffset(id):
    return S.Name(str(id) + '_offset')

def generateStride(id, level=None):
    if level is not None:
        return S.Name(str(id) + '_stride_' + str(level))
    return S.Name(str(id) + '_stride')

def type_from_name(name):
    id = str(name)
    return '_T' + markGenerated(id)

indexType = T.Monotype('int')
lengthType = T.Monotype('int')
globalIndex = generateName('globalIndex')
globalIndexType = indexType
tileBegin = generateName('tileBegin')
tileBeginType = indexType
tileIndex = generateName('tileIndex')
tileIndexType = indexType
tileId = generateName('tileId')
tileIdType = indexType



anonymousReturnValue = generateName('return')
counterType = indexType
blockIdx = S.Name('blockIdx.x')
threadIdx = S.Name('threadIdx.x')
blockDim = S.Name('blockDim.x')

BLOCKSIZE = S.Name('BLOCKSIZE')
def structifyName(id):
    return S.Name(markGenerated(str(id)) + 'Struct')

def instance_name(id):
    return S.Name(str(id) + 'inst')

applyOperator = S.Name('operator()')

import backendsyntax as B



def call(fn, *arguments):
    return S.Apply(S.Name(fn), list(arguments))







def arrayAccess(arrayName):
    return '[' + arrayName + ']'

def variableCached(variableName):
    return markGenerated(variableName) + 'Cached'


def markRegister(variableName):
    return markGenerated(variableName + 'Reg')

def markScratch(variableType):
    if (variableType.__class__.__name__ == 'str'):
        return markGenerated('scratch' + variableType)
    return markGenerated('scratch' + repr(variableType.getScalarType()))

def arrayType(variableType):
    return 'Array<' + variableType + '>'

def makeCacheStruct(variableType, variableName):
    return arrayType(variableType) + '(' + variableCached(variableName) + ', ' + variableLength(variableName) + ')'

def clampRange(indexName, arrayName):
    return '(' + indexName + ' > ' + variableLength(arrayName) + ') ? ' + variableLength(arrayName) + ' : ' + indexName


def phaseToDevice(functionName, phase):
    return functionName + str(phase) + 'Dev'

def pointerize(typename):
    return str(typename) + '*'

def declareAndLoadCache(printer, symbols, typeDict, variableName):
    cachedArrayName = variableCached(variableName)
    cachedArrayType = typeDict[repr(symbols[variableName].getScalarType())]
    printer.printTermLine(' '.join(['__shared__', cachedArrayType, cachedArrayName + '[BLOCKSIZE]']))
    cacheArrayChunkArgs = [variableData(variableName), cachedArrayName, blockMin, clampRange(blockMax, variableName), globalIndex]
    printer.printTermLine('cacheArrayChunk(' + ', '.join(cacheArrayChunkArgs) + ')')

def declareScratch(printer, typeDict, scratchType):
    externalType = typeDict[scratchType]
    printer.printTermLine(' '.join(['__shared__', externalType, markScratch(scratchType) + '[BLOCKSIZE]']))
    
def openActiveBlock(printer, arrayName):
    printer.openLine('if (' + blockMin + ' < ' + arrayName + '.length)')
    printer.openBlock()

def openIfBlock(printer, condition):
    printer.openLine('if (' + condition + ')')
    printer.openBlock()

def declareVariable(printer, name, typeString):
    printer.printTermLine(typeString + ' ' + name)

def initStragglers(array):
    return (globalIndex + ' >= ', array)

def blockGuard(array):
    return S.Name(str(blockIdx) + ' < ' + str(variableLength(array)))

def threadGuard(array):
    return S.Name(str(globalIndex) + ' < ' + str(variableLength(array)))

def thread_zero(stmt):
    return S.Name(str(threadIdx) + ' == 0')

def make_tuple(inputs):
    # The extra S.Name() in here is because we don't yet have a CApply
    # Look at the behavior of S.Apply.__str__()  for details
    return S.Apply(S.Name(B.CNamespace(S.Name('thrust'), S.Name('make_tuple'))), inputs)

def tuple_get(idx, idf, arg):
    unbox = S.Bind(idf, S.Apply(S.Name(B.TemplateInst(
        B.CNamespace(S.Name('thrust'), S.Name('get')),
        [S.Number(idx)])), [arg]))
    unbox.no_return_convert = True
    return unbox
