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

import backendsyntax as B
import backtypes as BT
import coretypes as T
import coresyntax as S
import codesnippets as C

def seq_convert(bind, functors):
    _return = bind.binder()
    apply_node = bind.value()
    if isinstance(apply_node, S.Map):
        return _map(apply_node.parameters, _return, functors)
    if isinstance(apply_node, S.Index) or isinstance(apply_node, S.Subscript):
        return bind
    try:
        fn_name = str(apply_node.function())
        fn = eval('_' + fn_name)
    except:
        #If we don't know how to convert this function, return it unchanged
        return bind
    return fn(apply_node.arguments(), _return, functors, context=apply_node.context)

def _gather(arguments, _return, functors, **kwargs):
    (source, indices) = arguments
    type = BT.DependentType(T.Monotype('gathered'), [C.type_from_name(source),
                                                     C.type_from_name(indices)])
    typedef = B.CTypedef(type, C.type_from_name(_return))
    typedecl = B.CTypeDecl(C.type_from_name(_return), _return)
    apply_node = B.CConstructor(S.Name('gather'),
                    [source, indices])
    # XXX This won't allow us to return a gather from a sequential function
    cbind = B.CBind(typedecl, apply_node)
    return [typedef, cbind]

def _map(arguments, _return, functors, **kwargs):
    fn, args = (arguments[0], arguments[1:])
    instantiate = hasattr(fn, 'type')
    functors.add(str(fn))
    arity = len(args)
    transformed = T.Monotype('transformed%s' % arity)
    if instantiate:
        # Don't mark this function as a generated function
        # Later on, the code generator turns all types into
        # Marked types.  But since this is a user generated object
        # it should not be marked.
        fn.no_mark = True
        dependent_list = [fn] + [C.type_from_name(x) for x in args]
    else:
        dependent_list = [C.type_from_name(x) for x in arguments]
    type = BT.DependentType(transformed, dependent_list)
    typedef = B.CTypedef(type, C.type_from_name(_return))
    typedecl = B.CTypeDecl(C.type_from_name(_return), _return)
    #XXX I need to compute the type of the result of the transform function
    # This information is hard to find, so for now I'm just using the type
    # of the first argument to map.
    # This precludes the use of transforms which have a different output type
    # than their inputs, and needs to be fixed
    
    result_type = B.CTypename(B.CNamespace(C.type_from_name(args[0]),
                                           S.Name('value_type')))

    if instantiate:
        xform_args = [B.CConstructor(arguments[0], [])] + arguments[1:]
    else:
        xform_args = arguments
    apply_node = B.CConstructor(B.TemplateInst(S.Name('transform'),
                                          [result_type]),
                           xform_args)
    # XXX This won't allow us to return a map from a sequential function
    
    cbind = B.CBind(typedecl, apply_node)
   
    return [typedef, cbind]

def _sum(arguments, _return, functors, **kwargs):
    result_id = _return.id
    summed_seq_name = C.markGenerated(result_id + '_impl_seq')
    constant_seq_name = C.markGenerated(result_id + '_result_seq')
    input_argument = arguments[0]
    
    input_argument_type_id = C.type_from_name(input_argument)
    summed_seq_type_id = C.type_from_name(summed_seq_name)
    input_argument_type = T.Monotype(input_argument_type_id)
    summed_seq_type = T.Monotype(summed_seq_type_id)
    constant_seq_type_id = C.type_from_name(constant_seq_name)
    constant_seq_type = T.Monotype(constant_seq_type_id)
    
    summed_seq_typedef = B.CTypedef(BT.DependentType(
        T.Monotype('summed_sequence'), [input_argument_type]),
                                    summed_seq_type_id)
    constant_seq_typedef = B.CTypedef(BT.DependentType(
        T.Monotype('constant_sequence'),
        [B.CTypename(B.CNamespace(summed_seq_type_id,
                                  S.Name('value_type')))]),
                                      constant_seq_type_id)
    summed_seq_inst = B.CTypeDecl(summed_seq_type,
                                  B.CConstructor(S.Name(summed_seq_name),
                                                 [input_argument]))
    constant_seq_inst = B.CTypeDecl(constant_seq_type,
                                    B.CConstructor(S.Name(constant_seq_name),
                                                   [S.Number(0),
                                                    S.Number(1)]))
    return [summed_seq_typedef, summed_seq_inst,
            constant_seq_typedef, constant_seq_inst]
   

def _reduce(arguments, _return, functors, **kwargs):
    raise NotImplementedException("XXX Sequential/Blockwise reduce not implemented")
    # #Instantiates either a sequential or block-wise reduction
    # fn, data, prefix = arguments
    # instantiate = hasattr(fn, 'type')
    # functors.add(str(fn))
    # if instantiate:
    #     fn = B.CConstructor(fn, [])
    # apply_node = B.CConstructor(B.CNamespace(str(kwargs['context']),
    #                                     S.Name('reduce')),
    #                        (fn, data, prefix))
    # if _return.id != C.anonymousReturnValue.id:
    #     result_type = B.CTypename(B.CNamespace(C.type_from_name(arguments[0]),
    #                                            S.Name('value_type')))
    #     typedecl = B.CTypeDecl(result_type, _return)
    #     cbind = B.CBind(typedecl, apply_node)
    # else:
    #     cbind = B.CBind(_return, apply_node)
    # return cbind
