import backendcompiler as BC
import conversions

def execute(ast, M):
    assert(len(M.entry_points) == 1)
    entry_point = M.entry_points[0]
    backend_ast = conversions.front_to_back_node(ast)
    c = BC.Compiler(entry_point)
    result = c(backend_ast)
    M.device_code = result
    M.wrap_info = ((BC.wrap_result_type(), BC.wrap_name()),
                   zip(BC.wrap_arg_types(), BC.wrap_arg_names()))
    return []
