import backendcompiler as BC
import conversions

def execute(ast, M):
    assert(len(M.entry_points) == 1)
    entry_point = M.entry_points[0]
    backend_ast = conversions.front_to_back_node(ast)
    c = BC.Compiler(entry_point)
    result = c(backend_ast)
    M.device_code = result
    M.wrap_info = ((c.wrap_result_type(), c.wrap_name()),
                   zip(c.wrap_arg_types(), c.wrap_arg_names()))
    return []
