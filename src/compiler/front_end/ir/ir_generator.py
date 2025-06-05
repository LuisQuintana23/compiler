from llvmlite import ir
from ..analysis.semantic import SymbolTable, Symbol, TypeKind, SemanticAnalyzer
from ..analysis.parser import parse_c_code, create_parser, read_c_file
from lark import Tree, Token


class LLVMGenerator:
    def __init__(self, symbol_table: SymbolTable):
        self.symbol_table = symbol_table
        self.module = ir.Module(name="main_module")
        self.builder = None
        self.printf = None
        self.string_count = 0
        self.named_values = {}

    def declare_printf(self):
        voidptr_ty = ir.IntType(8).as_pointer()
        printf_ty = ir.FunctionType(ir.IntType(32), [voidptr_ty], var_arg=True)
        self.printf = ir.Function(self.module, printf_ty, name="printf")

    def emit_global_string(self, string_val: str) -> ir.Value:
        byte_val = bytearray(string_val.encode("utf8"))
        byte_val.append(0)  # null terminator
        str_type = ir.ArrayType(ir.IntType(8), len(byte_val))
        str_const = ir.Constant(str_type, byte_val)
        var_name = f".str{self.string_count}"
        self.string_count += 1

        global_str = ir.GlobalVariable(self.module, str_type, name=var_name)
        global_str.linkage = 'internal'
        global_str.global_constant = True
        global_str.initializer = str_const

        return self.builder.bitcast(global_str, ir.IntType(8).as_pointer())

    def extract_int_literal(self, node):
        """recursively searches for the first NUMBER token and converts it to int."""
        if isinstance(node, Token) and node.type == 'NUMBER':
            return int(node.value)
        elif isinstance(node, Tree):
            for child in node.children:
                val = self.extract_int_literal(child)
                if val is not None:
                    return val
        return None

    def generate_main(self):
        main_ty = ir.FunctionType(ir.IntType(32), [])
        main_fn = ir.Function(self.module, main_ty, name="main")
        block = main_fn.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(block)

        self.declare_printf()

        global_symbols = self.symbol_table.get_scope_symbols(0)
        for sym in global_symbols:
            if sym.name == "main" and sym.type.kind == TypeKind.FUNCTION:
                self.generate_function_body(sym)

    def generate_function_body(self, sym: Symbol):
        if sym.body is None:
            return

        # find function scope
        func_scope_id = None
        for succ in self.symbol_table.graph.successors(0):
            if self.symbol_table.graph.nodes[succ].get("name") == f"function_{sym.name}":
                func_scope_id = succ
                break

        block_scope_ids = list(self.symbol_table.graph.successors(func_scope_id or 0))
        for scope_id in block_scope_ids:
            for local_sym in self.symbol_table.get_scope_symbols(scope_id):
                if local_sym.type.kind == TypeKind.POINTER and local_sym.type.base_type.kind == TypeKind.INT:
                    ptr = self.builder.alloca(ir.IntType(32), name=local_sym.name)
                    self.named_values[local_sym.name] = ptr

        for stmt in sym.body.children:
            self.generate_statement_recursive(stmt)

    def generate_statement_recursive(self, node):
        if isinstance(node, Tree):
            if node.data in ['assignment_expression', 'function_call', 'return_statement']:
                self.generate_statement(node)
            else:
                for child in node.children:
                    self.generate_statement_recursive(child)

    def generate_statement(self, node: Tree):
        if node.data == 'assignment_expression':
            var_name = node.children[0].value  # IDENTIFIER
            value = self.extract_int_literal(node.children[1])
            var_ptr = self.named_values.get(var_name)
            if var_ptr and value is not None:
                self.builder.store(ir.Constant(ir.IntType(32), value), var_ptr)

        elif node.data == 'function_call':
            func_name = node.children[0].value
            if func_name == "printf":
                string_token = node.children[1].children[0].children[0]
                printf_str = string_token.value.strip('"').encode().decode('unicode_escape')
                str_ptr = self.emit_global_string(printf_str)
                self.builder.call(self.printf, [str_ptr])

        elif node.data == 'return_statement':
            return_val = self.extract_int_literal(node.children[1])
            if return_val is not None:
                self.builder.ret(ir.Constant(ir.IntType(32), return_val))

    def generate(self):
        self.generate_main()
        return self.module


def main():
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m compiler.front_end.ir.ir_generator <c_file>")
        sys.exit(1)

    code = read_c_file(sys.argv[1])
    if code is None:
        sys.exit(1)

    parser = create_parser()
    tree = parse_c_code(parser, code)

    if tree is None:
        print("Error parsing code")
        sys.exit(1)

    analyzer = SemanticAnalyzer()
    success = analyzer.analyze(tree)

    if success:
        llvm_gen = LLVMGenerator(analyzer.symbol_table)
        llvm_ir = llvm_gen.generate()
        print(str(llvm_ir))
    else:
        for err in analyzer.errors:
            print(err)
        sys.exit(1)


if __name__ == "__main__":
    main()
