from llvmlite import ir
from compiler.front_end.analysis.semantic import SymbolTable, Symbol, TypeKind, SemanticAnalyzer
from compiler.front_end.analysis.parser import parse_c_code, create_parser, read_c_file
from lark import Tree, Token
import argparse

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

    def generate_expression(self, node):
        if isinstance(node, Token):
            if node.type == 'NUMBER':
                return ir.Constant(ir.IntType(32), int(node.value))
            elif node.type == 'IDENTIFIER':
                ptr = self.named_values.get(node.value)
                if ptr is None:
                    raise ValueError(f"Identificador '{node.value}' no declarado.")
                return self.builder.load(ptr)

        elif isinstance(node, Tree):
            if node.data == 'expr':
                if len(node.children) == 1:
                    return self.generate_expression(node.children[0])
                left = self.generate_expression(node.children[0])
                op = node.children[1]
                right = self.generate_expression(node.children[2])
                if left is None or right is None:
                    raise ValueError(f"Expresión mal formada en: {node.pretty()}")
                if op.type == 'PLUS':
                    return self.builder.add(left, right)
                elif op.type == 'MINUS':
                    return self.builder.sub(left, right)
                elif op.type == 'EQ':
                    return self.builder.icmp_signed('==', left, right)
                elif op.type == 'NEQ':
                    return self.builder.icmp_signed('!=', left, right)

            elif node.data == 'term':
                if len(node.children) == 1:
                    return self.generate_expression(node.children[0])
                left = self.generate_expression(node.children[0])
                op = node.children[1]
                right = self.generate_expression(node.children[2])
                if left is None or right is None:
                    raise ValueError(f"Término mal formado en: {node.pretty()}")
                if op.type == 'STAR':
                    return self.builder.mul(left, right)
                elif op.type == 'SLASH':
                    return self.builder.sdiv(left, right)
                elif op.type == 'PERCENT':
                    return self.builder.srem(left, right)

            elif node.data == 'factor':
                return self.generate_expression(node.children[0])

        return None

    def generate_statement(self, node: Tree):
        if node.data == 'assignment_expression':
            var_name = node.children[0].value
            value = self.generate_expression(node.children[1])
            var_ptr = self.named_values.get(var_name)
            if var_ptr and value is not None:
                self.builder.store(value, var_ptr)

        elif node.data == 'function_call':
            func_name = node.children[0].value
            if func_name == "printf":
                args = []
                argument_list_node = None
                for child in node.children:
                    if isinstance(child, Tree) and child.data == 'argument_list':
                        argument_list_node = child
                        break

                if argument_list_node:
                    for i, arg in enumerate(argument_list_node.children):
                        if isinstance(arg, Tree) and arg.data == 'argument':
                            if i == 0 and isinstance(arg.children[0], Token) and arg.children[0].type == 'STRING':
                                fmt_str = arg.children[0].value.strip('"').encode().decode('unicode_escape')
                                str_ptr = self.emit_global_string(fmt_str)
                                args.append(str_ptr)
                            elif i > 0:
                                expr_node = arg.children[0] if arg.children else None
                                if expr_node:
                                    arg_val = self.generate_expression(expr_node)
                                    if arg_val is not None:
                                        args.append(arg_val)

                if len(args) >= 2:
                    self.builder.call(self.printf, args)
                else:
                    raise ValueError("printf llamado sin suficientes argumentos para formato.")

        elif node.data == 'return_statement':
            if len(node.children) > 1:
                return_val = self.generate_expression(node.children[1])
                if return_val is not None:
                    self.builder.ret(return_val)
            else:
                self.builder.ret(ir.Constant(ir.IntType(32), 0))

    def generate(self):
        self.generate_main()
        return self.module

def generate_ir(args: argparse.Namespace):
    import sys

    code = read_c_file(args.input)
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
        try:
            llvm_gen = LLVMGenerator(analyzer.symbol_table)
            llvm_ir = llvm_gen.generate()
            print(str(llvm_ir))
        except Exception as e:
            print(f"\u274c [GENERATOR ERROR] {e}")
            sys.exit(1)
    else:
        for err in analyzer.errors:
            print(err)
        sys.exit(1)
