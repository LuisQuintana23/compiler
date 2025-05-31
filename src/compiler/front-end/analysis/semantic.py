from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Union
from enum import Enum, auto
# to manage the symbol table with a graph
import networkx as nx
from lark import Tree, Token
import sys
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class TypeKind(Enum):
    INT = auto()
    CHAR = auto()
    VOID = auto()
    FLOAT = auto()
    DOUBLE = auto()
    POINTER = auto()
    ARRAY = auto()
    FUNCTION = auto()

    @classmethod
    def from_str(cls, type_str: str) -> 'TypeKind':
        type_map = {
            'int': cls.INT,
            'char': cls.CHAR,
            'void': cls.VOID,
            'float': cls.FLOAT,
            'double': cls.DOUBLE
        }
        return type_map.get(type_str, cls.INT)

@dataclass
class Type:
    kind: TypeKind
    base_type: Optional['Type'] = None
    array_size: Optional[int] = None
    return_type: Optional['Type'] = None
    param_types: List['Type'] = field(default_factory=list)

    def __str__(self) -> str:
        if self.kind == TypeKind.POINTER:
            return f"{self.base_type}*"
        elif self.kind == TypeKind.ARRAY:
            return f"{self.base_type}[{self.array_size}]"
        elif self.kind == TypeKind.FUNCTION:
            params = ", ".join(str(t) for t in self.param_types)
            return f"{self.return_type}({params})"
        return self.kind.name.lower()

@dataclass
class Symbol:
    name: str
    type: Type
    is_initialized: bool = False
    is_constant: bool = False
    scope_level: int = 0
    line: int = 0
    column: int = 0
    body: Optional[Tree] = None

    def __str__(self) -> str:
        return f"{self.name}: {self.type} (scope: {self.scope_level})"

class SymbolTable:
    def __init__(self):
        # here is where the nx graph is initialized
        self.graph = nx.DiGraph()
        self.current_scope = 0
        # global scope is 0
        # scopes are used to manage variable visibility and lifetime
        # each scope represents a region where variables are valid
        # scopes can be nested (global -> function -> block)
        # variables declared in inner scopes can shadow outer ones
        self.scope_stack: List[int] = [0]
        # add global scope
        # the global scope is the root of our scope hierarchy
        # all other scopes are descendants of the global scope
        self._add_scope(0, "global")
        logger.debug("Initialized symbol table with global scope")
        logger.debug(f"Graph nodes: {list(self.graph.nodes())}")
        logger.debug(f"Graph edges: {list(self.graph.edges())}")

    def _add_scope(self, scope_id: int, name: str):
        """add a new scope node to the graph
        scopes are nodes that represent variable visibility levels
        connected in a tree where parent scopes are visible to children"""
        self.graph.add_node(scope_id, name=name, type="scope")
        logger.debug(f"Added scope {scope_id} with name {name}")
        logger.debug(f"Graph nodes after adding scope: {list(self.graph.nodes())}")
        logger.debug(f"Graph edges after adding scope: {list(self.graph.edges())}")

    def enter_scope(self, name: str = None):
        """enter a new scope level
        called when entering a new code block (function, if, while)
        creates a child scope where variables are visible to inner scopes only"""
        new_scope = self.current_scope + 1
        # Create a new scope node with a unique ID
        scope_id = len(self.graph.nodes())
        self._add_scope(scope_id, name or f"scope_{scope_id}")
        # connect the new scope to its parent
        # this edge represents the "contains" relationship
        # parent scopes contain their children
        self.graph.add_edge(self.current_scope, scope_id, type="contains")
        self.scope_stack.append(scope_id)
        self.current_scope = scope_id
        logger.debug(f"Entered scope {scope_id} (parent: {self.current_scope})")
        logger.debug(f"Graph nodes after entering scope: {list(self.graph.nodes())}")
        logger.debug(f"Graph edges after entering scope: {list(self.graph.edges())}")
        # Log the attributes of all nodes to verify they haven't changed
        for node in self.graph.nodes():
            logger.debug(f"Node {node} attributes after entering scope: {self.graph.nodes[node]}")

    def exit_scope(self):
        """exit current scope
        called when leaving a code block
        variables in this scope are no longer accessible"""
        if len(self.scope_stack) > 1:
            old_scope = self.current_scope
            self.scope_stack.pop()
            self.current_scope = self.scope_stack[-1]
            logger.debug(f"Exited scope {old_scope}, now in scope {self.current_scope}")
            logger.debug(f"Graph nodes after exiting scope: {list(self.graph.nodes())}")
            logger.debug(f"Graph edges after exiting scope: {list(self.graph.edges())}")

    def add_symbol(self, symbol: Symbol) -> bool:
        """add a symbol to current scope
        symbols are only visible in current scope and children
        returns false if symbol already exists in current scope"""
        # check if symbol exists in current scope
        # this prevents redeclaration of variables in the same scope
        for node in self.graph.nodes():
            if self.graph.nodes[node].get('type') == 'symbol' and \
               self.graph.nodes[node].get('name') == symbol.name and \
               node in self.graph.successors(self.current_scope):
                logger.debug(f"Symbol {symbol.name} already exists in current scope")
                return False

        # add symbol node
        # each symbol is a node in our graph
        # it's connected to its scope with a "defines" edge
        symbol_id = len(self.graph.nodes())
        logger.debug(f"Adding symbol {symbol.name} with ID {symbol_id}")
        
        # Create the symbol node with explicit type='symbol'
        self.graph.add_node(symbol_id, 
                          type='symbol',  # Explicitly set type to 'symbol'
                          name=symbol.name,
                          symbol=symbol)
        
        # Connect the symbol to its scope
        self.graph.add_edge(self.current_scope, symbol_id, type="defines")
        
        # Log the node attributes to verify they are set correctly
        logger.debug(f"Added symbol node with attributes: {self.graph.nodes[symbol_id]}")
        logger.debug(f"Added symbol {symbol.name} to scope {self.current_scope}")
        logger.debug(f"Graph nodes after adding symbol: {list(self.graph.nodes())}")
        logger.debug(f"Graph edges after adding symbol: {list(self.graph.edges())}")
        logger.debug(f"Symbol table contents for scope {self.current_scope}:")
        for node in self.graph.successors(self.current_scope):
            node_attrs = self.graph.nodes[node]
            logger.debug(f"  Node {node} attributes: {node_attrs}")
            if node_attrs.get('type') == 'symbol':
                sym = node_attrs['symbol']
                logger.debug(f"  Symbol: {sym.name} (type: {sym.type.kind})")
        return True

    def lookup(self, name: str) -> Optional[Symbol]:
        """look up a symbol in current and outer scopes
        implements lexical scoping: inner scopes can access outer variables
        inner variables shadow (hide) outer ones with same name"""
        # get all ancestor scopes
        # this gives us all scopes that contain the current scope
        scopes = nx.ancestors(self.graph, self.current_scope)
        scopes.add(self.current_scope)
        logger.debug(f"Looking up symbol {name} in scopes: {scopes}")
        logger.debug(f"Current graph nodes: {list(self.graph.nodes())}")
        logger.debug(f"Current graph edges: {list(self.graph.edges())}")

        # search for symbol in all scopes
        # we search from innermost to outermost scope
        # this implements variable shadowing (inner variables hide outer ones)
        for scope in scopes:
            logger.debug(f"Checking scope {scope} for symbol {name}")
            for node in self.graph.successors(scope):
                if self.graph.nodes[node].get('type') == 'symbol' and \
                   self.graph.nodes[node].get('name') == name:
                    symbol = self.graph.nodes[node]['symbol']
                    logger.debug(f"Found symbol {name} in scope {scope}")
                    return symbol
        logger.debug(f"Symbol {name} not found in any scope")
        return None

    def get_scope_symbols(self, scope_id: int) -> List[Symbol]:
        """get all symbols in a specific scope
        returns variables and functions declared in the given scope"""
        symbols = []
        logger.debug(f"Getting symbols for scope {scope_id}")
        logger.debug(f"Current graph nodes: {list(self.graph.nodes())}")
        logger.debug(f"Current graph edges: {list(self.graph.edges())}")
        
        # Get all nodes that are directly connected to this scope
        for node in self.graph.successors(scope_id):
            node_attrs = self.graph.nodes[node]
            logger.debug(f"Checking node {node} in scope {scope_id}")
            logger.debug(f"Node attributes: {node_attrs}")
            
            # A symbol node should have type='symbol' and a 'symbol' attribute
            if node_attrs.get('type') == 'symbol' and 'symbol' in node_attrs:
                symbol = node_attrs['symbol']
                logger.debug(f"Found symbol in scope {scope_id}: {symbol.name} (type: {symbol.type.kind})")
                symbols.append(symbol)
            else:
                logger.debug(f"Node {node} is not a symbol (type: {node_attrs.get('type')})")
                # Log all attributes to help debug
                for key, value in node_attrs.items():
                    logger.debug(f"  {key}: {value}")
        
        logger.debug(f"Found {len(symbols)} symbols in scope {scope_id}")
        for symbol in symbols:
            logger.debug(f"  Symbol: {symbol.name} (type: {symbol.type.kind})")
        return symbols

    def visualize(self, filename: str = "symbol_table"):
        """generate a visualization of the symbol table"""
        import matplotlib.pyplot as plt
        
        pos = nx.spring_layout(self.graph)
        plt.figure(figsize=(12, 8))
        
        # draw scope nodes
        scope_nodes = [n for n in self.graph.nodes() 
                      if self.graph.nodes[n].get('type') == 'scope']
        nx.draw_networkx_nodes(self.graph, pos, 
                             nodelist=scope_nodes,
                             node_color='lightblue',
                             node_size=1000)
        
        # draw symbol nodes
        symbol_nodes = [n for n in self.graph.nodes() 
                       if self.graph.nodes[n].get('type') == 'symbol']
        nx.draw_networkx_nodes(self.graph, pos,
                             nodelist=symbol_nodes,
                             node_color='lightgreen',
                             node_size=500)
        
        # draw edges
        nx.draw_networkx_edges(self.graph, pos)
        
        # add labels
        labels = {n: self.graph.nodes[n].get('name', '') 
                 for n in self.graph.nodes()}
        nx.draw_networkx_labels(self.graph, pos, labels)
        
        plt.savefig(f"{filename}.png")
        plt.close()

class SemanticError(Exception):
    def __init__(self, message: str, line: int = None, column: int = None):
        self.message = message
        self.line = line
        self.column = column
        super().__init__(f"semantic error at line {line}, column {column}: {message}")

class SemanticAnalyzer:
    def __init__(self):
        self.symbol_table = SymbolTable()
        self.current_function = None
        self.errors: List[SemanticError] = []
        self.warnings: List[str] = []

    def print_tree(self, node, indent=0):
        """print the tree structure for debugging"""
        if isinstance(node, Tree):
            print('  ' * indent + f"tree: {node.data}")
            for child in node.children:
                self.print_tree(child, indent + 1)
        elif isinstance(node, Token):
            print('  ' * indent + f"token: {node.type} = {node.value}")

    def analyze(self, tree: Tree) -> bool:
        """analyze the parse tree for semantic errors"""
        try:
            self.visit(tree)
            return len(self.errors) == 0
        except SemanticError as e:
            self.errors.append(e)
            return False

    def visit(self, node: Any):
        """visit a node in the parse tree"""
        if isinstance(node, Tree):
            method_name = f'visit_{node.data}'
            visitor = getattr(self, method_name, self.generic_visit)
            return visitor(node)
        return node

    def generic_visit(self, node: Tree):
        """default visitor for nodes without specific handlers"""
        for child in node.children:
            self.visit(child)

    def get_type_from_node(self, node: Tree) -> Type:
        """extract type information from a type node"""
        if isinstance(node, Token):
            type_map = {
                'int': TypeKind.INT,
                'float': TypeKind.FLOAT,
                'char': TypeKind.CHAR,
                'void': TypeKind.VOID,
                'double': TypeKind.DOUBLE
            }
            type_kind = type_map.get(node.value)
            if type_kind is None:
                raise SemanticError(f"invalid type: {node.value}", 
                                  getattr(node, 'line', None),
                                  getattr(node, 'column', None))
            return Type(type_kind)
        elif isinstance(node, Tree):
            if node.data == 'type_specifier':
                # the type specifier tree should have a single child with the type token
                if len(node.children) == 1 and isinstance(node.children[0], Token):
                    return self.get_type_from_node(node.children[0])
        return Type(TypeKind.INT)

    def get_identifier_from_node(self, node: Tree) -> str:
        """extract identifier from a node, handling different tree structures"""
        if isinstance(node, Token):
            return node.value
        elif isinstance(node, Tree):
            if node.data == 'identifier':
                return node.children[0].value
            elif len(node.children) > 0:
                # recursively look for identifier in children
                return self.get_identifier_from_node(node.children[0])
        raise SemanticError("could not find identifier in node", 
                          getattr(node, 'line', None),
                          getattr(node, 'column', None))

    def visit_type_specifier(self, node: Tree):
        """handle type specifiers"""
        return self.get_type_from_node(node.children[0])

    def visit_declaration(self, node: Tree):
        """handle variable declarations"""
        type_node = node.children[0]
        if isinstance(type_node, Tree) and type_node.data == 'type_specifier':
            type_info = self.visit_type_specifier(type_node)
        else:
            type_info = self.get_type_from_node(type_node)
        
        identifier = node.children[1]
        
        # handle both token and tree cases for identifier
        if isinstance(identifier, Token):
            var_name = identifier.value
        else:
            var_name = self.get_identifier_from_node(identifier)
        
        # For basic types like int, we want to create a pointer type
        # since variables in C are always pointers to their type
        if type_info.kind in [TypeKind.INT, TypeKind.FLOAT, TypeKind.CHAR, TypeKind.DOUBLE]:
            type_info = Type(kind=TypeKind.POINTER, base_type=type_info)
        
        symbol = Symbol(
            name=var_name,
            type=type_info,
            line=getattr(identifier, 'line', 0),
            column=getattr(identifier, 'column', 0)
        )
        
        if not self.symbol_table.add_symbol(symbol):
            raise SemanticError(
                f"variable '{var_name}' already declared in this scope",
                symbol.line,
                symbol.column
            )

    def handle_function_declaration(self, declarator: Union[Tree, Token], return_type: Type):
        """handle function declarations"""
        if isinstance(declarator, Token):
            func_name = declarator.value
            param_types = []  # No parameters for simple function declarations
        else:
            func_name = self.get_identifier_from_node(declarator)
            # get parameter types
            param_types = []
            if len(declarator.children) > 1 and isinstance(declarator.children[1], Tree) and declarator.children[1].data == 'parameter_list':
                for param in declarator.children[1].children:
                    if param.data == 'parameter_declaration':
                        param_type = self.get_type_from_node(param.children[0])
                        param_types.append(param_type)
        
        func_type = Type(
            kind=TypeKind.FUNCTION,
            return_type=return_type,
            param_types=param_types
        )
        
        symbol = Symbol(
            name=func_name,
            type=func_type,
            line=getattr(declarator, 'line', 0),
            column=getattr(declarator, 'column', 0)
        )
        
        # Add the function symbol to the current scope (global scope for functions)
        if not self.symbol_table.add_symbol(symbol):
            raise SemanticError(
                f"function '{func_name}' already declared",
                symbol.line,
                symbol.column
            )
        logger.debug(f"Added function symbol {func_name} to scope {self.symbol_table.current_scope}")

    def visit_function_definition(self, node: Tree):
        """handle function definitions"""
        # the declarator is the second child of the function definition
        declarator = node.children[1]
        func_name = self.get_identifier_from_node(declarator)
        logger.debug(f"Processing function definition: {func_name}")
        
        # Get the function symbol
        symbol = self.symbol_table.lookup(func_name)
        if not symbol:
            # If the function wasn't declared before, create it now
            logger.debug(f"Function {func_name} not found in symbol table, creating new symbol")
            return_type = self.get_type_from_node(node.children[0])
            self.handle_function_declaration(declarator, return_type)
            symbol = self.symbol_table.lookup(func_name)
            logger.debug(f"Created new symbol for {func_name}: {symbol}")
        else:
            logger.debug(f"Found existing symbol for {func_name}: {symbol}")
        
        # Store the function body in the symbol
        if len(node.children) > 2:
            logger.debug(f"Storing body for function {func_name}")
            logger.debug(f"Body type: {type(node.children[2])}")
            logger.debug(f"Body data: {node.children[2].data}")
            symbol.body = node.children[2]
            logger.debug(f"Stored body for function '{func_name}': type={type(symbol.body)}, content={repr(symbol.body)}")
            logger.debug(f"Verifying body was stored: {symbol.body is not None}")
            # Verify the symbol node still has type='symbol'
            for node_id in self.symbol_table.graph.nodes():
                node_attrs = self.symbol_table.graph.nodes[node_id]
                if node_attrs.get('name') == func_name:
                    logger.debug(f"Function symbol node {node_id} attributes: {node_attrs}")
                    if node_attrs.get('type') != 'symbol':
                        logger.error(f"Function symbol node has wrong type: {node_attrs.get('type')}")
        else:
            logger.debug(f"No body found for function {func_name}")
        
        # Create a new scope for the function body
        # Note: We don't store the function symbol in this scope, it's already in the global scope
        self.symbol_table.enter_scope(f"function_{func_name}")
        self.current_function = func_name
        
        # process function body (compound statement)
        if len(node.children) > 2:
            logger.debug(f"Processing body for function {func_name}")
            self.visit(node.children[2])
        
        self.symbol_table.exit_scope()
        self.current_function = None
        logger.debug(f"Finished processing function {func_name}")

    def visit_compound_statement(self, node: Tree):
        """handle compound statements (blocks)"""
        self.symbol_table.enter_scope(f"block_{id(node)}")
        for child in node.children:
            self.visit(child)
        self.symbol_table.exit_scope()

    def get_expression_type(self, node: Tree) -> Type:
        """determine the type of an expression"""
        if isinstance(node, Token):
            if node.type == 'NUMBER':
                return Type(TypeKind.INT)
            elif node.type == 'IDENTIFIER':
                symbol = self.symbol_table.lookup(node.value)
                if not symbol:
                    raise SemanticError(f"undeclared variable '{node.value}'", 
                                      getattr(node, 'line', None),
                                      getattr(node, 'column', None))
                return symbol.type
        elif isinstance(node, Tree):
            if node.data == 'expr':
                if len(node.children) == 1:
                    return self.get_expression_type(node.children[0])
                # binary operation
                left_type = self.get_expression_type(node.children[0])
                right_type = self.get_expression_type(node.children[2])
                # for arithmetic operations, result type is the "wider" type
                if left_type.kind == TypeKind.FLOAT or right_type.kind == TypeKind.FLOAT:
                    return Type(TypeKind.FLOAT)
                return Type(TypeKind.INT)
            elif node.data == 'term':
                if len(node.children) == 1:
                    return self.get_expression_type(node.children[0])
                # binary operation
                left_type = self.get_expression_type(node.children[0])
                right_type = self.get_expression_type(node.children[2])
                # for arithmetic operations, result type is the "wider" type
                if left_type.kind == TypeKind.FLOAT or right_type.kind == TypeKind.FLOAT:
                    return Type(TypeKind.FLOAT)
                return Type(TypeKind.INT)
            elif node.data == 'factor':
                return self.get_expression_type(node.children[0])
        return Type(TypeKind.INT)  # default to int if we can't determine

    def visit_statement(self, node: Tree):
        """handle statements with proper type checking"""
        if node.data == 'assignment_expression':
            self.visit_assignment_expression(node)
        else:
            self.generic_visit(node)

    def visit_assignment_expression(self, node: Tree):
        """handle assignment expressions with type checking"""
        var_name = self.get_identifier_from_node(node.children[0])
        
        symbol = self.symbol_table.lookup(var_name)
        if not symbol:
            raise SemanticError(
                f"undeclared variable '{var_name}'",
                getattr(node, 'line', None),
                getattr(node, 'column', None)
            )
        
        # visit the right-hand side first to set any flags
        expr_node = self.visit(node.children[1])
        
        # get the type of the right-hand side expression
        expr_type = self.get_expression_type(node.children[1])
        
        # get line number from the assignment operator or the expression
        line = getattr(node.children[1], 'line', None)
        if line is None and len(node.children) > 1:
            line = getattr(node.children[1].children[0], 'line', None)
        
        # For pointer types (variables), we need to check against the base type
        if symbol.type.kind == TypeKind.POINTER:
            if symbol.type.base_type.kind != expr_type.kind:
                if symbol.type.base_type.kind == TypeKind.FLOAT and expr_type.kind == TypeKind.INT:
                    # check if this was an integer division
                    if isinstance(expr_node, Tree):
                        if getattr(expr_node, 'is_integer_division', False):
                            self.warnings.append(
                                f"warning: integer division assigned to float variable '{var_name}' at line {line} "
                                "this may cause precision loss"
                            )
                else:
                    raise SemanticError(
                        f"type mismatch in assignment: cannot assign {expr_type} to {symbol.type.base_type}",
                        line,
                        getattr(node, 'column', None)
                    )
        else:
            # For non-pointer types, check directly
            if symbol.type.kind != expr_type.kind:
                if symbol.type.kind == TypeKind.FLOAT and expr_type.kind == TypeKind.INT:
                    # check if this was an integer division
                    if isinstance(expr_node, Tree):
                        if getattr(expr_node, 'is_integer_division', False):
                            self.warnings.append(
                                f"warning: integer division assigned to float variable '{var_name}' at line {line} "
                                "this may cause precision loss"
                            )
                else:
                    raise SemanticError(
                        f"type mismatch in assignment: cannot assign {expr_type} to {symbol.type}",
                        line,
                        getattr(node, 'column', None)
                    )
        
        symbol.is_initialized = True
        return node

    def visit_identifier(self, node: Tree):
        """handle identifier usage"""
        var_name = node.value
        symbol = self.symbol_table.lookup(var_name)
        
        if not symbol:
            raise SemanticError(
                f"undeclared variable '{var_name}'",
                getattr(node, 'line', None),
                getattr(node, 'column', None)
            )
        
        if not symbol.is_initialized and not symbol.is_constant:
            self.warnings.append(
                f"warning: variable '{var_name}' might be used before initialization"
            )
        
        return symbol

    def visit_function_call(self, node: Tree):
        """handle function calls"""
        func_name = node.children[0].value if isinstance(node.children[0], Token) else self.get_identifier_from_node(node.children[0])
        
        # for printf, we'll skip the lookup since it's a built-in function
        if func_name == 'printf':
            # visit arguments to check their validity
            if len(node.children) > 1 and isinstance(node.children[1], Tree):
                for arg in node.children[1].children:
                    self.visit(arg)
            return
        
        symbol = self.symbol_table.lookup(func_name)
        if not symbol:
            raise SemanticError(
                f"undeclared function '{func_name}'",
                getattr(node, 'line', None),
                getattr(node, 'column', None)
            )
        
        if symbol.type.kind != TypeKind.FUNCTION:
            raise SemanticError(
                f"'{func_name}' is not a function",
                getattr(node, 'line', None),
                getattr(node, 'column', None)
            )
        
        # check arguments
        args = []
        if len(node.children) > 1 and isinstance(node.children[1], Tree):
            args = node.children[1].children
        
        if len(args) != len(symbol.type.param_types):
            raise SemanticError(
                f"function '{func_name}' expects {len(symbol.type.param_types)} arguments, "
                f"but got {len(args)}",
                getattr(node, 'line', None),
                getattr(node, 'column', None)
            )
        
        # visit arguments
        for arg in args:
            self.visit(arg)

    def visit_term(self, node: Tree):
        """handle term expressions with type checking"""
        if len(node.children) == 1:
            return self.visit(node.children[0])
        
        # binary operation
        left_type = self.get_expression_type(node.children[0])
        right_type = self.get_expression_type(node.children[2])
        
        # visit both operands
        self.visit(node.children[0])
        self.visit(node.children[2])
        
        # check for division by zero in constant cases
        if node.children[1].type == 'SLASH':
            if isinstance(node.children[2], Token) and node.children[2].type == 'NUMBER':
                if int(node.children[2].value) == 0:
                    raise SemanticError(
                        "division by zero",
                        getattr(node, 'line', None),
                        getattr(node, 'column', None)
                    )
            
            # check for integer division
            if left_type.kind == TypeKind.INT and right_type.kind == TypeKind.INT:
                # create a new node to preserve the integer division flag
                result_node = Tree('term', node.children)
                result_node.is_integer_division = True
                return result_node
        
        return node

    def visit_expr(self, node: Tree):
        """handle expressions with type checking"""
        if len(node.children) == 1:
            return self.visit(node.children[0])
        
        # binary operation
        left_type = self.get_expression_type(node.children[0])
        right_type = self.get_expression_type(node.children[2])
        
        # visit both operands
        self.visit(node.children[0])
        self.visit(node.children[2])
        
        return node

def analyze_c_code(tree: Tree, visualize: bool = False) -> bool:
    """
    analyze c code for semantic errors
    
    args:
        tree: lark parsetree object from the parser
        visualize: whether to generate a visualization of the symbol table
        
    returns:
        bool: true if no semantic errors were found, false otherwise
    """
    analyzer = SemanticAnalyzer()
    success = analyzer.analyze(tree)
    
    if visualize:
        analyzer.symbol_table.visualize()
    
    # print warnings
    for warning in analyzer.warnings:
        print(warning)
    
    # print errors
    for error in analyzer.errors:
        print(error)
    
    return success

def main():
    """main function to run semantic analysis on a c file from command line"""
    import argparse
    from .parser import create_parser, parse_c_code, read_c_file
    
    parser = argparse.ArgumentParser(description='semantic analyzer for c code')
    parser.add_argument('file', help='c source file to analyze')
    parser.add_argument('--visualize', action='store_true', 
                       help='generate visualization of symbol table')
    
    args = parser.parse_args()
    
    # read and parse the c file
    code = read_c_file(args.file)
    if code is None:
        sys.exit(1)
    
    parser = create_parser()
    tree = parse_c_code(parser, code)
    
    if tree:
        if analyze_c_code(tree, args.visualize):
            print("semantic analysis completed successfully")
        else:
            print("semantic analysis failed")
            sys.exit(1)

if __name__ == "__main__":
    # when running as a script, we need to use absolute imports
    import os
    import sys
    # add the parent directory to the python path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    from compiler.front_end.analysis.parser import create_parser, parse_c_code, read_c_file
    main()