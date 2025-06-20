from lark import Lark, Tree
import os
from pathlib import Path
import sys
import argparse
from graphviz import Digraph

def create_parser():
    """
    Read the grammar from c_parser.lark
    """
    current_dir = Path(__file__).parent
    grammar_file = current_dir / "c_parser.lark"
    
    with open(grammar_file, 'r') as f:
        grammar = f.read()
    
    # it's using lalr parser
    return Lark(grammar, parser='lalr', start='start')

def parse_c_code(parser, code):
    """
    Parse c code using the provided parser
    
    Args:
        parser: Lark parser instance
        code (str): c source code to parse
        
    Returns:
        ParseTree: The parse tree of the code
    """
    try:
        return parser.parse(code)
    except Exception as e:
        print(f"Error parsing code: {e}")
        return None

def read_c_file(file_path):
    """
    Read c source code from a file
    
    Args:
        file_path (str): Path to the c source file
        
    Returns:
        str: Contents of the file, or None if there was an error
    """
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        return None
    except PermissionError:
        print(f"Error: Permission denied when reading '{file_path}'")
        return None
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}")
        return None

def visualize_tree(tree, output_file):
    """
    Create a visual representation of the parse tree using Graphviz
    
    Args:
        tree: Lark ParseTree object
        output_file (str): Path where to save the image (without extension)
    """
    dot = Digraph(comment='Parse Tree')
    dot.attr(rankdir='TB')
    
    def add_node(node, parent_id=None):
        if isinstance(node, Tree):
            # create a unique id for this node
            node_id = str(id(node))
            
            # add the node with its rule name
            dot.node(node_id, node.data, shape='box')
            
            # connect to parent if it exists
            if parent_id is not None:
                dot.edge(parent_id, node_id)
            
            # add all children
            for child in node.children:
                add_node(child, node_id)
        else:
            # for tokens, create a leaf node
            node_id = str(id(node))
            label = f"{node.type}: {node.value}" if hasattr(node, 'type') else str(node)
            dot.node(node_id, label, shape='ellipse')
            if parent_id is not None:
                dot.edge(parent_id, node_id)
    
    # start building the tree from the root
    add_node(tree)
    
    # save the visualization
    dot.render(output_file, format='png', cleanup=True)
    print(f"Tree visualization saved as {output_file}.png")

def print_concise_tree(tree, indent=0):
    """
    Print a concise version of the parse tree, focusing on main rules
    
    Args:
        tree: Lark ParseTree object
        indent: Current indentation level
    """
    # skip printing for certain rules that are too detailed
    skip_rules = {
        'logical_or_expression', 'logical_and_expression',
        'inclusive_or_expression', 'exclusive_or_expression',
        'and_expression', 'equality_expression', 'relational_expression',
        'shift_expression', 'additive_expression', 'multiplicative_expression',
        'cast_expression', 'unary_expression', 'postfix_expression',
        'primary_expression', 'assignment_expression', 'conditional_expression'
    }
    
    if isinstance(tree, Tree):
        if tree.data not in skip_rules:
            print('  ' * indent + f"{tree.data}")
            for child in tree.children:
                print_concise_tree(child, indent + 1)
        else:
            # for skipped rules, just print their tokens
            for child in tree.children:
                if not isinstance(child, Tree):
                    print('  ' * indent + f"Token: {child.type} = {child.value}")
                else:
                    print_concise_tree(child, indent)
    elif hasattr(tree, 'type'):
        print('  ' * indent + f"Token: {tree.type} = {tree.value}")

def parser(args: argparse.Namespace):
    # read the c code from file
    code = read_c_file(args.input)
    if code is None:
        sys.exit(1)
    
    # create the parser
    parser = create_parser()
    
    # parse the code
    tree = parse_c_code(parser, code)
    
    if tree:
        print(f"Successfully parsed c code from '{args.input}'!")
        
        if args.tree_image:
            # generate tree visualization
            output_file = Path(args.input).stem + '_tree'
            visualize_tree(tree, output_file)
        
        if args.verbose:
            # print detailed tree structure
            print("\nDetailed parse tree:")
            print(tree.pretty())
            print("\nTree structure:")
            for subtree in tree.iter_subtrees():
                print(f"Rule: {subtree.data}")
                for child in subtree.children:
                    if hasattr(child, 'type'):
                        print(f"  Token: {child.type} = {child.value}")
                    else:
                        print(f"  Child: {child}")
        else:
            # print concise tree structure
            print("\nConcise parse tree:")
            print_concise_tree(tree)
