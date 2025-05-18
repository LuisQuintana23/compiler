import ply.yacc as yacc
from .lexer import lexer

# grammar rules
def p_program(p):
    '''program : external_declaration_list
              | empty'''
    if p[1] is None:  # empty
        p[0] = ('program', [])
    else:
        p[0] = ('program', p[1])

def p_external_declaration_list(p):
    '''external_declaration_list : external_declaration
                               | external_declaration_list external_declaration'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[2]]

def p_external_declaration(p):
    '''external_declaration : function_definition'''
    p[0] = p[1]

def p_function_definition(p):
    '''function_definition : type_specifier IDENTIFIER parameters compound_statement'''
    p[0] = ('function_definition', p[1], p[2], p[3], p[4])

def p_parameters(p):
    '''parameters : LPAREN parameter_list_opt RPAREN'''
    p[0] = p[2] if p[2] is not None else []

def p_parameter_list_opt(p):
    '''parameter_list_opt : parameter_list
                         | KEYWORD
                         | empty'''
    if p[1] is None or p[1] == 'void':  # empty or void
        p[0] = []
    else:
        p[0] = p[1]

def p_parameter_list(p):
    '''parameter_list : parameter
                     | parameter_list COMMA parameter'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[3]]

def p_parameter(p):
    '''parameter : type_specifier IDENTIFIER'''
    p[0] = ('parameter', p[1], p[2])

def p_type_specifier(p):
    '''type_specifier : KEYWORD'''
    p[0] = p[1]

def p_compound_statement(p):
    '''compound_statement : LBRACE block_item_list RBRACE'''
    p[0] = ('compound_statement', p[2])

def p_block_item_list(p):
    '''block_item_list : block_item
                      | block_item_list block_item
                      | empty'''
    if len(p) == 2:
        if p[1] is None:  # empty
            p[0] = []
        else:
            p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[2]]

def p_block_item(p):
    '''block_item : declaration
                 | statement'''
    p[0] = p[1]

def p_declaration(p):
    '''declaration : type_specifier IDENTIFIER SEMI
                  | type_specifier IDENTIFIER SEMI expression SEMI'''
    if len(p) == 4:
        p[0] = ('var_declaration', p[1], p[2])
    else:
        p[0] = ('var_declaration_init', p[1], p[2], p[4])

def p_statement(p):
    '''statement : compound_statement
                | expression_statement
                | selection_statement
                | iteration_statement
                | jump_statement
                | empty_statement'''
    p[0] = p[1]

def p_empty_statement(p):
    '''empty_statement : SEMI'''
    p[0] = ('empty_statement',)

def p_expression_statement(p):
    '''expression_statement : expression SEMI'''
    p[0] = ('expression_statement', p[1])

def p_selection_statement(p):
    '''selection_statement : if_statement'''
    p[0] = p[1]

def p_if_statement(p):
    '''if_statement : KEYWORD LPAREN expression RPAREN statement
                   | KEYWORD LPAREN expression RPAREN statement KEYWORD statement'''
    if len(p) == 6:
        p[0] = ('if_statement', p[3], p[5])
    else:
        p[0] = ('if_else_statement', p[3], p[5], p[7])

def p_iteration_statement(p):
    '''iteration_statement : while_statement'''
    p[0] = p[1]

def p_while_statement(p):
    '''while_statement : KEYWORD LPAREN expression RPAREN statement'''
    p[0] = ('while_statement', p[3], p[5])

def p_jump_statement(p):
    '''jump_statement : return_statement'''
    p[0] = p[1]

def p_return_statement(p):
    '''return_statement : KEYWORD expression SEMI'''
    p[0] = ('return_statement', p[2])

def p_expression(p):
    '''expression : assignment_expression
                 | function_call'''
    p[0] = p[1]

def p_assignment_expression(p):
    '''assignment_expression : conditional_expression
                           | IDENTIFIER OPERATOR assignment_expression'''
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = ('assignment', p[1], p[2], p[3])

def p_conditional_expression(p):
    '''conditional_expression : logical_or_expression'''
    p[0] = p[1]

def p_logical_or_expression(p):
    '''logical_or_expression : logical_and_expression
                            | logical_or_expression OPERATOR logical_and_expression'''
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = ('binary_op', p[2], p[1], p[3])

def p_logical_and_expression(p):
    '''logical_and_expression : equality_expression
                             | logical_and_expression OPERATOR equality_expression'''
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = ('binary_op', p[2], p[1], p[3])

def p_equality_expression(p):
    '''equality_expression : relational_expression
                          | equality_expression OPERATOR relational_expression'''
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = ('binary_op', p[2], p[1], p[3])

def p_relational_expression(p):
    '''relational_expression : additive_expression
                            | relational_expression OPERATOR additive_expression'''
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = ('binary_op', p[2], p[1], p[3])

def p_additive_expression(p):
    '''additive_expression : multiplicative_expression
                          | additive_expression OPERATOR multiplicative_expression'''
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = ('binary_op', p[2], p[1], p[3])

def p_multiplicative_expression(p):
    '''multiplicative_expression : unary_expression
                                | multiplicative_expression OPERATOR unary_expression'''
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = ('binary_op', p[2], p[1], p[3])

def p_unary_expression(p):
    '''unary_expression : postfix_expression
                       | OPERATOR unary_expression'''
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = ('unary_op', p[1], p[2])

def p_postfix_expression(p):
    '''postfix_expression : function_call
                         | primary_expression'''
    p[0] = p[1]

def p_function_call(p):
    '''function_call : IDENTIFIER LPAREN argument_list_opt RPAREN'''
    p[0] = ('function_call', p[1], p[3])

def p_argument_list_opt(p):
    '''argument_list_opt : argument_list
                        | empty'''
    p[0] = p[1] if p[1] is not None else []

def p_argument_list(p):
    '''argument_list : expression
                    | argument_list COMMA expression'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[3]]

def p_primary_expression(p):
    '''primary_expression : IDENTIFIER
                         | CONSTANT
                         | LITERAL
                         | parenthesized_expression'''
    if isinstance(p[1], str):
        # is it a constant?
        if p[1].isdigit():
            p[0] = ('constant', p[1])
        # is it a string literal?
        elif p[1].startswith('"'):
            p[0] = ('string_literal', p[1])
        else:
            p[0] = p[1]
    else:
        p[0] = p[1]

def p_parenthesized_expression(p):
    '''parenthesized_expression : LPAREN expression RPAREN'''
    p[0] = p[2]

def p_empty(p):
    '''empty :'''
    p[0] = None

# errors during the syntax analysis
def p_error(p):
    if p:
        print(f"Syntax error at '{p.value}' on line {p.lineno}")
    else:
        print("Syntax error at EOF")

# build the parser
parser = yacc.yacc()

# parse a file and return the abstract syntax tree
def parse_file(file_path):
    with open(file_path, 'r') as f:
        data = f.read()
    
    try:
        result = parser.parse(data, lexer=lexer)
        return result
    except Exception as e:
        print(f"Error parsing file: {e}")
        return None


def main():
    import sys

    if len(sys.argv) != 2:
        print("Usage: python parser.py <file.c>")
        sys.exit(1)
        
    result = parse_file(sys.argv[1])
    if result:
        print("Parsing successful!")
        print("Abstract Syntax Tree:")
        print(result) 



if __name__ == "__main__":
    main()