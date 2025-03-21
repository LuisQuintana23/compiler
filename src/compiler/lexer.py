import ply.lex as lex
import sys
import os
from collections import defaultdict

tokens = (
    'KEYWORD', 
    'IDENTIFIER', 
    'OPERATOR', 
    'CONSTANT', 
    'PUNCTUATION',
    'LITERAL',
)

# Regular expressions
t_KEYWORD = r'\b(?:auto|const|double|float|int|short|struct|unsigned|break|continue|else|for|long|signed|switch|void|case|default|enum|goto|register|sizeof|typedef|volatile|char|do|extern|if|return|static|union|while)\b'
t_OPERATOR = r'(\+\+|--|->|==|!=|<=|>=|&&|\|\||[+\-*/%=<>!&|^~])'
t_CONSTANT = r'(\d+(\.\d+)?([eE][+-]?\d+)?|0[xX][0-9a-fA-F]+|0[0-7]*)'
t_PUNCTUATION = r'[()[\]{},;.]'
t_IDENTIFIER = r'[a-zA-Z_][a-zA-Z_0-9]*'
t_LITERAL = r'\"([^\\\n]|(\\.))*?\"'

t_ignore = ' \t\r\f\v'

def t_NEWLINE(t):
    r'\n+'
    t.lexer.lineno += len(t.value)

def t_error(t):
    print(f"Invalid character: {t.value[0]}")
    t.lexer.skip(1)

lexer = lex.lex()

# Load the input file
def process_file(file_path):
    if not os.path.exists(file_path):
        print(f"File '{file_path}' not found")
        sys.exit(1)

    with open(file_path, 'r') as f:
        data = f.read()

    lexer.input(data)

    # dictionary to group the tokens
    token_dict = defaultdict(list)
    token_count = 0

    while True:
        tok = lexer.token()
        if not tok:
            break
        token_dict[tok.type].append(tok.value)
        token_count += 1

    # displays tokens group
    print("\n{:<15} {}".format("Token Type", "Values"))
    print("=" * 70)

    for token_type, values in token_dict.items():
        print("{:<15} {}".format(token_type, ", ".join(values)))

    print("=" * 70)
    print(f"Total Tokens: {token_count}")
    print(f"Number of lines: {lexer.lineno}")

def main():
    if len(sys.argv) != 2:
        print(f"Usage: unam.fi.compilers.g5.06[.exe] <file.c>")
        sys.exit(1)
    file_path = sys.argv[1]
    process_file(file_path)

if __name__ == "__main__":
    main()