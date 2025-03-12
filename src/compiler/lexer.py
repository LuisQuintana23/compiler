import ply.lex as lex
import os

tokens = (
    'KEYWORD', 
    'IDENTIFIER', 
    'OPERATOR', 
    'CONSTANT', 
    'PUNCTUATION',
    'CHAR',
    'STRING',
)

# regular expressions
t_KEYWORD = r'\b(?:auto|const|double|float|int|short|struct|unsigned|break|continue|else|for|long|signed|switch|void|case|default|enum|goto|register|sizeof|typedef|volatile|char|dor|extern|if|return|static|union|while)\b'
t_OPERATOR = r'(\+\+|--|->|==|!=|<=|>=|&&|\|\||[+\-*/%=<>!&|^~])'
t_CONSTANT = r'(\d+(\.\d+)?([eE][+-]?\d+)?|0[xX][0-9a-fA-F]+|0[0-7]*)'
t_PUNCTUATION = r'[()[\]{},;.]'
t_IDENTIFIER = r'[a-zA-Z_][a-zA-Z_0-9]*'
t_STRING = r'\"([^\\\n]|(\\.))*?\"'
t_CHAR = r"'\\" + r"[abfnrtv\\'\"0]'|'.'"

t_ignore = ''

def t_WHITESPACE(t):
    r'[ \t\r\f\v]+'
    pass

def t_NEWLINE(t):
    r'\n+'
    t.lexer.lineno += len(t.value)

def t_error(t):
    print(f"Invalid character: {t.value[0]}")
    t.lexer.skip(1)

lexer = lex.lex()

# load the input file
def process_file(file_name):
    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_path, file_name)

    with open(file_path, 'r') as f:
        data = f.read()

    lexer.input(data)

    # show tokens
    while True:
        tok = lexer.token()
        if not tok:
            break
        print(tok)

    print(f"Number of lines: {lexer.lineno}")


def start():
    file_name = 'program.c'
    process_file(file_name=file_name)

if __name__ == "__main__":
    start()
