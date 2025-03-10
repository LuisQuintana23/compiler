import ply.lex as lex
import os

tokens = (
    'KEYWORD', 
    'IDENTIFIER', 
    'OPERATOR', 
    'CONSTANT', 
    'PUNCTUATION',
    'STRING'
)

# regular expressions
t_KEYWORD = r'\b(?:if|else|while|for|return|int)\b'
t_OPERATOR = r'[+\-*/%=]'
t_CONSTANT = r'\d+(\.\d+)?'
t_PUNCTUATION = r'[(),;.}{]'
t_IDENTIFIER = r'[a-zA-Z_][a-zA-Z_0-9]*'
t_STRING = r'\'([^\\\n]|(\\.))*?\'|\"([^\\\n]|(\\.))*?\"'

t_ignore = ' \t'

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


def start():
    file_name = 'program.c'
    process_file(file_name=file_name)

if __name__ == "__main__":
    start()
