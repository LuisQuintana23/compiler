import ply.lex as lex
import sys
import os
import toml

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
def process_file(file_path):
    if not os.path.exists(file_path):
        print(f"File '{file_path}' not found")
        sys.exit(1)

    with open(file_path, 'r') as f:
        data = f.read()

    lexer.input(data)

    # show tokens
    while True:
        tok = lexer.token()
        if not tok:
            break
        print(tok)

def get_app_name():
    try:
        with open('pyproject.toml', 'r') as f:
            config = toml.load(f)
        return config['project']['name']
    except Exception as e:
        print(f"Can't read pyproject.toml: {e}")
        return None

def main():
    APP_NAME = get_app_name()
    if len(sys.argv) != 2:
        print(f"Use: {APP_NAME} <file.c>")
        sys.exit(1)
    file_path = sys.argv[1]
    process_file(file_path)

if __name__ == "__main__":
    main()
