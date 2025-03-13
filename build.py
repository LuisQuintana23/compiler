import os
import sys

def build():
    os.system("poetry install")
    os.system("pyinstaller --onefile --clean --name unam.fi.compilers.g5.06 src/compiler/lexer.py")

if __name__ == "__main__":
    build()
