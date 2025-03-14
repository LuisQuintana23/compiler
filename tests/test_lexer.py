import pytest
import ply.lex as lex
from src.compiler.lexer import lexer

def test_keywords():
    data = "auto const double float int short struct unsigned break continue else for long signed switch void case default enum goto register sizeof typedef volatile char do extern if return static union while"
    lexer.input(data)
    tokens = [tok.type for tok in lexer]
    expected = ['KEYWORD'] * len(data.split(' '))
    assert tokens == expected

def test_identifiers():
    data = "variable myFunction another_variable"
    lexer.input(data)
    tokens = [tok.type for tok in lexer]
    expected = ['IDENTIFIER'] * 3
    assert tokens == expected

def test_operators():
    data = "+ - * / % ="
    lexer.input(data)
    tokens = [tok.type for tok in lexer]
    expected = ['OPERATOR'] * 6
    assert tokens == expected

def test_constants():
    data = "123 456.78 0.5"
    lexer.input(data)
    tokens = [tok.type for tok in lexer]
    expected = ['CONSTANT'] * 3
    assert tokens == expected

def test_punctuation():
    data = "( ) , ; ."
    lexer.input(data)
    tokens = [tok.type for tok in lexer]
    expected = ['PUNCTUATION'] * 5
    assert tokens == expected

def test_literal():
    data = "\"Hello world\" \"This a string\""
    lexer.input(data)
    tokens = [tok.type for tok in lexer]
    expected = ['LITERAL'] * 2
    assert tokens == expected
