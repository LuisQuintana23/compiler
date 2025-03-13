# Compiler

## Requirements
- `python` >= 3.12  
- `poetry`: You can install it from the following [link](https://python-poetry.org/docs/#installing-with-the-official-installer)

## Lexer

When using it for the first time, run the install command

```bash
poetry install
```

To use the lexer, run the following command:  

```bash
poetry run lexer <file.c>
```

### Testing

To test your changes, run:  

```bash
poetry run pytest
```

## Alternative Usage (Not Recommended)

### Manually Installing Dependencies  

Use pip to install the required dependencies:  

```bash
python3 -m pip install ply pytest
```

**Note:** On Windows, use py or python instead of python3.  

### Running the Lexer  

Execute the following command:  

```bash
python3 ./src/compiler/lexer
```
