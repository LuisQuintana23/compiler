# Compiler

# Installation

Go to the latest release tag and [download](https://github.com/LuisQuintana23/compiler/releases) the proper version for your system (Windows and Linux are available)

Then, run in a terminal the following command

On Linux

```bash
./unam.fi.compilers.g5.06 <file.c>
```

On Windows (Powershell)

```ps1
.\unam.fi.compilers.g5.06.exe <file.c>
```

# Develop

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
pip install ply pytest
```

**Note:** On Windows, use py or python instead of python3.  

### Running the Lexer  

Execute the following command:  

```bash
python3 ./src/compiler/lexer.py
```
