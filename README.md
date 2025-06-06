# Compiler

# Installation

Go to the latest release tag and [download](https://github.com/LuisQuintana23/compiler/releases) the proper version for your system (Windows and Linux are available)

Then, run in a terminal the following command

On Linux

```bash
./unam.fi.compilers.g5.06 <command> <file.c> [options]
```


# Usage

The compiler provides several commands to handle different phases of compilation:

## Commands

### Parse
Parse a c file and generate the AST:
```bash
./unam.fi.compilers.g5.06 parse <file.c> [--tree-image]
```
Options:
- `--tree-image`: Generate a png visualization of the parse tree

### Semantic
Run semantic analysis on a c file:
```bash
./unam.fi.compilers.g5.06 semantic <file.c> [--visualize]
```
Options:
- `--visualize`: Generate visualization of symbol table

### IR
Generate the IR of a c file:
```bash
./unam.fi.compilers.g5.06 ir <file.c>
```

### Codegen
Generate machine code from a c file:
```bash
./unam.fi.compilers.g5.06 codegen <file.c> [--output <name>] [--no-optimize]
```
Options:
- `--output`, `-o`: Specify output file name (default: a.out)

## Examples

1. Parse a file and generate tree visualization:
```bash
./unam.fi.compilers.g5.06 parse program.c --tree-image
```

2. Run semantic analysis with symbol table visualization:
```bash
./unam.fi.compilers.g5.06 semantic program.c --visualize
```

3. Generate IR:
```bash
./unam.fi.compilers.g5.06 ir program.c
```

4. Generate executable:
```bash
./unam.fi.compilers.g5.06 codegen program.c
```

5. Generate executable with custom name:
```bash
./unam.fi.compilers.g5.06 codegen program.c -o myprogram
```

# Develop

## Requirements
- `python` >= 3.12
- `poetry`: You can install it from the following [link](https://python-poetry.org/docs/#installing-with-the-official-installer)
- `graphviz`: It is used to generate a tree image

## Installation

When using it for the first time, run the install command

```bash
poetry install
```

## Running with Poetry

You can also run the compiler directly using poetry, which provides the same functionality as the binary:

```bash
# Parse a file
poetry run python -m compiler.cli.cli parse program.c

# Run semantic analysis
poetry run python -m compiler.cli.cli semantic program.c

# Generate IR
poetry run python -m compiler.cli.cli ir program.c

# Generate executable
poetry run python -m compiler.cli.cli codegen program.c
```

All commands and options available in the binary are also available when running with poetry, this is useful during development or when you want to run the compiler without building the binary

## Building

To build the binary:

```bash
python build.py
```

This will create the binary in the current directory.
