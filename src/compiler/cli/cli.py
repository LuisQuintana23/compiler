import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from compiler.utils import logger
from compiler.codegen.code_generator import main as codegen_main

def setup_parser() -> argparse.ArgumentParser:
    """Set up the argument parser for the compiler CLI"""
    parser = argparse.ArgumentParser(
        description="Compiler cli for c language",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        """
    )
    
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('input', type=str, help='Input c file')
    parser.add_argument('-o', '--output', type=str, help='Output file name')
    
    # create subparsers for different phases
    subparsers = parser.add_subparsers(dest='command', help='Available compilation phases')
    
    # parse command
    parse_parser = subparsers.add_parser(
        'parse',
        help='Run only the parsing phase',
        description='Parse a C source file and generate the AST.'
    )
    parse_parser.add_argument('-o', '--output', type=str, help='Output AST file (optional)')
    
    # analyze command (semantic analysis)
    analyze_parser = subparsers.add_parser(
        'analyze',
        help='Run only the semantic analysis phase',
        description='Perform semantic analysis on a C source file.'
    )
    analyze_parser.add_argument('-o', '--output', type=str, help='Output symbol table file (optional)')
    
    # ir command
    ir_parser = subparsers.add_parser(
        'ir',
        help='Run only the IR generation phase',
        description='Generate LLVM IR from a C source file.'
    )
    ir_parser.add_argument('-o', '--output', type=str, help='Output IR file (optional)')
    
    return parser

def validate_input_file(input_file: str) -> Optional[Path]:
    """Validate the input file exists and has the correct extension"""
    path = Path(input_file)
    if not path.exists():
        logger.error(f"Input file not found: {input_file}")
        return None
    if path.suffix != '.c':
        logger.warning(f"Input file does not have .c extension: {input_file}")
    return path

def handle_codegen(args: argparse.Namespace) -> int:
    """Handle the codegen command"""
    input_path = validate_input_file(args.input)
    if not input_path:
        return 1
    
    logger.info("Running code generation phase", file=str(input_path))
    return 0 if codegen_main(str(input_path), output=args.output or f"{input_path.stem}.out") else 1

def main() -> int:
    """Main entry point for the compiler CLI"""
    parser = setup_parser()
    args = parser.parse_args()
    
    # set log level based on verbose flag
    if args.verbose:
        logger.logger.setLevel(logging.DEBUG)
    
    # validate input file
    input_path = validate_input_file(args.input)
    if not input_path:
        return 1
    
    if args.command is None:
        logger.info("Starting compilation process", file=str(input_path))
        return handle_codegen(args)
    else:
        parser.print_help()
        return 1

if __name__ == '__main__':
    sys.exit(main()) 