import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from compiler.utils import logger
from compiler.codegen.code_generator import codegen
from compiler.front_end.analysis.parser import parser
from compiler.front_end.analysis.semantic import semantic
from compiler.front_end.ir.ir_generator import generate_ir as ir

def setup_parser() -> argparse.ArgumentParser:
    """Set up the argument parser for the compiler CLI"""
    parser = argparse.ArgumentParser(
        description="Compiler cli for c language",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        """
    )
    
    # create subparsers for different phases
    subparsers = parser.add_subparsers(dest='command', help='Available compilation phases')
    
    # parse command
    parse_parser = subparsers.add_parser(
        'parse',
        help='Run only the parsing phase',
        description='Parse a c source file and generate the AST.'
    )
    parse_parser.add_argument('input', type=str, help='Input c file')
    parse_parser.add_argument('--tree-image', action='store_true', help='Generate a PNG visualization of the parse tree')

    # semantic command
    parse_semantic = subparsers.add_parser(
        'semantic',
        help='Run only the semantic phase',
        description='Semantic analysis of a c source file.'
    )
    parse_semantic.add_argument('input', type=str, help='Input c file')
    parse_semantic.add_argument('--visualize', action='store_true', help='generate visualization of symbol table')

    # ir command
    ir_parser = subparsers.add_parser(
        'ir',
        help='Run only the ir phase',
        description='Generate the IR of a c source file.'
    )
    ir_parser.add_argument('input', type=str, help='Input c file')

    # codegen command
    parse_codegen = subparsers.add_parser(
        'codegen',
        help='Run only the codegen phase',
        description='Generate the machine code of a c file.'
    )
    parse_codegen.add_argument('input', type=str, help='Input c file')
    parse_codegen.add_argument('--output', '-o', type=str, help='Output file name')
    parse_codegen.add_argument('--no-optimize', action='store_true',
                        help='Disable optimizations')

    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    
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
    success = codegen(args)
    return 0 if success else 1

def handle_parse(args: argparse.Namespace) -> int:
    """Handle the parse command"""
    input_path = validate_input_file(args.input)
    if not input_path:
        return 1
    
    logger.info("Running parse phase", file=str(input_path))
    parser(args)
    return 0

def handle_semantic(args: argparse.Namespace) -> int:
    """Handle the semantic command"""
    input_path = validate_input_file(args.input)
    if not input_path:
        return 1
    
    logger.info("Running semantic phase", file=str(input_path))
    semantic(args)
    return 0

def handle_ir(args: argparse.Namespace) -> int:
    """Handle the ir command"""
    input_path = validate_input_file(args.input)
    if not input_path:
        return 1
    
    logger.info("Running ir phase", file=str(input_path))
    ir(args)
    return 0

def main() -> int:
    """Main entry point for the compiler CLI"""
    parser = setup_parser()
    args = parser.parse_args()
    
    # set log level based on verbose flag
    if args.verbose:
        logger.logger.setLevel(logging.DEBUG)
    
    if args.command is None:
        parser.print_help()
        return 1
    elif args.command == 'codegen':
        logger.info("Starting compilation process")
        return handle_codegen(args)
    elif args.command == 'parse':
        return handle_parse(args)
    elif args.command == 'semantic':
        return handle_semantic(args)
    elif args.command == 'ir':
        return handle_ir(args)
    else:
        parser.print_help()
        return 1

if __name__ == '__main__':
    sys.exit(main()) 