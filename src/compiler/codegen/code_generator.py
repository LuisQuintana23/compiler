import llvmlite.binding as llvm
import llvmlite.ir as ir
from pathlib import Path
import sys
import os
from typing import Optional, Union

from ..utils import logger
from ..front_end.ir.ir_generator import LLVMGenerator
from ..front_end.analysis.semantic import SemanticAnalyzer

class CodeGenerator:
    """Generates machine code from LLVM IR using llvmlite"""
    
    def __init__(self):
        """Initialize the code generator and LLVM environment"""
        # initialize llvm
        llvm.initialize()
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()
        
        # create target machine
        self.target = llvm.Target.from_default_triple()
        self.target_machine = self.target.create_target_machine()
        
        # create pass manager
        self.pass_manager = llvm.create_module_pass_manager()
        self.target_machine.add_analysis_passes(self.pass_manager)
        
        # add optimization passes
        self.pass_manager.add_constant_merge_pass()
        self.pass_manager.add_dead_arg_elimination_pass()
        self.pass_manager.add_function_attrs_pass()
        self.pass_manager.add_function_inlining_pass(threshold=225)
        self.pass_manager.add_global_dce_pass()
        self.pass_manager.add_global_optimizer_pass()
        self.pass_manager.add_ipsccp_pass()
        self.pass_manager.add_dead_code_elimination_pass()
        self.pass_manager.add_cfg_simplification_pass()
        self.pass_manager.add_gvn_pass()
        self.pass_manager.add_instruction_combining_pass()
        self.pass_manager.add_licm_pass()
        self.pass_manager.add_sccp_pass()
        self.pass_manager.add_sroa_pass()
        self.pass_manager.add_type_based_alias_analysis_pass()
        self.pass_manager.add_basic_alias_analysis_pass()
        
        logger.debug("CodeGenerator initialized with target machine and pass manager")

    def load_ir_from_file(self, ir_file: str) -> llvm.ModuleRef:
        """Load LLVM IR from a file"""
        try:
            logger.debug(f"Loading IR from file: {ir_file}")
            with open(ir_file, 'r') as f:
                ir_content = f.read()
            
            # parse the ir
            module = llvm.parse_assembly(ir_content)
            module.verify()
            logger.debug("IR module loaded and verified successfully")
            return module
        except Exception as e:
            logger.error(f"Error loading IR file: {e}")
            raise

    def generate_from_ir_module(self, module: Union[ir.Module, llvm.ModuleRef], output_dir: str = None, output_name: str = None, optimize: bool = True) -> None:
        """Generate code directly from an IR module"""
        try:
            # convert ir.module to llvm.moduleref if needed
            if isinstance(module, ir.Module):
                module = llvm.parse_assembly(str(module))
                module.verify()
            
            # apply optimizations if requested
            if optimize:
                self.optimize_module(module)
            
            # determine output directory
            if output_dir is None:
                output_dir = "."
            else:
                os.makedirs(output_dir, exist_ok=True)
            
            # generate executable with exact name provided
            exe_file = str(Path(output_dir) / output_name) if output_name else str(Path(output_dir) / "a.out")
            
            # generate object code directly to executable
            obj_data = self.target_machine.emit_object(module)
            temp_obj = str(Path(output_dir) / ".temp.o")
            with open(temp_obj, 'wb') as f:
                f.write(obj_data)
            
            # link directly to executable using exact name
            cmd = f"gcc {temp_obj} -o '{exe_file}'"  # added quotes to handle names with dots
            if os.system(cmd) != 0:
                raise RuntimeError("Linker failed to create executable")
            
            # clean up temporary object file
            os.remove(temp_obj)
            
            logger.info("Code generation completed successfully")
            logger.info(f"Executable: {exe_file}")
            
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            raise

    def optimize_module(self, module: llvm.ModuleRef) -> None:
        """Apply optimizations to the module"""
        try:
            logger.debug("Optimizing module")
            self.pass_manager.run(module)
            logger.debug("Module optimization completed")
        except Exception as e:
            logger.error(f"Error optimizing module: {e}")
            raise


def generate_code(input_file: str, output_dir: str = None, output_name: str = None, optimize: bool = True) -> None:
    """
    Generate machine code from a c source file
        input_file: Path to the c source file
        output_dir: Directory to store output files (default: same as input file)
        output_name: Base name for output files (default: same as input file)
        optimize: Whether to apply optimizations
    """
    try:
        from ..front_end.analysis.parser import create_parser, parse_c_code, read_c_file
        from ..front_end.analysis.semantic import SemanticAnalyzer

        code = read_c_file(input_file)
        if code is None:
            logger.error(f"Could not read file: {input_file}")
            return

        parser = create_parser()
        tree = parse_c_code(parser, code)
        if tree is None:
            logger.error("Error parsing code")
            return

        analyzer = SemanticAnalyzer()
        success = analyzer.analyze(tree)
        if not success:
            logger.error("Semantic analysis failed:")
            for err in analyzer.errors:
                logger.error(str(err))
            return

        ir_generator = LLVMGenerator(analyzer.symbol_table)
        ir_module = ir_generator.generate()

        generator = CodeGenerator()
        generator.generate_from_ir_module(ir_module, output_dir, output_name, optimize)

    except Exception as e:
        logger.error(f"Code generation failed: {e}")
        raise

def main(input_file: Optional[str] = None, output: Optional[str] = None) -> bool:
    """Main function to generate code from a C source file
    
    Args:
        input_file: Path to the C source file (optional if called from CLI)
        output: Output file name (optional)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # if called from cli, parse arguments
        if input_file is None:
            import argparse
            
            parser = argparse.ArgumentParser(description='Generate machine code from C source file')
            parser.add_argument('input_file', help='C source file to compile')
            parser.add_argument('--output', '-o', help='Output file name')
            parser.add_argument('--no-optimize', action='store_true',
                              help='Disable optimizations')
            
            args = parser.parse_args()
            input_file = args.input_file
            output = args.output
        
        # determine output directory and use exact output name
        output_dir = str(Path(output).parent) if output else str(Path(input_file).parent)
        output_name = output if output else "a.out"  # use exact output name
        
        # generate code
        generate_code(input_file, output_dir, output_name)
        return True
    except Exception as e:
        logger.error(f"Code generation failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 