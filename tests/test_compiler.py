import os
import subprocess
import pytest

def test_compile_program():
    # get the absolute paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    input_file = os.path.join(project_root, "examples", "test.c")
    
    # verify input file exists
    assert os.path.exists(input_file), f"Input file {input_file} does not exist"
    
    # run the compiler using poetry
    result = subprocess.run(
        ["poetry", "run", "python", "-m", "compiler.cli.cli", "codegen", input_file],
        capture_output=True,
        text=True
    )
    
    # print compiler output for debugging
    print("\nCompiler stdout:", result.stdout)
    print("\nCompiler stderr:", result.stderr)
    
    # check if compilation was successful
    assert result.returncode == 0, f"Compilation failed with error: {result.stderr}"
    
    # the program should output "It works again yeah! 13"
    # the compiler generates the executable in the same directory as the input file
    executable = os.path.join(os.path.dirname(input_file), "a.out")
    print("\nLooking for executable at:", executable)
    print("Current directory contents:", os.listdir(os.path.dirname(input_file)))
    
    assert os.path.exists(executable), "Compiler did not generate executable"
    
    # run the compiled program
    program_output = subprocess.run(
        [executable],
        capture_output=True,
        text=True
    )
    
    # verify the output
    expected_output = "It works again yeah! 13 \n"
    assert program_output.stdout == expected_output, \
        f"Expected output '{expected_output}', got '{program_output.stdout}'"
