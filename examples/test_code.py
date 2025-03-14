#!/usr/bin/env python3
"""
Example evaluation script that tests code against a set of test cases.
"""
import sys
import ast
import io
import contextlib
from typing import List, Tuple

# Test cases: (input, expected_output)
TEST_CASES = [
    ("5", "120"),
    ("0", "1"),
    ("1", "1"),
    ("10", "3628800")
]

def evaluate_code(code: str) -> Tuple[int, List[str]]:
    """
    Evaluate the code by running it against test cases
    
    Returns:
        Tuple of (passing_tests, error_messages)
    """
    # Check if the code is syntactically valid
    try:
        ast.parse(code)
    except SyntaxError as e:
        return 0, [f"Syntax error: {e}"]
    
    # Try to extract the function
    function_name = "factorial"
    if function_name not in code:
        return 0, [f"Function '{function_name}' not found in code"]
    
    # Create a namespace to execute the code
    namespace = {}
    
    # Execute the code
    try:
        exec(code, namespace)
    except Exception as e:
        return 0, [f"Error executing code: {e}"]
    
    # Check if the function exists
    if function_name not in namespace:
        return 0, [f"Function '{function_name}' not defined"]
    
    # Run test cases
    passing_tests = 0
    errors = []
    
    for i, (input_val, expected) in enumerate(TEST_CASES):
        try:
            # Capture stdout
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                result = str(namespace[function_name](int(input_val)))
            
            if result == expected:
                passing_tests += 1
            else:
                errors.append(f"Test case {i+1} failed: input={input_val}, expected={expected}, got={result}")
        except Exception as e:
            errors.append(f"Error in test case {i+1}: {e}")
    
    return passing_tests, errors

if __name__ == "__main__":
    # Read input from stdin
    code = sys.stdin.read()
    
    # Evaluate the code
    passing_tests, errors = evaluate_code(code)
    
    # Print detailed results
    print(f"Passing tests: {passing_tests}/{len(TEST_CASES)}")
    for error in errors:
        print(f"Error: {error}")
    
    # Calculate reward (last line must be a number)
    reward = passing_tests / len(TEST_CASES) * 10
    print(reward)
