import pytest
import sys
import os
import tempfile
import subprocess
import shutil

# Add the src directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

def is_command_available(command):
    """Check if a command is available on the system"""
    return shutil.which(command) is not None

def test_optimization_with_context():
    """Test optimization with context passed via --context"""
    # Skip this test in CI environments
    if os.environ.get('CI') == 'true':
        pytest.skip("Skipping test that requires CLI in CI environment")
    
    # Create a simple evaluation script that uses context
    script_content = """#!/usr/bin/env python3
import sys
import os

# Get context from environment variable
context = os.environ.get('AGENT_CONTEXT', '')

# Get agent output from stdin
agent_output = sys.stdin.read()

# Calculate reward based on how many characters from the context are in the output
matching_chars = sum(1 for c in context if c in agent_output)
reward = matching_chars

# Print detailed information
print(f"Context: {context}")
print(f"Agent output: {agent_output}")
print(f"Matching characters: {matching_chars}")
print(reward)  # This must be the last line and a number
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as script_file:
        script_path = script_file.name
        script_file.write(script_content)
    
    try:
        # Make the script executable
        os.chmod(script_path, 0o755)
        
        # Create a context file
        context = "abcdefghijklmnopqrstuvwxyz"
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as context_file:
            context_file_path = context_file.name
            context_file.write(context)
        
        # Run the command with context
        result = subprocess.run(
            [
                "python", "-m", "llm_agent_evolution",
                "--use-mock",
                "--eval-command", f"python {script_path}",
                "--context", context,
                "--population-size", "10",
                "--parallel-agents", "2",
                "--max-evaluations", "5"  # Reduce evaluations for faster test
            ],
            capture_output=True,
            text=True,
            timeout=120  # Increase timeout
        )
        
        # Print output for debugging
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        
        # Check that it ran successfully
        assert result.returncode == 0
        
        # Check for context in output - more specific check
        assert "Using context for evaluation: abcdefghijklmnopqrstuvwxyz" in result.stdout or \
               "Using context: abcdefghijklmnopqrstuvwxyz" in result.stdout or \
               "Context: abcdefghijklmnopqrstuvwxyz" in result.stdout, \
            f"Context '{context}' not found in output. Output: {result.stdout[:500]}..."
        
        # Now test with context file
        result = subprocess.run(
            [
                "python", "-m", "llm_agent_evolution",
                "--use-mock",
                "--eval-command", f"python {script_path}",
                "--context-file", context_file_path,
                "--population-size", "10",
                "--parallel-agents", "2",
                "--max-evaluations", "20"
            ],
            capture_output=True,
            text=True
        )
        
        # Check that it ran successfully
        assert result.returncode == 0
        assert "Context: abcdefghijklmnopqrstuvwxyz" in result.stdout
        
    finally:
        # Clean up
        if os.path.exists(script_path):
            os.remove(script_path)
        if os.path.exists(context_file_path):
            os.remove(context_file_path)

def test_optimization_with_stdin_pipe():
    """Test optimization with input piped via stdin"""
    # Skip this test in CI environments
    if os.environ.get('CI') == 'true':
        pytest.skip("Skipping test that requires CLI in CI environment")
    
    # Create a simple evaluation script that uses context
    script_content = """#!/usr/bin/env python3
import sys
import os

# Get context from environment variable
context = os.environ.get('AGENT_CONTEXT', '')

# Get agent output from stdin
agent_output = sys.stdin.read()

# Calculate reward based on how many 'a's are in the output
# Only count lowercase 'a', not apostrophes or uppercase 'A'
a_count = sum(1 for c in agent_output if c == 'a')
reward = a_count

# Print detailed information
print(f"Context: {context}")
print(f"Agent output: {agent_output}")
print(f"'a' count: {a_count}")
print(reward)  # This must be the last line and a number
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as script_file:
        script_path = script_file.name
        script_file.write(script_content)
    
    shell_script_path = None
    
    try:
        # Make the script executable
        os.chmod(script_path, 0o755)
        
        # Create a shell script that pipes input to the optimizer
        shell_script = """#!/bin/sh
echo "This is stdin input with special characters: @#$%" | python -m llm_agent_evolution --use-mock --eval-command "python {}" --population-size 5 --parallel-agents 2 --max-evaluations 3
""".format(script_path)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as shell_file:
            shell_script_path = shell_file.name
            shell_file.write(shell_script)
        
        # Make the shell script executable
        os.chmod(shell_script_path, 0o755)
        
        # Run the shell script
        result = subprocess.run(
            [shell_script_path],
            capture_output=True,
            text=True,
            timeout=120  # Increase timeout to prevent hanging
        )
        
        # Print output for debugging
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        
        # Check that it ran successfully
        assert result.returncode == 0
        
        # More specific check for stdin input
        assert "This is stdin input with special characters: @#$%" in result.stdout or \
               "Read context from stdin: This is stdin input with special characters: @#$%" in result.stdout, \
               "Stdin input not found in output"
        
        # Test with a more direct approach using Python's subprocess
        # This is more reliable than shell=True with pipes
        context_text = "Direct subprocess test with JSON data: {\"key\": \"value\"}"
        
        # Create a process that will provide the context via stdin
        process = subprocess.Popen(
            [
                "python", "-m", "llm_agent_evolution",
                "--use-mock",
                "--eval-command", f"python {script_path}",
                "--population-size", "5",
                "--parallel-agents", "2",
                "--max-evaluations", "3"
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Send the context via stdin
        stdout, stderr = process.communicate(input=context_text, timeout=60)
        
        # Print output for debugging
        print(f"STDOUT: {stdout}")
        print(f"STDERR: {stderr}")
        
        # Check that it ran successfully
        assert process.returncode == 0
        
        # Check that the context was used
        assert context_text in stdout or \
               f"Read context from stdin: {context_text}" in stdout or \
               "Direct subprocess test with JSON data" in stdout, \
               "Context from stdin not found in output"
        
        # Test with a more complex JSON input
        json_context = """
        {
            "task": "Optimize for maximum 'a' characters",
            "constraints": {
                "max_length": 23,
                "allowed_chars": "abcdefghijklmnopqrstuvwxyz"
            },
            "examples": [
                {"input": "test", "output": "aaaaa"},
                {"input": "more", "output": "aaaaaaaaaaa"}
            ]
        }
        """
        
        # Create a process that will provide the JSON context via stdin
        process = subprocess.Popen(
            [
                "python", "-m", "llm_agent_evolution",
                "--use-mock",
                "--eval-command", f"python {script_path}",
                "--population-size", "5",
                "--parallel-agents", "2",
                "--max-evaluations", "3"
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Send the JSON context via stdin
        stdout, stderr = process.communicate(input=json_context, timeout=60)
        
        # Print output for debugging
        print(f"STDOUT: {stdout}")
        print(f"STDERR: {stderr}")
        
        # Check that it ran successfully
        assert process.returncode == 0
        
        # Check that the context was used - look for snippets that would indicate the JSON was processed
        assert "task" in stdout or "Optimize for maximum" in stdout or \
               "Read context from stdin" in stdout, \
               "JSON context from stdin not found in output"
        
    finally:
        # Clean up
        if os.path.exists(script_path):
            os.remove(script_path)
        if shell_script_path and os.path.exists(shell_script_path):
            os.remove(shell_script_path)
