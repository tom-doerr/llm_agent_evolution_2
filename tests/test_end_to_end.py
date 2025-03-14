import pytest
import sys
import os
import subprocess
import tempfile
import shutil

# Add the src directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

def is_command_available(command):
    """Check if a command is available on the system"""
    return shutil.which(command) is not None

def test_llm_evolve_help():
    """Test that the llm-evolve command works and shows help"""
    # Skip this test in CI environments
    if os.environ.get('CI') == 'true':
        pytest.skip("Skipping test that requires CLI in CI environment")
    
    # Run the command
    result = subprocess.run(
        ["python", "-m", "llm_agent_evolution", "--help"],
        capture_output=True,
        text=True
    )
    
    # Check that it ran successfully
    assert result.returncode == 0
    assert "LLM Agent Evolution" in result.stdout
    assert "Command to run" in result.stdout
    
    # Also test the direct command without module
    if is_command_available("llm-evolve"):
        result = subprocess.run(
            ["llm-evolve", "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "LLM Agent Evolution" in result.stdout
    else:
        # Skip this check if llm-evolve is not in PATH
        pytest.skip("llm-evolve command not found in PATH")

def test_llm_evolve_quick_test():
    """Test that the quick test mode works"""
    # Skip this test in CI environments
    if os.environ.get('CI') == 'true':
        pytest.skip("Skipping test that requires CLI in CI environment")
    
    # Create a temporary log file
    with tempfile.NamedTemporaryFile(suffix='.log') as temp_log:
        # Run the command with quick test mode without explicit subcommand
        result = subprocess.run(
            [
                "python", "-m", "llm_agent_evolution", 
                "--quick-test",
                "--log-file", temp_log.name,
                "--max-evaluations", "10"  # Limit to 10 evaluations for speed
            ],
            capture_output=True,
            text=True
        )
        
        # Check that it ran successfully
        assert result.returncode == 0
        assert "Running quick test with mock LLM adapter" in result.stdout
        
        # Check that the log file was created and has content
        with open(temp_log.name, 'r') as f:
            log_content = f.read()
            assert "LLM Agent Evolution Log" in log_content
            
    # Now test without the subcommand
    with tempfile.NamedTemporaryFile(suffix='.log') as temp_log:
        # Run the command with quick test mode without subcommand
        result = subprocess.run(
            [
                "python", "-m", "llm_agent_evolution", 
                "--quick-test",
                "--log-file", temp_log.name,
                "--max-evaluations", "10"  # Limit to 10 evaluations for speed
            ],
            capture_output=True,
            text=True
        )
        
        # Check that it ran successfully
        assert result.returncode == 0
        assert "Running quick test with mock LLM adapter" in result.stdout
        
        # Check that the log file was created and has content
        with open(temp_log.name, 'r') as f:
            log_content = f.read()
            assert "LLM Agent Evolution Log" in log_content

def test_llm_evolve_standalone():
    """Test that the standalone optimizer works"""
    # Skip this test in CI environments
    if os.environ.get('CI') == 'true':
        pytest.skip("Skipping test that requires CLI in CI environment")
    
    # Create a simple evaluation script
    script_content = """#!/usr/bin/env python3
import sys
text = sys.stdin.read()
print(len(text))  # Reward is the length of the text
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as script_file:
        script_path = script_file.name
        script_file.write(script_content)
    
    try:
        # Make the script executable
        os.chmod(script_path, 0o755)
        
        # Create a temporary log file for output
        with tempfile.NamedTemporaryFile(suffix='.log') as temp_log:
            # Run the command with eval-command instead of standalone subcommand
            result = subprocess.run(
                [
                    "python", "-m", "llm_agent_evolution", 
                    "--use-mock",
                    f"python {script_path}",  # Use as positional argument
                    "--population-size", "10",
                    "--parallel-agents", "2",
                    "--max-evaluations", "20",
                    "--log-file", temp_log.name
                ],
                capture_output=True,
                text=True
            )
            
            # Check that it ran successfully
            assert result.returncode == 0
            
            # Check for expected output - either optimization or evolution
            assert "Starting optimization" in result.stdout or "Starting evolution" in result.stdout
            assert "completed" in result.stdout.lower()
    finally:
        # Clean up
        if os.path.exists(script_path):
            os.remove(script_path)
