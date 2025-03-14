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

def test_cli_help():
    """Test that the CLI shows help"""
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
    assert "--population-size" in result.stdout
    assert "--parallel-agents" in result.stdout
    
    # Also test the installed command if available
    if is_command_available("llm-evolve"):
        result = subprocess.run(
            ["llm-evolve", "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "LLM Agent Evolution" in result.stdout

def test_cli_quick_test():
    """Test that the quick test mode works"""
    # Skip this test in CI environments
    if os.environ.get('CI') == 'true':
        pytest.skip("Skipping test that requires CLI in CI environment")
    
    # Create a temporary log file
    with tempfile.NamedTemporaryFile(suffix='.log') as temp_log:
        # Run the command with quick test mode
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

def test_cli_without_subcommand():
    """Test that the CLI works without specifying a subcommand"""
    # Skip this test in CI environments
    if os.environ.get('CI') == 'true':
        pytest.skip("Skipping test that requires CLI in CI environment")
    
    # Run the command without a subcommand
    result = subprocess.run(
        [
            "python", "-m", "llm_agent_evolution", 
            "--quick-test",
            "--max-evaluations", "5"  # Limit to 5 evaluations for speed
        ],
        capture_output=True,
        text=True
    )
    
    # Check that it ran successfully
    assert result.returncode == 0
    assert "Running quick test with mock LLM adapter" in result.stdout
    assert "Starting evolution" in result.stdout
