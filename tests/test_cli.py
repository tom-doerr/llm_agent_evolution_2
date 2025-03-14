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

def test_cli_with_eval_command():
    """Test that the CLI works with an evaluation command"""
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
        
        # Run the command with the evaluation command
        result = subprocess.run(
            [
                "python", "-m", "llm_agent_evolution", 
                f"python {script_path}",  # Positional argument as eval command
                "--use-mock",
                "--population-size", "10",
                "--parallel-agents", "2",
                "--max-evaluations", "5"
            ],
            capture_output=True,
            text=True
        )
        
        # Check that it ran successfully
        assert result.returncode == 0
        assert "Using evaluation command: python" in result.stdout
        assert "Starting optimization" in result.stdout or "Starting evolution" in result.stdout
        
        # Test the inference command
        # First create a simple agent file
        import tomli_w
        agent_data = {
            "agent": {
                "id": "test-agent-id",
                "reward": 42.0,
                "task_chromosome": {
                    "content": "Test inference content",
                    "type": "task"
                },
                "mate_selection_chromosome": {
                    "content": "Select the mate with the highest reward",
                    "type": "mate_selection"
                },
                "mutation_chromosome": {
                    "content": "Add more a's to the content",
                    "type": "mutation"
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(suffix='.toml', delete=False) as agent_file:
            agent_file_path = agent_file.name
            tomli_w.dump(agent_data, agent_file)
        
        # Run the inference command
        result = subprocess.run(
            [
                "python", "-m", "llm_agent_evolution", 
                "inference",
                "--load", agent_file_path,
                "--eval-command", f"python {script_path}",
                "--context", "Test context"
            ],
            capture_output=True,
            text=True
        )
        
        # Check that it ran successfully
        assert result.returncode == 0
        assert "RUNNING INFERENCE WITH LOADED AGENT" in result.stdout
        assert "Loaded agent with ID: test-agent-id" in result.stdout
        assert "Agent output: Test inference content" in result.stdout
        assert "Reward: 21.0" in result.stdout  # Length of "Test inference content"
        
    finally:
        # Clean up
        if os.path.exists(script_path):
            os.remove(script_path)
        if os.path.exists(agent_file_path):
            os.remove(agent_file_path)
