import pytest
import sys
import os
import tempfile
import tomli_w
import subprocess
import shutil

# Add the src directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from llm_agent_evolution.domain.model import Agent, Chromosome

def is_command_available(command):
    """Check if a command is available on the system"""
    return shutil.which(command) is not None

def test_save_and_load_agent():
    """Test saving and loading an agent using TOML format"""
    # Skip this test in CI environments
    if os.environ.get('CI') == 'true':
        pytest.skip("Skipping test that requires file operations in CI environment")
    
    # Create an agent
    agent = Agent(
        task_chromosome=Chromosome(content="This is a test task", type="task"),
        mate_selection_chromosome=Chromosome(content="Select the best mate", type="mate_selection"),
        mutation_chromosome=Chromosome(content="Mutate wisely", type="mutation"),
        reward=42.0
    )
    
    # Create a temporary file for the agent
    with tempfile.NamedTemporaryFile(suffix='.toml', delete=False) as temp_file:
        agent_file = temp_file.name
        
        # Save the agent to the file
        agent_data = {
            "agent": {
                "id": agent.id,
                "reward": agent.reward,
                "task_chromosome": {
                    "content": agent.task_chromosome.content,
                    "type": agent.task_chromosome.type
                },
                "mate_selection_chromosome": {
                    "content": agent.mate_selection_chromosome.content,
                    "type": agent.mate_selection_chromosome.type
                },
                "mutation_chromosome": {
                    "content": agent.mutation_chromosome.content,
                    "type": agent.mutation_chromosome.type
                }
            }
        }
        
        tomli_w.dump(agent_data, temp_file)
    
    script_path = None
    
    try:
        # Create a simple evaluation script
        script_content = """#!/usr/bin/env python3
import sys
import os

# Get context from environment variable
context = os.environ.get('AGENT_CONTEXT', '')

# Get agent output from stdin
agent_output = sys.stdin.read()

# Print the output and context for verification
print(f"Agent output: {agent_output}")
print(f"Context: {context}")

# Return a fixed reward for testing
print(42.0)
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as script_file:
            script_path = script_file.name
            script_file.write(script_content)
        
        # Make the script executable
        os.chmod(script_path, 0o755)
        
        # Run the command with the loaded agent
        result = subprocess.run(
            [
                "python", "-m", "llm_agent_evolution",
                "--use-mock",
                "--load", agent_file,
                "--eval-command", f"python {script_path}",
                "--context", "Test context"
            ],
            capture_output=True,
            text=True,
            timeout=60  # Add timeout to prevent hanging
        )
        
        # Print output for debugging
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        
        # Check that it ran successfully
        assert result.returncode == 0
        assert "Agent output: This is a test task" in result.stdout
        assert "Context: Test context" in result.stdout
        
        # Now test with inference only (no optimization)
        result = subprocess.run(
            [
                "python", "-m", "llm_agent_evolution",
                "inference",  # New subcommand for just running inference
                "--use-mock",
                "--load", agent_file,
                "--eval-command", f"python {script_path}",
                "--context", "Inference context"
            ],
            capture_output=True,
            text=True,
            timeout=60  # Add timeout to prevent hanging
        )
        
        # Print output for debugging
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        
        # Check that it ran successfully in inference mode
        assert result.returncode == 0
        assert "Agent output: This is a test task" in result.stdout
        assert "Context: Inference context" in result.stdout
        assert "RUNNING INFERENCE WITH LOADED AGENT" in result.stdout
        
    finally:
        # Clean up
        if os.path.exists(agent_file):
            os.remove(agent_file)
        if script_path and os.path.exists(script_path):
            os.remove(script_path)

def test_e2e_agent_loading():
    """Test end-to-end agent loading and inference"""
    # Skip this test in CI environments
    if os.environ.get('CI') == 'true':
        pytest.skip("Skipping test that requires CLI in CI environment")
    
    # Create a simple evaluation script instead of relying on examples/count_a.py
    script_content = """#!/usr/bin/env python3
import sys
import os

# Get context from environment variable
context = os.environ.get('AGENT_CONTEXT', '')

# Get agent output from stdin
agent_output = sys.stdin.read()

# Count 'a's in the output
a_count = agent_output.count('a')

# Print the output and context for verification
print(f"Agent output: {agent_output}")
print(f"Context: {context}")
print(f"a count: {a_count}")

# Return the count as reward
print(a_count)
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as script_file:
        script_path = script_file.name
        script_file.write(script_content)
    
    # Make the script executable
    os.chmod(script_path, 0o755)
    
    # Create a temporary agent file
    agent_data = {
        "agent": {
            "id": "test-agent-id",
            "reward": 42.0,
            "task_chromosome": {
                "content": "aaaaaaaaaaaaaaaaaaaaaaa",  # 23 a's (optimal)
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
    
    with tempfile.NamedTemporaryFile(suffix='.toml', delete=False) as temp_file:
        agent_file = temp_file.name
        tomli_w.dump(agent_data, temp_file)
    
    try:
        # Run the command with the loaded agent
        result = subprocess.run(
            [
                "python", "-m", "llm_agent_evolution",
                "--use-mock",
                "--load", agent_file,
                "--eval-command", f"python {script_path}"
            ],
            capture_output=True,
            text=True,
            timeout=60  # Add timeout to prevent hanging
        )
        
        # Print output for debugging
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        
        # Check that it ran successfully
        assert result.returncode == 0
        assert "Agent output: aaaaaaaaaaaaaaaaaaaaaaa" in result.stdout
        assert "23" in result.stdout  # The reward
        
        # Now test with context
        result = subprocess.run(
            [
                "python", "-m", "llm_agent_evolution",
                "--use-mock",
                "--load", agent_file,
                "--eval-command", f"python {script_path}",
                "--context", "This is a test context"
            ],
            capture_output=True,
            text=True,
            timeout=60  # Add timeout to prevent hanging
        )
        
        # Print output for debugging
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        
        # Check that it ran successfully with context
        assert result.returncode == 0
        assert "Agent output: aaaaaaaaaaaaaaaaaaaaaaa" in result.stdout
        assert "Context: This is a test context" in result.stdout
        
    finally:
        # Clean up
        if os.path.exists(agent_file):
            os.remove(agent_file)
        if os.path.exists(script_path):
            os.remove(script_path)
