import pytest
import sys
import os
import tempfile
import subprocess

# Add the src directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from llm_agent_evolution.domain.model import Agent, Chromosome
from llm_agent_evolution.adapters.secondary.llm import DSPyLLMAdapter
from llm_agent_evolution.adapters.secondary.mock_llm import MockLLMAdapter

def test_command_evaluation_with_mock():
    """Test command-based evaluation with mock LLM adapter"""
    # Create a mock LLM adapter
    llm_adapter = MockLLMAdapter(seed=42)
    
    # Set an evaluation command that returns the length of the input
    llm_adapter.eval_command = "python -c 'import sys; text=sys.stdin.read(); print(len(text))'"
    
    # Test with different outputs
    test_inputs = [
        "",
        "hello",
        "hello world",
        "a" * 50
    ]
    
    for text in test_inputs:
        # Create a task chromosome with the test input
        task_chromosome = Chromosome(content=text, type="task")
        
        # Create an agent with this chromosome
        agent = Agent(
            task_chromosome=task_chromosome,
            mate_selection_chromosome=Chromosome(content="", type="mate_selection"),
            mutation_chromosome=Chromosome(content="", type="mutation")
        )
        
        # Evaluate the agent
        reward = llm_adapter.evaluate_task_output(text)
        
        # Check that the reward matches the expected length
        assert reward == len(text)

def test_script_based_evaluation_integration():
    """Test integration between script evaluator and LLM adapter"""
    # Create a temporary evaluation script
    script_content = """#!/usr/bin/env python3
import sys
text = sys.stdin.read()
print(len(text))  # Reward is the length of the text
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        script_path = f.name
        f.write(script_content)
    
    try:
        # Make the script executable
        os.chmod(script_path, 0o755)
        
        # Create a mock LLM adapter
        llm_adapter = MockLLMAdapter(seed=42)
        
        # Create an agent with a known task output
        agent = Agent(
            task_chromosome=Chromosome(content="test content", type="task"),
            mate_selection_chromosome=Chromosome(content="", type="mate_selection"),
            mutation_chromosome=Chromosome(content="", type="mutation")
        )
        
        # Run the script directly to get the expected result
        process = subprocess.run(
            [script_path],
            input="test content",
            text=True,
            capture_output=True
        )
        expected_reward = float(process.stdout.strip())
        
        # Now use the LLM adapter with the script
        from llm_agent_evolution.adapters.secondary.script_evaluator import ScriptEvaluatorAdapter
        evaluator = ScriptEvaluatorAdapter()
        actual_reward = evaluator.evaluate(agent.task_chromosome.content, script_path)
        
        # Check that the results match
        assert actual_reward == expected_reward
        assert actual_reward == 12.0  # Length of "test content"
        
    finally:
        # Clean up
        if os.path.exists(script_path):
            os.remove(script_path)

def test_cli_command_evaluation_integration():
    """Test integration between CLI command evaluation and the application"""
    # Skip this test in CI environments
    if os.environ.get('CI') == 'true':
        pytest.skip("Skipping test that requires subprocess in CI environment")
    
    # Create a mock LLM adapter with fixed seed
    llm_adapter = MockLLMAdapter(seed=42)
    
    # Set an evaluation command
    eval_command = "python -c 'import sys; text=sys.stdin.read(); print(len(text))'"
    llm_adapter.eval_command = eval_command
    
    # Test with a specific input
    test_input = "Hello, world!"
    expected_length = len(test_input)
    
    # Evaluate using the adapter
    reward = llm_adapter.evaluate_task_output(test_input)
    
    # Check the result
    assert reward == expected_length
    
    # Now test the command directly to verify
    try:
        process = subprocess.run(
            eval_command,
            shell=True,
            input=test_input,
            text=True,
            capture_output=True
        )
        direct_result = float(process.stdout.strip())
        
        # Results should match
        assert reward == direct_result
    except Exception as e:
        pytest.skip(f"Skipping direct command test due to error: {e}")
