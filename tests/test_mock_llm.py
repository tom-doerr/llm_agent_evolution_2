import pytest
import sys
import os

# Add the src directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from llm_agent_evolution.domain.model import Agent, Chromosome
from llm_agent_evolution.adapters.secondary.mock_llm import MockLLMAdapter

def test_mock_llm_initialization():
    """Test creating a mock LLM adapter"""
    adapter = MockLLMAdapter(seed=42)
    assert adapter is not None

def test_mock_generate_mutation():
    """Test generating mutations with the mock adapter"""
    adapter = MockLLMAdapter(seed=42)
    
    # Test task chromosome mutation
    task_chromosome = Chromosome(content="", type="task")
    mutated = adapter.generate_mutation(task_chromosome, "")
    assert mutated.type == "task"
    assert isinstance(mutated.content, str)
    
    # Test mate selection chromosome mutation
    mate_chromosome = Chromosome(content="", type="mate_selection")
    mutated = adapter.generate_mutation(mate_chromosome, "")
    assert mutated.type == "mate_selection"
    assert isinstance(mutated.content, str)
    
    # Test mutation chromosome mutation
    mutation_chromosome = Chromosome(content="", type="mutation")
    mutated = adapter.generate_mutation(mutation_chromosome, "")
    assert mutated.type == "mutation"
    assert isinstance(mutated.content, str)

def test_mock_select_mate():
    """Test mate selection with the mock adapter"""
    adapter = MockLLMAdapter(seed=42)
    
    # Create agent and candidates
    agent = Agent(
        task_chromosome=Chromosome(content="", type="task"),
        mate_selection_chromosome=Chromosome(content="", type="mate_selection"),
        mutation_chromosome=Chromosome(content="", type="mutation")
    )
    
    candidates = [
        Agent(
            task_chromosome=Chromosome(content=f"task{i}", type="task"),
            mate_selection_chromosome=Chromosome(content=f"mate{i}", type="mate_selection"),
            mutation_chromosome=Chromosome(content=f"mutation{i}", type="mutation")
        ) for i in range(3)
    ]
    
    # Test selection
    selected = adapter.select_mate(agent, candidates)
    assert selected in candidates

def test_mock_evaluate_task_output():
    """Test task evaluation with the mock adapter"""
    adapter = MockLLMAdapter(seed=42)  # Use fixed seed for deterministic results
    
    # Test with different outputs
    assert adapter.evaluate_task_output("") == 0
    assert adapter.evaluate_task_output("a" * 10) == 10
    assert adapter.evaluate_task_output("a" * 23) == 23
    assert adapter.evaluate_task_output("a" * 30) == 16  # 23 'a's but 7 chars over limit
    
    # Test with mixed content
    mixed = "a" * 15 + "b" * 5
    assert adapter.evaluate_task_output(mixed) == 15  # 15 'a's, length 20 (under limit)
    
    mixed_over = "a" * 15 + "b" * 15
    assert adapter.evaluate_task_output(mixed_over) == 8  # 15 'a's, 7 chars over limit
    
    # Test with command-based evaluation
    adapter.eval_command = "echo 42"
    result = adapter.evaluate_task_output("test")
    assert result == 42.0
