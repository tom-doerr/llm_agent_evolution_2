import sys
import os
import pytest

# Add the src directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from llm_agent_evolution.domain.model import Chromosome
from llm_agent_evolution.adapters.secondary.llm import DSPyLLMAdapter
from llm_agent_evolution.adapters.secondary.mock_llm import MockLLMAdapter

def test_mock_llm_generate_mutation():
    """Test that the mock LLM can generate mutations"""
    llm = MockLLMAdapter(seed=42)
    chromosome = Chromosome(content="test content", type="task")
    mutation_instructions = "mutate this"
    
    result = llm.generate_mutation(chromosome, mutation_instructions)
    
    assert isinstance(result, Chromosome)
    assert isinstance(result.content, str)
    assert result.type == "task"

def test_dspy_llm_generate_mutation():
    """Test that the DSPy LLM adapter properly handles string inputs and outputs"""
    # Skip this test in CI environments
    if os.environ.get('CI') == 'true':
        pytest.skip("Skipping test that requires real LLM in CI environment")
    
    try:
        llm = DSPyLLMAdapter(model_name="openrouter/google/gemini-2.0-flash-001")
        chromosome = Chromosome(content="test content", type="task")
        mutation_instructions = "Add more a's to this content"
        
        result = llm.generate_mutation(chromosome, mutation_instructions)
        
        assert isinstance(result, Chromosome)
        assert isinstance(result.content, str)
        assert result.type == "task"
        print(f"Generated content: {result.content}")
    except Exception as e:
        pytest.fail(f"DSPy LLM adapter failed: {e}")
