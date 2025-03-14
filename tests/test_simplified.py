import pytest
import sys
import os
import tempfile
import random

# Add the src directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from llm_agent_evolution.domain.model import Agent, Chromosome, TARGET_LENGTH
from llm_agent_evolution.domain.services import select_parents_pareto, combine_chromosomes, mate_agents
from llm_agent_evolution.adapters.secondary.mock_llm import MockLLMAdapter
from llm_agent_evolution.evolution import EvolutionEngine

def test_domain_model():
    """Test the domain model classes"""
    # Test Chromosome
    chromosome = Chromosome(content="test content", type="task")
    assert chromosome.content == "test content"
    assert chromosome.type == "task"
    
    # Test Agent
    agent = Agent(
        task_chromosome=Chromosome(content="task content", type="task"),
        mate_selection_chromosome=Chromosome(content="mate content", type="mate_selection"),
        mutation_chromosome=Chromosome(content="mutation content", type="mutation")
    )
    assert agent.id is not None
    assert agent.task_chromosome.content == "task content"
    assert agent.chromosomes["task"] == agent.task_chromosome
    assert agent.get_chromosome("task") == agent.task_chromosome

def test_parent_selection():
    """Test parent selection using Pareto distribution"""
    # Create a population with varying rewards
    population = [
        Agent(
            task_chromosome=Chromosome(content=f"task{i}", type="task"),
            mate_selection_chromosome=Chromosome(content=f"mate{i}", type="mate_selection"),
            mutation_chromosome=Chromosome(content=f"mutation{i}", type="mutation"),
            reward=i * 10
        ) for i in range(10)
    ]
    
    # Select parents
    parents = select_parents_pareto(population, 5)
    assert len(parents) == 5
    assert all(isinstance(p, Agent) for p in parents)
    
    # Test with negative rewards
    for i, agent in enumerate(population):
        agent.reward = i * 10 - 50  # Rewards from -50 to 40
    
    parents = select_parents_pareto(population, 5)
    assert len(parents) == 5

def test_chromosome_combination():
    """Test combining chromosomes"""
    # Test with task chromosomes
    parent1 = Chromosome(content="a" * 10, type="task")
    parent2 = Chromosome(content="a" * TARGET_LENGTH, type="task")
    
    # Set a fixed seed for deterministic testing
    random.seed(42)
    
    # Run multiple combinations to check statistical properties
    results = []
    for _ in range(20):
        result = combine_chromosomes(parent1, parent2)
        results.append(len(result.content))
    
    # Check that at least some results are exactly TARGET_LENGTH
    assert TARGET_LENGTH in results
    
    # Reset the random seed
    random.seed(None)

def test_mock_llm_adapter():
    """Test the mock LLM adapter"""
    adapter = MockLLMAdapter(seed=42)
    
    # Test mutation
    chromosome = Chromosome(content="test content", type="task")
    mutated = adapter.generate_mutation(chromosome, "mutate this")
    assert isinstance(mutated, Chromosome)
    assert mutated.type == "task"
    
    # Test mate selection
    agent = Agent(
        task_chromosome=Chromosome(content="agent dna", type="task"),
        mate_selection_chromosome=Chromosome(content="select best", type="mate_selection"),
        mutation_chromosome=Chromosome(content="mutate well", type="mutation")
    )
    
    candidates = [
        Agent(
            task_chromosome=Chromosome(content=f"candidate{i}", type="task"),
            mate_selection_chromosome=Chromosome(content="", type="mate_selection"),
            mutation_chromosome=Chromosome(content="", type="mutation")
        ) for i in range(3)
    ]
    
    selected = adapter.select_mate(agent, candidates)
    assert selected in candidates
    
    # Test evaluation
    reward = adapter.evaluate_task_output("a" * TARGET_LENGTH)
    assert reward == TARGET_LENGTH
    
    reward = adapter.evaluate_task_output("a" * (TARGET_LENGTH + 5))
    assert reward == TARGET_LENGTH - 5  # Penalty for exceeding target length

def test_evolution_engine():
    """Test the evolution engine"""
    # Create a mock LLM adapter
    adapter = MockLLMAdapter(seed=42)
    
    # Create an evolution engine
    engine = EvolutionEngine(
        llm_adapter=adapter,
        population_size=10,
        parallel_agents=2,
        initial_content="a",
        verbose=False
    )
    
    # Initialize population
    population = engine.initialize_population()
    assert len(population) == 10
    
    # Evaluate an agent
    agent = population[0]
    reward = engine.evaluate_agent(agent)
    assert isinstance(reward, float)
    assert agent.reward == reward
    
    # Get statistics
    stats = engine.get_stats()
    assert stats["count"] == 1
    assert stats["population_size"] == 10
    assert stats["best"] == reward
    assert stats["worst"] == reward
