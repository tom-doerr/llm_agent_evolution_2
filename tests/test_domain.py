import pytest
import sys
import os

# Add the src directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from llm_agent_evolution.domain.model import Agent, Chromosome
from llm_agent_evolution.domain.services import select_parents_pareto, combine_chromosomes, mate_agents

def test_chromosome_creation():
    """Test creating a chromosome"""
    chromosome = Chromosome(content="test content", type="task")
    assert chromosome.content == "test content"
    assert chromosome.type == "task"

def test_agent_creation():
    """Test creating an agent"""
    task_chromosome = Chromosome(content="task content", type="task")
    mate_selection_chromosome = Chromosome(content="mate selection content", type="mate_selection")
    mutation_chromosome = Chromosome(content="mutation content", type="mutation")
    
    agent = Agent(
        task_chromosome=task_chromosome,
        mate_selection_chromosome=mate_selection_chromosome,
        mutation_chromosome=mutation_chromosome
    )
    
    assert agent.id is not None
    assert agent.reward is None
    assert agent.task_chromosome.content == "task content"
    assert agent.mate_selection_chromosome.content == "mate selection content"
    assert agent.mutation_chromosome.content == "mutation content"

def test_combine_chromosomes():
    """Test combining chromosomes"""
    parent1 = Chromosome(content="This is the first chromosome.", type="task")
    parent2 = Chromosome(content="This is the second chromosome.", type="task")
    
    # Since combination has randomness, we just check basic properties
    result = combine_chromosomes(parent1, parent2)
    assert isinstance(result, Chromosome)
    assert result.type == "task"
    assert len(result.content) > 0

def test_combine_chromosomes_with_empty_content():
    """Test combining chromosomes when one has empty content"""
    parent1 = Chromosome(content="", type="task")
    parent2 = Chromosome(content="This is the second chromosome.", type="task")
    
    result = combine_chromosomes(parent1, parent2)
    assert result.content == parent2.content
    
    # Test the reverse case
    result = combine_chromosomes(parent2, parent1)
    assert result.content == parent2.content
    
    # Test both empty
    parent1 = Chromosome(content="", type="task")
    parent2 = Chromosome(content="", type="task")
    result = combine_chromosomes(parent1, parent2)
    assert result.content == ""

def test_combine_task_chromosomes_length_handling():
    """Test that task chromosomes are handled appropriately for length"""
    from llm_agent_evolution.domain.services import TARGET_LENGTH
    # Create a parent with optimal length
    parent1 = Chromosome(content="a" * TARGET_LENGTH, type="task")
    # Create a parent with too much content
    parent2 = Chromosome(content="a" * (TARGET_LENGTH * 2), type="task")
    
    # Set a fixed seed for deterministic testing
    import random
    random.seed(42)
    
    # Run multiple combinations to check statistical properties
    results = []
    for _ in range(20):
        result = combine_chromosomes(parent1, parent2)
        results.append(len(result.content))
    
    # The average length should be closer to TARGET_LENGTH than to the longer parent
    avg_length = sum(results) / len(results)
    assert abs(avg_length - TARGET_LENGTH) < abs(avg_length - len(parent2.content)), \
        f"Average length {avg_length} should be closer to target {TARGET_LENGTH} than to longer parent {len(parent2.content)}"
    
    # Check that at least some results are exactly TARGET_LENGTH
    assert TARGET_LENGTH in results, f"None of the results had the target length {TARGET_LENGTH}"
    
    # Check that a significant portion of results are at or near TARGET_LENGTH
    near_target = [r for r in results if abs(r - TARGET_LENGTH) <= 2]
    assert len(near_target) >= len(results) * 0.5, \
        f"Only {len(near_target)} out of {len(results)} results were near the target length"
    
    # Reset the random seed
    random.seed(None)

def test_mate_agents():
    """Test mating two agents"""
    agent1 = Agent(
        task_chromosome=Chromosome(content="task1", type="task"),
        mate_selection_chromosome=Chromosome(content="mate1", type="mate_selection"),
        mutation_chromosome=Chromosome(content="mutation1", type="mutation")
    )
    
    agent2 = Agent(
        task_chromosome=Chromosome(content="task2", type="task"),
        mate_selection_chromosome=Chromosome(content="mate2", type="mate_selection"),
        mutation_chromosome=Chromosome(content="mutation2", type="mutation")
    )
    
    offspring = mate_agents(agent1, agent2)
    assert isinstance(offspring, Agent)
    assert offspring.id != agent1.id
    assert offspring.id != agent2.id

def test_select_parents_pareto():
    """Test parent selection using Pareto distribution"""
    # Create a population with varying rewards
    population = [
        Agent(
            task_chromosome=Chromosome(content=f"task{i}", type="task"),
            mate_selection_chromosome=Chromosome(content=f"mate{i}", type="mate_selection"),
            mutation_chromosome=Chromosome(content=f"mutation{i}", type="mutation"),
        ) for i in range(10)
    ]
    
    # Assign rewards
    for i, agent in enumerate(population):
        agent.reward = i * 10
    
    # Select parents
    parents = select_parents_pareto(population, 5)
    assert len(parents) == 5
    assert all(isinstance(p, Agent) for p in parents)
    
    # Run multiple selections to verify statistical properties
    selections = []
    for _ in range(100):
        selected = select_parents_pareto(population, 1)[0]
        selections.append(selected.reward)
    
    # Higher rewards should be selected more frequently
    mean_selection = sum(selections) / len(selections)
    assert mean_selection > 45  # Should be biased toward higher rewards

def test_select_parents_with_negative_rewards():
    """Test parent selection with negative rewards"""
    # Create a population with negative rewards
    population = [
        Agent(
            task_chromosome=Chromosome(content=f"task{i}", type="task"),
            mate_selection_chromosome=Chromosome(content=f"mate{i}", type="mate_selection"),
            mutation_chromosome=Chromosome(content=f"mutation{i}", type="mutation"),
        ) for i in range(10)
    ]
    
    # Assign rewards including negative values
    rewards = [-50, -30, -10, 0, 10, 20, 30, 40, 50, 60]
    for i, agent in enumerate(population):
        agent.reward = rewards[i]
    
    # Select parents
    parents = select_parents_pareto(population, 5)
    assert len(parents) == 5
    
    # Run multiple selections to verify statistical properties
    selections = []
    for _ in range(100):
        selected = select_parents_pareto(population, 1)[0]
        selections.append(selected.reward)
    
    # Higher rewards should be selected more frequently
    mean_selection = sum(selections) / len(selections)
    assert mean_selection > 0  # Should be biased toward higher rewards
