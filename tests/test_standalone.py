import pytest
import sys
import os
import tempfile
import subprocess

# Add the src directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from llm_agent_evolution.standalone import (
    Chromosome, Agent, combine_chromosomes, select_parents, mate_agents, mutate_agent
)

def test_chromosome_creation():
    """Test creating a chromosome"""
    chromosome = Chromosome(content="test content", type_="task")
    assert chromosome.content == "test content"
    assert chromosome.type == "task"

def test_agent_creation():
    """Test creating an agent"""
    task_chromosome = Chromosome(content="task content", type_="task")
    mate_selection_chromosome = Chromosome(content="mate selection content", type_="mate_selection")
    mutation_chromosome = Chromosome(content="mutation content", type_="mutation")
    
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
    parent1 = Chromosome(content="This is the first chromosome.", type_="task")
    parent2 = Chromosome(content="This is the second chromosome.", type_="task")
    
    # Since combination has randomness, we just check basic properties
    result = combine_chromosomes(parent1, parent2)
    assert isinstance(result, Chromosome)
    assert result.type == "task"
    assert len(result.content) > 0

def test_select_parents():
    """Test parent selection"""
    # Create a population with varying rewards
    population = [
        Agent(
            task_chromosome=Chromosome(content=f"task{i}", type_="task"),
            mate_selection_chromosome=Chromosome(content=f"mate{i}", type_="mate_selection"),
            mutation_chromosome=Chromosome(content=f"mutation{i}", type_="mutation"),
        ) for i in range(10)
    ]
    
    # Assign rewards
    for i, agent in enumerate(population):
        agent.reward = i * 10
    
    # Select parents
    parents = select_parents(population, 5)
    assert len(parents) == 5
    assert all(isinstance(p, Agent) for p in parents)

def test_mate_agents():
    """Test mating two agents"""
    agent1 = Agent(
        task_chromosome=Chromosome(content="task1", type_="task"),
        mate_selection_chromosome=Chromosome(content="mate1", type_="mate_selection"),
        mutation_chromosome=Chromosome(content="mutation1", type_="mutation")
    )
    
    agent2 = Agent(
        task_chromosome=Chromosome(content="task2", type_="task"),
        mate_selection_chromosome=Chromosome(content="mate2", type_="mate_selection"),
        mutation_chromosome=Chromosome(content="mutation2", type_="mutation")
    )
    
    offspring = mate_agents(agent1, agent2)
    assert isinstance(offspring, Agent)
    assert offspring.id != agent1.id
    assert offspring.id != agent2.id

def test_mutate_agent():
    """Test mutating an agent"""
    agent = Agent(
        task_chromosome=Chromosome(content="task content", type_="task"),
        mate_selection_chromosome=Chromosome(content="mate selection content", type_="mate_selection"),
        mutation_chromosome=Chromosome(content="mutation content", type_="mutation")
    )
    
    mutated = mutate_agent(agent)
    assert isinstance(mutated, Agent)
    assert mutated.id != agent.id
    # The task content might change due to mutation
    assert isinstance(mutated.task_chromosome.content, str)
