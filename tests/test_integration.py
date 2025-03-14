import pytest
import sys
import os
import tempfile

# Add the src directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from llm_agent_evolution.domain.model import Agent, Chromosome
from llm_agent_evolution.domain.services import mate_agents
from llm_agent_evolution.adapters.secondary.mock_llm import MockLLMAdapter
from llm_agent_evolution.adapters.secondary.logging import FileLoggingAdapter
from llm_agent_evolution.adapters.secondary.statistics import StatisticsAdapter
from llm_agent_evolution.application import EvolutionService

@pytest.fixture
def evolution_service():
    """Create an evolution service with mock adapters for testing"""
    # Create a temporary log file
    temp_dir = tempfile.gettempdir()
    log_file = os.path.join(temp_dir, "test_evolution.log")
    
    # Create adapters
    llm_adapter = MockLLMAdapter(seed=42)
    logging_adapter = FileLoggingAdapter(log_file=log_file)
    statistics_adapter = StatisticsAdapter()
    
    # Create and return the service
    service = EvolutionService(
        llm_port=llm_adapter,
        logging_port=logging_adapter,
        statistics_port=statistics_adapter
    )
    
    yield service
    
    # Cleanup
    if os.path.exists(log_file):
        os.remove(log_file)

def test_initialize_population(evolution_service):
    """Test population initialization"""
    population = evolution_service.initialize_population(10)
    assert len(population) == 10
    assert all(isinstance(agent, Agent) for agent in population)

def test_mutate_agent(evolution_service):
    """Test agent mutation"""
    # Create an agent
    agent = Agent(
        task_chromosome=Chromosome(content="original task", type="task"),
        mate_selection_chromosome=Chromosome(content="original mate selection", type="mate_selection"),
        mutation_chromosome=Chromosome(content="original mutation", type="mutation")
    )
    
    # Mutate the agent
    mutated = evolution_service.mutate_agent(agent)
    
    # Check the result
    assert isinstance(mutated, Agent)
    assert mutated.id != agent.id
    assert mutated.task_chromosome.content != agent.task_chromosome.content
    assert mutated.mate_selection_chromosome.content != agent.mate_selection_chromosome.content
    assert mutated.mutation_chromosome.content != agent.mutation_chromosome.content

def test_evaluate_agent(evolution_service):
    """Test agent evaluation"""
    # Create an agent with a known task output
    agent = Agent(
        task_chromosome=Chromosome(content="a" * 15, type="task"),
        mate_selection_chromosome=Chromosome(content="mate selection", type="mate_selection"),
        mutation_chromosome=Chromosome(content="mutation", type="mutation")
    )
    
    # Evaluate the agent
    reward = evolution_service.evaluate_agent(agent)
    
    # Check the result
    assert reward == 15
    assert agent.reward == 15

def test_mini_evolution_cycle(evolution_service):
    """Test a mini evolution cycle"""
    # Initialize a small population
    population = evolution_service.initialize_population(5)
    
    # Evaluate all agents
    for agent in population:
        evolution_service.evaluate_agent(agent)
    
    # Select parents
    parents = evolution_service.select_parents(population, 2)
    assert len(parents) == 2
    
    # Create a new agent through mating
    parent1, parent2 = parents
    offspring = evolution_service.mate_agents(parent1, parent2)
    
    # Mutate the offspring
    mutated = evolution_service.mutate_agent(offspring)
    
    # Evaluate the mutated offspring
    reward = evolution_service.evaluate_agent(mutated)
    
    # Add to population
    new_population = evolution_service.add_to_population(population, mutated)
    
    # Check results
    assert len(new_population) == 6
    assert mutated in new_population
    assert mutated.reward is not None
