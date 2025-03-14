from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ..domain.model import Agent

class EvolutionUseCase(ABC):
    """Primary port for driving the evolution process"""
    
    @abstractmethod
    def initialize_population(self, size: int) -> List[Agent]:
        """Initialize a population of the given size"""
        pass
    
    @abstractmethod
    def select_parents(self, population: List[Agent], num_parents: int) -> List[Agent]:
        """Select parents from the population for mating"""
        pass
    
    @abstractmethod
    def mate_agents(self, parent1: Agent, parent2: Agent) -> Agent:
        """Create a new agent by mating two parents"""
        pass
    
    @abstractmethod
    def mutate_agent(self, agent: Agent) -> Agent:
        """Mutate an agent using its mutation chromosome"""
        pass
    
    @abstractmethod
    def evaluate_agent(self, agent: Agent) -> float:
        """Evaluate an agent and return its reward"""
        pass
    
    @abstractmethod
    def add_to_population(self, population: List[Agent], agent: Agent) -> List[Agent]:
        """Add an agent to the population, maintaining population constraints"""
        pass
    
    @abstractmethod
    def get_population_stats(self, population: List[Agent]) -> Dict[str, Any]:
        """Get statistics about the current population"""
        pass
    
    @abstractmethod
    def run_evolution(self, 
                     population_size: int, 
                     parallel_agents: int,
                     max_evaluations: Optional[int] = None,
                     progress_callback: Optional[callable] = None,
                     initial_population: Optional[List[Agent]] = None) -> List[Agent]:
        """Run the evolution process with the given parameters"""
        pass
