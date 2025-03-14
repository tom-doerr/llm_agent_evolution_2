from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ..domain.model import Agent, Chromosome

class LLMPort(ABC):
    """Secondary port for LLM interactions"""
    
    @abstractmethod
    def generate_mutation(self, chromosome: Chromosome, mutation_instructions: str) -> Chromosome:
        """Generate a mutation for a chromosome based on instructions"""
        pass
    
    @abstractmethod
    def select_mate(self, agent: Agent, candidates: List[Agent]) -> Agent:
        """Select a mate from candidates based on agent's mate selection chromosome"""
        pass
    
    @abstractmethod
    def evaluate_task_output(self, output: str) -> float:
        """Evaluate the output of a task and return a reward"""
        pass

class LoggingPort(ABC):
    """Secondary port for logging"""
    
    @abstractmethod
    def initialize_log(self) -> None:
        """Initialize or clear the log file"""
        pass
    
    @abstractmethod
    def log_evaluation(self, agent: Agent) -> None:
        """Log an agent evaluation"""
        pass
    
    @abstractmethod
    def log_population_stats(self, stats: Dict[str, Any]) -> None:
        """Log population statistics"""
        pass
    
    @abstractmethod
    def log_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Log a general event"""
        pass

class StatisticsPort(ABC):
    """Secondary port for statistics tracking"""
    
    @abstractmethod
    def track_reward(self, reward: float) -> None:
        """Track a new reward value"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        pass
    
    @abstractmethod
    def get_sliding_window_stats(self, window_size: int = 100) -> Dict[str, Any]:
        """Get statistics for the sliding window of recent evaluations"""
        pass
    
    @abstractmethod
    def get_improvement_history(self) -> List[Dict[str, Any]]:
        """Get the history of improvements for visualization"""
        pass
