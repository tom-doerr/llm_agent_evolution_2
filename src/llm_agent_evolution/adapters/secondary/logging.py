import os
import time
from typing import Dict, Any
from ....domain.model import Agent
from ....ports.secondary import LoggingPort

class FileLoggingAdapter(LoggingPort):
    """Adapter for logging to a file"""
    
    def __init__(self, log_file: str = "evolution.log"):
        """Initialize the logging adapter with the specified log file"""
        self.log_file = log_file
    
    def initialize_log(self) -> None:
        """Initialize or clear the log file"""
        # Create an empty log file
        with open(self.log_file, 'w') as f:
            f.write(f"# LLM Agent Evolution Log - Started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    def log_evaluation(self, agent: Agent) -> None:
        """Log an agent evaluation"""
        with open(self.log_file, 'a') as f:
            f.write(f"--- Agent Evaluation: {agent.id} ---\n")
            f.write(f"Reward: {agent.reward}\n")
            f.write(f"Task Chromosome: {agent.task_chromosome.content}\n")
            f.write(f"Mate Selection Chromosome: {agent.mate_selection_chromosome.content}\n")
            f.write(f"Mutation Chromosome: {agent.mutation_chromosome.content}\n")
            f.write("\n")
    
    def log_population_stats(self, stats: Dict[str, Any]) -> None:
        """Log population statistics"""
        with open(self.log_file, 'a') as f:
            f.write(f"--- Population Statistics at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
    
    def log_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Log a general event"""
        with open(self.log_file, 'a') as f:
            f.write(f"--- {event_type} at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            for key, value in details.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
