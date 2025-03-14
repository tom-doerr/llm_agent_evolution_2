import threading
import time
import sys
import os
import argparse
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import random

from .domain.model import Agent, Chromosome
from .domain.services import select_parents_pareto, mate_agents
from .ports.primary import EvolutionUseCase
from .ports.secondary import LLMPort, LoggingPort, StatisticsPort
from .adapters.secondary.llm import DSPyLLMAdapter
from .adapters.secondary.mock_llm import MockLLMAdapter
from .adapters.secondary.logging import FileLoggingAdapter
from .adapters.secondary.statistics import StatisticsAdapter
from .adapters.primary.cli import CLIAdapter

class EvolutionService(EvolutionUseCase):
    """Implementation of the evolution use case"""
    
    def __init__(self, 
                llm_port: LLMPort, 
                logging_port: LoggingPort, 
                statistics_port: StatisticsPort):
        """Initialize the evolution service with required ports"""
        self.llm_port = llm_port
        self.logging_port = logging_port
        self.statistics_port = statistics_port
        self.population_lock = threading.Lock()
        self.stop_event = threading.Event()
    
    def initialize_population(self, size: int) -> List[Agent]:
        """Initialize a population of the given size with empty chromosomes"""
        population = []
        
        # Create initial agents with empty chromosomes
        for _ in range(size):
            agent = Agent(
                task_chromosome=Chromosome(content="", type="task"),
                mate_selection_chromosome=Chromosome(content="", type="mate_selection"),
                mutation_chromosome=Chromosome(content="", type="mutation")
            )
            population.append(agent)
        
        return population
    
    def select_parents(self, population: List[Agent], num_parents: int) -> List[Agent]:
        """Select parents from the population using Pareto distribution"""
        return select_parents_pareto(population, num_parents)
    
    def mate_agents(self, parent1: Agent, parent2: Agent) -> Agent:
        """Create a new agent by mating two parents"""
        return mate_agents(parent1, parent2)
    
    def mutate_agent(self, agent: Agent) -> Agent:
        """Mutate an agent using its mutation chromosome"""
        # Use the agent's own mutation chromosome as instructions
        mutation_instructions = agent.mutation_chromosome.content
        
        # Mutate each chromosome
        task_chromosome = self.llm_port.generate_mutation(
            agent.task_chromosome, 
            mutation_instructions
        )
        
        mate_selection_chromosome = self.llm_port.generate_mutation(
            agent.mate_selection_chromosome,
            mutation_instructions
        )
        
        mutation_chromosome = self.llm_port.generate_mutation(
            agent.mutation_chromosome,
            mutation_instructions
        )
        
        # Create and return the mutated agent
        return Agent(
            task_chromosome=task_chromosome,
            mate_selection_chromosome=mate_selection_chromosome,
            mutation_chromosome=mutation_chromosome
        )
    
    def evaluate_agent(self, agent: Agent) -> float:
        """Evaluate an agent and return its reward"""
        # Get the task output from the task chromosome
        task_output = agent.task_chromosome.content
        
        # Evaluate the output
        reward = self.llm_port.evaluate_task_output(task_output)
        
        # Update the agent's reward
        agent.reward = reward
        
        # Track the reward in statistics
        self.statistics_port.track_reward(reward)
        
        # Log the evaluation
        self.logging_port.log_evaluation(agent)
        
        return reward
    
    def add_to_population(self, population: List[Agent], agent: Agent) -> List[Agent]:
        """Add an agent to the population, maintaining population constraints"""
        with self.population_lock:
            # Add the new agent
            population.append(agent)
            
            # If population exceeds limit, remove the worst agent
            if len(population) > 1000000:  # MAX_POPULATION_SIZE
                # Sort by reward (None rewards are treated as worst)
                sorted_population = sorted(
                    population,
                    key=lambda a: a.reward if a.reward is not None else float('-inf'),
                    reverse=True
                )
                # Keep only the top agents
                population = sorted_population[:1000000]  # MAX_POPULATION_SIZE
        
        return population
    
    def get_population_stats(self, population: List[Agent]) -> Dict[str, Any]:
        """Get statistics about the current population"""
        # Get overall statistics
        stats = self.statistics_port.get_stats()
        
        # Get sliding window statistics
        window_stats = self.statistics_port.get_sliding_window_stats(window_size=100)
        
        # Combine the statistics
        combined_stats = {
            **stats,
            "population_size": len(population),
            "window_stats": window_stats
        }
        
        # Log the statistics
        self.logging_port.log_population_stats(combined_stats)
        
        return combined_stats
    
    def _evolution_worker(self, population: List[Agent]) -> None:
        """Worker function for evolution thread"""
        while not self.stop_event.is_set():
            try:
                # Select parents
                parents = self.select_parents(population, 2)
                if len(parents) < 2:
                    # Not enough parents, create random agents
                    new_agent = Agent(
                        task_chromosome=Chromosome(content="", type="task"),
                        mate_selection_chromosome=Chromosome(content="", type="mate_selection"),
                        mutation_chromosome=Chromosome(content="", type="mutation")
                    )
                else:
                    # Select mate using the first parent's mate selection chromosome
                    parent1 = parents[0]
                    parent2 = self.llm_port.select_mate(parent1, [p for p in parents[1:]])
                    
                    # Create new agent through mating
                    new_agent = self.mate_agents(parent1, parent2)
                
                # Mutate the new agent
                mutated_agent = self.mutate_agent(new_agent)
                
                # Evaluate the agent
                self.evaluate_agent(mutated_agent)
                
                # Add to population
                self.add_to_population(population, mutated_agent)
                
            except Exception as e:
                print(f"Worker error: {e}")
                time.sleep(1)  # Avoid tight loop on errors
    
    def run_evolution(self, 
                     population_size: int, 
                     parallel_agents: int,
                     max_evaluations: Optional[int] = None,
                     progress_callback: Optional[callable] = None) -> List[Agent]:
        """Run the evolution process with the given parameters"""
        # Initialize log
        self.logging_port.initialize_log()
        
        # Initialize population
        population = self.initialize_population(population_size)
        
        # Log start event
        self.logging_port.log_event("Evolution Started", {
            "population_size": population_size,
            "parallel_agents": parallel_agents,
            "max_evaluations": max_evaluations or "unlimited"
        })
        
        # Start worker threads
        with ThreadPoolExecutor(max_workers=parallel_agents) as executor:
            workers = [
                executor.submit(self._evolution_worker, population)
                for _ in range(parallel_agents)
            ]
            
            try:
                # Monitor progress
                evaluation_count = 0
                last_stats_time = time.time()
                last_progress_time = time.time()
                
                while True:
                    # Check if we've reached max evaluations
                    current_count = len(self.statistics_port.rewards)
                    if max_evaluations and current_count >= max_evaluations:
                        break
                    
                    # Call progress callback if provided (not too frequently)
                    if progress_callback and time.time() - last_progress_time > 0.5:
                        progress_callback(current_count)
                        last_progress_time = time.time()
                    
                    # Display stats periodically
                    if time.time() - last_stats_time > 10:  # Every 10 seconds
                        stats = self.get_population_stats(population)
                        last_stats_time = time.time()
                    
                    time.sleep(0.1)
                    
            except KeyboardInterrupt:
                print("\nStopping evolution...")
            finally:
                # Signal workers to stop
                self.stop_event.set()
                
                # Wait for workers to finish
                for worker in workers:
                    worker.result()
        
        # Log end event
        self.logging_port.log_event("Evolution Completed", {
            "total_evaluations": len(self.statistics_port.rewards),
            "final_population_size": len(population)
        })
        
        return population

def create_application(model_name: str = "openrouter/google/gemini-2.0-flash-001",
                      log_file: str = "evolution.log",
                      use_mock: bool = False,
                      random_seed: Optional[int] = None) -> CLIAdapter:
    """Create and wire the application components"""
    # Create adapters
    if use_mock:
        llm_adapter = MockLLMAdapter(seed=random_seed)
    else:
        llm_adapter = DSPyLLMAdapter(model_name=model_name)
        
    logging_adapter = FileLoggingAdapter(log_file=log_file)
    statistics_adapter = StatisticsAdapter()
    
    # Create evolution service
    evolution_service = EvolutionService(
        llm_port=llm_adapter,
        logging_port=logging_adapter,
        statistics_port=statistics_adapter
    )
    
    # Create CLI adapter
    cli_adapter = CLIAdapter(evolution_use_case=evolution_service)
    
    return cli_adapter

def main():
    """Main entry point for the application"""
    # Parse arguments to get model and log file
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", default=os.environ.get("MODEL", "openrouter/google/gemini-2.0-flash-001"))
    parser.add_argument("--log-file", default=os.environ.get("LOG_FILE", "evolution.log"))
    parser.add_argument("--use-mock", action="store_true", help="Use mock LLM adapter for testing")
    parser.add_argument("--seed", type=int, 
                       default=int(os.environ.get("RANDOM_SEED", "0")) or None, 
                       help="Random seed for reproducibility")
    parser.add_argument("--population-size", type=int, 
                       default=int(os.environ.get("POPULATION_SIZE", "100")))
    parser.add_argument("--parallel-agents", type=int, 
                       default=int(os.environ.get("PARALLEL_AGENTS", "10")))
    parser.add_argument("--max-evaluations", type=int, 
                       default=int(os.environ.get("MAX_EVALUATIONS", "0")) or None)
    
    # Check if USE_MOCK is set in environment
    if os.environ.get("USE_MOCK") == "1":
        parser.set_defaults(use_mock=True)
    
    # Only parse known args to get these values
    args, _ = parser.parse_known_args()
    
    # Create the application
    cli = create_application(
        model_name=args.model, 
        log_file=args.log_file,
        use_mock=args.use_mock,
        random_seed=args.seed
    )
    
    # Run the application
    return cli.run()

if __name__ == "__main__":
    sys.exit(main())
