#!/usr/bin/env python3
"""
Universal Optimization CLI - Optimize any text output using script-based evaluation
"""
import sys
import os
import argparse
import time
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from llm_agent_evolution.domain.model import Agent, Chromosome
from llm_agent_evolution.domain.services import select_parents_pareto, mate_agents
from llm_agent_evolution.adapters.secondary.script_evaluator import ScriptEvaluatorAdapter
from llm_agent_evolution.adapters.secondary.logging import FileLoggingAdapter
from llm_agent_evolution.adapters.secondary.statistics import StatisticsAdapter
from llm_agent_evolution.adapters.secondary.llm import DSPyLLMAdapter
from llm_agent_evolution.adapters.secondary.mock_llm import MockLLMAdapter

class UniversalOptimizer:
    """Universal optimizer using script-based evaluation"""
    
    def __init__(self, 
                 eval_script: str,
                 population_size: int = 50,
                 parallel_agents: int = 8,
                 use_mock_llm: bool = False,
                 model_name: str = "openrouter/google/gemini-2.0-flash-001",
                 log_file: str = "universal_optimize.log",
                 random_seed: Optional[int] = None,
                 script_timeout: int = 30,
                 initial_content: str = "",
                 max_chars: int = 1000):
        """
        Initialize the universal optimizer
        
        Args:
            eval_script: Path to the evaluation script
            population_size: Initial population size
            parallel_agents: Number of agents to evaluate in parallel
            use_mock_llm: Whether to use the mock LLM adapter
            model_name: Name of the LLM model to use
            log_file: Path to the log file
            random_seed: Random seed for reproducibility
            script_timeout: Maximum execution time for the evaluation script
            initial_content: Initial content for the chromosomes
            max_chars: Maximum number of characters for chromosomes
        """
        # Set random seed if provided
        if random_seed is not None:
            import random
            import numpy as np
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # Initialize adapters
        self.script_evaluator = ScriptEvaluatorAdapter()
        self.logging_adapter = FileLoggingAdapter(log_file=log_file)
        self.statistics_adapter = StatisticsAdapter()
        
        if use_mock_llm:
            self.llm_adapter = MockLLMAdapter(seed=random_seed)
        else:
            self.llm_adapter = DSPyLLMAdapter(model_name=model_name)
        
        # Store configuration
        self.eval_script = eval_script
        self.population_size = population_size
        self.parallel_agents = parallel_agents
        self.script_timeout = script_timeout
        self.initial_content = initial_content
        self.max_chars = max_chars
        
        # Runtime state
        self.population = []
        self.stop_event = threading.Event()
        self.population_lock = threading.Lock()
    
    def initialize_population(self) -> List[Agent]:
        """Initialize a population with the given size"""
        population = []
        
        # Create initial agents with empty or provided content
        for _ in range(self.population_size):
            agent = Agent(
                task_chromosome=Chromosome(content=self.initial_content, type="task"),
                mate_selection_chromosome=Chromosome(content="", type="mate_selection"),
                mutation_chromosome=Chromosome(content="", type="mutation")
            )
            population.append(agent)
        
        return population
    
    def evaluate_agent(self, agent: Agent) -> float:
        """Evaluate an agent using the script evaluator"""
        # Get the task output from the task chromosome
        task_output = agent.task_chromosome.content
        
        try:
            # Evaluate using the script
            reward = self.script_evaluator.evaluate(
                task_output, 
                self.eval_script,
                timeout=self.script_timeout
            )
        except Exception as e:
            print(f"Evaluation error: {e}")
            reward = 0.0
        
        # Update the agent's reward
        agent.reward = reward
        
        # Track the reward in statistics
        self.statistics_adapter.track_reward(reward)
        
        # Log the evaluation
        self.logging_adapter.log_evaluation(agent)
        
        return reward
    
    def mutate_agent(self, agent: Agent) -> Agent:
        """Mutate an agent using its mutation chromosome"""
        # Use the agent's own mutation chromosome as instructions
        mutation_instructions = agent.mutation_chromosome.content
        
        # Mutate each chromosome
        task_chromosome = self.llm_adapter.generate_mutation(
            agent.task_chromosome, 
            mutation_instructions
        )
        
        mate_selection_chromosome = self.llm_adapter.generate_mutation(
            agent.mate_selection_chromosome,
            mutation_instructions
        )
        
        mutation_chromosome = self.llm_adapter.generate_mutation(
            agent.mutation_chromosome,
            mutation_instructions
        )
        
        # Create and return the mutated agent
        return Agent(
            task_chromosome=task_chromosome,
            mate_selection_chromosome=mate_selection_chromosome,
            mutation_chromosome=mutation_chromosome
        )
    
    def _evolution_worker(self) -> None:
        """Worker function for evolution thread"""
        while not self.stop_event.is_set():
            try:
                with self.population_lock:
                    if not self.population:
                        time.sleep(0.1)
                        continue
                    
                    # Select parents
                    parents = select_parents_pareto(self.population, 2)
                
                if len(parents) < 2:
                    # Not enough parents, create random agents
                    new_agent = Agent(
                        task_chromosome=Chromosome(content=self.initial_content, type="task"),
                        mate_selection_chromosome=Chromosome(content="", type="mate_selection"),
                        mutation_chromosome=Chromosome(content="", type="mutation")
                    )
                else:
                    # Select mate using the first parent's mate selection chromosome
                    parent1 = parents[0]
                    parent2 = self.llm_adapter.select_mate(parent1, [p for p in parents[1:]])
                    
                    # Create new agent through mating
                    new_agent = mate_agents(parent1, parent2)
                
                # Mutate the new agent
                mutated_agent = self.mutate_agent(new_agent)
                
                # Evaluate the agent
                self.evaluate_agent(mutated_agent)
                
                # Add to population
                with self.population_lock:
                    self.population.append(mutated_agent)
                    
                    # If population exceeds limit, remove the worst agent
                    if len(self.population) > 1000000:  # MAX_POPULATION_SIZE
                        # Sort by reward (None rewards are treated as worst)
                        sorted_population = sorted(
                            self.population,
                            key=lambda a: a.reward if a.reward is not None else float('-inf'),
                            reverse=True
                        )
                        # Keep only the top agents
                        self.population = sorted_population[:1000000]  # MAX_POPULATION_SIZE
                
            except Exception as e:
                print(f"Worker error: {e}")
                time.sleep(1)  # Avoid tight loop on errors
    
    def run(self, max_evaluations: Optional[int] = None, 
           progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Run the optimization process
        
        Args:
            max_evaluations: Maximum number of evaluations to run
            progress_callback: Callback function for progress updates
            
        Returns:
            Dictionary with optimization results
        """
        # Initialize log
        self.logging_adapter.initialize_log()
        
        # Initialize population
        self.population = self.initialize_population()
        
        # Log start event
        self.logging_adapter.log_event("Optimization Started", {
            "eval_script": self.eval_script,
            "population_size": self.population_size,
            "parallel_agents": self.parallel_agents,
            "max_evaluations": max_evaluations or "unlimited"
        })
        
        # Start worker threads
        workers = []
        for _ in range(self.parallel_agents):
            thread = threading.Thread(target=self._evolution_worker)
            thread.daemon = True
            thread.start()
            workers.append(thread)
        
        # Evaluate initial population to get starting rewards
        print("Evaluating initial population...")
        with ThreadPoolExecutor(max_workers=self.parallel_agents) as executor:
            list(executor.map(self.evaluate_agent, self.population))
        print(f"Initial population evaluated. Starting evolution...")
        
        try:
            # Monitor progress
            last_stats_time = time.time()
            last_progress_time = time.time()
            
            while True:
                # Check if we've reached max evaluations
                current_count = len(self.statistics_adapter.rewards)
                if max_evaluations and current_count >= max_evaluations:
                    break
                
                # Call progress callback if provided (not too frequently)
                if progress_callback and time.time() - last_progress_time > 0.5:
                    progress_callback(current_count, max_evaluations)
                    last_progress_time = time.time()
                
                # Display stats periodically
                if time.time() - last_stats_time > 10:  # Every 10 seconds
                    stats = self.get_stats()
                    print(f"\nPopulation: {stats['population_size']}, "
                          f"Evaluations: {stats['count']}, "
                          f"Best: {stats['best']:.2f}, "
                          f"Mean: {stats['mean']:.2f}")
                    last_stats_time = time.time()
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nStopping optimization...")
        finally:
            # Signal workers to stop
            self.stop_event.set()
            
            # Wait for workers to finish (with timeout)
            for worker in workers:
                worker.join(timeout=1.0)
        
        # Get final results
        results = self.get_results()
        
        # Log end event
        self.logging_adapter.log_event("Optimization Completed", {
            "total_evaluations": len(self.statistics_adapter.rewards),
            "final_population_size": len(self.population),
            "best_reward": results["best_agent"]["reward"] if results["best_agent"] else None
        })
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        # Get overall statistics
        stats = self.statistics_adapter.get_stats()
        
        # Get sliding window statistics
        window_stats = self.statistics_adapter.get_sliding_window_stats(window_size=100)
        
        # Combine the statistics
        with self.population_lock:
            combined_stats = {
                **stats,
                "population_size": len(self.population),
                "window_stats": window_stats,
                "cache_stats": self.script_evaluator.get_cache_stats()
            }
        
        # Log the statistics
        self.logging_adapter.log_population_stats(combined_stats)
        
        return combined_stats
    
    def get_results(self) -> Dict[str, Any]:
        """Get the optimization results"""
        with self.population_lock:
            # Sort population by reward
            sorted_population = sorted(
                self.population,
                key=lambda a: a.reward if a.reward is not None else float('-inf'),
                reverse=True
            )
            
            # Get the best agent
            best_agent = sorted_population[0] if sorted_population else None
            
            # Get top agents
            top_agents = []
            for agent in sorted_population[:10]:  # Top 10 agents
                top_agents.append({
                    "id": agent.id,
                    "reward": agent.reward,
                    "content": agent.task_chromosome.content
                })
            
            # Get statistics
            stats = self.get_stats()
            
            return {
                "best_agent": {
                    "id": best_agent.id,
                    "reward": best_agent.reward,
                    "content": best_agent.task_chromosome.content
                } if best_agent else None,
                "top_agents": top_agents,
                "stats": stats,
                "evaluations": len(self.statistics_adapter.rewards)
            }

def progress_bar(current: int, total: Optional[int] = None) -> None:
    """Display a simple progress bar"""
    if total:
        percent = min(100, int(current / total * 100))
        bar_length = 30
        filled_length = int(bar_length * current / total)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        sys.stdout.write(f"\r[{bar}] {percent}% ({current}/{total})")
    else:
        sys.stdout.write(f"\rEvaluations: {current}")
    sys.stdout.flush()

def main():
    """Main entry point for the universal optimizer CLI"""
    parser = argparse.ArgumentParser(
        description="Universal Optimization CLI - Optimize any text output using script-based evaluation"
    )
    
    parser.add_argument(
        "--eval-script", 
        required=True,
        help="Path to the evaluation script"
    )
    
    parser.add_argument(
        "--population-size", 
        type=int, 
        default=50,
        help="Initial population size (default: 50)"
    )
    
    parser.add_argument(
        "--parallel-agents", 
        type=int, 
        default=8,
        help="Number of agents to evaluate in parallel (default: 8)"
    )
    
    parser.add_argument(
        "--max-evaluations", 
        type=int, 
        default=None,
        help="Maximum number of evaluations to run (default: unlimited)"
    )
    
    parser.add_argument(
        "--use-mock-llm",
        action="store_true",
        help="Use mock LLM adapter for testing"
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="openrouter/google/gemini-2.0-flash-001",
        help="LLM model to use (default: openrouter/google/gemini-2.0-flash-001)"
    )
    
    parser.add_argument(
        "--log-file", 
        type=str, 
        default="universal_optimize.log",
        help="Log file path (default: universal_optimize.log)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--script-timeout",
        type=int,
        default=30,
        help="Maximum execution time for the evaluation script in seconds (default: 30)"
    )
    
    parser.add_argument(
        "--initial-content",
        type=str,
        default="",
        help="Initial content for the chromosomes"
    )
    
    parser.add_argument(
        "--initial-file",
        type=str,
        default=None,
        help="File containing initial content for the chromosomes"
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="File to write the best result to"
    )
    
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)"
    )
    
    parser.add_argument(
        "--max-chars",
        type=int,
        default=1000,
        help="Maximum number of characters for chromosomes (default: 1000)"
    )
    
    args = parser.parse_args()
    
    # Check if evaluation script exists
    if not os.path.exists(args.eval_script):
        print(f"Error: Evaluation script not found: {args.eval_script}")
        return 1
    
    # Get initial content from file if specified
    initial_content = args.initial_content
    if args.initial_file:
        if not os.path.exists(args.initial_file):
            print(f"Error: Initial content file not found: {args.initial_file}")
            return 1
        with open(args.initial_file, 'r') as f:
            initial_content = f.read()
    
    # Make the evaluation script executable if it's not already
    if not os.access(args.eval_script, os.X_OK):
        try:
            os.chmod(args.eval_script, os.stat(args.eval_script).st_mode | 0o111)
            print(f"Made evaluation script executable: {args.eval_script}")
        except Exception as e:
            print(f"Warning: Could not make evaluation script executable: {e}")
    
    # Create and run the optimizer
    optimizer = UniversalOptimizer(
        eval_script=args.eval_script,
        population_size=args.population_size,
        parallel_agents=args.parallel_agents,
        use_mock_llm=args.use_mock_llm,
        model_name=args.model,
        log_file=args.log_file,
        random_seed=args.seed,
        script_timeout=args.script_timeout,
        initial_content=initial_content,
        max_chars=args.max_chars
    )
    
    print(f"Starting optimization with {args.population_size} agents and {args.parallel_agents} parallel workers")
    print(f"Evaluation script: {args.eval_script}")
    print(f"Using {'mock' if args.use_mock_llm else 'real'} LLM")
    print(f"Press Ctrl+C to stop\n")
    
    # Run the optimization
    results = optimizer.run(
        max_evaluations=args.max_evaluations,
        progress_callback=progress_bar
    )
    
    # Print results
    print("\n\nOptimization completed!")
    print(f"Total evaluations: {results['evaluations']}")
    
    if results['best_agent']:
        print(f"\nBest agent (ID: {results['best_agent']['id']})")
        print(f"Reward: {results['best_agent']['reward']}")
        print("\nContent:")
        print("=" * 40)
        print(results['best_agent']['content'])
        print("=" * 40)
        
        # Write to output file if specified
        if args.output_file:
            if args.output_format == "text":
                with open(args.output_file, 'w') as f:
                    f.write(results['best_agent']['content'])
            else:  # json
                with open(args.output_file, 'w') as f:
                    json.dump(results, f, indent=2)
            print(f"\nResults written to {args.output_file}")
    else:
        print("\nNo valid results found")
    
    # Print statistics
    print("\nStatistics:")
    print(f"Mean reward: {results['stats']['mean']:.2f}")
    print(f"Median reward: {results['stats']['median']:.2f}")
    print(f"Standard deviation: {results['stats']['std_dev']:.2f}")
    
    # Print cache statistics
    cache_stats = results['stats']['cache_stats']
    print(f"\nCache statistics:")
    print(f"Size: {cache_stats['size']}/{cache_stats['max_size']}")
    print(f"Hit ratio: {cache_stats['hit_ratio']:.2f}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
