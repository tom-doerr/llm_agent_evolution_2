#!/usr/bin/env python3
"""
Universal Optimizer module for the LLM Agent Evolution package
"""
import sys
import os
import time
import json
import threading
import signal
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional

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
                 max_chars: int = 1000,
                 verbose: bool = False):
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
            verbose: Whether to enable verbose output
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
        self.verbose = verbose
        
        # Runtime state
        self.population = []
        self.stop_event = threading.Event()
        self.population_lock = threading.Lock()
    
    def initialize_population(self) -> List[Agent]:
        """Initialize a population with the given size"""
        population = []
        
        # Create initial agents with meaningful chromosomes
        for _ in range(self.population_size):
            # Initial task chromosome with empty content
            task_content = self.initial_content if self.initial_content else ""
            
            # Initial mate selection chromosome with instructions
            mate_selection_content = """
            Select the mate with the highest reward.
            Choose mates that have shown improvement.
            Consider diversity in the population.
            Look for complementary strengths.
            """
            
            # Initial mutation chromosome with instructions
            mutation_content = """
            Improve the content to maximize the evaluation score.
            Try different approaches and patterns.
            Keep the content concise and focused.
            Experiment with different structures.
            """
            
            agent = Agent(
                task_chromosome=Chromosome(content=task_content, type="task"),
                mate_selection_chromosome=Chromosome(content=mate_selection_content, type="mate_selection"),
                mutation_chromosome=Chromosome(content=mutation_content, type="mutation")
            )
            population.append(agent)
        
        return population
    
    def evaluate_agent(self, agent: Agent) -> float:
        """Evaluate an agent using the script evaluator"""
        # Get the task output from the task chromosome
        task_output = agent.task_chromosome.content
        
        # Ensure task_output is a string
        if not isinstance(task_output, str):
            print(f"Warning: task_output is not a string, it's a {type(task_output)}. Converting to string.")
            try:
                if isinstance(task_output, list):
                    task_output = " ".join(str(item) for item in task_output)
                else:
                    task_output = str(task_output)
            except Exception as e:
                print(f"Error converting task_output to string: {e}")
                task_output = ""
        
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
        
        try:
            # Ensure mutation_instructions is a string
            if not isinstance(mutation_instructions, str):
                mutation_instructions = str(mutation_instructions)
            
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
        except Exception as e:
            print(f"Mutation error: {e}")
            # Return a copy of the original agent if mutation fails
            return Agent(
                task_chromosome=Chromosome(content=agent.task_chromosome.content, type="task"),
                mate_selection_chromosome=Chromosome(content=agent.mate_selection_chromosome.content, type="mate_selection"),
                mutation_chromosome=Chromosome(content=agent.mutation_chromosome.content, type="mutation")
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
                    try:
                        parent2 = self.llm_adapter.select_mate(parent1, [p for p in parents[1:]])
                        # Create new agent through mating
                        new_agent = mate_agents(parent1, parent2)
                    except Exception as e:
                        print(f"Mate selection error: {e}")
                        # If mate selection fails, just use the second parent
                        parent2 = parents[1] if len(parents) > 1 else parent1
                        new_agent = mate_agents(parent1, parent2)
                
                # Verbose output for parent selection and mating
                if self.verbose:
                    print("\n" + "=" * 60)
                    print(f"EVOLUTION STEP")
                    print("=" * 60)
                    print("\n1. PARENT SELECTION")
                    print(f"Parent 1 (ID: {parent1.id}):")
                    print(f"Reward: {parent1.reward}")
                    print(f"Task Chromosome:")
                    print(f"{parent1.task_chromosome.content}")
                    print(f"\nMate Selection Chromosome:")
                    print(f"{parent1.mate_selection_chromosome.content}")
                    print(f"\nMutation Chromosome:")
                    print(f"{parent1.mutation_chromosome.content}")
        
                    print(f"\nParent 2 (ID: {parent2.id}):")
                    print(f"Reward: {parent2.reward}")
                    print(f"Task Chromosome:")
                    print(f"{parent2.task_chromosome.content}")
                    print(f"\nMate Selection Chromosome:")
                    print(f"{parent2.mate_selection_chromosome.content}")
                    print(f"\nMutation Chromosome:")
                    print(f"{parent2.mutation_chromosome.content}")
                    
                    print("\n2. MATING")
                    print(f"New agent after mating (ID: {new_agent.id}):")
                    print(f"Task Chromosome:")
                    print(f"{new_agent.task_chromosome.content}")
                    print(f"\nMate Selection Chromosome:")
                    print(f"{new_agent.mate_selection_chromosome.content}")
                    print(f"\nMutation Chromosome:")
                    print(f"{new_agent.mutation_chromosome.content}")
                
                # Mutate the new agent
                if self.verbose:
                    print("\n3. MUTATION")
                    print(f"Mutation instructions: '{new_agent.mutation_chromosome.content[:50]}{'...' if len(new_agent.mutation_chromosome.content) > 50 else ''}'")
                    print(f"Before mutation:")
                    print(f"Task Chromosome:")
                    print(f"{new_agent.task_chromosome.content}")
                    print(f"\nMate Selection Chromosome:")
                    print(f"{new_agent.mate_selection_chromosome.content}")
                    print(f"\nMutation Chromosome:")
                    print(f"{new_agent.mutation_chromosome.content}")
                
                mutated_agent = self.mutate_agent(new_agent)
                
                if self.verbose:
                    print(f"After mutation:")
                    print(f"Task Chromosome:")
                    print(f"{mutated_agent.task_chromosome.content}")
                    print(f"\nMate Selection Chromosome:")
                    print(f"{mutated_agent.mate_selection_chromosome.content}")
                    print(f"\nMutation Chromosome:")
                    print(f"{mutated_agent.mutation_chromosome.content}")
                
                # Evaluate the agent
                if self.verbose:
                    print("\n4. EVALUATION")
                
                self.evaluate_agent(mutated_agent)
                
                if self.verbose:
                    print(f"Reward: {mutated_agent.reward}")
                
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
                        
                    # Limit verbose output to first 5 agents after initial population
                    if self.verbose and len(self.population) > self.population_size + 5:
                        self.verbose = False
                        print("\n" + "=" * 40)
                        print("Limiting verbose output to first 5 agents")
                        print("=" * 40)
                
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
        try:
            self.logging_adapter.initialize_log()
        except Exception as e:
            print(f"Warning: Could not initialize log: {e}")
        
        # Initialize population
        self.population = self.initialize_population()
        
        # Log start event
        try:
            self.logging_adapter.log_event("Optimization Started", {
                "eval_script": self.eval_script,
                "population_size": self.population_size,
                "parallel_agents": self.parallel_agents,
                "max_evaluations": max_evaluations or "unlimited"
            })
        except Exception as e:
            print(f"Warning: Could not log start event: {e}")
        
        # Add signal handler for graceful shutdown
        original_sigint_handler = signal.getsignal(signal.SIGINT)
        def sigint_handler(sig, frame):
            print("\nGracefully stopping optimization...")
            self.stop_event.set()
            # Restore original handler to allow forced exit with another Ctrl+C
            signal.signal(signal.SIGINT, original_sigint_handler)
        
        signal.signal(signal.SIGINT, sigint_handler)
        
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
            
            while not self.stop_event.is_set():
                # Check if we've reached max evaluations
                current_count = len(self.statistics_adapter.rewards)
                if max_evaluations and current_count >= max_evaluations:
                    print("\nReached maximum evaluations")
                    break
                
                # Call progress callback if provided (not too frequently)
                if progress_callback and time.time() - last_progress_time > 0.5:
                    try:
                        progress_callback(current_count, max_evaluations)
                    except Exception as e:
                        print(f"Error in progress callback: {e}")
                    last_progress_time = time.time()
                
                # Display stats periodically
                if time.time() - last_stats_time > 10:  # Every 10 seconds
                    try:
                        stats = self.get_stats()
                        print(f"\nPopulation: {stats['population_size']}, "
                              f"Evaluations: {stats['count']}, "
                              f"Best: {stats['best']:.2f}, "
                              f"Mean: {stats['mean']:.2f}")
                    except Exception as e:
                        print(f"Error displaying stats: {e}")
                    last_stats_time = time.time()
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nStopping optimization...")
        except Exception as e:
            print(f"Error in optimization loop: {e}")
        finally:
            # Signal workers to stop
            self.stop_event.set()
            
            # Wait for workers to finish (with timeout)
            for worker in workers:
                try:
                    worker.join(timeout=1.0)
                except Exception:
                    pass
        
        # Get final results
        try:
            results = self.get_results()
        except Exception as e:
            print(f"Error getting results: {e}")
            results = {
                "best_agent": None,
                "top_agents": [],
                "stats": {"mean": 0, "median": 0, "std_dev": 0, "count": 0, "best": 0},
                "evaluations": 0
            }
        
        # Log end event
        try:
            self.logging_adapter.log_event("Optimization Completed", {
                "total_evaluations": len(self.statistics_adapter.rewards),
                "final_population_size": len(self.population),
                "best_reward": results["best_agent"]["reward"] if results["best_agent"] else None
            })
        except Exception as e:
            print(f"Warning: Could not log end event: {e}")
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        try:
            # Get overall statistics
            stats = self.statistics_adapter.get_stats()
            
            # Get sliding window statistics
            window_stats = self.statistics_adapter.get_sliding_window_stats(window_size=100)
            
            # Get cache stats
            try:
                cache_stats = self.script_evaluator.get_cache_stats()
            except Exception:
                cache_stats = {"size": 0, "max_size": 0, "hits": 0, "misses": 0, "hit_ratio": 0}
            
            # Get population size without holding the lock for too long
            population_size = 0
            try:
                with self.population_lock:
                    population_size = len(self.population)
            except Exception:
                pass
            
            # Combine the statistics
            combined_stats = {
                **stats,
                "population_size": population_size,
                "window_stats": window_stats,
                "cache_stats": cache_stats
            }
            
            # Log the statistics
            try:
                self.logging_adapter.log_population_stats(combined_stats)
            except Exception as e:
                print(f"Error logging stats: {e}")
            
            return combined_stats
        except Exception as e:
            print(f"Error calculating stats: {e}")
            return {
                "mean": 0,
                "median": 0,
                "std_dev": 0,
                "count": 0,
                "best": 0,
                "population_size": 0,
                "window_stats": {},
                "cache_stats": {"size": 0, "max_size": 0, "hits": 0, "misses": 0, "hit_ratio": 0}
            }
    
    def get_results(self) -> Dict[str, Any]:
        """Get the optimization results"""
        try:
            with self.population_lock:
                if not self.population:
                    return {
                        "best_agent": None,
                        "top_agents": [],
                        "stats": {"mean": 0, "median": 0, "std_dev": 0, "count": 0, "best": 0},
                        "evaluations": len(self.statistics_adapter.rewards)
                    }
                
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
            
            # Get statistics without holding the lock
            try:
                stats = self.statistics_adapter.get_stats()
            except Exception as e:
                print(f"Error getting statistics: {e}")
                stats = {"mean": 0, "median": 0, "std_dev": 0, "count": 0, "best": 0}
            
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
        except Exception as e:
            print(f"Error getting results: {e}")
            return {
                "best_agent": None,
                "top_agents": [],
                "stats": {"mean": 0, "median": 0, "std_dev": 0, "count": 0, "best": 0},
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

def run_optimizer(
    eval_script: str,
    population_size: int = 50,
    parallel_agents: int = 8,
    max_evaluations: Optional[int] = None,
    use_mock_llm: bool = False,
    model_name: str = "openrouter/google/gemini-2.0-flash-001",
    log_file: str = "universal_optimize.log",
    random_seed: Optional[int] = None,
    script_timeout: int = 30,
    initial_content: str = "",
    output_file: Optional[str] = None,
    output_format: str = "text",
    max_chars: int = 1000,
    verbose: bool = False,
    eval_command: Optional[str] = None
) -> int:
    """Run the universal optimizer with the given parameters"""
    # Check if evaluation script exists
    if not os.path.exists(eval_script):
        print(f"Error: Evaluation script not found: {eval_script}")
        return 1
    
    # Make the evaluation script executable if it's not already
    if not os.access(eval_script, os.X_OK):
        try:
            os.chmod(eval_script, os.stat(eval_script).st_mode | 0o111)
            print(f"Made evaluation script executable: {eval_script}")
        except Exception as e:
            print(f"Warning: Could not make evaluation script executable: {e}")
    
    # Create and run the optimizer
    optimizer = UniversalOptimizer(
        eval_script=eval_script,
        population_size=population_size,
        parallel_agents=parallel_agents,
        use_mock_llm=use_mock_llm,
        model_name=model_name,
        log_file=log_file,
        random_seed=random_seed,
        script_timeout=script_timeout,
        initial_content=initial_content,
        max_chars=max_chars,
        verbose=verbose
    )
    
    # Set the eval command if provided
    if eval_command and optimizer.llm_adapter:
        optimizer.llm_adapter.eval_command = eval_command
    
    print(f"Starting optimization with {population_size} agents and {parallel_agents} parallel workers")
    print(f"Evaluation script: {eval_script}")
    print(f"Using {'mock' if use_mock_llm else 'real'} LLM")
    print(f"Press Ctrl+C to stop\n")
    
    # Run the optimization
    results = optimizer.run(
        max_evaluations=max_evaluations,
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
        if output_file:
            if output_format == "text":
                with open(output_file, 'w') as f:
                    f.write(results['best_agent']['content'])
            else:  # json
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
            print(f"\nResults written to {output_file}")
    else:
        print("\nNo valid results found")
    
    # Print statistics
    print("\nStatistics:")
    print(f"Mean reward: {results['stats']['mean']:.2f}")
    print(f"Median reward: {results['stats']['median']:.2f}")
    print(f"Standard deviation: {results['stats']['std_dev']:.2f}")
    
    # Print cache statistics
    if 'cache_stats' in results['stats']:
        cache_stats = results['stats']['cache_stats']
        print(f"\nCache statistics:")
        print(f"Size: {cache_stats['size']}/{cache_stats['max_size']}")
        print(f"Hit ratio: {cache_stats['hit_ratio']:.2f}")
    
    return 0

if __name__ == "__main__":
    # If run directly, use the CLI arguments
    import argparse
    
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
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose mode with detailed output of each evolution step"
    )
    
    args = parser.parse_args()
    
    # Get initial content from file if specified
    initial_content = args.initial_content
    if args.initial_file:
        if not os.path.exists(args.initial_file):
            print(f"Error: Initial content file not found: {args.initial_file}")
            sys.exit(1)
        with open(args.initial_file, 'r') as f:
            initial_content = f.read()
    
    # Run the optimizer
    sys.exit(run_optimizer(
        eval_script=args.eval_script,
        population_size=args.population_size,
        parallel_agents=args.parallel_agents,
        max_evaluations=args.max_evaluations,
        use_mock_llm=args.use_mock_llm,
        model_name=args.model,
        log_file=args.log_file,
        random_seed=args.seed,
        script_timeout=args.script_timeout,
        initial_content=initial_content,
        output_file=args.output_file,
        output_format=args.output_format,
        max_chars=args.max_chars,
        verbose=args.verbose
    ))
