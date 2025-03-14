"""
Core implementation of the Universal Optimizer
"""
import os
import time
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
            verbose: Whether to enable verbose output with detailed information for five fixed agents
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
        
        # Verbose output control - track exactly 5 fixed agents as specified in spec
        self.max_verbose_agents = 5  # Maximum number of agents to show verbose output for
        self.verbose_agent_ids = set()  # Set of agent IDs to show verbose output for
        self.verbose_agent_count = 0  # Counter for agents with verbose output
    
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
                # Get parents for mating
                parents = self._select_parents_for_mating()
                if not parents:
                    time.sleep(0.1)
                    continue
                
                # Create new agent through mating or random initialization
                new_agent = self._create_new_agent(parents)
                
                # Handle verbose output if enabled - check if we should track this agent
                show_verbose = self._should_show_verbose_output(new_agent.id)
                
                # Show verbose output for parent selection and mating if enabled
                if show_verbose:
                    self._print_verbose_mating_info(parents, new_agent)
                
                # Mutate the new agent
                if show_verbose:
                    self._print_verbose_mutation_start(new_agent)
                
                mutated_agent = self.mutate_agent(new_agent)
                
                if show_verbose:
                    self._print_verbose_mutation_result(mutated_agent)
                    
                    # Add the mutated agent ID to verbose tracking
                    self.verbose_agent_ids.add(mutated_agent.id)
                
                # Evaluate the agent
                if show_verbose:
                    print("\n4. EVALUATION")
                
                self.evaluate_agent(mutated_agent)
                
                if show_verbose:
                    print(f"Reward: {mutated_agent.reward}")
                
                # Add to population
                with self.population_lock:
                    self.population.append(mutated_agent)
                    
                    # If population exceeds limit, remove the worst agent
                    from llm_agent_evolution.domain.model import MAX_POPULATION_SIZE
                    if len(self.population) > MAX_POPULATION_SIZE:
                        # Sort by reward (None rewards are treated as worst)
                        sorted_population = sorted(
                            self.population,
                            key=lambda a: a.reward if a.reward is not None else float('-inf'),
                            reverse=True
                        )
                        # Keep only the top agents
                        self.population = sorted_population[:MAX_POPULATION_SIZE]
                        
                    # Limit verbose output to first N agents
                    if self.verbose and self.verbose_agent_count >= self.max_verbose_agents and show_verbose:
                        print("\n" + "=" * 40)
                        print(f"Limiting verbose output to first {self.max_verbose_agents} agents")
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
            progress_callback: Callback function for progress updates (current_count, max_count)
            
        Returns:
            Dictionary with optimization results
        """
        # Initialize log
        try:
            self.logging_adapter.initialize_log()
        except Exception as e:
            print(f"Warning: Could not initialize log: {e}")
            import traceback
            print(traceback.format_exc())
        
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
                        # Try with both arguments first
                        if max_evaluations:
                            progress_callback(current_count, max_evaluations)
                        else:
                            # If max_evaluations is None, try with just current_count
                            progress_callback(current_count)
                    except TypeError:
                        # If that fails, try with just the current count
                        try:
                            progress_callback(current_count)
                        except Exception as e:
                            print(f"Error in progress callback: {e}")
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
    
    def _select_parents_for_mating(self) -> List[Agent]:
        """Select parents for mating from the population"""
        with self.population_lock:
            if not self.population:
                return []
            
            # Select parents
            return select_parents_pareto(self.population, 2)
    
    def _create_new_agent(self, parents: List[Agent]) -> Agent:
        """Create a new agent either through mating or random initialization"""
        if len(parents) < 2:
            # Not enough parents, create random agent
            return Agent(
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
                return mate_agents(parent1, parent2)
            except Exception as e:
                print(f"Mate selection error: {e}")
                # If mate selection fails, just use the second parent
                parent2 = parents[1] if len(parents) > 1 else parent1
                return mate_agents(parent1, parent2)
    
    def _should_show_verbose_output(self, agent_id: str) -> bool:
        """Determine if verbose output should be shown for the agent"""
        if not self.verbose:
            return False
            
        with self.population_lock:
            # If we're tracking this agent already, show output
            if agent_id in self.verbose_agent_ids:
                return True
                
            # If we haven't reached max agents to track, add this one
            if len(self.verbose_agent_ids) < self.max_verbose_agents:
                self.verbose_agent_ids.add(agent_id)
                self.verbose_agent_count += 1
                print(f"\nNow tracking agent {agent_id} for verbose output (agent {self.verbose_agent_count} of {self.max_verbose_agents})")
                return True
                
            return False
    
    def _print_verbose_mating_info(self, parents: List[Agent], new_agent: Agent) -> None:
        """Print verbose information about the mating process"""
        parent1 = parents[0]
        parent2 = parents[1] if len(parents) > 1 else parent1
        
        print("\n" + "=" * 60)
        print(f"EVOLUTION STEP - DETAILED INFORMATION")
        print("=" * 60)
        print("\n1. PARENT SELECTION")
        print(f"Parent 1 (ID: {parent1.id}):")
        print(f"Reward: {parent1.reward}")
        print(f"Task Chromosome ({len(parent1.task_chromosome.content)} chars):")
        print(f"{parent1.task_chromosome.content}")
        print(f"\nMate Selection Chromosome ({len(parent1.mate_selection_chromosome.content)} chars):")
        print(f"{parent1.mate_selection_chromosome.content}")
        print(f"\nMutation Chromosome ({len(parent1.mutation_chromosome.content)} chars):")
        print(f"{parent1.mutation_chromosome.content}")

        print(f"\nParent 2 (ID: {parent2.id}):")
        print(f"Reward: {parent2.reward}")
        print(f"Task Chromosome ({len(parent2.task_chromosome.content)} chars):")
        print(f"{parent2.task_chromosome.content}")
        print(f"\nMate Selection Chromosome ({len(parent2.mate_selection_chromosome.content)} chars):")
        print(f"{parent2.mate_selection_chromosome.content}")
        print(f"\nMutation Chromosome ({len(parent2.mutation_chromosome.content)} chars):")
        print(f"{parent2.mutation_chromosome.content}")
        
        print("\n2. MATING")
        print("Using chromosome combination with hotspot switching")
        print(f"New agent after mating (ID: {new_agent.id}):")
        print(f"Task Chromosome ({len(new_agent.task_chromosome.content)} chars):")
        print(f"{new_agent.task_chromosome.content}")
        print(f"\nMate Selection Chromosome ({len(new_agent.mate_selection_chromosome.content)} chars):")
        print(f"{new_agent.mate_selection_chromosome.content}")
        print(f"\nMutation Chromosome ({len(new_agent.mutation_chromosome.content)} chars):")
        print(f"{new_agent.mutation_chromosome.content}")
    
    def _print_verbose_mutation_start(self, agent: Agent) -> None:
        """Print verbose information before mutation"""
        print("\n3. MUTATION")
        print(f"Mutation instructions: '{agent.mutation_chromosome.content[:50]}{'...' if len(agent.mutation_chromosome.content) > 50 else ''}'")
        print(f"Before mutation:")
        print(f"Task Chromosome:")
        print(f"{agent.task_chromosome.content}")
        print(f"\nMate Selection Chromosome:")
        print(f"{agent.mate_selection_chromosome.content}")
        print(f"\nMutation Chromosome:")
        print(f"{agent.mutation_chromosome.content}")
    
    def _print_verbose_mutation_result(self, agent: Agent) -> None:
        """Print verbose information after mutation"""
        print(f"After mutation:")
        print(f"Task Chromosome:")
        print(f"{agent.task_chromosome.content}")
        print(f"\nMate Selection Chromosome:")
        print(f"{agent.mate_selection_chromosome.content}")
        print(f"\nMutation Chromosome:")
        print(f"{agent.mutation_chromosome.content}")
    
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
