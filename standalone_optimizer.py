#!/usr/bin/env python3
"""
Standalone Universal Optimizer - A simplified version that can be run directly
"""
import os
import sys
import time
import random
import argparse
import subprocess
import threading
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional
import uuid
import signal

# Constants
MAX_CHARS = 1000  # Maximum characters for chromosomes
CHROMOSOME_SWITCH_PROBABILITY = 0.3  # Probability of switching chromosomes at hotspots
HOTSPOT_CHARS = ".,;:!?()[]{}'\"\n "  # Punctuation and spaces as hotspots

class Chromosome:
    """Represents a single chromosome with content and type"""
    def __init__(self, content: str, type_: str):
        self.content = content[:MAX_CHARS]  # Ensure chromosome doesn't exceed max length
        self.type = type_
        
class Agent:
    """Represents an agent with three chromosomes and a reward score"""
    def __init__(self, task_chromosome, mate_selection_chromosome, mutation_chromosome, id_=None, reward=None):
        self.task_chromosome = task_chromosome
        self.mate_selection_chromosome = mate_selection_chromosome
        self.mutation_chromosome = mutation_chromosome
        self.id = id_ if id_ else str(uuid.uuid4())
        self.reward = reward

class ScriptEvaluator:
    """Evaluates agent outputs using external scripts"""
    def __init__(self, cache_size=1000):
        self.cache = {}  # {hash: reward}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
        
    def _hash_input(self, output: str, script_path: str) -> str:
        """Generate a simple hash for the output and script path"""
        return f"{hash(script_path)}_{hash(output)}"
    
    def evaluate(self, output: str, script_path: str, timeout: int = 30) -> float:
        """Evaluate the output using the specified script"""
        # Check cache first
        input_hash = self._hash_input(output, script_path)
        if input_hash in self.cache:
            self.cache_hits += 1
            return self.cache[input_hash]
        
        self.cache_misses += 1
        
        # Ensure script exists and is executable
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Evaluation script not found: {script_path}")
        
        if not os.access(script_path, os.X_OK):
            # Make sure the script is executable
            os.chmod(script_path, 0o755)
        
        try:
            # Run the script with the output as stdin
            process = subprocess.Popen(
                [script_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Send the output to the script
            stdout, stderr = process.communicate(input=output, timeout=timeout)
            
            # Check for errors
            if process.returncode != 0:
                raise RuntimeError(f"Evaluation script failed with error: {stderr}")
            
            # Parse the reward from the last line of output
            lines = stdout.strip().split('\n')
            if not lines:
                raise ValueError("Evaluation script produced no output")
            
            try:
                reward = float(lines[-1].strip())
            except ValueError:
                raise ValueError(f"Evaluation script did not return a valid numerical reward: {lines[-1]}")
            
            # Update cache
            if len(self.cache) >= self.cache_size:
                # Remove a random entry to keep the cache size in check
                self.cache.pop(next(iter(self.cache)))
            self.cache[input_hash] = reward
            
            return reward
            
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Evaluation script timed out after {timeout} seconds")

class MockLLM:
    """Mock LLM for testing purposes"""
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
    
    def generate_mutation(self, chromosome, mutation_instructions):
        """Generate a mock mutation for a chromosome"""
        if chromosome.type == "task":
            options = [
                "Hello world",
                "Testing the mutation",
                "This is a sample output",
                "Random text for evaluation",
                "The quick brown fox jumps over the lazy dog"
            ]
            content = random.choice(options)
        else:
            options = [
                "Select the candidate with the highest reward",
                "Choose a mate with diverse characteristics",
                "Prefer candidates with shorter chromosomes",
                "Try to improve the content quality",
                "Keep the chromosome short and focused"
            ]
            content = random.choice(options)
        
        return Chromosome(content=content, type_=chromosome.type)
    
    def select_mate(self, agent, candidates):
        """Select a mate from candidates randomly"""
        if not candidates:
            return None
        return random.choice(candidates)

def select_parents_pareto(population, num_parents):
    """Select parents using Pareto distribution weighting by fitness^2"""
    if not population or num_parents <= 0:
        return []
    
    # Filter out agents without rewards
    valid_agents = [agent for agent in population if agent.reward is not None]
    if not valid_agents:
        return random.sample(population, min(num_parents, len(population)))
    
    # Get reward statistics
    rewards = [agent.reward for agent in valid_agents]
    min_reward = min(rewards)
    max_reward = max(rewards)
    reward_range = max(1.0, max_reward - min_reward)  # Avoid division by zero
    
    # Normalize rewards to [0,1] range and then square them for Pareto distribution
    # Add a small epsilon to ensure even worst agents have some chance
    epsilon = 0.01
    adjusted_rewards = [((agent.reward - min_reward) / reward_range + epsilon) ** 2 
                        for agent in valid_agents]
    
    # Normalize weights
    total_weight = sum(adjusted_rewards)
    weights = [w / total_weight for w in adjusted_rewards]
    
    # Weighted sampling without replacement
    selected_indices = []
    for _ in range(min(num_parents, len(valid_agents))):
        if not weights:
            break
        # Select an index based on weights
        index = random.choices(range(len(weights)), weights=weights, k=1)[0]
        selected_indices.append(index)
        # Remove the selected index from consideration
        weights.pop(index)
        valid_agents.pop(index)
    
    return [valid_agents[i] for i in selected_indices]

def combine_chromosomes(parent1, parent2):
    """Combine two chromosomes by switching at hotspots"""
    if not parent1.content and not parent2.content:
        return Chromosome(content="", type_=parent1.type)
    
    if not parent1.content:
        return Chromosome(content=parent2.content, type_=parent1.type)
    
    if not parent2.content:
        return Chromosome(content=parent1.content, type_=parent1.type)
    
    # Randomly select primary parent
    if random.random() < 0.5:
        primary_parent, secondary_parent = parent1, parent2
    else:
        primary_parent, secondary_parent = parent2, parent1
    
    # Find all hotspots in primary parent's content
    hotspots = [i for i, char in enumerate(primary_parent.content) if char in HOTSPOT_CHARS]
    if not hotspots:
        # If no hotspots, add some arbitrary points
        content_len = len(primary_parent.content)
        hotspots = [i for i in range(0, content_len, max(1, content_len // 5))]
    
    # Start with primary parent's content
    result = list(primary_parent.content)
    using_primary = True
    
    # Sort hotspots and select a subset for switching
    hotspots.sort()
    if len(hotspots) > 0:
        switch_points = sorted(random.sample(hotspots, min(3, len(hotspots))))
        
        # Perform the switches
        for point in switch_points:
            if random.random() < CHROMOSOME_SWITCH_PROBABILITY:
                using_primary = not using_primary
                
                if not using_primary and point < len(secondary_parent.content):
                    # Take content from secondary parent from this point
                    secondary_content = list(secondary_parent.content[point:])
                    
                    # Adjust result length
                    if point + len(secondary_content) > len(result):
                        result = result[:point] + secondary_content
                    else:
                        result[point:point+len(secondary_content)] = secondary_content
    
    return Chromosome(content=''.join(result), type_=parent1.type)

def mate_agents(parent1, parent2):
    """Create a new agent by combining chromosomes from two parents"""
    # Combine each chromosome type
    task_chromosome = combine_chromosomes(
        parent1.task_chromosome, 
        parent2.task_chromosome
    )
    
    mate_selection_chromosome = combine_chromosomes(
        parent1.mate_selection_chromosome,
        parent2.mate_selection_chromosome
    )
    
    mutation_chromosome = combine_chromosomes(
        parent1.mutation_chromosome,
        parent2.mutation_chromosome
    )
    
    # Create and return new agent
    return Agent(
        task_chromosome=task_chromosome,
        mate_selection_chromosome=mate_selection_chromosome,
        mutation_chromosome=mutation_chromosome
    )

class UniversalOptimizer:
    """Universal optimizer using script-based evaluation"""
    
    def __init__(self, 
                 eval_script,
                 population_size=50,
                 parallel_agents=8,
                 use_mock_llm=False,
                 script_timeout=30,
                 initial_content="",
                 random_seed=None,
                 verbose=False):
        """Initialize the universal optimizer"""
        # Set random seed if provided
        if random_seed is not None:
            random.seed(random_seed)
        
        # Initialize components
        self.script_evaluator = ScriptEvaluator()
        self.llm = MockLLM(seed=random_seed) if use_mock_llm else None
        
        # Store configuration
        self.eval_script = eval_script
        self.population_size = population_size
        self.parallel_agents = parallel_agents
        self.script_timeout = script_timeout
        self.initial_content = initial_content
        self.verbose = verbose
        
        # Runtime state
        self.population = []
        self.stop_event = threading.Event()
        self.population_lock = threading.Lock()
        self.rewards = []
    
    def initialize_population(self):
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
            """
            
            # Initial mutation chromosome with instructions
            mutation_content = """
            Improve the content to maximize the evaluation score.
            Try different approaches and patterns.
            Keep the content concise and focused.
            """
            
            agent = Agent(
                task_chromosome=Chromosome(content=task_content, type_="task"),
                mate_selection_chromosome=Chromosome(content=mate_selection_content, type_="mate_selection"),
                mutation_chromosome=Chromosome(content=mutation_content, type_="mutation")
            )
            population.append(agent)
        
        return population
    
    def evaluate_agent(self, agent):
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
        
        # Track the reward
        with self.population_lock:
            self.rewards.append(reward)
        
        return reward
    
    def mutate_agent(self, agent):
        """Mutate an agent using its mutation chromosome"""
        # Use the agent's own mutation chromosome as instructions
        mutation_instructions = agent.mutation_chromosome.content
        
        try:
            # Mutate each chromosome
            task_chromosome = self.llm.generate_mutation(
                agent.task_chromosome, 
                mutation_instructions
            )
            
            mate_selection_chromosome = self.llm.generate_mutation(
                agent.mate_selection_chromosome,
                mutation_instructions
            )
            
            mutation_chromosome = self.llm.generate_mutation(
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
                task_chromosome=Chromosome(content=agent.task_chromosome.content, type_="task"),
                mate_selection_chromosome=Chromosome(content=agent.mate_selection_chromosome.content, type_="mate_selection"),
                mutation_chromosome=Chromosome(content=agent.mutation_chromosome.content, type_="mutation")
            )
    
    def _evolution_worker(self):
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
                        task_chromosome=Chromosome(content=self.initial_content, type_="task"),
                        mate_selection_chromosome=Chromosome(content="", type_="mate_selection"),
                        mutation_chromosome=Chromosome(content="", type_="mutation")
                    )
                else:
                    # Select mate using the first parent's mate selection chromosome
                    parent1 = parents[0]
                    try:
                        parent2 = self.llm.select_mate(parent1, [p for p in parents[1:]])
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
                    
                    print(f"\nParent 2 (ID: {parent2.id}):")
                    print(f"Reward: {parent2.reward}")
                    print(f"Task Chromosome:")
                    print(f"{parent2.task_chromosome.content}")
                    
                    print("\n2. MATING")
                    print(f"New agent after mating (ID: {new_agent.id}):")
                    print(f"Task Chromosome:")
                    print(f"{new_agent.task_chromosome.content}")
                
                # Mutate the new agent
                if self.verbose:
                    print("\n3. MUTATION")
                    print(f"Before mutation:")
                    print(f"Task Chromosome:")
                    print(f"{new_agent.task_chromosome.content}")
                
                mutated_agent = self.mutate_agent(new_agent)
                
                if self.verbose:
                    print(f"After mutation:")
                    print(f"Task Chromosome:")
                    print(f"{mutated_agent.task_chromosome.content}")
                
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
    
    def run(self, max_evaluations=None):
        """Run the optimization process"""
        # Initialize population
        self.population = self.initialize_population()
        
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
            
            while not self.stop_event.is_set():
                # Check if we've reached max evaluations
                current_count = len(self.rewards)
                if max_evaluations and current_count >= max_evaluations:
                    print("\nReached maximum evaluations")
                    break
                
                # Display stats periodically
                if time.time() - last_stats_time > 10:  # Every 10 seconds
                    try:
                        # Get population size without holding the lock for too long
                        population_size = 0
                        with self.population_lock:
                            population_size = len(self.population)
                            
                        # Get reward statistics
                        rewards = self.rewards
                        if rewards:
                            mean_reward = sum(rewards) / len(rewards)
                            best_reward = max(rewards)
                            print(f"\nPopulation: {population_size}, "
                                f"Evaluations: {len(rewards)}, "
                                f"Best: {best_reward:.2f}, "
                                f"Mean: {mean_reward:.2f}")
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
            # Sort population by reward
            with self.population_lock:
                if not self.population:
                    return None
                
                # Sort population by reward
                sorted_population = sorted(
                    self.population,
                    key=lambda a: a.reward if a.reward is not None else float('-inf'),
                    reverse=True
                )
                
                # Get the best agent
                best_agent = sorted_population[0] if sorted_population else None
                
                # Calculate statistics
                rewards = self.rewards
                if rewards:
                    mean_reward = sum(rewards) / len(rewards)
                    # Calculate median
                    sorted_rewards = sorted(rewards)
                    mid = len(sorted_rewards) // 2
                    median_reward = sorted_rewards[mid] if len(sorted_rewards) % 2 == 1 else (sorted_rewards[mid-1] + sorted_rewards[mid]) / 2
                    # Calculate standard deviation
                    std_dev = (sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)) ** 0.5
                else:
                    mean_reward = median_reward = std_dev = 0
                
                return {
                    "best_agent": best_agent,
                    "stats": {
                        "mean": mean_reward,
                        "median": median_reward,
                        "std_dev": std_dev,
                        "count": len(rewards)
                    }
                }
        except Exception as e:
            print(f"Error getting results: {e}")
            return None

def main():
    """Main entry point for the standalone optimizer"""
    parser = argparse.ArgumentParser(
        description="Standalone Universal Optimizer - Optimize any text output using script-based evaluation"
    )
    
    parser.add_argument(
        "eval_command",
        nargs="?",
        help="Evaluation command (receives agent output via stdin, returns score as last line)"
    )
    
    parser.add_argument(
        "--eval-script", 
        help="Path to the evaluation script (alternative to eval_command)"
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
        "--verbose",
        action="store_true",
        help="Enable verbose mode with detailed output of each evolution step"
    )
    
    args = parser.parse_args()
    
    # Determine evaluation method
    eval_script = args.eval_script
    eval_command = args.eval_command
    
    if not eval_script and not eval_command:
        print("Error: Either eval_command or --eval-script must be specified")
        return 1
        
    # If both are provided, eval_command takes precedence
    if eval_command and not eval_script:
        # Create a temporary script that runs the eval command
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.sh') as f:
            eval_script = f.name
            f.write("#!/bin/sh\n")
            f.write(f"{eval_command}\n")
        
        # Make executable
        os.chmod(eval_script, 0o755)
    
    # Get initial content from file if specified
    initial_content = args.initial_content
    if args.initial_file:
        if not os.path.exists(args.initial_file):
            print(f"Error: Initial content file not found: {args.initial_file}")
            return 1
        with open(args.initial_file, 'r') as f:
            initial_content = f.read()
    
    # Create and run the optimizer
    print(f"Starting optimization with {args.population_size} agents and {args.parallel_agents} parallel workers")
    print(f"Evaluation script: {eval_script}")
    print(f"Using {'mock' if args.use_mock_llm else 'real'} LLM")
    print(f"Press Ctrl+C to stop\n")
    
    optimizer = UniversalOptimizer(
        eval_script=eval_script,
        population_size=args.population_size,
        parallel_agents=args.parallel_agents,
        use_mock_llm=args.use_mock_llm,
        script_timeout=args.script_timeout,
        initial_content=initial_content,
        random_seed=args.seed,
        verbose=args.verbose
    )
    
    # Run the optimization
    results = optimizer.run(max_evaluations=args.max_evaluations)
    
    # Print results
    if results:
        print("\n\nOptimization completed!")
        print(f"Total evaluations: {results['stats']['count']}")
        
        best_agent = results["best_agent"]
        if best_agent:
            print(f"\nBest agent (ID: {best_agent.id})")
            print(f"Reward: {best_agent.reward}")
            print("\nContent:")
            print("=" * 40)
            print(best_agent.task_chromosome.content)
            print("=" * 40)
            
            # Write to output file if specified
            if args.output_file:
                with open(args.output_file, 'w') as f:
                    f.write(best_agent.task_chromosome.content)
                print(f"\nResults written to {args.output_file}")
        else:
            print("\nNo valid results found")
        
        # Print statistics
        print("\nStatistics:")
        print(f"Mean reward: {results['stats']['mean']:.2f}")
        print(f"Median reward: {results['stats']['median']:.2f}")
        print(f"Standard deviation: {results['stats']['std_dev']:.2f}")
    else:
        print("\nNo results available")
    
    # Clean up temporary script if created
    if eval_command and not args.eval_script and os.path.exists(eval_script):
        os.remove(eval_script)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
