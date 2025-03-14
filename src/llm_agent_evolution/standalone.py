#!/usr/bin/env python3
"""
Standalone optimizer for LLM Agent Evolution
A simplified version that can be run directly without dependencies
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

# Constants
MAX_CHARS = 1000  # Maximum characters for chromosomes
TARGET_LENGTH = 23  # Target length for task optimization
MAX_OUTPUT_TOKENS = 40  # Limit token output for the DSPy LM
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
        
    def _hash_input(self, output: str) -> str:
        """Generate a simple hash for the output"""
        return str(hash(output))
    
    def evaluate(self, output: str, eval_command: str, timeout: int = 30) -> float:
        """Evaluate the output using the specified command"""
        # Check cache first
        input_hash = self._hash_input(output)
        if input_hash in self.cache:
            self.cache_hits += 1
            return self.cache[input_hash]
        
        self.cache_misses += 1
        
        try:
            # Create a temporary script that runs the eval command
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.sh') as f:
                script_path = f.name
                f.write("#!/bin/sh\n")
                f.write(f"{eval_command}\n")
            
            # Make executable
            os.chmod(script_path, 0o755)
            
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
                print(f"Warning: Evaluation script failed with error: {stderr}")
                reward = 0.0
            else:
                # Parse the reward from the last line of output
                lines = stdout.strip().split('\n')
                if not lines:
                    print("Warning: Evaluation script produced no output")
                    reward = 0.0
                else:
                    try:
                        reward = float(lines[-1].strip())
                    except ValueError:
                        print(f"Warning: Evaluation script did not return a valid numerical reward: {lines[-1]}")
                        reward = 0.0
            
            # Update cache
            if len(self.cache) >= self.cache_size:
                # Remove a random entry to keep the cache size in check
                self.cache.pop(next(iter(self.cache)))
            self.cache[input_hash] = reward
            
            # Clean up
            os.remove(script_path)
            
            return reward
            
        except subprocess.TimeoutExpired:
            print(f"Warning: Evaluation script timed out after {timeout} seconds")
            return 0.0
        except Exception as e:
            print(f"Warning: Evaluation error: {e}")
            return 0.0

def select_parents(population: List[Agent], num_parents: int) -> List[Agent]:
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
    
    # Handle case where all rewards are the same
    if min_reward == max_reward:
        return random.sample(valid_agents, min(num_parents, len(valid_agents)))
    
    # Shift rewards to positive range if needed
    if min_reward < 0:
        shift = abs(min_reward) + 1.0  # Add 1 to ensure all values are positive
        adjusted_rewards = [(r + shift) for r in rewards]
    else:
        # If all rewards are already positive, just use them directly
        adjusted_rewards = rewards.copy()
    
    # Square the rewards for Pareto distribution (emphasize higher rewards)
    squared_rewards = [r**2 for r in adjusted_rewards]
    
    # Normalize weights
    total_weight = sum(squared_rewards)
    weights = [w / total_weight for w in squared_rewards]
    
    # Weighted sampling without replacement
    selected_parents = []
    remaining_agents = valid_agents.copy()
    remaining_weights = weights.copy()
    
    # Select parents one by one
    for _ in range(min(num_parents, len(valid_agents))):
        if not remaining_agents:
            break
            
        # Normalize remaining weights
        weight_sum = sum(remaining_weights)
        if weight_sum <= 0:
            # If all weights are zero, select randomly
            idx = random.randrange(len(remaining_agents))
        else:
            normalized_weights = [w / weight_sum for w in remaining_weights]
            # Select based on weights
            idx = random.choices(range(len(remaining_agents)), weights=normalized_weights, k=1)[0]
        
        # Add selected agent to parents
        selected_parents.append(remaining_agents[idx])
        
        # Remove selected agent from candidates
        remaining_agents.pop(idx)
        remaining_weights.pop(idx)
    
    return selected_parents

def combine_chromosomes(parent1: Chromosome, parent2: Chromosome) -> Chromosome:
    """Combine two chromosomes by switching at hotspots"""
    # Handle empty content cases
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
    
    # Sort hotspots and select a subset for switching
    hotspots.sort()
    if len(hotspots) > 0:
        # Select 1-2 random hotspots for switching
        num_switches = random.randint(1, min(2, len(hotspots)))
        switch_points = sorted(random.sample(hotspots, num_switches))
        
        # Perform the switches
        for point in switch_points:
            if random.random() < CHROMOSOME_SWITCH_PROBABILITY:
                # Take content from secondary parent from this point
                if point < len(secondary_parent.content):
                    secondary_content = list(secondary_parent.content[point:])
                    
                    # Adjust result length
                    if point + len(secondary_content) > len(result):
                        result = result[:point] + secondary_content
                    else:
                        result[point:point+len(secondary_content)] = secondary_content
    
    # Ensure we don't exceed MAX_CHARS
    combined_content = ''.join(result)
    if len(combined_content) > MAX_CHARS:
        combined_content = combined_content[:MAX_CHARS]
    
    return Chromosome(content=combined_content, type_=parent1.type)

def mate_agents(parent1: Agent, parent2: Agent) -> Agent:
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

def mutate_agent(agent: Agent) -> Agent:
    """Simple mutation that makes random changes to chromosomes"""
    # For task chromosome, try to optimize for the hidden goal
    task_content = agent.task_chromosome.content
    
    # 50% chance to add more 'a's
    if random.random() < 0.5:
        # Add 'a's to reach TARGET_LENGTH
        current_length = len(task_content)
        if current_length < TARGET_LENGTH:
            # Add more 'a's
            task_content += 'a' * (TARGET_LENGTH - current_length)
        elif current_length > TARGET_LENGTH:
            # Trim to TARGET_LENGTH
            task_content = task_content[:TARGET_LENGTH]
    else:
        # Random mutation
        if random.random() < 0.3:
            # Replace with all 'a's
            task_content = 'a' * TARGET_LENGTH
        elif random.random() < 0.5 and task_content:
            # Shuffle characters
            chars = list(task_content)
            random.shuffle(chars)
            task_content = ''.join(chars)
    
    # Create new chromosomes
    new_task = Chromosome(content=task_content, type_="task")
    new_mate = Chromosome(content=agent.mate_selection_chromosome.content, type_="mate_selection")
    new_mutation = Chromosome(content=agent.mutation_chromosome.content, type_="mutation")
    
    # Create and return mutated agent
    return Agent(
        task_chromosome=new_task,
        mate_selection_chromosome=new_mate,
        mutation_chromosome=new_mutation
    )

def run_standalone_optimizer(
    eval_command: str,
    population_size: int = 50,
    parallel_agents: int = 8,
    max_evaluations: Optional[int] = None,
    initial_content: str = "",
    random_seed: Optional[int] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run the standalone optimizer
    
    Args:
        eval_command: Command to run for evaluation
        population_size: Initial population size
        parallel_agents: Number of agents to evaluate in parallel
        max_evaluations: Maximum number of evaluations to run
        initial_content: Initial content for the chromosomes
        random_seed: Random seed for reproducibility
        verbose: Whether to enable verbose output
        
    Returns:
        Dictionary with optimization results
    """
    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)
    
    # Initialize components
    evaluator = ScriptEvaluator()
    
    # Initialize population
    population = []
    for _ in range(population_size):
        # Initial task chromosome with some content
        task_content = initial_content if initial_content else ""
        
        # Initial mate selection chromosome with instructions
        mate_selection_content = "Select the mate with the highest reward."
        
        # Initial mutation chromosome with instructions
        mutation_content = "Improve the content to maximize the evaluation score."
        
        agent = Agent(
            task_chromosome=Chromosome(content=task_content, type_="task"),
            mate_selection_chromosome=Chromosome(content=mate_selection_content, type_="mate_selection"),
            mutation_chromosome=Chromosome(content=mutation_content, type_="mutation")
        )
        population.append(agent)
    
    # Evaluate initial population
    print("Evaluating initial population...")
    for agent in population:
        # Evaluate the agent
        reward = evaluator.evaluate(agent.task_chromosome.content, eval_command)
        agent.reward = reward
    
    # Setup for evolution
    stop_event = threading.Event()
    population_lock = threading.Lock()
    evaluation_count = population_size
    
    # Worker function for evolution
    def evolution_worker():
        nonlocal evaluation_count
        
        while not stop_event.is_set():
            try:
                # Check if we've reached max evaluations
                if max_evaluations and evaluation_count >= max_evaluations:
                    break
                
                # Select parents
                with population_lock:
                    parents = select_parents(population, 2)
                
                if len(parents) < 2:
                    # Not enough parents, create random agent
                    new_agent = Agent(
                        task_chromosome=Chromosome(content=initial_content, type_="task"),
                        mate_selection_chromosome=Chromosome(content="", type_="mate_selection"),
                        mutation_chromosome=Chromosome(content="", type_="mutation")
                    )
                else:
                    # Create new agent through mating
                    parent1, parent2 = parents
                    new_agent = mate_agents(parent1, parent2)
                
                # Mutate the new agent
                mutated_agent = mutate_agent(new_agent)
                
                # Evaluate the agent
                reward = evaluator.evaluate(mutated_agent.task_chromosome.content, eval_command)
                mutated_agent.reward = reward
                
                # Add to population
                with population_lock:
                    population.append(mutated_agent)
                    
                    # If population exceeds limit, remove the worst agent
                    if len(population) > population_size * 2:
                        # Sort by reward
                        sorted_population = sorted(
                            population,
                            key=lambda a: a.reward if a.reward is not None else float('-inf'),
                            reverse=True
                        )
                        # Keep only the top agents
                        population.clear()
                        population.extend(sorted_population[:population_size])
                    
                    # Increment evaluation count
                    evaluation_count += 1
                
                # Verbose output
                if verbose:
                    print(f"Agent {mutated_agent.id}: reward={reward}, content='{mutated_agent.task_chromosome.content}'")
                
            except Exception as e:
                print(f"Worker error: {e}")
                time.sleep(1)  # Avoid tight loop on errors
    
    # Start worker threads
    workers = []
    for _ in range(parallel_agents):
        thread = threading.Thread(target=evolution_worker)
        thread.daemon = True
        thread.start()
        workers.append(thread)
    
    try:
        # Monitor progress
        last_stats_time = time.time()
        start_time = time.time()
        
        while not stop_event.is_set():
            # Check if we've reached max evaluations
            if max_evaluations and evaluation_count >= max_evaluations:
                print("\nReached maximum evaluations")
                break
            
            # Display stats periodically
            if time.time() - last_stats_time > 5:  # Every 5 seconds
                with population_lock:
                    # Calculate statistics
                    rewards = [a.reward for a in population if a.reward is not None]
                    if rewards:
                        mean_reward = sum(rewards) / len(rewards)
                        sorted_rewards = sorted(rewards)
                        median_reward = sorted_rewards[len(sorted_rewards)//2]
                        best_reward = max(rewards)
                        
                        # Calculate rate
                        elapsed = time.time() - start_time
                        rate = evaluation_count / elapsed if elapsed > 0 else 0
                        
                        # Print stats
                        print(f"\nEvaluations: {evaluation_count}/{max_evaluations if max_evaluations else 'unlimited'} "
                              f"({rate:.1f}/sec)")
                        print(f"Population: {len(population)} agents")
                        print(f"Best: {best_reward:.2f}, Mean: {mean_reward:.2f}, Median: {median_reward:.2f}")
                        
                        # Print best agent
                        best_agent = max(population, key=lambda a: a.reward if a.reward is not None else float('-inf'))
                        print(f"Best agent content: '{best_agent.task_chromosome.content}'")
                
                last_stats_time = time.time()
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopping optimization...")
    finally:
        # Signal workers to stop
        stop_event.set()
        
        # Wait for workers to finish
        for worker in workers:
            worker.join(timeout=1.0)
    
    # Get final results
    with population_lock:
        # Sort population by reward
        sorted_population = sorted(
            population,
            key=lambda a: a.reward if a.reward is not None else float('-inf'),
            reverse=True
        )
        
        # Get the best agent
        best_agent = sorted_population[0] if sorted_population else None
        
        # Calculate statistics
        rewards = [a.reward for a in population if a.reward is not None]
        if rewards:
            mean_reward = sum(rewards) / len(rewards)
            sorted_rewards = sorted(rewards)
            median_reward = sorted_rewards[len(sorted_rewards)//2]
            std_dev = (sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)) ** 0.5
        else:
            mean_reward = median_reward = std_dev = 0
        
        return {
            "best_agent": {
                "id": best_agent.id if best_agent else None,
                "reward": best_agent.reward if best_agent else None,
                "content": best_agent.task_chromosome.content if best_agent else None
            },
            "stats": {
                "mean": mean_reward,
                "median": median_reward,
                "std_dev": std_dev,
                "count": evaluation_count,
                "population_size": len(population)
            }
        }

def main():
    """Main entry point for the standalone optimizer"""
    parser = argparse.ArgumentParser(
        description="Standalone Optimizer - A simplified version of LLM Agent Evolution",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "eval_command",
        help="Evaluation command (receives agent output via stdin, returns score as last line)"
    )
    
    parser.add_argument(
        "--population-size", "-p",
        type=int, 
        default=50,
        help="Initial population size"
    )
    
    parser.add_argument(
        "--parallel-agents", "-j",
        type=int, 
        default=8,
        help="Number of agents to evaluate in parallel"
    )
    
    parser.add_argument(
        "--max-evaluations", "-n",
        type=int, 
        default=1000,
        help="Maximum number of evaluations to run"
    )
    
    parser.add_argument(
        "--initial-content", "-i",
        type=str,
        default="",
        help="Initial content for the chromosomes"
    )
    
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--output-file", "-o",
        type=str,
        default=None,
        help="File to write the best result to"
    )
    
    args = parser.parse_args()
    
    # Print banner
    print("\n" + "=" * 60)
    print("STANDALONE OPTIMIZER".center(60))
    print("=" * 60)
    
    # Print configuration
    print("\nConfiguration:")
    print(f"- Population size: {args.population_size}")
    print(f"- Parallel agents: {args.parallel_agents}")
    print(f"- Max evaluations: {args.max_evaluations}")
    print(f"- Evaluation command: {args.eval_command}")
    if args.initial_content:
        print(f"- Initial content: '{args.initial_content}'")
    if args.seed is not None:
        print(f"- Random seed: {args.seed}")
    
    print("\nStarting optimization...")
    start_time = time.time()
    
    # Run the optimizer
    results = run_standalone_optimizer(
        eval_command=args.eval_command,
        population_size=args.population_size,
        parallel_agents=args.parallel_agents,
        max_evaluations=args.max_evaluations,
        initial_content=args.initial_content,
        random_seed=args.seed,
        verbose=args.verbose
    )
    
    # Calculate total runtime
    total_runtime = time.time() - start_time
    
    # Print results
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS".center(60))
    print("=" * 60)
    
    print(f"\nTotal evaluations: {results['stats']['count']}")
    print(f"Final population size: {results['stats']['population_size']}")
    print(f"Runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")
    
    print("\nStatistics:")
    print(f"- Mean reward: {results['stats']['mean']:.2f}")
    print(f"- Median reward: {results['stats']['median']:.2f}")
    print(f"- Standard deviation: {results['stats']['std_dev']:.2f}")
    
    # Print best agent
    best_agent = results["best_agent"]
    if best_agent and best_agent["reward"] is not None:
        print("\nBest Agent:")
        print(f"- ID: {best_agent['id']}")
        print(f"- Reward: {best_agent['reward']:.2f}")
        print(f"- Content ({len(best_agent['content'])} chars):")
        print("-" * 60)
        print(best_agent['content'])
        print("-" * 60)
        
        # Write to output file if specified
        if args.output_file:
            with open(args.output_file, 'w') as f:
                f.write(best_agent['content'])
            print(f"\nBest result written to: {args.output_file}")
    else:
        print("\nNo valid results found")
    
    print("\n" + "=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
