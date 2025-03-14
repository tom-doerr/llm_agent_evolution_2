import random
import threading
import time
import os
import sys
import tempfile
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass

from llm_agent_evolution.domain.model import Agent, Chromosome, MAX_POPULATION_SIZE, TARGET_LENGTH
from llm_agent_evolution.domain.services import select_parents_pareto, mate_agents

class EvolutionEngine:
    def __init__(self, 
                llm_adapter,
                population_size: int = 100,
                parallel_agents: int = 10,
                eval_command: Optional[str] = None,
                initial_content: str = "",
                verbose: bool = False):
        self.llm_adapter = llm_adapter
        self.population_size = population_size
        self.parallel_agents = parallel_agents
        self.initial_content = initial_content
        self.verbose = verbose
        
        if eval_command:
            self.llm_adapter.eval_command = eval_command
            
        self.population = []
        self.rewards_history = []
        self.best_reward = None
        self.worst_reward = None
        self.recent_rewards = []  # Last 100 rewards
        self.evaluation_count = 0
        self.stop_event = threading.Event()
        self.population_lock = threading.Lock()
        
        # For verbose mode tracking
        self.verbose_agent_ids = set()
        self.verbose_agent_count = 0
        self.max_verbose_agents = 5
    
    def initialize_population(self) -> List[Agent]:
        population = []
        
        for _ in range(self.population_size):
            task_content = self.initial_content
            
            mate_selection_content = """
            Select the mate with the highest reward.
            Consider diversity in the population.
            """
            
            mutation_content = """
            Improve the content to maximize the evaluation score.
            Try different approaches.
            """
            
            agent = Agent(
                task_chromosome=Chromosome(content=task_content, type="task"),
                mate_selection_chromosome=Chromosome(content=mate_selection_content, type="mate_selection"),
                mutation_chromosome=Chromosome(content=mutation_content, type="mutation")
            )
            population.append(agent)
        
        return population
    
    def evaluate_agent(self, agent: Agent) -> float:
        task_output = agent.task_chromosome.content
        
        reward = self.llm_adapter.evaluate_task_output(task_output)
        agent.reward = reward
        
        with self.population_lock:
            self.evaluation_count += 1
            self.rewards_history.append(reward)
            
            if len(self.recent_rewards) >= 100:
                self.recent_rewards.pop(0)
            self.recent_rewards.append(reward)
            
            if self.best_reward is None or reward > self.best_reward:
                self.best_reward = reward
                
            if self.worst_reward is None or reward < self.worst_reward:
                self.worst_reward = reward
        
        return reward
    
    def _evolution_worker(self) -> None:
        while not self.stop_event.is_set():
            try:
                # Select parents
                with self.population_lock:
                    if len(self.population) < 2:
                        time.sleep(0.1)
                        continue
                    parents = select_parents_pareto(self.population, 2)
                
                # Create new agent through mating
                parent1 = parents[0]
                parent2 = self.llm_adapter.select_mate(parent1, [p for p in parents[1:]])
                new_agent = mate_agents(parent1, parent2)
                
                # Track for verbose output if needed
                show_verbose = self._should_show_verbose_output(new_agent.id)
                
                if show_verbose:
                    self._print_verbose_mating_info(parent1, parent2, new_agent)
                
                # Mutate the new agent
                if show_verbose:
                    print("\n3. MUTATION")
                    print(f"Mutation instructions: {new_agent.mutation_chromosome.content[:50]}...")
                    print(f"Before mutation: {new_agent.task_chromosome.content[:50]}...")
                
                task_chromosome = self.llm_adapter.generate_mutation(
                    new_agent.task_chromosome,
                    new_agent.mutation_chromosome.content
                )
                
                mate_selection_chromosome = self.llm_adapter.generate_mutation(
                    new_agent.mate_selection_chromosome,
                    new_agent.mutation_chromosome.content
                )
                
                mutation_chromosome = self.llm_adapter.generate_mutation(
                    new_agent.mutation_chromosome,
                    new_agent.mutation_chromosome.content
                )
                
                mutated_agent = Agent(
                    task_chromosome=task_chromosome,
                    mate_selection_chromosome=mate_selection_chromosome,
                    mutation_chromosome=mutation_chromosome
                )
                
                if show_verbose:
                    print(f"After mutation: {mutated_agent.task_chromosome.content[:50]}...")
                
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
                    if len(self.population) > MAX_POPULATION_SIZE:
                        sorted_population = sorted(
                            self.population,
                            key=lambda a: a.reward if a.reward is not None else float('-inf'),
                            reverse=True
                        )
                        self.population = sorted_population[:MAX_POPULATION_SIZE]
                
            except Exception as e:
                print(f"Worker error: {e}")
                time.sleep(1)
    
    def _should_show_verbose_output(self, agent_id: str) -> bool:
        if not self.verbose:
            return False
            
        with self.population_lock:
            if agent_id in self.verbose_agent_ids:
                return True
                
            if self.verbose_agent_count < self.max_verbose_agents:
                self.verbose_agent_ids.add(agent_id)
                self.verbose_agent_count += 1
                print(f"\nNow tracking agent {agent_id} for verbose output (agent {self.verbose_agent_count} of {self.max_verbose_agents})")
                return True
                
            return False
    
    def _print_verbose_mating_info(self, parent1: Agent, parent2: Agent, new_agent: Agent) -> None:
        print("\n" + "=" * 60)
        print(f"EVOLUTION STEP - DETAILED INFORMATION FOR AGENT {new_agent.id}")
        print("=" * 60)
        print("\n1. PARENT SELECTION")
        print(f"Parent 1 (ID: {parent1.id}):")
        print(f"Reward: {parent1.reward}")
        print(f"Task Chromosome: {parent1.task_chromosome.content[:50]}...")
        
        print(f"\nParent 2 (ID: {parent2.id}):")
        print(f"Reward: {parent2.reward}")
        print(f"Task Chromosome: {parent2.task_chromosome.content[:50]}...")
        
        print("\n2. MATING")
        print(f"New agent after mating (ID: {new_agent.id}):")
        print(f"Task Chromosome: {new_agent.task_chromosome.content[:50]}...")
    
    def get_stats(self) -> Dict[str, Any]:
        with self.population_lock:
            if not self.rewards_history:
                return {
                    "count": 0,
                    "population_size": len(self.population),
                    "mean": None,
                    "median": None,
                    "std_dev": None,
                    "best": None,
                    "worst": None
                }
            
            import numpy as np
            
            # Calculate overall statistics
            rewards = self.rewards_history
            mean = np.mean(rewards)
            median = np.median(rewards)
            std_dev = np.std(rewards)
            
            # Calculate recent window statistics
            if self.recent_rewards:
                window_mean = np.mean(self.recent_rewards)
                window_median = np.median(self.recent_rewards)
                window_std_dev = np.std(self.recent_rewards)
            else:
                window_mean = window_median = window_std_dev = None
            
            return {
                "count": len(rewards),
                "population_size": len(self.population),
                "mean": mean,
                "median": median,
                "std_dev": std_dev,
                "best": self.best_reward,
                "worst": self.worst_reward,
                "window_stats": {
                    "count": len(self.recent_rewards),
                    "mean": window_mean,
                    "median": window_median,
                    "std_dev": window_std_dev
                }
            }
    
    def run_evolution(self, 
                     max_evaluations: Optional[int] = None,
                     progress_callback: Optional[Callable] = None) -> List[Agent]:
        # Initialize population
        self.population = self.initialize_population()
        
        # Start worker threads
        workers = []
        for _ in range(self.parallel_agents):
            thread = threading.Thread(target=self._evolution_worker)
            thread.daemon = True
            thread.start()
            workers.append(thread)
        
        # Evaluate initial population
        print("Evaluating initial population...")
        with ThreadPoolExecutor(max_workers=self.parallel_agents) as executor:
            list(executor.map(self.evaluate_agent, self.population))
        
        try:
            # Monitor progress
            last_stats_time = time.time()
            last_progress_time = time.time()
            
            while not self.stop_event.is_set():
                # Check if we've reached max evaluations
                if max_evaluations and self.evaluation_count >= max_evaluations:
                    print("\nReached maximum evaluations")
                    break
                
                # Call progress callback if provided
                if progress_callback and time.time() - last_progress_time > 0.5:
                    try:
                        if max_evaluations:
                            progress_callback(self.evaluation_count, max_evaluations)
                        else:
                            progress_callback(self.evaluation_count)
                    except Exception as e:
                        print(f"Error in progress callback: {e}")
                    last_progress_time = time.time()
                
                # Display stats periodically
                if time.time() - last_stats_time > 10:
                    stats = self.get_stats()
                    if stats["mean"] is not None:
                        print(f"\nEvaluations: {stats['count']}, Population: {stats['population_size']}")
                        print(f"Best: {stats['best']:.2f}, Mean: {stats['mean']:.2f}, Median: {stats['median']:.2f}")
                    last_stats_time = time.time()
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nStopping evolution...")
        finally:
            # Signal workers to stop
            self.stop_event.set()
            
            # Wait for workers to finish
            for worker in workers:
                worker.join(timeout=1.0)
        
        # Return the final population
        return self.population

def run_optimizer(
    eval_command: str,
    population_size: int = 50,
    parallel_agents: int = 8,
    max_evaluations: Optional[int] = None,
    use_mock_llm: bool = False,
    model_name: str = "openrouter/google/gemini-2.0-flash-001",
    initial_content: str = "",
    verbose: bool = False,
    random_seed: Optional[int] = None
) -> Dict[str, Any]:
    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)
    
    # Create LLM adapter
    if use_mock_llm:
        from llm_agent_evolution.adapters.secondary.mock_llm import MockLLMAdapter
        llm_adapter = MockLLMAdapter(seed=random_seed)
    else:
        from llm_agent_evolution.adapters.secondary.llm import DSPyLLMAdapter
        llm_adapter = DSPyLLMAdapter(model_name=model_name)
    
    # Create and run the evolution engine
    engine = EvolutionEngine(
        llm_adapter=llm_adapter,
        population_size=population_size,
        parallel_agents=parallel_agents,
        eval_command=eval_command,
        initial_content=initial_content,
        verbose=verbose
    )
    
    # Simple progress indicator
    def progress_indicator(current: int, total: Optional[int] = None) -> None:
        if total:
            if current % max(1, int(total * 0.1)) == 0:
                print(f"Progress: {current}/{total} ({current/total*100:.1f}%)")
        else:
            if current % 100 == 0:
                print(f"Progress: {current} evaluations")
    
    # Run evolution
    print(f"Starting optimization with {population_size} agents and {parallel_agents} parallel workers")
    print(f"Evaluation command: {eval_command}")
    print(f"Using {'mock' if use_mock_llm else 'real'} LLM")
    
    start_time = time.time()
    population = engine.run_evolution(
        max_evaluations=max_evaluations,
        progress_callback=progress_indicator
    )
    total_runtime = time.time() - start_time
    
    # Get statistics
    stats = engine.get_stats()
    
    # Get best agent
    if population:
        sorted_population = sorted(
            population,
            key=lambda a: a.reward if a.reward is not None else float('-inf'),
            reverse=True
        )
        best_agent = sorted_population[0]
    else:
        best_agent = None
    
    # Print results
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS".center(60))
    print("=" * 60)
    
    print(f"\nTotal evaluations: {stats['count']}")
    print(f"Final population size: {stats['population_size']}")
    print(f"Runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")
    
    if stats["mean"] is not None:
        print("\nStatistics:")
        print(f"- Mean reward: {stats['mean']:.2f}")
        print(f"- Median reward: {stats['median']:.2f}")
        print(f"- Standard deviation: {stats['std_dev']:.2f}")
        print(f"- Best reward: {stats['best']:.2f}")
        print(f"- Worst reward: {stats['worst']:.2f}")
    
    if best_agent:
        print("\nBest Agent:")
        print(f"- ID: {best_agent.id}")
        print(f"- Reward: {best_agent.reward:.2f}")
        print(f"- Content ({len(best_agent.task_chromosome.content)} chars):")
        print("-" * 60)
        print(best_agent.task_chromosome.content)
        print("-" * 60)
    
    # Return results
    return {
        "best_agent": {
            "id": best_agent.id if best_agent else None,
            "reward": best_agent.reward if best_agent else None,
            "content": best_agent.task_chromosome.content if best_agent else None
        },
        "stats": stats,
        "runtime": total_runtime
    }

def load_agent(file_path: str) -> Optional[Agent]:
    try:
        import tomli
        with open(file_path, 'rb') as f:
            agent_data = tomli.load(f)
        
        if 'agent' in agent_data:
            agent_info = agent_data['agent']
            loaded_agent = Agent(
                task_chromosome=Chromosome(
                    content=agent_info['task_chromosome']['content'],
                    type=agent_info['task_chromosome']['type']
                ),
                mate_selection_chromosome=Chromosome(
                    content=agent_info['mate_selection_chromosome']['content'],
                    type=agent_info['mate_selection_chromosome']['type']
                ),
                mutation_chromosome=Chromosome(
                    content=agent_info['mutation_chromosome']['content'],
                    type=agent_info['mutation_chromosome']['type']
                ),
                id=agent_info.get('id'),
                reward=agent_info.get('reward')
            )
            return loaded_agent
    except Exception as e:
        print(f"Error loading agent: {e}")
    return None

def save_agent(agent: Agent, file_path: str) -> bool:
    try:
        import tomli_w
        agent_data = {
            "agent": {
                "id": agent.id,
                "reward": agent.reward,
                "task_chromosome": {
                    "content": agent.task_chromosome.content,
                    "type": agent.task_chromosome.type
                },
                "mate_selection_chromosome": {
                    "content": agent.mate_selection_chromosome.content,
                    "type": agent.mate_selection_chromosome.type
                },
                "mutation_chromosome": {
                    "content": agent.mutation_chromosome.content,
                    "type": agent.mutation_chromosome.type
                }
            }
        }
        
        with open(file_path, 'wb') as f:
            tomli_w.dump(agent_data, f)
        return True
    except Exception as e:
        print(f"Error saving agent: {e}")
        return False

def evaluate_agent_with_command(agent: Agent, eval_command: str, context: Optional[str] = None) -> float:
    try:
        # Set up environment for context if provided
        env = os.environ.copy()
        if context:
            env['AGENT_CONTEXT'] = context
        
        # Run the evaluation command
        process = subprocess.run(
            eval_command,
            shell=True,
            input=agent.task_chromosome.content,
            text=True,
            capture_output=True,
            env=env
        )
        
        # Extract the reward from the last line of output
        output_lines = process.stdout.strip().split('\n')
        try:
            reward = float(output_lines[-1])
            detailed_output = '\n'.join(output_lines[:-1])
        except (ValueError, IndexError):
            reward = 0.0
            detailed_output = process.stdout
        
        print("\nAgent evaluation complete")
        print(f"Reward: {reward}")
        
        if detailed_output:
            print("\nDetailed evaluation output:")
            print(detailed_output)
        
        return reward
    except Exception as e:
        print(f"Error evaluating agent: {e}")
        return 0.0
