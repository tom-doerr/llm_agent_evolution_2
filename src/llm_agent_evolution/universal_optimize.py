"""
Universal Optimizer module for the LLM Agent Evolution package
"""
import sys
import os
import time
import tomli_w
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
from llm_agent_evolution.universal_optimizer_core import UniversalOptimizer

# Import the core optimizer class
from llm_agent_evolution.universal_optimizer_core import UniversalOptimizer

def progress_bar(current_count: int, max_count: Optional[int] = None) -> None:
    """Simple progress indicator (not a bar) to minimize output volume"""
    if max_count:
        # Only print every 10% progress to reduce output volume
        if current_count % max(1, int(max_count * 0.1)) == 0 or current_count == max_count:
            percent = current_count / max_count * 100
            print(f"Progress: {current_count}/{max_count} ({percent:.1f}%)")
    else:
        # Only print at regular intervals
        if current_count % 100 == 0:
            print(f"Progress: {current_count} evaluations")

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
    save: Optional[str] = None,
    output_format: str = "toml",
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
        
        # Write to save file if specified
        if save:
            if output_format == "text":
                with open(save, 'w') as f:
                    f.write(results['best_agent']['content'])
            else:  # toml
                with open(save, 'wb') as f:
                    tomli_w.dump(results, f)
            print(f"\nResults saved to {save}")
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
        "--save", "-o",
        type=str,
        default=None,
        help="File to save the best result to"
    )
    
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["text", "toml"],
        default="toml",
        help="Output format (default: toml)"
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
        save=args.save,
        output_format=args.output_format,
        max_chars=args.max_chars,
        verbose=args.verbose
    ))
