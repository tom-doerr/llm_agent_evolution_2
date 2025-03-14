import argparse
import sys
import os
from typing import List, Dict, Any
import time
from llm_agent_evolution.ports.primary import EvolutionUseCase

class CLIAdapter:
    """Command-line interface adapter for the evolution system"""
    
    def __init__(self, evolution_use_case: EvolutionUseCase):
        """Initialize the CLI adapter with the evolution use case"""
        self.evolution_use_case = evolution_use_case
    
    def parse_args(self) -> argparse.Namespace:
        """Parse command-line arguments"""
        parser = argparse.ArgumentParser(
            description="LLM Agent Evolution - Evolve LLM-based agents through evolutionary algorithms"
        )
        
        parser.add_argument(
            "--population-size", 
            type=int, 
            default=100,
            help="Initial population size (default: 100)"
        )
        
        parser.add_argument(
            "--parallel-agents", 
            type=int, 
            default=10,
            help="Number of agents to evaluate in parallel (default: 10)"
        )
        
        parser.add_argument(
            "--max-evaluations", 
            type=int, 
            default=None,
            help="Maximum number of evaluations to run (default: unlimited)"
        )
        
        parser.add_argument(
            "--log-file", 
            type=str, 
            default="evolution.log",
            help="Log file path (default: evolution.log)"
        )
        
        parser.add_argument(
            "--model", 
            type=str, 
            default="openrouter/google/gemini-2.0-flash-001",
            help="LLM model to use (default: openrouter/google/gemini-2.0-flash-001)"
        )
        
        parser.add_argument(
            "--use-mock",
            action="store_true",
            help="Use mock LLM adapter for testing"
        )
        
        parser.add_argument(
            "--eval-command",
            type=str,
            default=None,
            help="Command to run for evaluation (receives agent output via stdin, returns score as last line)"
        )
        
        parser.add_argument(
            "--seed",
            type=int,
            default=None,
            help="Random seed for reproducibility"
        )
        
        parser.add_argument(
            "--quick-test",
            action="store_true",
            help="Run a quick test with mock LLM (100 evaluations)"
        )
        
        parser.add_argument(
            "--no-visualization",
            action="store_true",
            help="Disable visualization generation"
        )
        
        return parser.parse_args()
    
    def display_stats(self, stats: Dict[str, Any]) -> None:
        """Display statistics to the console"""
        print("\n=== Population Statistics ===")
        
        # Print overall stats
        print(f"Population size: {stats.get('population_size', 0)}")
        print(f"Total evaluations: {stats.get('count', 0)}")
        
        if stats.get('mean') is not None:
            print(f"Mean reward: {stats.get('mean'):.2f}")
            print(f"Median reward: {stats.get('median'):.2f}")
            print(f"Std deviation: {stats.get('std_dev'):.2f}")
            print(f"Best reward: {stats.get('best'):.2f}")
            print(f"Worst reward: {stats.get('worst'):.2f}")
            
            # Display improvement metrics if available
            if stats.get('improvement_rate') is not None:
                print(f"Improvement rate: {stats.get('improvement_rate'):.4f} per minute")
                
            if stats.get('time_since_last_best') is not None:
                minutes = stats.get('time_since_last_best') / 60
                print(f"Time since last best: {minutes:.2f} minutes")
        
        # Recent window stats
        window_stats = stats.get('window_stats', {})
        if window_stats and window_stats.get('count', 0) > 0:
            print(f"\nRecent Evaluations (Last {window_stats.get('window_size', 100)})")
            print(f"Count: {window_stats.get('count', 0)}")
            
            if window_stats.get('mean') is not None:
                print(f"Mean reward: {window_stats.get('mean'):.2f}")
                print(f"Median reward: {window_stats.get('median'):.2f}")
                print(f"Std deviation: {window_stats.get('std_dev'):.2f}")
        
        # Check for visualizations
        viz_dir = "visualizations"
        if os.path.exists(viz_dir) and os.listdir(viz_dir):
            print(f"\nVisualizations available in: {os.path.abspath(viz_dir)}")
            print("Latest visualization files:")
            files = sorted(
                [f for f in os.listdir(viz_dir) if f.endswith('.png')],
                key=lambda x: os.path.getmtime(os.path.join(viz_dir, x)),
                reverse=True
            )
            for file in files[:3]:  # Show the 3 most recent files
                print(f"- {file}")
    
    def run(self) -> int:
        """Run the CLI application"""
        args = self.parse_args()
        
        # Handle quick test mode
        if args.quick_test:
            print("Running quick test with mock LLM adapter")
            args.use_mock = True
            args.max_evaluations = 100
            args.population_size = 20
            args.parallel_agents = 4
            if args.seed is None:
                args.seed = 42  # Use fixed seed for reproducible tests
        
        try:
            # Show startup message
            print("LLM Agent Evolution")
            print(f"Population size: {args.population_size}")
            print(f"Parallel agents: {args.parallel_agents}")
            print(f"Max evaluations: {args.max_evaluations or 'unlimited'}")
            print(f"Using {'mock' if args.use_mock else 'real'} LLM adapter")
            
            # Simple progress tracking
            last_count = 0
            last_print_time = time.time()
            
            def progress_callback(current_count):
                nonlocal last_count, last_print_time
                now = time.time()
                # Only print progress every 5 seconds to avoid flooding the console
                if now - last_print_time >= 5:
                    if args.max_evaluations:
                        print(f"Progress: {current_count}/{args.max_evaluations} evaluations "
                              f"({current_count/args.max_evaluations*100:.1f}%)")
                    else:
                        print(f"Progress: {current_count} evaluations")
                    last_count = current_count
                    last_print_time = now
            
            # Run evolution
            population = self.evolution_use_case.run_evolution(
                population_size=args.population_size,
                parallel_agents=args.parallel_agents,
                max_evaluations=args.max_evaluations,
                progress_callback=progress_callback
            )
            
            # Get and display final statistics
            stats = self.evolution_use_case.get_population_stats(population)
            self.display_stats(stats)
            
            # Display best agent
            best_agent = max(population, key=lambda a: a.reward if a.reward is not None else float('-inf'))
            print("\n=== Best Agent ===")
            print(f"ID: {best_agent.id}")
            print(f"Reward: {best_agent.reward}")
            print("Task Chromosome:")
            print(best_agent.task_chromosome.content)
            
            return 0
        except KeyboardInterrupt:
            print("\nEvolution process interrupted by user.")
            return 1
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            print(traceback.format_exc())
            return 1

def main():
    """Entry point for the CLI application"""
    # This will be implemented in application.py to wire everything together
    pass

if __name__ == "__main__":
    sys.exit(main())
