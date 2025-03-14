import argparse
import sys
from typing import List, Dict, Any
from ....ports.primary import EvolutionUseCase

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
        
        return parser.parse_args()
    
    def display_stats(self, stats: Dict[str, Any]) -> None:
        """Display statistics to the console"""
        print("\n=== Population Statistics ===")
        print(f"Population size: {stats.get('count', 0)}")
        print(f"Mean reward: {stats.get('mean'):.2f}")
        print(f"Median reward: {stats.get('median'):.2f}")
        print(f"Std deviation: {stats.get('std_dev'):.2f}")
        print(f"Best reward: {stats.get('best'):.2f}")
        print(f"Worst reward: {stats.get('worst'):.2f}")
        
        # Recent window stats
        window_stats = stats.get('window_stats', {})
        if window_stats:
            print("\n=== Recent Evaluations (Sliding Window) ===")
            print(f"Window size: {window_stats.get('window_size', 0)}")
            print(f"Count: {window_stats.get('count', 0)}")
            if window_stats.get('mean') is not None:
                print(f"Mean reward: {window_stats.get('mean'):.2f}")
                print(f"Median reward: {window_stats.get('median'):.2f}")
                print(f"Std deviation: {window_stats.get('std_dev'):.2f}")
    
    def run(self) -> int:
        """Run the CLI application"""
        args = self.parse_args()
        
        try:
            # Run the evolution process
            population = self.evolution_use_case.run_evolution(
                population_size=args.population_size,
                parallel_agents=args.parallel_agents,
                max_evaluations=args.max_evaluations
            )
            
            # Get and display final statistics
            stats = self.evolution_use_case.get_population_stats(population)
            self.display_stats(stats)
            
            return 0
        except KeyboardInterrupt:
            print("\nEvolution process interrupted by user.")
            return 1
        except Exception as e:
            print(f"\nError: {e}")
            return 1

def main():
    """Entry point for the CLI application"""
    # This will be implemented in application.py to wire everything together
    pass

if __name__ == "__main__":
    sys.exit(main())
