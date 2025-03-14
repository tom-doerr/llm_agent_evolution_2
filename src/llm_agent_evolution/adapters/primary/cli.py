import argparse
import sys
import os
from typing import List, Dict, Any
import time
from llm_agent_evolution.ports.primary import EvolutionUseCase
from llm_agent_evolution.domain.model import Agent, Chromosome

class CLIAdapter:
    """Command-line interface adapter for the evolution system"""
    
    def __init__(self, evolution_use_case):
        """Initialize the CLI adapter with the evolution service"""
        self.evolution_use_case = evolution_use_case
    
    def parse_args(self) -> argparse.Namespace:
        """Parse command-line arguments"""
        parser = argparse.ArgumentParser(
            description="LLM Agent Evolution - Evolve LLM-based agents through evolutionary algorithms",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        # Core parameters
        group_core = parser.add_argument_group('Core Parameters')
        group_core.add_argument(
            "--population-size", "-p",
            type=int, 
            default=100,
            help="Initial population size"
        )
        
        group_core.add_argument(
            "--parallel-agents", "-j",
            type=int, 
            default=10,
            help="Number of agents to evaluate in parallel"
        )
        
        group_core.add_argument(
            "--max-evaluations", "-n",
            type=int, 
            default=None,
            help="Maximum number of evaluations to run (unlimited if not specified)"
        )
        
        # LLM configuration
        group_llm = parser.add_argument_group('LLM Configuration')
        group_llm.add_argument(
            "--model", "-m",
            type=str, 
            default="openrouter/google/gemini-2.0-flash-001",
            help="LLM model to use"
        )
        
        group_llm.add_argument(
            "--use-mock", "--mock",
            action="store_true",
            help="Use mock LLM adapter for testing (no API calls)"
        )
        
        # Evaluation
        group_eval = parser.add_argument_group('Evaluation')
        group_eval.add_argument(
            "--eval-command", "-e",
            type=str,
            default=None,
            help="Command to run for evaluation (receives agent output via stdin, returns score as last line)"
        )
        
        # Agent loading and context
        group_agent = parser.add_argument_group('Agent Loading and Context')
        group_agent.add_argument(
            "--load", "-l",
            type=str,
            default=None,
            help="Load a previously saved agent from file"
        )
        
        group_agent.add_argument(
            "--save", "-o",
            type=str,
            default=None,
            help="File to save the best result to"
        )
        
        group_agent.add_argument(
            "--context", "-c",
            type=str,
            default=None,
            help="Context to pass to the agent (available as AGENT_CONTEXT environment variable)"
        )
        
        group_agent.add_argument(
            "--context-file", "-cf",
            type=str,
            default=None,
            help="File containing context to pass to the agent"
        )
        
        # Output and logging
        group_output = parser.add_argument_group('Output and Logging')
        group_output.add_argument(
            "--log-file",
            type=str, 
            default="evolution.log",
            help="Log file path"
        )
        
        group_output.add_argument(
            "--verbose", "-v",
            action="store_true",
            help="Enable verbose output"
        )
        
        # Misc options
        group_misc = parser.add_argument_group('Miscellaneous')
        group_misc.add_argument(
            "--seed", "-s",
            type=int,
            default=None,
            help="Random seed for reproducibility"
        )
        
        group_misc.add_argument(
            "--quick-test", "-q",
            action="store_true",
            help="Run a quick test with mock LLM (100 evaluations)"
        )
        
        return parser.parse_args()
    
    def display_stats(self, stats: Dict[str, Any]) -> None:
        """Display statistics to the console in a clear, information-dense format"""
        print("\nPOPULATION STATS: ", end="")
        print(f"Size: {stats.get('population_size', 0):,} | ", end="")
        print(f"Evals: {stats.get('count', 0):,}")
        
        if stats.get('mean') is not None:
            print(f"Mean: {stats.get('mean', 0.0):.2f} | ", end="")
            print(f"Median: {stats.get('median', 0.0):.2f} | ", end="")
            print(f"StdDev: {stats.get('std_dev', 0.0):.2f}")
            print(f"Best: {stats.get('best', 0.0):.2f} | ", end="")
            print(f"Worst: {stats.get('worst', 0.0):.2f}")
            
            # Window stats
            window_stats = stats.get('window_stats', {})
            if window_stats.get('mean') is not None:
                print(f"Recent (n={window_stats.get('count', 0)}): ", end="")
                print(f"Mean: {window_stats.get('mean', 0.0):.2f} | ", end="")
                print(f"Median: {window_stats.get('median', 0.0):.2f}")
    
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
        
        # Get context from stdin if no context is provided
        if not hasattr(args, 'context'):
            args.context = None
            
        if not hasattr(args, 'context_file'):
            args.context_file = None
            
        # Get context from file if specified
        context = args.context
        if args.context_file:
            try:
                with open(args.context_file, 'r') as f:
                    context = f.read()
            except Exception as e:
                print(f"Error reading context file: {e}")
                return 1
                
        # Check if we should read from stdin
        if context is None and not sys.stdin.isatty():
            try:
                context = sys.stdin.read()
                print(f"Read context from stdin: {context[:50]}{'...' if len(context) > 50 else ''}")
            except Exception as e:
                print(f"Error reading from stdin: {e}")
        
        try:
            # Show startup banner
            print("LLM AGENT EVOLUTION")
            
            # Configuration summary - more concise
            print(f"Population: {args.population_size} | Parallel: {args.parallel_agents} | Model: {args.model}")
            print(f"Using {'mock' if args.use_mock else 'real'} LLM | Max evals: {args.max_evaluations or 'unlimited'}")
            if args.eval_command:
                print(f"Eval command: {args.eval_command}")
            if args.seed is not None:
                print(f"Random seed: {args.seed}")
            if context:
                print(f"Context: {context[:50]}{'...' if len(context) > 50 else ''}")
            if args.load:
                print(f"Loading agent from: {args.load}")
            
            print("\nStarting evolution process...")
            start_time = time.time()
            
            # Set context in environment if provided
            if context:
                os.environ['AGENT_CONTEXT'] = context
            
            # Progress tracking with rate calculation
            last_count = 0
            last_print_time = time.time()
            
            def progress_callback(current_count, max_count=None):
                nonlocal last_count, last_print_time
                now = time.time()
                
                # Only print progress every 5 seconds to avoid flooding the console
                if now - last_print_time >= 5:
                    elapsed = now - last_print_time
                    evals_since_last = current_count - last_count
                    
                    # Calculate rate (evals per second)
                    rate = evals_since_last / elapsed if elapsed > 0 else 0
                    
                    # Format progress message - simplified to reduce output volume
                    if args.max_evaluations:
                        percent = current_count / args.max_evaluations * 100
                        print(f"Progress: {current_count:,}/{args.max_evaluations:,} ({percent:.1f}%)")
                    else:
                        print(f"Progress: {current_count:,} evals | Time: {(now - start_time)/60:.1f} min")
                    
                    last_count = current_count
                    last_print_time = now
            
            # Load agent if specified
            initial_population = None
            if hasattr(args, 'load') and args.load:
                try:
                    import tomli
                    with open(args.load, 'rb') as f:
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
                        initial_population = [loaded_agent]
                        print(f"Loaded agent with ID: {loaded_agent.id}")
                        
                        # If we're just running inference with a loaded agent
                        if args.eval_command and not args.max_evaluations:
                            # Evaluate the loaded agent
                            reward = self.evolution_use_case.evaluate_agent(loaded_agent)
                            print(f"\nAgent evaluation complete")
                            print(f"Reward: {reward}")
                            print(f"\nAgent output: {loaded_agent.task_chromosome.content}")
                            return 0
                except Exception as e:
                    print(f"Error loading agent: {e}")
                    return 1
            
            # Run evolution
            population = self.evolution_use_case.run_evolution(
                population_size=args.population_size,
                parallel_agents=args.parallel_agents,
                max_evaluations=args.max_evaluations,
                progress_callback=progress_callback,
                initial_population=initial_population
            )
            
            # Calculate total runtime
            total_runtime = time.time() - start_time
            
            # Get and display final statistics
            stats = self.evolution_use_case.get_population_stats(population)
            self.display_stats(stats)
            
            # Display best agent
            best_agent = max(population, key=lambda a: a.reward if a.reward is not None else float('-inf'))
            
            print("\nBEST AGENT")
            print(f"ID: {best_agent.id} | Reward: {best_agent.reward}")
            
            # Show task chromosome with character count
            task_content = best_agent.task_chromosome.content
            print(f"Task Chromosome ({len(task_content)} chars):")
            print(task_content)
            
            # Save agent if requested
            if hasattr(args, 'save') and args.save:
                try:
                    import tomli_w
                    agent_data = {
                        "agent": {
                            "id": best_agent.id,
                            "reward": best_agent.reward,
                            "task_chromosome": {
                                "content": best_agent.task_chromosome.content,
                                "type": best_agent.task_chromosome.type
                            },
                            "mate_selection_chromosome": {
                                "content": best_agent.mate_selection_chromosome.content,
                                "type": best_agent.mate_selection_chromosome.type
                            },
                            "mutation_chromosome": {
                                "content": best_agent.mutation_chromosome.content,
                                "type": best_agent.mutation_chromosome.type
                            }
                        }
                    }
                    
                    with open(args.save, 'wb') as f:
                        tomli_w.dump(agent_data, f)
                    print(f"Best agent saved to: {args.save}")
                except Exception as e:
                    print(f"Error saving agent: {e}")
            
            # Show summary
            print(f"\nEvolution completed in {total_runtime/60:.2f} minutes")
            print(f"Total evaluations: {stats.get('count', 0):,}")
            print(f"Final population size: {stats.get('population_size', 0):,}")
            print(f"Best reward achieved: {stats.get('best', 0):.2f}")
            
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
    from llm_agent_evolution.application import create_application
    
    # Create the application and run it
    cli = create_application()
    return cli.run()

if __name__ == "__main__":
    sys.exit(main())
