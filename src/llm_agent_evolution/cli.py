import sys
import os
import argparse
import tempfile
from typing import List, Optional, Tuple

from llm_agent_evolution.evolution import run_optimizer, load_agent, evaluate_agent_with_command

def main(args: Optional[List[str]] = None) -> int:
    try:
        # Parse arguments
        parser = _create_main_parser()
        parsed_args = parser.parse_args(args)
        
        # Process context from various sources
        context = None
        
        # Get context from file if specified
        if parsed_args.context_file:
            try:
                with open(parsed_args.context_file, 'r') as f:
                    context = f.read()
            except Exception as e:
                print(f"Error reading context file: {e}")
                return 1
                
        # Use context from argument if provided
        if parsed_args.context:
            context = parsed_args.context
            
        # Check if we should read from stdin
        if context is None and not sys.stdin.isatty():
            try:
                context = sys.stdin.read()
                print(f"Read context from stdin: {context[:50]}{'...' if len(context) > 50 else ''}")
            except Exception as e:
                print(f"Error reading from stdin: {e}")
        
        if context:
            os.environ['AGENT_CONTEXT'] = context
        
        # Handle quick test mode
        if parsed_args.quick_test:
            from llm_agent_evolution.quick_test import main as run_quick_test
            return run_quick_test(seed=parsed_args.seed, log_file=parsed_args.log_file)
        
        # Handle loaded agent mode
        if parsed_args.load and parsed_args.eval_command:
            agent = load_agent(parsed_args.load)
            if not agent:
                print(f"Error: Could not load agent from {parsed_args.load}")
                return 1
                
            print(f"Loaded agent with ID: {agent.id}")
            reward = evaluate_agent_with_command(agent, parsed_args.eval_command, context)
            return 0
        
        # Handle optimization mode
        if parsed_args.eval_command:
            result = run_optimizer(
                eval_command=parsed_args.eval_command,
                population_size=parsed_args.population_size,
                parallel_agents=parsed_args.parallel_agents,
                max_evaluations=parsed_args.max_evaluations,
                use_mock_llm=parsed_args.use_mock,
                model_name=parsed_args.model,
                initial_content=parsed_args.initial_content,
                verbose=parsed_args.verbose,
                random_seed=parsed_args.seed,
                log_file=parsed_args.log_file
            )
            
            # Save the best agent if requested
            if parsed_args.save and result["best_agent"]["content"]:
                from llm_agent_evolution.domain.model import Agent, Chromosome
                
                best_agent = Agent(
                    task_chromosome=Chromosome(content=result["best_agent"]["content"], type="task"),
                    mate_selection_chromosome=Chromosome(content="", type="mate_selection"),
                    mutation_chromosome=Chromosome(content="", type="mutation"),
                    id=result["best_agent"]["id"],
                    reward=result["best_agent"]["reward"]
                )
                
                if save_agent(best_agent, parsed_args.save):
                    print(f"\nBest agent saved to: {parsed_args.save}")
            
            return 0
        
        # If no specific action, show help
        parser.print_help()
        return 1
            
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        print(traceback.format_exc())
        return 1

def _create_parser_and_parse_args(args: Optional[List[str]]) -> Tuple[argparse.ArgumentParser, argparse.Namespace]:
    # Create the argument parser
    parser = _create_main_parser()
    
    # Check for positional eval_command
    eval_command = None
    if args is None and len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        eval_command = sys.argv[1]
        sys.argv.pop(1)
    
    # Parse arguments
    parsed_args = parser.parse_args(args)
    
    # If we found a positional eval_command, add it to the parsed args
    if eval_command:
        parsed_args.eval_command = eval_command
    
    # Handle quick test mode
    if parsed_args.quick_test:
        parsed_args.use_mock = True
        if not parsed_args.max_evaluations:
            parsed_args.max_evaluations = 100
        parsed_args.population_size = 20
        parsed_args.parallel_agents = 4
        if parsed_args.seed is None:
            parsed_args.seed = 42
    
    return parser, parsed_args

def _get_context(args: argparse.Namespace) -> Optional[str]:
    # Get context from file if specified
    context = args.context
    if args.context_file:
        try:
            with open(args.context_file, 'r') as f:
                context = f.read()
        except Exception as e:
            print(f"Error reading context file: {e}")
            return None
    
    # Check if we should read from stdin
    if context is None and not sys.stdin.isatty():
        try:
            context = sys.stdin.read()
            print(f"Read context from stdin: {context[:50]}{'...' if len(context) > 50 else ''}")
        except Exception as e:
            print(f"Error reading from stdin: {e}")
    
    return context

def _run_loaded_agent(args: argparse.Namespace, context: Optional[str]) -> int:
    agent = load_agent(args.load)
    if not agent:
        print(f"Error: Could not load agent from {args.load}")
        return 1
        
    print(f"Loaded agent with ID: {agent.id}")
    print(f"Agent content: {agent.task_chromosome.content[:50]}...")
    
    reward = evaluate_agent_with_command(agent, args.eval_command, context)
    print(f"\nAgent output: {agent.task_chromosome.content}")
    return 0

def _run_optimizer(args: argparse.Namespace, context: Optional[str]) -> int:
    result = run_optimizer(
        eval_command=args.eval_command,
        population_size=args.population_size,
        parallel_agents=args.parallel_agents,
        max_evaluations=args.max_evaluations,
        use_mock_llm=args.use_mock,
        model_name=args.model,
        initial_content=args.initial_content,
        verbose=args.verbose,
        random_seed=args.seed
    )
    
    # Save the best agent if requested
    if args.save and result["best_agent"]["content"]:
        from llm_agent_evolution.evolution import save_agent
        from llm_agent_evolution.domain.model import Agent, Chromosome
        
        best_agent = Agent(
            task_chromosome=Chromosome(content=result["best_agent"]["content"], type="task"),
            mate_selection_chromosome=Chromosome(content="", type="mate_selection"),
            mutation_chromosome=Chromosome(content="", type="mutation"),
            id=result["best_agent"]["id"],
            reward=result["best_agent"]["reward"]
        )
        
        if save_agent(best_agent, args.save):
            print(f"\nBest agent saved to: {args.save}")
    
    return 0

def _run_quick_test(args: argparse.Namespace) -> int:
    from llm_agent_evolution.quick_test import main as run_quick_test
    return run_quick_test(seed=args.seed)
def _create_main_parser():
    """Create the main argument parser"""
    parser = argparse.ArgumentParser(
        description="LLM Agent Evolution - Evolve LLM agents through natural selection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add the main arguments to the parser
    parser.add_argument("--population-size", "-p", type=int, default=100,
                       help="Initial population size")
    parser.add_argument("--parallel-agents", "-j", type=int, default=10,
                       help="Number of agents to evaluate in parallel")
    parser.add_argument("--max-evaluations", "-n", type=int, default=None,
                       help="Maximum number of evaluations to run")
    parser.add_argument("--model", "-m", type=str, default="openrouter/google/gemini-2.0-flash-001",
                       help="LLM model to use")
    parser.add_argument("--use-mock", "--mock", action="store_true",
                       help="Use mock LLM adapter for testing")
    parser.add_argument("--eval-command", "-e", type=str, default=None,
                       help="Command to run for evaluation")
    parser.add_argument("--load", "-l", type=str, default=None,
                       help="Load a previously saved agent from file")
    parser.add_argument("--save", "-o", type=str, default=None,
                       help="File to save the best result to")
    parser.add_argument("--context", "-c", type=str, default=None,
                       help="Context to pass to the agent")
    parser.add_argument("--context-file", "-cf", type=str, default=None,
                       help="File containing context to pass to the agent")
    parser.add_argument("--initial-content", "-i", type=str, default="",
                       help="Initial content for the chromosomes")
    parser.add_argument("--seed", "-s", type=int, default=None,
                       help="Random seed for reproducibility")
    parser.add_argument("--quick-test", "-q", action="store_true",
                       help="Run a quick test with mock LLM")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--log-file", type=str, default="evolution.log",
                       help="Log file path")
    
    return parser

# These functions are no longer needed with the simplified CLI

if __name__ == "__main__":
    sys.exit(main())
