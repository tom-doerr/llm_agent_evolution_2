"""
Command-line interface for the LLM Agent Evolution package
"""
import sys
import os
import argparse

def main():
    """Main CLI entry point"""
    # Create the argument parser
    parser = _create_main_parser()
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Add subcommand parsers
    _add_evolve_subparser(subparsers)
    _add_optimize_subparser(subparsers)
    _add_standalone_subparser(subparsers)
    _add_demo_subparser(subparsers)
    
    # Parse arguments and handle commands
    args = parser.parse_args()
    
    # Handle default command
    if not args.command:
        args.command = "evolve"
        # This allows running without the explicit 'evolve' subcommand
    
    # Remove subcommand from sys.argv if it's 'evolve' to avoid unrecognized argument error
    if args.command == "evolve" and len(sys.argv) > 1 and sys.argv[1] == "evolve":
        sys.argv.remove("evolve")
    
    # Dispatch to the appropriate command handler
    if args.command == "evolve":
        return _handle_evolve_command(args)
    elif args.command == "optimize":
        return _handle_optimize_command(args)
    elif args.command == "standalone":
        return _handle_standalone_command(args)
    elif args.command == "demo":
        return _handle_demo_command(args)
    else:
        parser.print_help()
        return 1
def _create_main_parser():
    """Create the main argument parser"""
    parser = argparse.ArgumentParser(
        description="LLM Agent Evolution - Command Line Interface",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add eval_command as an optional positional argument to the main parser
    parser.add_argument(
        "eval_command",
        nargs="?",
        help="Command to run for evaluation (receives agent output via stdin, returns score as last line)"
    )
    
    # Add the main arguments to the top-level parser
    _add_common_arguments(parser)
    
    return parser

def _add_common_arguments(parser):
    """Add common arguments to a parser"""
    parser.add_argument(
        "--population-size", "-p",
        type=int, 
        default=100,
        help="Initial population size"
    )
    
    parser.add_argument(
        "--parallel-agents", "-j",
        type=int, 
        default=10,
        help="Number of agents to evaluate in parallel"
    )
    
    parser.add_argument(
        "--max-evaluations", "-n",
        type=int, 
        default=None,
        help="Maximum number of evaluations to run"
    )
    
    parser.add_argument(
        "--use-mock", "--mock",
        action="store_true",
        help="Use mock LLM adapter for testing"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str, 
        default="openrouter/google/gemini-2.0-flash-001",
        help="LLM model to use"
    )
    
    parser.add_argument(
        "--log-file",
        type=str, 
        default="evolution.log",
        help="Log file path"
    )
    
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--eval-command", "-e",
        type=str,
        default=None,
        help="Command to run for evaluation"
    )
    
    parser.add_argument(
        "--quick-test", "-q",
        action="store_true",
        help="Run a quick test with mock LLM"
    )
    
    parser.add_argument(
        "--save", "-o",
        type=str,
        default=None,
        help="File to save the best result to"
    )
    
    parser.add_argument(
        "--load", "-l",
        type=str,
        default=None,
        help="Load a previously saved agent from file"
    )
    
    parser.add_argument(
        "--context", "-c",
        type=str,
        default=None,
        help="Context to pass to the agent (available as AGENT_CONTEXT environment variable)"
    )
    
    parser.add_argument(
        "--context-file", "-cf",
        type=str,
        default=None,
        help="File containing context to pass to the agent"
    )

def _add_evolve_subparser(subparsers):
    """Add the evolve subparser"""
    evolve_parser = subparsers.add_parser(
        "evolve", 
        help="Run the main evolution process"
    )
    
    # Add arguments for the evolve command
    _add_common_arguments(evolve_parser)
    
def _add_optimize_subparser(subparsers):
    """Add the optimize subparser"""
    optimize_parser = subparsers.add_parser(
        "optimize", 
        help="Run the universal optimizer"
    )
    
    # Add arguments for the optimize command
    optimize_parser.add_argument(
        "eval_command",
        nargs="?",
        help="Evaluation command (receives agent output via stdin, returns score as last line)"
    )
    
    optimize_parser.add_argument(
        "--eval-script", 
        help="Path to the evaluation script (alternative to eval_command)"
    )
    
    # Add common arguments
    _add_common_arguments(optimize_parser)
    
    # Add optimize-specific arguments
    optimize_parser.add_argument(
        "--script-timeout", "-t",
        type=int,
        default=30,
        help="Maximum execution time for the evaluation script in seconds"
    )
    
    optimize_parser.add_argument(
        "--initial-content", "-i",
        type=str,
        default="",
        help="Initial content for the chromosomes"
    )
    
    optimize_parser.add_argument(
        "--initial-file", "-f",
        type=str,
        default=None,
        help="File containing initial content for the chromosomes"
    )
    
    optimize_parser.add_argument(
        "--output-format",
        type=str,
        choices=["text", "toml"],
        default="text",
        help="Output format (text or TOML)"
    )
    
    optimize_parser.add_argument(
        "--max-chars",
        type=int,
        default=1000,
        help="Maximum number of characters for chromosomes"
    )
    
    optimize_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose mode with detailed output"
    )
    
def _add_standalone_subparser(subparsers):
    """Add the standalone subparser"""
    standalone_parser = subparsers.add_parser(
        "standalone", 
        help="Run the simplified standalone optimizer (no LLM API calls)"
    )
    
    standalone_parser.add_argument(
        "eval_command",
        help="Evaluation command (receives agent output via stdin, returns score as last line)"
    )
    
    standalone_parser.add_argument(
        "--population-size", "-p",
        type=int, 
        default=50,
        help="Initial population size"
    )
    
    standalone_parser.add_argument(
        "--parallel-agents", "-j",
        type=int, 
        default=8,
        help="Number of agents to evaluate in parallel"
    )
    
    standalone_parser.add_argument(
        "--max-evaluations", "-n",
        type=int, 
        default=1000,
        help="Maximum number of evaluations to run"
    )
    
    standalone_parser.add_argument(
        "--initial-content", "-i",
        type=str,
        default="",
        help="Initial content for the chromosomes"
    )
    
    standalone_parser.add_argument(
        "--seed", "-s",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    standalone_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    standalone_parser.add_argument(
        "--save", "-o",
        type=str,
        default=None,
        help="File to save the best result to"
    )

def _add_demo_subparser(subparsers):
    """Add the demo subparser"""
    demo_parser = subparsers.add_parser(
        "demo", 
        help="Run the evolution demo with detailed step-by-step output"
    )
    
    demo_parser.add_argument(
        "--use-mock", "--mock",
        action="store_true",
        help="Use mock LLM instead of real LLM"
    )
    
    demo_parser.add_argument(
        "--initial-content", "-i",
        type=str,
        default="a",
        help="Initial content for task chromosome"
    )
    
    # Parse arguments and handle commands
    args = parser.parse_args()
    
    # Handle default command
    if not args.command:
        args.command = "evolve"
        # This allows running without the explicit 'evolve' subcommand
    
    # Remove subcommand from sys.argv if it's 'evolve' to avoid unrecognized argument error
    if args.command == "evolve" and len(sys.argv) > 1 and sys.argv[1] == "evolve":
        sys.argv.remove("evolve")
    
    # Dispatch to the appropriate command handler
    if args.command == "evolve":
        return _handle_evolve_command(args)
    elif args.command == "optimize":
        return _handle_optimize_command(args)
    elif args.command == "standalone":
        return _handle_standalone_command(args)
    elif args.command == "demo":
        return _handle_demo_command(args)
    else:
        parser.print_help()
        return 1

def _handle_evolve_command(args):
    """Handle the evolve command"""
    # Import and run the main application
    from llm_agent_evolution.application import create_application
    
    # Create the application
    cli = create_application(
        model_name=args.model,
        log_file=args.log_file,
        use_mock=args.use_mock,
        random_seed=args.seed,
        eval_command=args.eval_command
    )
    
    # Override arguments
    cli_args = argparse.Namespace()
    cli_args.population_size = args.population_size
    cli_args.parallel_agents = args.parallel_agents
    cli_args.max_evaluations = args.max_evaluations
    cli_args.use_mock = args.use_mock
    cli_args.quick_test = args.quick_test
    cli_args.seed = args.seed
    cli_args.model = args.model
    cli_args.log_file = args.log_file
    cli_args.eval_command = args.eval_command
    cli_args.save = args.save if hasattr(args, 'save') else None
    cli_args.load = args.load if hasattr(args, 'load') else None
    cli_args.context = args.context if hasattr(args, 'context') else None
    cli_args.context_file = args.context_file if hasattr(args, 'context_file') else None
    
    # Store original parse_args method
    original_parse_args = cli.parse_args
    
    # Override parse_args to return our fixed arguments
    cli.parse_args = lambda: cli_args
    
    # Run the application
    return cli.run()
    
def _handle_optimize_command(args):
    """Handle the optimize command"""
    # Import the universal optimizer
    from llm_agent_evolution.universal_optimize import run_optimizer
    
    # Get initial content from file if specified
    initial_content = args.initial_content
    if args.initial_file:
        if not os.path.exists(args.initial_file):
            print(f"Error: Initial content file not found: {args.initial_file}")
            return 1
        with open(args.initial_file, 'r') as f:
            initial_content = f.read()
    
    # Determine evaluation method
    eval_script = args.eval_script
    eval_command = args.eval_command
    
    if not eval_script and not eval_command:
        print("Error: Either eval_command or --eval-script must be specified")
        return 1
        
    # If both are provided, eval_command takes precedence
    if eval_command and not eval_script:
        # Create a temporary script that runs the eval command
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.sh') as f:
            eval_script = f.name
            f.write("#!/bin/sh\n")
            f.write(f"{eval_command}\n")
        
        # Make executable
        os.chmod(eval_script, 0o755)
    
    try:
        # Run the optimizer
        result = run_optimizer(
            eval_script=eval_script,
            population_size=args.population_size,
            parallel_agents=args.parallel_agents,
            max_evaluations=args.max_evaluations,
            use_mock_llm=args.use_mock,  # Use the common flag name
            model_name=args.model,
            log_file=args.log_file,
            random_seed=args.seed,
            script_timeout=args.script_timeout,
            initial_content=initial_content,
            output_file=args.save,  # Use the common save parameter
            output_format=args.output_format,
            max_chars=args.max_chars,
            verbose=args.verbose,
            eval_command=eval_command
        )
        
        return result
    finally:
        # Clean up temporary script if created
        if eval_command and not args.eval_script and os.path.exists(eval_script):
            os.remove(eval_script)

def _handle_standalone_command(args):
    """Handle the standalone command"""
    # Import the standalone optimizer
    from llm_agent_evolution.standalone import run_standalone_optimizer
    
    # Run the standalone optimizer
    try:
        results = run_standalone_optimizer(
            eval_command=args.eval_command,
            population_size=args.population_size,
            parallel_agents=args.parallel_agents,
            max_evaluations=args.max_evaluations,
            initial_content=args.initial_content,
            random_seed=args.seed,
            verbose=args.verbose
        )
        
        # Write to save file if specified
        if args.save and results["best_agent"]["content"]:
            with open(args.save, 'w') as f:
                f.write(results["best_agent"]["content"])
            print(f"\nBest result saved to: {args.save}")
            
        return 0
    except Exception as e:
        print(f"Error running standalone optimizer: {e}")
        import traceback
        print(traceback.format_exc())
        return 1

def _handle_demo_command(args):
    """Handle the demo command"""
    # Import the evolution demo
    from llm_agent_evolution.evolution_demo import run_evolution_demo
    
    # Run the demo
    return run_evolution_demo(
        use_mock=args.use_mock,
        initial_content=args.initial_content
    )

if __name__ == "__main__":
    sys.exit(main())
