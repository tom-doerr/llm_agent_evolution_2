#!/usr/bin/env python3
"""
Command-line interface for the LLM Agent Evolution package
"""
import sys
import os
import argparse

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="LLM Agent Evolution - Command Line Interface"
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Optimize command
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
    
    optimize_parser.add_argument(
        "--population-size", 
        type=int, 
        default=50,
        help="Initial population size (default: 50)"
    )
    
    optimize_parser.add_argument(
        "--parallel-agents", 
        type=int, 
        default=8,
        help="Number of agents to evaluate in parallel (default: 8)"
    )
    
    optimize_parser.add_argument(
        "--max-evaluations", 
        type=int, 
        default=None,
        help="Maximum number of evaluations to run (default: unlimited)"
    )
    
    optimize_parser.add_argument(
        "--use-mock-llm",
        action="store_true",
        help="Use mock LLM adapter for testing"
    )
    
    optimize_parser.add_argument(
        "--model", 
        type=str, 
        default="openrouter/google/gemini-2.0-flash-001",
        help="LLM model to use (default: openrouter/google/gemini-2.0-flash-001)"
    )
    
    optimize_parser.add_argument(
        "--log-file", 
        type=str, 
        default="universal_optimize.log",
        help="Log file path (default: universal_optimize.log)"
    )
    
    optimize_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    optimize_parser.add_argument(
        "--script-timeout",
        type=int,
        default=30,
        help="Maximum execution time for the evaluation script in seconds (default: 30)"
    )
    
    optimize_parser.add_argument(
        "--initial-content",
        type=str,
        default="",
        help="Initial content for the chromosomes"
    )
    
    optimize_parser.add_argument(
        "--initial-file",
        type=str,
        default=None,
        help="File containing initial content for the chromosomes"
    )
    
    optimize_parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="File to write the best result to"
    )
    
    optimize_parser.add_argument(
        "--output-format",
        type=str,
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)"
    )
    
    optimize_parser.add_argument(
        "--max-chars",
        type=int,
        default=1000,
        help="Maximum number of characters for chromosomes (default: 1000)"
    )
    
    optimize_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose mode with detailed output of each evolution step"
    )
    
    # Demo command
    demo_parser = subparsers.add_parser(
        "demo", 
        help="Run the evolution demo with detailed step-by-step output"
    )
    
    demo_parser.add_argument(
        "--use-mock",
        action="store_true",
        help="Use mock LLM instead of real LLM"
    )
    
    demo_parser.add_argument(
        "--initial-content",
        type=str,
        default="a",
        help="Initial content for task chromosome"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command == "optimize":
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
        
        # Run the optimizer
        result = run_optimizer(
            eval_script=eval_script,
            population_size=args.population_size,
            parallel_agents=args.parallel_agents,
            max_evaluations=args.max_evaluations,
            use_mock_llm=args.use_mock_llm,
            model_name=args.model,
            log_file=args.log_file,
            random_seed=args.seed,
            script_timeout=args.script_timeout,
            initial_content=initial_content,
            output_file=args.output_file,
            output_format=args.output_format,
            max_chars=args.max_chars,
            verbose=args.verbose,
            eval_command=eval_command
        )
        
        # Clean up temporary script if created
        if eval_command and not args.eval_script and os.path.exists(eval_script):
            os.remove(eval_script)
            
        return result
    
    elif args.command == "demo":
        # Import the evolution demo
        from llm_agent_evolution.evolution_demo import run_evolution_demo
        
        # Run the demo
        return run_evolution_demo(
            use_mock=args.use_mock,
            initial_content=args.initial_content
        )
    
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())
