#!/usr/bin/env python3
"""
Simple runner script for LLM Agent Evolution
"""
import sys
import os
import argparse

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def main():
    """Parse arguments and run the appropriate command"""
    parser = argparse.ArgumentParser(description="LLM Agent Evolution Runner")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Quick test command
    quick_test_parser = subparsers.add_parser("quick-test", help="Run a quick test with mock LLM")
    quick_test_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run the evolution process")
    run_parser.add_argument("--population-size", type=int, default=100, help="Population size")
    run_parser.add_argument("--parallel-agents", type=int, default=10, help="Number of parallel agents")
    run_parser.add_argument("--max-evaluations", type=int, default=None, help="Maximum evaluations")
    run_parser.add_argument("--use-mock", action="store_true", help="Use mock LLM")
    run_parser.add_argument("--model", default="openrouter/google/gemini-2.0-flash-001", help="LLM model name")
    run_parser.add_argument("--log-file", default="evolution.log", help="Log file path")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    if args.command == "quick-test":
        from src.llm_agent_evolution.quick_test import main as quick_test_main
        return quick_test_main(seed=args.seed)
    elif args.command == "run":
        # Set environment variables for the application
        os.environ["POPULATION_SIZE"] = str(args.population_size)
        os.environ["PARALLEL_AGENTS"] = str(args.parallel_agents)
        if args.max_evaluations:
            os.environ["MAX_EVALUATIONS"] = str(args.max_evaluations)
        if args.use_mock:
            os.environ["USE_MOCK"] = "1"
        os.environ["MODEL"] = args.model
        os.environ["LOG_FILE"] = args.log_file
        
        # Import and run the application
        from src.llm_agent_evolution.application import main as app_main
        return app_main()
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())
