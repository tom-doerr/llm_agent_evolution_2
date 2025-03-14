#!/usr/bin/env python3
"""
Script to run LLM Agent Evolution with DSPy
"""
import sys
import os
import argparse

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

def main():
    """Run LLM Agent Evolution with DSPy"""
    parser = argparse.ArgumentParser(description="Run LLM Agent Evolution with DSPy")
    parser.add_argument("--population-size", type=int, default=50, help="Population size")
    parser.add_argument("--parallel-agents", type=int, default=4, help="Number of parallel agents")
    parser.add_argument("--max-evaluations", type=int, default=1000, help="Maximum evaluations")
    parser.add_argument("--model", default="openrouter/google/gemini-2.0-flash-001", help="DSPy model name")
    parser.add_argument("--log-file", default="dspy_evolution.log", help="Log file path")
    
    args = parser.parse_args()
    
    # Print banner
    print("LLM Agent Evolution with DSPy")
    print(f"Population size: {args.population_size}")
    print(f"Parallel agents: {args.parallel_agents}")
    print(f"Max evaluations: {args.max_evaluations}")
    print(f"Model: {args.model}")
    print(f"Log file: {args.log_file}")
    
    # Confirm before running with real LLM
    response = input("\nThis will make real API calls to the LLM. Continue? (y/n): ")
    if not response.lower().startswith('y'):
        print("Aborted.")
        return 1
    
    # Set environment variables
    os.environ["POPULATION_SIZE"] = str(args.population_size)
    os.environ["PARALLEL_AGENTS"] = str(args.parallel_agents)
    os.environ["MAX_EVALUATIONS"] = str(args.max_evaluations)
    os.environ["MODEL"] = args.model
    os.environ["LOG_FILE"] = args.log_file
    
    # Import and run the application
    from llm_agent_evolution.application import main
    return main()

if __name__ == "__main__":
    sys.exit(main())
