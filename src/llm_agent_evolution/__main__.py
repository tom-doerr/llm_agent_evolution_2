"""
Main entry point for running the LLM Agent Evolution package directly
"""
import sys
import os
from .application import main

def run_main():
    """Run the main application with environment variables"""
    # Print banner
    print("=" * 60)
    print("LLM Agent Evolution")
    print("A framework for evolving LLM-based agents")
    print("=" * 60)
    
    # Show configuration from environment
    print("\nConfiguration:")
    print(f"- Population size: {os.environ.get('POPULATION_SIZE', '100')}")
    print(f"- Parallel agents: {os.environ.get('PARALLEL_AGENTS', '10')}")
    print(f"- Max evaluations: {os.environ.get('MAX_EVALUATIONS', 'unlimited')}")
    print(f"- Using {'mock' if os.environ.get('USE_MOCK') == '1' else 'real'} LLM")
    print(f"- Model: {os.environ.get('MODEL', 'openrouter/google/gemini-2.0-flash-001')}")
    print(f"- Log file: {os.environ.get('LOG_FILE', 'evolution.log')}")
    print("\nStarting evolution...\n")
    
    # Handle subcommands in sys.argv
    if len(sys.argv) > 1 and sys.argv[1] in ["evolve", "standalone", "optimize", "demo"]:
        # Remove the subcommand to avoid unrecognized argument error
        sys.argv.remove(sys.argv[1])
    
    # Run the main application
    return main()

if __name__ == "__main__":
    sys.exit(run_main())
