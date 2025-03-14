#!/usr/bin/env python3
"""
Script to run LLM Agent Evolution with DSPy
"""
import sys
import os
import argparse
from rich.console import Console

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

def main():
    """Run LLM Agent Evolution with DSPy"""
    console = Console()
    
    parser = argparse.ArgumentParser(description="Run LLM Agent Evolution with DSPy")
    parser.add_argument("--population-size", type=int, default=50, help="Population size")
    parser.add_argument("--parallel-agents", type=int, default=4, help="Number of parallel agents")
    parser.add_argument("--max-evaluations", type=int, default=1000, help="Maximum evaluations")
    parser.add_argument("--model", default="openrouter/google/gemini-2.0-flash-001", help="DSPy model name")
    parser.add_argument("--log-file", default="dspy_evolution.log", help="Log file path")
    
    args = parser.parse_args()
    
    # Print banner
    console.print("[bold green]LLM Agent Evolution with DSPy[/bold green]")
    console.print(f"Population size: [cyan]{args.population_size}[/cyan]")
    console.print(f"Parallel agents: [cyan]{args.parallel_agents}[/cyan]")
    console.print(f"Max evaluations: [cyan]{args.max_evaluations}[/cyan]")
    console.print(f"Model: [cyan]{args.model}[/cyan]")
    console.print(f"Log file: [cyan]{args.log_file}[/cyan]")
    
    # Confirm before running with real LLM
    if not console.input("\n[yellow]This will make real API calls to the LLM. Continue? (y/n): [/yellow]").lower().startswith('y'):
        console.print("[red]Aborted.[/red]")
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
