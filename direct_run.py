#!/usr/bin/env python3
"""
Direct runner script for LLM Agent Evolution that doesn't rely on imports
"""
import sys
import os
import argparse
import subprocess

def main():
    """Run the LLM Agent Evolution directly using Python module"""
    parser = argparse.ArgumentParser(description="LLM Agent Evolution Direct Runner")
    parser.add_argument("--quick-test", action="store_true", help="Run a quick test with mock LLM")
    parser.add_argument("--population-size", type=int, default=20, help="Population size")
    parser.add_argument("--parallel-agents", type=int, default=4, help="Number of parallel agents")
    parser.add_argument("--max-evaluations", type=int, default=100, help="Maximum evaluations")
    parser.add_argument("--use-mock", action="store_true", help="Use mock LLM")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set environment variables
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(os.path.dirname(__file__)) + ":" + env.get("PYTHONPATH", "")
    
    if args.quick_test or args.use_mock:
        env["USE_MOCK"] = "1"
    
    env["POPULATION_SIZE"] = str(args.population_size)
    env["PARALLEL_AGENTS"] = str(args.parallel_agents)
    env["MAX_EVALUATIONS"] = str(args.max_evaluations)
    env["RANDOM_SEED"] = str(args.seed)
    
    # Run the Python module directly
    cmd = [sys.executable, "-m", "llm_agent_evolution"]
    
    try:
        subprocess.run(cmd, env=env, check=True)
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
