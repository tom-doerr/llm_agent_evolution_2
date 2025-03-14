#!/usr/bin/env python3
"""
Quick test script for LLM Agent Evolution
"""
import sys
from src.llm_agent_evolution.application import create_application

def main():
    """Run a quick test with mock LLM adapter"""
    # Create application with mock adapter
    cli = create_application(use_mock=True, random_seed=42)
    
    # Override arguments for quick test
    import argparse
    args = argparse.Namespace()
    args.population_size = 20
    args.parallel_agents = 4
    args.max_evaluations = 100
    args.use_mock = True
    args.quick_test = True
    
    # Store original parse_args method
    original_parse_args = cli.parse_args
    
    # Override parse_args to return our fixed arguments
    cli.parse_args = lambda: args
    
    # Run the application
    return cli.run()

if __name__ == "__main__":
    sys.exit(main())
