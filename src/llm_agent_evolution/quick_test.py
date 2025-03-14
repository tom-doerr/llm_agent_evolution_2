"""
Quick test module for LLM Agent Evolution
"""
import sys
import os
import argparse
import tempfile
from .application import create_application

def main(seed=42, log_file=None):
    """Run a quick test with mock LLM adapter"""
    # Set environment variables for consistent behavior
    os.environ["USE_MOCK"] = "1"
    os.environ["POPULATION_SIZE"] = "20"
    os.environ["PARALLEL_AGENTS"] = "4"
    os.environ["MAX_EVALUATIONS"] = "100"
    if seed:
        os.environ["RANDOM_SEED"] = str(seed)
    
    # Create a valid log file path in a directory that should be writable
    if not log_file:
        log_file = os.path.join(tempfile.gettempdir(), "quick_test.log")
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Create application with mock adapter
    cli = create_application(use_mock=True, random_seed=seed, log_file=log_file)
    
    # Override arguments for quick test
    args = argparse.Namespace()
    args.population_size = 20
    args.parallel_agents = 4
    args.max_evaluations = 100
    args.use_mock = True
    args.quick_test = True
    args.seed = seed
    args.log_file = log_file
    args.model = "openrouter/google/gemini-2.0-flash-001"  # Add model attribute
    args.eval_command = None  # Add eval_command attribute
    args.load = None  # Add load attribute
    args.context = None  # Add context attribute
    args.context_file = None  # Add context_file attribute
    args.initial_content = ""  # Add initial_content attribute
    args.verbose = False  # Add verbose attribute
    args.save = None  # Add save attribute
    
    # Store original parse_args method
    original_parse_args = cli.parse_args
    
    # Override parse_args to return our fixed arguments
    cli.parse_args = lambda: args
    
    # Run the application
    return cli.run()

if __name__ == "__main__":
    sys.exit(main())
