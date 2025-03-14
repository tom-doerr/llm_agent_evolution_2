"""
Main entry point for running the LLM Agent Evolution package directly
"""
import sys
from .cli import main

def run_main():
    """Run the main application"""
    # Run the main CLI function
    return main()

if __name__ == "__main__":
    sys.exit(run_main())
