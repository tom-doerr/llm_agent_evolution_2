"""
Main entry point for running the LLM Agent Evolution package directly
"""
import sys
import os

# Add the parent directory to the path if running as script
if __name__ == "__main__" and __package__ is None:
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    from llm_agent_evolution.cli import main
else:
    from .cli import main

if __name__ == "__main__":
    sys.exit(main())
