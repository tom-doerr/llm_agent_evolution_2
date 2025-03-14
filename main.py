#!/usr/bin/env python3
"""
LLM Agent Evolution - Main entry point
"""
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from llm_agent_evolution.application import main

if __name__ == "__main__":
    sys.exit(main())
