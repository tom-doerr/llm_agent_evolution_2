#!/bin/bash
# Simple wrapper script for running LLM Agent Evolution

# Add the current directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Function to show usage
function show_usage {
    echo "Usage: ./evolve.sh [command] [options]"
    echo ""
    echo "Commands:"
    echo "  quick-test    Run a quick test with mock LLM"
    echo "  run           Run the evolution process"
    echo ""
    echo "For more options, run: ./evolve.sh [command] --help"
}

# Check if a command was provided
if [ $# -eq 0 ]; then
    show_usage
    exit 1
fi

# Run the appropriate command
case "$1" in
    quick-test)
        shift
        python -m llm_agent_evolution.quick_test "$@"
        ;;
    run)
        shift
        python -m llm_agent_evolution "$@"
        ;;
    *)
        echo "Unknown command: $1"
        show_usage
        exit 1
        ;;
esac
