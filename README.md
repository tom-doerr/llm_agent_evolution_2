# LLM Agent Evolution

Framework for evolving LLM-based agents through evolutionary algorithms.

## Overview

This project implements an evolutionary algorithm for LLM-based agents. The system evolves agents with three chromosomes:
- Task chromosome: The output that gets evaluated for fitness
- Mate selection chromosome: Instructions for selecting mates
- Mutation chromosome: Instructions for how to mutate chromosomes

The system uses a continuous evolution process rather than discrete generations, with parent selection using a Pareto distribution weighted by fitness squared.

## Installation

```bash
# Install the package in development mode
pip install -e .

# Install with test dependencies
pip install -e ".[test]"
```

## Usage

### Quick Test with Mock LLM

Run a quick test using the mock LLM adapter (no real API calls):

```bash
./evolve.sh quick-test
```

### Run with Real LLM

Run the evolution process with a real LLM:

```bash
./evolve.sh run --population-size 50 --parallel-agents 8 --model "openrouter/google/gemini-2.0-flash-001"
```

### Command Line Options

```
Usage: ./evolve.sh [command] [options]

Commands:
  quick-test    Run a quick test with mock LLM
  run           Run the evolution process

Options:
  --population-size INT    Initial population size (default: 100)
  --parallel-agents INT    Number of agents to evaluate in parallel (default: 10)
  --max-evaluations INT    Maximum number of evaluations to run (default: unlimited)
  --model STRING           LLM model to use (default: openrouter/google/gemini-2.0-flash-001)
  --log-file STRING        Log file path (default: evolution.log)
  --use-mock               Use mock LLM adapter for testing
  --seed INT               Random seed for reproducibility
```

## Architecture

The project follows a hexagonal (ports and adapters) architecture:

- **Domain**: Core entities and business logic
  - `model.py`: Agent and Chromosome classes
  - `services.py`: Evolution services like parent selection and mating

- **Ports**: Interface definitions
  - `primary.py`: Use case interfaces
  - `secondary.py`: External system interfaces

- **Adapters**: Implementation of interfaces
  - Primary: CLI interface
  - Secondary: LLM, logging, and statistics implementations

## Development

```bash
# Run tests
python -m pytest

# Run a specific test
python -m pytest tests/test_domain.py

# Run with coverage
python -m pytest --cov=llm_agent_evolution
```

## License

MIT
