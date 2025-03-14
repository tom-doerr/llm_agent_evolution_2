# LLM Agent Evolution

A framework for evolving LLM-based agents through evolutionary algorithms with real-time monitoring.

## Overview

This project implements an evolutionary algorithm for LLM-based agents. The system evolves agents with three chromosomes:
- **Task chromosome**: The output that gets evaluated for fitness
- **Mate selection chromosome**: Instructions for selecting mates
- **Mutation chromosome**: Instructions for how to mutate chromosomes

The system uses a continuous evolution process rather than discrete generations, with parent selection using a Pareto distribution weighted by fitness squared.

## Key Features

- **Hexagonal Architecture**: Clean separation of concerns with domain logic isolated from external systems
- **Continuous Evolution**: No discrete generations, population evolves continuously
- **Multithreading Support**: Run multiple evolution threads in parallel
- **Real-time Monitoring**: Live dashboard for tracking evolution progress
- **Visualization Tools**: Generate charts and graphs of evolution metrics
- **Mock LLM Support**: Test without making real API calls

## Installation

### Prerequisites

- Python 3.8+
- pip

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llm_agent_evolution.git
cd llm_agent_evolution

# Install the package in development mode
pip install -e .

# Install with test dependencies
pip install -e ".[test]"

# Install visualization and dashboard dependencies
pip install matplotlib streamlit pandas
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

### Real-time Monitoring Dashboard

Launch the Streamlit dashboard to monitor evolution in real-time:

```bash
./run_streamlit.sh
```

This will start the dashboard on port 8765. Open your browser to http://localhost:8765 to view it.

The dashboard automatically refreshes and provides:
- Current population statistics
- Evolution charts (rewards over time, distribution, top agents)
- Best agent details
- Live updates as evolution progresses

### Visualization Tools

Generate static visualizations from log files:

```bash
./visualization_cli.py --log-file evolution.log
```

## How the Optimization Works

The system optimizes agents through an evolutionary process:

1. **Initialization**: Create a population of agents with empty chromosomes
2. **Evaluation**: Evaluate each agent's task chromosome to get a reward
3. **Parent Selection**: Select parents using Pareto distribution weighted by fitness^2
4. **Mating**: Combine chromosomes from parents at hotspots (punctuation, spaces)
5. **Mutation**: Use the agent's mutation chromosome as instructions for the LLM to modify it
6. **Population Management**: Add new agents to population, removing worst if size limit reached

The current implementation optimizes for a specific hidden goal: maximizing the number of 'a' characters in the first 23 positions, with penalties for exceeding 23 characters.

## Customizing for Other Optimization Goals

To use this framework for different optimization goals:

1. **Create a Custom LLM Adapter**:
   - Subclass `LLMPort` and implement the `evaluate_task_output` method
   - Define your own reward function based on your optimization criteria

```python
class CustomLLMAdapter(LLMPort):
    def evaluate_task_output(self, output: str) -> float:
        # Your custom evaluation logic here
        # Return a float representing the reward
        pass
```

2. **Wire Your Adapter**:
   - Modify `create_application()` in `application.py` to use your adapter
   - Or create a new runner script that uses your adapter

3. **Configure Evolution Parameters**:
   - Adjust population size, parallel agents, and other parameters
   - Consider the complexity of your optimization goal

## Command Line Options

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
  --no-visualization       Disable visualization generation
```

## Architecture

The project follows a hexagonal (ports and adapters) architecture:

- **Domain**: Core entities and business logic
  - `model.py`: Agent and Chromosome classes
  - `services.py`: Evolution services like parent selection and mating

- **Ports**: Interface definitions
  - `primary.py`: Use case interfaces (evolution)
  - `secondary.py`: External system interfaces (LLM, logging, statistics)

- **Adapters**: Implementation of interfaces
  - Primary: CLI interface
  - Secondary: LLM, logging, statistics, and visualization implementations

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
